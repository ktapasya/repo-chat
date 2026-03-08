"""Vector search for repo-chat."""

from typing import List
import numpy as np

from .storage import Storage
from .embed import Embedder
from .models import Chunk, SearchResult


class Search:
    """Retrieve relevant chunks using vector similarity."""

    def __init__(self, repo_root: str):
        """Initialize the search engine.

        Args:
            repo_root: Path to the repository root.
        """
        self.storage = Storage(repo_root)
        self.embedder = Embedder()

    def search(self, query: str, top_k: int = 6) -> List[SearchResult]:
        """Return top-k relevant chunks for a query.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by relevance (highest first).
        """
        # Step 1: Vector search
        vector_results = self._vector_search(query, top_k)

        # Store query embedding for computing neighbor scores
        query_embedding_bytes = self.embedder.embed(query)
        self._query_embedding = Embedder.bytes_to_embedding(query_embedding_bytes)

        # Step 2: Expand graph from vector results (with neighbor tracking)
        graph_chunks_with_source = self._expand_graph(vector_results)

        # Step 3: Filter neighbors by score (keep top 5 per source chunk)
        filtered_graph_chunks = self._filter_neighbors_by_score(graph_chunks_with_source, top_k_neighbors=5)

        # Step 4: Merge and deduplicate
        merged = self._merge_results(vector_results, filtered_graph_chunks)

        # DEBUG: Print retrieval structure
        self._print_debug_output(vector_results, graph_chunks_with_source)

        # Step 5: Return merged results
        return merged

    def _format_neighbor(self, chunk: Chunk, source_chunk_id: int) -> str:
        """Format neighbor chunk for debug output."""
        return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"

    def _filter_neighbors_by_score(
        self,
        graph_chunks_with_source: List[tuple[Chunk, int]],
        top_k_neighbors: int = 5
    ) -> List[Chunk]:
        """Filter neighbor chunks by semantic score, keeping top-K per source chunk.

        Args:
            graph_chunks_with_source: List of (neighbor_chunk, source_chunk_id) tuples.
            top_k_neighbors: Number of top neighbors to keep per source chunk.

        Returns:
            List of filtered neighbor chunks.
        """
        # Group neighbors by source chunk
        neighbors_by_source = {}
        for chunk, source_id in graph_chunks_with_source:
            if source_id not in neighbors_by_source:
                neighbors_by_source[source_id] = []
            neighbors_by_source[source_id].append(chunk)

        # For each source chunk, keep top-K neighbors by score
        filtered_chunks = []
        for source_id, neighbors in neighbors_by_source.items():
            # Compute scores for all neighbors of this source
            scored_neighbors = []
            for nc in neighbors:
                if nc.embedding:
                    neighbor_emb = Embedder.bytes_to_embedding(nc.embedding)
                    score = float(self._query_embedding @ neighbor_emb)
                    scored_neighbors.append((nc, score))

            # Sort by score and keep top-K
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk for chunk, score in scored_neighbors[:top_k_neighbors]]
            filtered_chunks.extend(top_chunks)

        return filtered_chunks

    def _print_debug_output(
        self,
        vector_results: List[SearchResult],
        graph_chunks_with_source: List[tuple[Chunk, int]]
    ) -> None:
        """Print debug output showing retrieval structure.

        Args:
            vector_results: Vector search results.
            graph_chunks_with_source: Graph-expanded chunks with source tracking.
        """
        print("\n===== RETRIEVAL DEBUG =====")
        for i, vr in enumerate(vector_results):
            print(f"\n[{i}] {vr.chunk.file_path}:{vr.chunk.start_line}-{vr.chunk.end_line} (score={vr.score:.3f})")

            # Find nodes in this chunk
            nodes = self.storage.get_nodes_in_chunk(
                vr.chunk.file_path,
                vr.chunk.start_line,
                vr.chunk.end_line
            )

            # Find neighbors from this chunk's nodes with their scores
            chunk_neighbors = []
            for gc, source_chunk_id in graph_chunks_with_source:
                if gc.id != vr.chunk.id and source_chunk_id == vr.chunk.id:
                    # Compute real similarity score for neighbor
                    if gc.embedding:
                        neighbor_emb = Embedder.bytes_to_embedding(gc.embedding)
                        score = float(self._query_embedding @ neighbor_emb)
                    else:
                        score = 0.0
                    chunk_neighbors.append((gc, score))

            # Sort neighbors by score and print top 5
            chunk_neighbors.sort(key=lambda x: x[1], reverse=True)
            for neighbor, score in chunk_neighbors[:5]:
                print(f"  - {self._format_neighbor(neighbor, vr.chunk.id)} (score={score:.3f})")

            if not chunk_neighbors:
                print(f"  (no neighbors - {len(nodes)} nodes in chunk)")
        print("========================\n")

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform pure vector similarity search.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects from vector search.
        """
        # Load chunks with embeddings
        chunks = self.storage.get_chunks_with_embeddings()

        if not chunks:
            return []

        # Convert embeddings to numpy matrix
        embeddings = Embedder.bytes_to_embeddings_list([
            chunk.embedding for chunk in chunks
        ])

        # Embed query (convert bytes to numpy for similarity)
        query_bytes = self.embedder.embed(query)
        query_vec = Embedder.bytes_to_embedding(query_bytes)

        # Compute cosine similarity
        # Note: embeddings are already normalized by embedder
        scores = embeddings @ query_vec

        # Get top-k indices (sorted by score, descending)
        top_k = min(top_k, len(chunks))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Build results
        results = [
            SearchResult(chunk=chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

        return results

    def _expand_graph(self, results: List[SearchResult]) -> List[tuple[Chunk, int]]:
        """Expand search results using graph traversal and track source chunks.

        Args:
            results: Initial vector search results.

        Returns:
            List of (neighbor_chunk, source_chunk_id) tuples.
        """
        additional_chunks = []
        visited_nodes = set()

        for result in results:
            chunk = result.chunk

            # Find nodes in this chunk
            nodes = self.storage.get_nodes_in_chunk(
                chunk.file_path,
                chunk.start_line,
                chunk.end_line
            )

            # Expand to ALL neighbors (no arbitrary limit)
            for node in nodes:
                if node.id in visited_nodes:
                    continue
                visited_nodes.add(node.id)

                neighbors = self.storage.get_neighbors(node.id)

                # Get chunks for neighbor nodes
                neighbor_chunks = self.storage.get_chunks_for_nodes(neighbors)
                for nc in neighbor_chunks:
                    additional_chunks.append((nc, chunk.id))

        return additional_chunks

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        graph_chunks: List[Chunk]
    ) -> List[SearchResult]:
        """Merge vector and graph results, deduplicating by chunk ID.

        Args:
            vector_results: Results from vector search.
            graph_chunks: Additional chunks from graph expansion.

        Returns:
            Merged and deduplicated results.
        """
        # Start with vector results
        seen_ids = {r.chunk.id for r in vector_results}
        merged = list(vector_results)

        # Add graph chunks (give small score boost)
        for chunk in graph_chunks:
            if chunk.id not in seen_ids:
                merged.append(SearchResult(chunk=chunk, score=0.1))
                seen_ids.add(chunk.id)

        # Sort by score
        merged.sort(key=lambda r: r.score, reverse=True)

        return merged

    def close(self):
        """Close the storage connection."""
        self.storage.close()
