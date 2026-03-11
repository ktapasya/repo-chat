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
        # Embed query once for both vector search and neighbor scoring
        query_embedding_bytes = self.embedder.embed(query)
        query_emb = Embedder.bytes_to_embedding(query_embedding_bytes)

        # Step 1: Vector search
        vector_results = self._vector_search(query_emb, top_k)

        # Step 2: Expand graph from vector results (load, calc scores, filter, return SearchResult)
        graph_results_with_source = self._expand_graph(query_emb, vector_results, top_k_neighbors=5)

        # DEBUG: Print retrieval structure
        self._print_debug_output(vector_results, graph_results_with_source)

        # Step 3: Merge and deduplicate
        merged = self._merge_results(vector_results, graph_results_with_source)

        # Step 4: Return merged results
        return merged

    def _print_debug_output(
        self,
        vector_results: List[SearchResult],
        graph_results_with_source: List[tuple[SearchResult, int]]
    ) -> None:
        """Print debug output showing retrieval structure.

        Args:
            vector_results: Vector search results.
            graph_results_with_source: Graph-expanded SearchResults with source tracking.
        """
        print("\n===== RETRIEVAL DEBUG =====")
        for i, vr in enumerate(vector_results):
            print(f"\n[{i}] {vr.chunk.file_path}:{vr.chunk.start_line}-{vr.chunk.end_line} (score={vr.score:.3f})")

            # Find neighbors from this chunk
            chunk_neighbors = [
                (result.chunk, result.score)
                for result, source_id in graph_results_with_source
                if source_id == vr.chunk.id and result.chunk.id != vr.chunk.id
            ]

            # Sort by score and print top 5
            chunk_neighbors.sort(key=lambda x: x[1], reverse=True)
            for neighbor, score in chunk_neighbors[:5]:
                print(f"  - {neighbor.file_path}:{neighbor.start_line}-{neighbor.end_line} (score={score:.3f})")

            if not chunk_neighbors:
                print(f"  (no neighbors)")
        print("========================\n")

    def _vector_search(self, query_emb: np.ndarray, top_k: int) -> List[SearchResult]:
        """Perform pure vector similarity search.

        Args:
            query_emb: Query embedding as numpy array.
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

        # Compute cosine similarity
        # Note: embeddings are already normalized by embedder
        scores = embeddings @ query_emb

        # Get top-k indices (sorted by score, descending)
        top_k = min(top_k, len(chunks))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Build results
        return [
            SearchResult(chunk=chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def _expand_graph(
        self,
        query_emb: np.ndarray,
        results: List[SearchResult],
        top_k_neighbors: int = 5
    ) -> List[tuple[SearchResult, int]]:
        """Expand search results using graph traversal.

        Args:
            query_emb: Query embedding for scoring neighbors.
            results: Initial vector search results.
            top_k_neighbors: Number of top neighbors to keep per source chunk.

        Returns:
            List of (SearchResult, source_chunk_id) tuples, grouped by source.
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

            # Load neighbor chunks via graph traversal
            neighbor_chunks = []
            for node in nodes:
                if node.id in visited_nodes:
                    continue
                visited_nodes.add(node.id)

                neighbors = self.storage.get_neighbors(node.id)
                neighbor_chunks.extend(
                    self.storage.get_chunks_for_nodes(neighbors)
                )

            # Deduplicate neighbors by chunk ID
            seen_ids = {chunk.id}
            unique_neighbors = []
            for nc in neighbor_chunks:
                if nc.id not in seen_ids:
                    unique_neighbors.append(nc)
                    seen_ids.add(nc.id)

            # Calc scores for all neighbors
            scored_neighbors = []
            for nc in unique_neighbors:
                if nc.embedding:
                    neighbor_emb = Embedder.bytes_to_embedding(nc.embedding)
                    score = float(query_emb @ neighbor_emb)
                    scored_neighbors.append((nc, score))

            # Filter to top-K per source chunk
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = scored_neighbors[:top_k_neighbors]

            # Return SearchResult objects grouped by source
            for nc, score in top_neighbors:
                additional_chunks.append(
                    (SearchResult(chunk=nc, score=score), chunk.id)
                )

        return additional_chunks

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        graph_results_with_source: List[tuple[SearchResult, int]]
    ) -> List[SearchResult]:
        """Merge vector and graph results, deduplicating by chunk ID.

        Args:
            vector_results: Results from vector search.
            graph_results_with_source: Additional SearchResults from graph expansion.

        Returns:
            Merged and deduplicated results.
        """
        # Start with vector results
        seen_ids = {r.chunk.id for r in vector_results}
        merged = list(vector_results)

        # Add graph results (already have scores)
        for result, _ in graph_results_with_source:
            if result.chunk.id not in seen_ids:
                merged.append(result)
                seen_ids.add(result.chunk.id)

        # Sort by score
        merged.sort(key=lambda r: r.score, reverse=True)

        return merged

    def close(self):
        """Close the storage connection."""
        self.storage.close()
