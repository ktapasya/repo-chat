"""Vector search and retrieval for repo-chat."""

import re
from typing import List, Optional, Tuple

import numpy as np

from .embed import Embedder
from .models import CodeChunk, Symbol
from .storage import Storage


def _normalize_identifiers(text: str) -> str:
    """Normalize code identifiers for better embedding similarity.

    Transforms code-specific patterns into natural language:
    - DEFAULT_MODEL -> "default model"
    - LiquidAI/LFM2.5 -> "liquidai lfm 2 5"
    - _private_func -> "private func"

    Args:
        text: Raw code text.

    Returns:
        Normalized text with identifiers split into natural tokens.
    """
    # Replace underscores and slashes with spaces
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    # Split camelCase and PascalCase
    # Insert space before uppercase letters that follow lowercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split numbers from letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # Remove special characters except alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace and lowercase
    text = ' '.join(text.split())
    return text.lower()


class Retriever:
    """Retrieves relevant code chunks using vector search and symbol lookup."""

    def __init__(self, repo_root: str):
        """Initialize the retriever.

        Args:
            repo_root: Path to the repository root.
        """
        self.repo_root = repo_root
        self.storage = Storage(repo_root)
        self.embedder = Embedder()

    def retrieve(self, query: str, top_k: int = 6) -> List[CodeChunk]:
        """Retrieve relevant code chunks for a query.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of relevant CodeChunks sorted by relevance.
        """
        # First check if query references a specific symbol
        symbol = self._extract_symbol(query)
        if symbol:
            symbol_data = self.storage.get_symbol(symbol)
            if symbol_data:
                # Fetch the specific code block for this symbol
                return self._fetch_symbol_chunk(symbol_data)

        # Fall back to vector search
        return self._vector_search(query, top_k)

    def _extract_symbol(self, query: str) -> Optional[str]:
        """Extract a potential symbol name from the query.

        Simple heuristic: look for identifiers like "login_user" or "AuthService"

        Args:
            query: User's query.

        Returns:
            Symbol name if found, None otherwise.
        """
        # Common patterns in questions
        # "Where is X defined?" -> X is the symbol
        # "How does X work?" -> X is the symbol
        # "Show me X" -> X is the symbol

        # Look for alphanumeric identifiers (CamelCase or snake_case)
        import re

        # Pattern for function names (snake_case)
        func_pattern = r'\b[a-z_][a-z0-9_]{2,}\b'
        # Pattern for class names (CamelCase)
        class_pattern = r'\b[A-Z][a-zA-Z0-9]{2,}\b'

        # Check for function names
        func_matches = re.findall(func_pattern, query)
        if func_matches:
            # Check if any of these exist in the symbol table
            for match in func_matches:
                if self.storage.get_symbol(match):
                    return match

        # Check for class names
        class_matches = re.findall(class_pattern, query)
        if class_matches:
            for match in class_matches:
                if self.storage.get_symbol(match):
                    return match

        return None

    def _fetch_symbol_chunk(self, symbol: Symbol) -> List[CodeChunk]:
        """Fetch code chunks for a specific symbol.

        Args:
            symbol: Symbol to fetch code for.

        Returns:
            List of CodeChunks containing the symbol definition.
        """
        # Find chunks that overlap with the symbol's line range
        all_chunks = self.storage.get_all_chunks()

        matching_chunks = []
        for chunk in all_chunks:
            if chunk.file_path == symbol.file_path:
                # Check if chunk overlaps with symbol range
                if not (chunk.line_end < symbol.line_start or
                        chunk.line_start > symbol.line_end):
                    matching_chunks.append(chunk)

        return matching_chunks

    def _token_overlap_score(self, query: str, content: str) -> float:
        """Calculate token overlap score between query and content.

        Args:
            query: User's query.
            content: Code chunk content.

        Returns:
            Overlap score between 0 and 1.
        """
        # Simple tokenization: split on non-alphanumeric characters
        import re
        query_tokens = set(re.findall(r'\w+', query.lower()))
        content_tokens = set(re.findall(r'\w+', content.lower()))

        if not query_tokens:
            return 0.0

        # Jaccard-like overlap: intersection / union
        intersection = len(query_tokens & content_tokens)
        union = len(query_tokens | content_tokens)

        return intersection / union if union > 0 else 0.0

    def _vector_search(self, query: str, top_k: int) -> List[CodeChunk]:
        """Perform hybrid search using vector similarity and token overlap.

        Args:
            query: User's query.
            top_k: Number of top results to return.

        Returns:
            List of CodeChunks sorted by relevance.
        """
        # Get all chunks with embeddings
        all_chunks = self.storage.get_all_chunks()

        # Filter out chunks without embeddings
        chunks_with_embeddings = [
            chunk for chunk in all_chunks
            if chunk.embedding is not None
        ]

        if not chunks_with_embeddings:
            return []

        # Extract embeddings and chunk IDs
        embeddings_list = [chunk.embedding for chunk in chunks_with_embeddings]

        # Convert bytes to numpy array
        embeddings_matrix = Embedder.bytes_to_embeddings_list(embeddings_list)

        # Normalize query before embedding for better code identifier matching
        normalized_query = _normalize_identifiers(query)
        query_embedding = self.embedder.embed(normalized_query)
        query_vector = Embedder.bytes_to_embedding(query_embedding)

        # Compute cosine similarity
        # Cosine similarity = dot product of normalized vectors
        # Or equivalently: (A . B) / (||A|| * ||B||)

        # Normalize vectors
        norms = np.linalg.norm(embeddings_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)

        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        query_norm = query_norm if query_norm != 0 else 1

        # Compute vector similarities
        vector_similarities = np.dot(embeddings_matrix, query_vector) / (norms * query_norm)

        # Compute token overlap scores for all chunks
        token_scores = np.array([
            self._token_overlap_score(query, chunk.content)
            for chunk in chunks_with_embeddings
        ])

        # Hybrid scoring: 70% vector + 30% lexical
        # Weights can be tuned
        hybrid_scores = 0.7 * vector_similarities + 0.3 * token_scores

        # Get top-k indices by hybrid score
        top_k_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        results = [chunks_with_embeddings[idx] for idx in top_k_indices]

        return results

    def get_context(self, query: str, top_k: int = 6) -> str:
        """Assemble context from retrieved chunks.

        Args:
            query: User's query.
            top_k: Number of chunks to retrieve.

        Returns:
            Formatted context string with code chunks and file references.
        """
        chunks = self.retrieve(query, top_k)
        print("Inside search.py")
        for c in chunks:
            print("----")
            print(c.file_path, c.line_start, c.line_end)
            print(c.content[:400])

        if not chunks:
            return ""

        context_parts = []

        for chunk in chunks:
            # Format: File path and line range
            ref = f"{chunk.file_path}:{chunk.line_start}-{chunk.line_end}"
            context_parts.append(f"### {ref}\n")
            context_parts.append(chunk.content)
            context_parts.append("\n")

        return "\n".join(context_parts)
