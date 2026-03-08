"""Embedding generation for repo-chat."""

from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Generate embeddings for code chunks using sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the embedder with a sentence-transformers model.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> bytes:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding as float32 bytes for storage.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return self._to_bytes(embedding)

    def embed_batch(self, texts: List[str]) -> List[bytes]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings as float32 bytes.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return [self._to_bytes(emb) for emb in embeddings]

    def _to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert numpy embedding to float32 bytes for storage.

        Args:
            embedding: Numpy array embedding.

        Returns:
            Embedding as bytes.
        """
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(embedding_bytes: bytes) -> np.ndarray:
        """Convert float32 bytes back to numpy array.

        Args:
            embedding_bytes: Embedding as bytes.

        Returns:
            Numpy array embedding.
        """
        return np.frombuffer(embedding_bytes, dtype=np.float32)

    @staticmethod
    def bytes_to_embeddings_list(embeddings_bytes: List[bytes]) -> np.ndarray:
        """Convert list of embedding bytes to numpy matrix.

        Args:
            embeddings_bytes: List of embeddings as bytes.

        Returns:
            Numpy matrix of shape (n_embeddings, embedding_dim).
        """
        return np.array([
            np.frombuffer(emb, dtype=np.float32)
            for emb in embeddings_bytes
        ])
