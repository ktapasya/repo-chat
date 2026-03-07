"""Embedding generation for code chunks using sentence-transformers."""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class Embedder:
    """Handles embedding generation for code chunks.

    Uses sentence-transformers with models auto-downloaded from HuggingFace.
    Models are cached locally in ~/.cache/repochat/models for fast reuse.
    """

    # Recommended models with ~384 dim for fast local inference
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    ALTERNATIVE_MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model name. Defaults to BAAI/bge-small-en-v1.5.
            cache_dir: Directory to cache downloaded models.
                      Defaults to ~/.cache/repochat/models.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/repochat/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._embedding_dim = None

    @property
    def model(self):
        """Lazy-load the embedding model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        if self._embedding_dim is None:
            # Trigger model load to get dimension
            _ = self.model
        return self._embedding_dim

    def _load_model(self) -> None:
        """Load the sentence-transformers model with caching.

        Downloads from HuggingFace on first run, caches locally.
        Raises RuntimeError if model loading fails.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from e

        try:
            # Load with local caching and trust remote code for nomic models
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                trust_remote_code=True,
            )
            # Get embedding dimension by encoding a dummy text
            test_emb = self._model.encode("test", show_progress_bar=False)
            self._embedding_dim = len(test_emb)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {e}"
            ) from e

    def embed(self, text: str) -> bytes:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding as bytes (float32 numpy array serialized).
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Ensure float32 for consistent storage
        return embedding.astype(np.float32).tobytes()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[bytes]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            List of embeddings as bytes (float32 numpy arrays serialized).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Convert each embedding to bytes
        return [
            emb.astype(np.float32).tobytes()
            for emb in embeddings
        ]

    @staticmethod
    def bytes_to_embedding(data: bytes) -> np.ndarray:
        """Convert bytes back to numpy array.

        Args:
            data: Bytes from storage.

        Returns:
            numpy array of float32 embeddings.
        """
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def bytes_to_embeddings_list(data_list: List[bytes]) -> np.ndarray:
        """Convert list of bytes back to 2D numpy array.

        Useful for batch operations like cosine similarity.

        Args:
            data_list: List of byte strings from storage.

        Returns:
            2D numpy array of shape (n, embedding_dim).
        """
        if not data_list:
            return np.array([]).reshape(0, -1)

        return np.vstack([
            np.frombuffer(data, dtype=np.float32)
            for data in data_list
        ])
