import logging
import threading
from typing import Any, Dict, List, Optional
import numpy as np

from config.settings import get_settings

logger = logging.getLogger("ekn.embeddings")


class EmbeddingService:
    """Manages local Sentence Transformer embedding generation and token-aware utilities.

    Loads the configured embedding model once per worker process and reuses it across documents.
    Defaults to CPU execution for broad compatibility.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.embedding_model
        self.device = self.settings.embedding_device
        self.default_batch_size = self.settings.embedding_batch_size
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._dimension: Optional[int] = None
        self._query_cache: Dict[str, List[float]] = {}
        self._query_cache_lock = threading.Lock()

    def get_model(self) -> Any:
        """Retrieve or lazily initialize the SentenceTransformer model singleton."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading SentenceTransformer model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            # Try loading cached local files first to avoid remote network latency
            try:
                model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                model = SentenceTransformer(self.model_name, device=self.device)

            self._model = model
            if self._tokenizer is None:
                self._tokenizer = getattr(model, "tokenizer", None)
            if self._tokenizer is not None:
                try:
                    self._tokenizer.model_max_length = int(1e9)
                except Exception:
                    pass

            # Resolve embedding dimension
            if hasattr(model, "get_embedding_dimension"):
                self._dimension = model.get_embedding_dimension()
            elif hasattr(model, "get_sentence_embedding_dimension"):
                self._dimension = model.get_sentence_embedding_dimension()
            else:
                self._dimension = self.settings.embedding_dimension

            logger.info(
                "Embedding model '%s' loaded successfully (dim=%d, max_seq_length=%d)",
                self.model_name,
                self._dimension,
                getattr(model, "max_seq_length", 256),
            )
            return self._model
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
            raise RuntimeError(f"Embedding model initialization failed: {exc}") from exc

    def get_tokenizer(self) -> Any:
        """Retrieve the HuggingFace tokenizer associated with the model.

        Loads fast local tokenizer directly for sub-millisecond token counting without full PyTorch weight loading.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        try:
            from transformers import AutoTokenizer

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer is not None:
                try:
                    self._tokenizer.model_max_length = int(1e9)
                except Exception:
                    pass
            return self._tokenizer
        except Exception as exc:
            logger.debug("Fast AutoTokenizer load skipped: %s. Falling back to full model load.", exc)
            self.get_model()
            return self._tokenizer

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        if self._dimension is None:
            if self._model is not None:
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = self._model.get_sentence_embedding_dimension()
            if self._dimension is None:
                self._dimension = self.settings.embedding_dimension
        return self._dimension

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using the model's tokenizer without special tokens."""
        if not text or not text.strip():
            return 0
        tokenizer = self.get_tokenizer()
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as exc:
            logger.warning("Tokenizer failed for text snippet: %s. Falling back to word estimate.", exc)
            return len(text.split())

    def encode_tokens(self, text: str) -> List[int]:
        """Encode text into a list of token IDs (without special tokens)."""
        tokenizer = self.get_tokenizer()
        return tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into clean text."""
        tokenizer = self.get_tokenizer()
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Generate normalized cosine-ready embedding vectors in configurable batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Optional batch size override; defaults to settings.embedding_batch_size.

        Returns:
            List of embedding vectors as Python float lists.
        """
        if not texts:
            return []

        model = self.get_model()
        effective_batch_size = batch_size or self.default_batch_size

        try:
            embeddings = model.encode(
                texts,
                batch_size=effective_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Ensure 2D numpy array
            if len(embeddings.shape) == 1:
                embeddings = np.expand_dims(embeddings, axis=0)

            # Validate dimensions
            expected_dim = self.dimension
            if embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Generated embedding dimension {embeddings.shape[1]} does not match expected {expected_dim}"
                )

            return embeddings.tolist()
        except Exception as exc:
            logger.error("Failed during batch embedding generation for %d items: %s", len(texts), exc)
            raise RuntimeError(f"Embedding computation error: {exc}") from exc

    def embed_query(self, query: str) -> List[float]:
        """Generate a single normalized query embedding vector with bounded in-memory caching."""
        if not query or not query.strip():
            return [0.0] * self.dimension

        cache_key = query.strip().lower()
        with self._query_cache_lock:
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

        vector = self.generate_embeddings([query])[0]

        with self._query_cache_lock:
            if len(self._query_cache) > 512:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[cache_key] = vector

        return vector

    def clear_query_cache(self) -> None:
        """Clear the in-memory query embedding cache."""
        with self._query_cache_lock:
            self._query_cache.clear()

    def set_mock_model(self, mock_model: Any, dimension: int = 384) -> None:
        """Testing hook to inject a mock embedding model for isolated unit tests."""
        self._model = mock_model
        self._tokenizer = getattr(mock_model, "tokenizer", None)
        self._dimension = dimension


# Global singleton instance
embedding_service = EmbeddingService()
