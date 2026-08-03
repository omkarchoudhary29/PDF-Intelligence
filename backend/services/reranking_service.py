import logging
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("ekn.reranking")


class RerankingService:
    """Manages Cross-Encoder reranking for hybrid retrieval candidates.

    Loads the configured cross-encoder model once per process and re-scores
    the fused RRF candidate chunks using query-passage cross-attention.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.cross_encoder_model
        self.device = self.settings.cross_encoder_device
        self.batch_size = self.settings.cross_encoder_batch_size
        self._model: Optional[Any] = None

    def get_model(self) -> Any:
        """Lazily initialize and return the CrossEncoder singleton."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading CrossEncoder model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            try:
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name, device=self.device)

            logger.info("CrossEncoder model '%s' initialized successfully", self.model_name)
            return self._model
        except Exception as exc:
            logger.error("Failed to load CrossEncoder '%s': %s", self.model_name, exc)
            raise RuntimeError(f"CrossEncoder initialization failed: {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Re-score candidate chunks using Cross-Encoder attention.

        Args:
            query: The search question or keywords.
            candidates: List of chunk dictionaries containing 'text'.
            top_k: Number of highest-scoring candidates to return.

        Returns:
            Ranked list of candidates with 'cross_encoder_score' and 'final_rank' (1-based).
        """
        if not candidates:
            return []

        model = self.get_model()

        # Build query-passage pairs
        pairs = [(query, c.get("text", "")) for c in candidates]

        try:
            scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:
            logger.error("Cross-Encoder prediction failed: %s", exc)
            raise RuntimeError(f"Cross-Encoder reranking error: {exc}") from exc

        # Assign raw cross-encoder relevance scores
        for idx, candidate in enumerate(candidates):
            candidate["cross_encoder_score"] = float(scores[idx])

        # Sort descending by cross-encoder score
        candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        top_candidates = candidates[:top_k]

        # Assign 1-based final rank
        for rank, candidate in enumerate(top_candidates, start=1):
            candidate["final_rank"] = rank

        return top_candidates


reranking_service = RerankingService()
