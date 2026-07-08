from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Singleton getter for the sentence transformer embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model
