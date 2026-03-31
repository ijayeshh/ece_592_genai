"""RAG with metadata-enriched indexing (policy layer groundwork)."""

from .config import RAGConfig
from .pipeline import RAGPipeline

__all__ = ["RAGConfig", "RAGPipeline"]
