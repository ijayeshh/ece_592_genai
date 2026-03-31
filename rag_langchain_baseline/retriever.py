"""Retriever construction and result formatting."""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def build_retriever(vectorstore: Chroma, top_k: int = 5) -> BaseRetriever:
    """Return a similarity-search retriever for *top_k* chunks."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    logger.debug("Retriever built with top_k=%d.", top_k)
    return retriever


def retrieve_with_scores(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Return (Document, score) pairs for *query* using similarity search."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    return results


def format_retrieved_chunks(
    docs_and_scores: list[tuple[Document, float]],
) -> list[dict]:
    """Convert raw retrieval results into serialisable dicts.

    Each dict contains:
        rank          – 1-based rank
        doc_id        – source document name
        chunk_index   – chunk position within document
        score         – similarity score (higher = more similar)
        preview       – first 200 chars of chunk text
        text          – full chunk text
    """
    formatted = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        formatted.append(
            {
                "rank": rank,
                "doc_id": meta.get("doc_id", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "score": round(float(score), 4),
                "preview": doc.page_content[:200].strip(),
                "text": doc.page_content,
            }
        )
    return formatted
