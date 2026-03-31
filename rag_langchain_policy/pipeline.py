"""High-level RAGPipeline: single entry point for query and chat usage."""

import logging

from .chains import build_rag_chain
from .config import RAGConfig
from .llm import build_llm
from .vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline with metadata-enriched Chroma index.

    Usage::

        config = RAGConfig()
        pipeline = RAGPipeline(config)
        result = pipeline.answer("How much jail time for financial fraud?")
        print(result["answer"])
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._vectorstore = load_vectorstore(config.persist_dir, config.embedding_model)
        self._llm = build_llm(config)
        self._chain = build_rag_chain(
            self._vectorstore, self._llm, top_k=config.top_k
        )
        logger.info("RAGPipeline ready.")

    def answer(self, query: str) -> dict:
        """Run the RAG pipeline for *query*.

        Returns a dict with:
            answer           – generated string
            retrieved_chunks – list of {rank, doc_id, chunk_index, score,
                               jurisdiction, effective_date, authority_rank,
                               preview, text}
            context_text     – the formatted CONTEXT block fed to the prompt
            final_prompt     – the complete prompt string (useful for debug)
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.info("Running query: %r", query)
        result = self._chain.invoke({"question": query})

        return {
            "answer": result.get("answer", ""),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "context_text": result.get("context_text", ""),
            "final_prompt": result.get("final_prompt", ""),
        }
