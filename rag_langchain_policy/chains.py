"""LCEL chain: retrieve → policy filters → format context → prompt → LLM."""

import logging

from langchain_core.runnables import RunnableLambda

from .config import RAGConfig
from .policy_filters import apply_policy_filters
from .prompts import RAG_PROMPT, format_context
from .retriever import format_retrieved_chunks, retrieve_with_scores

logger = logging.getLogger(__name__)

# Over-retrieval multiplier: fetch this many times top_k from Chroma before
# filtering, so jurisdiction + dedup can discard candidates without leaving
# the final context short.
_OVER_RETRIEVE_FACTOR = 3


def build_rag_chain(vectorstore, llm, top_k: int = 5, config: RAGConfig | None = None):
    """Return an LCEL runnable that implements the full policy-aware RAG pipeline.

    Input:  {"question": str}
    Output: {"answer": str, "retrieved_chunks": list[dict],
             "context_text": str, "final_prompt": str}

    Pipeline stages:
      1. Over-retrieve top_k * 3 candidates by cosine similarity
      2. Apply policy filters (jurisdiction / authority rerank / temporal dedup)
      3. Format the surviving top_k chunks into a CONTEXT block
      4. Fill the RAG prompt template
      5. Call the LLM
    """
    # Use a no-op config if none provided (all filters disabled by default)
    if config is None:
        config = RAGConfig(
            top_k=top_k,
            jurisdiction_filter=None,
            apply_authority_rerank=False,
            apply_temporal_dedup=False,
        )

    fetch_k = top_k * _OVER_RETRIEVE_FACTOR

    def _retrieve_and_filter(inputs: dict) -> dict:
        question = inputs["question"]

        # Step 1: over-retrieve
        docs_and_scores = retrieve_with_scores(vectorstore, question, top_k=fetch_k)
        raw_chunks = format_retrieved_chunks(docs_and_scores)

        # Step 2: apply policy filters
        filtered_chunks = apply_policy_filters(
            chunks=raw_chunks,
            top_k=top_k,
            jurisdiction_filter=config.jurisdiction_filter,
            apply_authority_rerank=config.apply_authority_rerank,
            apply_temporal_dedup=config.apply_temporal_dedup,
        )

        context_text = format_context(filtered_chunks)
        return {
            "question": question,
            "context": context_text,
            "retrieved_chunks": filtered_chunks,
            "context_text": context_text,
        }

    def _build_prompt_str(inputs: dict) -> dict:
        prompt_str = RAG_PROMPT.format(
            context=inputs["context"],
            question=inputs["question"],
        )
        inputs["final_prompt"] = prompt_str
        return inputs

    def _generate(inputs: dict) -> dict:
        answer = llm.invoke(inputs["final_prompt"])
        if hasattr(answer, "content"):
            answer = answer.content
        inputs["answer"] = str(answer).strip()
        return inputs

    chain = (
        RunnableLambda(_retrieve_and_filter)
        | RunnableLambda(_build_prompt_str)
        | RunnableLambda(_generate)
    )

    logger.debug(
        "Policy RAG chain built (top_k=%d, fetch_k=%d, jurisdiction=%s, "
        "rerank=%s, dedup=%s).",
        top_k,
        fetch_k,
        config.jurisdiction_filter,
        config.apply_authority_rerank,
        config.apply_temporal_dedup,
    )
    return chain
