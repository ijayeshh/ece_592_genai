"""LCEL chain: retrieve → format context → prompt → LLM → string output."""

import logging

from langchain_core.runnables import RunnableLambda

from .prompts import RAG_PROMPT, format_context
from .retriever import format_retrieved_chunks, retrieve_with_scores

logger = logging.getLogger(__name__)


def build_rag_chain(vectorstore, llm, top_k: int = 5):
    """Return an LCEL runnable that implements the full RAG pipeline.

    Input:  {"question": str}
    Output: {"answer": str, "retrieved_chunks": list[dict],
             "context_text": str, "final_prompt": str}

    The chain is intentionally kept as pure similarity top-k retrieval → generate
    (no reranking, no metadata filtering).
    """
    def _retrieve_and_format(inputs: dict) -> dict:
        question = inputs["question"]
        docs_and_scores = retrieve_with_scores(vectorstore, question, top_k=top_k)
        retrieved_chunks = format_retrieved_chunks(docs_and_scores)
        context_text = format_context(retrieved_chunks)
        return {
            "question": question,
            "context": context_text,
            "retrieved_chunks": retrieved_chunks,
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
        # StrOutputParser already returns a str; handle both str and AIMessage
        if hasattr(answer, "content"):
            answer = answer.content
        inputs["answer"] = str(answer).strip()
        return inputs

    chain = (
        RunnableLambda(_retrieve_and_format)
        | RunnableLambda(_build_prompt_str)
        | RunnableLambda(_generate)
    )

    logger.debug("RAG LCEL chain built.")
    return chain
