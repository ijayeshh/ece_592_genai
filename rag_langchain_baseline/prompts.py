"""Prompt templates for the RAG baseline."""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(retrieved_chunks: list[dict]) -> str:
    """Build the CONTEXT block from retrieved chunk dicts.

    Vanilla baseline: feed ONLY raw text to the LLM (no doc IDs),
    so the model won't copy filenames/chunk IDs into the answer.
    """
    return "\n\n".join(chunk["text"].strip() for chunk in retrieved_chunks)


# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
You are a precise legal-document assistant. Use ONLY the provided CONTEXT below to answer the question.
Do NOT use any prior knowledge outside the context.
Do not cite sources or mention filenames/chunk IDs.
If the answer is not present in the context, respond with exactly:
  I don't know based on the provided documents.


CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)
