# Legal-Domain RAG System using LangChain

This repository implements a retrieval-augmented generation (RAG) system over a
synthetic U.S. legal corpus. The work is structured in two stages:

1. **Vanilla RAG baseline** — pure similarity top-k retrieval → generation, no
   metadata awareness.
2. **Metadata-enriched policy layer** — same retrieval pipeline, but the Chroma
   index is enriched with document-level metadata (`jurisdiction`, `effective_date`,
   `authority_rank`). Every retrieved chunk exposes these fields, enabling
   jurisdiction filtering, recency-based selection, and authority-rank prioritisation
   before the context is passed to the LLM.

Both stages are fully implemented and runnable.

---

## Repository structure

```
ece_592_genai/
│
├── synthetic_data_legal/               # Corpus (16 synthetic .txt files)
│   └── metadata_manifest.csv          # Doc-level metadata for policy layer
│
├── rag_langchain_baseline/             # Stage 1: vanilla RAG package
│   ├── config.py                       # RAGConfig dataclass
│   ├── loaders.py                      # DirectoryLoader + TextLoader
│   ├── splitting.py                    # RecursiveCharacterTextSplitter
│   ├── vectorstore.py                  # Chroma build/load helpers
│   ├── retriever.py                    # Similarity search + result formatter
│   ├── llm.py                          # Groq (default) or HuggingFace backend
│   ├── prompts.py                      # RAG PromptTemplate
│   ├── chains.py                       # LCEL chain (retrieve→prompt→generate)
│   ├── pipeline.py                     # RAGPipeline entry point
│   └── _env.py                         # .env loader + HF token login
│
├── scripts/                            # Stage 1 CLI scripts
│   ├── build_index.py                  # Embed + persist Chroma index
│   ├── query.py                        # One-shot query runner
│   └── chat.py                         # Interactive REPL
│
├── rag_langchain_policy/               # Stage 2: policy-layer package
│   ├── config.py                       # RAGConfig with metadata_path added
│   ├── loaders.py                      # Adds manifest injection to loader
│   ├── splitting.py                    # Adds post-split metadata validation
│   ├── retriever.py                    # Result dicts include policy metadata fields
│   ├── chains.py                       # LCEL chain (retrieve→prompt→generate)
│   ├── llm.py                          # Groq (default) or HuggingFace backend
│   ├── prompts.py                      # RAG PromptTemplate
│   ├── pipeline.py                     # RAGPipeline entry point
│   ├── vectorstore.py                  # Chroma build/load helpers
│   └── _env.py                         # .env loader + HF token login
│
├── scripts_policy_layer/               # Stage 2 CLI scripts
│   ├── build_index.py                  # Adds --metadata_path arg
│   ├── query.py                        # Debug output includes policy metadata
│   ├── chat.py                         # Interactive REPL with policy index
│   └── test_index.py                   # Sanity-checks the policy Chroma index
│
├── .env                                # Secrets (not committed)
├── .env.example                        # Template for .env
├── .gitignore
└── pyproject.toml                      # Project dependencies (uv)
```

---

## Dataset

### Synthetic Dataset generated using https://chatgpt.com/. Model: ChatGPT 5.2.

`synthetic_data_legal/` contains 16 short synthetic documents representing
U.S. federal and California state legal texts across three crime categories:

| Prefix | Jurisdiction | Document types |
|--------|-------------|----------------|
| `USFED_` | Federal (`US-FED`) | Statutes, DOJ Guidance, Public FAQ |
| `USCA_`  | California (`US-CA`) | Statutes, AG Bulletin, Public Handbook |

Crime categories: Financial Fraud, Identity Theft, Benefits Fraud.

The documents are intentionally written to have **conflicting penalties across
years and jurisdictions** this is the core challenge the system is designed
to handle correctly.

### `metadata_manifest.csv`

Each document has a corresponding row in `synthetic_data_legal/metadata_manifest.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | string | Filename, e.g. `USFED_FinancialFraud_Statute_2018.txt` |
| `jurisdiction` | string | `US-FED` or `US-CA` |
| `effective_date` | ISO date | Date the document came into effect |
| `authority_rank` | int | `2` = guidance/bulletin/FAQ, `3` = statute |

---

## Installation

### Prerequisites

- Python 3.13+
- For Groq (default LLM): free API key from [console.groq.com/keys](https://console.groq.com/keys)
- For HuggingFace local inference: access to
  [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### Install dependencies

```bash
pip install uv       # if not already installed
uv sync
```

### PyTorch

`torch` is pulled in as a transitive dependency. To control which build gets
installed (CPU vs CUDA), install it **before** running `uv sync`:

```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=gsk_...        # required for default Groq backend
HF_TOKEN=hf_...             # required only for HuggingFace local inference
```

---

## Stage 1 — Vanilla RAG Baseline

### What it does

Loads all 16 documents → splits into chunks → embeds with
`BAAI/bge-small-en-v1.5` → stores in Chroma → at query time retrieves top-k
chunks by cosine similarity → feeds them into a strict RAG prompt →
generates an answer.

No metadata filtering, no reranking, no conflict resolution.

### How it works

```
query
  │
  ▼
Chroma.similarity_search_with_relevance_scores(query, k=top_k)
  │
  ▼
format_context()  →  plain text CONTEXT block (no citations)
  │
  ▼
RAG_PROMPT_TEMPLATE.format(context=..., question=...)
  │
  ▼
ChatGroq / HuggingFacePipeline
  │
  ▼
{ answer, retrieved_chunks, context_text, final_prompt }
```

### Build the index

```bash
python scripts/build_index.py --data_dir synthetic_data_legal --persist_dir .chroma_langchain_baseline --chunk_size 350 --chunk_overlap 60 --embedding_model BAAI/bge-small-en-v1.5
```

Persists to `.chroma_langchain_baseline/`.

### One-shot query

```bash
python scripts/query.py --query "How much jail time for financial fraud?" --top_k 5 --persist_dir .chroma_langchain_baseline
```

Add `--debug` to print retrieved chunks, the full context block, and the
complete prompt sent to the LLM.

### Interactive chat

```bash
python scripts/chat.py
```

| Command | Effect |
|---------|--------|
| `:debug on` | Print chunks + context + prompt for each response |
| `:debug off` | Turn off debug output |
| `:exit` | Quit |

### LLM backend

The default backend is **Groq** (`llama-3.1-8b-instant`). To use local
HuggingFace inference instead:

```bash
python scripts/query.py --query "..." --llm_backend huggingface --hf_model meta-llama/Llama-3.2-3B-Instruct
```

The HuggingFace loader tries: 4-bit (bitsandbytes) → fp16/bf16 on GPU → CPU
with a warning.

### Configuration defaults

| Parameter | Default |
|-----------|---------|
| `data_dir` | `synthetic_data_legal` |
| `persist_dir` | `.chroma_langchain_baseline` |
| `embedding_model` | `BAAI/bge-small-en-v1.5` |
| `chunk_size` | `350` |
| `chunk_overlap` | `60` |
| `top_k` | `5` |
| `llm_backend` | `groq` |
| `llm_model` | `llama-3.1-8b-instant` |
| `hf_model` | `meta-llama/Llama-3.2-3B-Instruct` |
| `max_new_tokens` | `256` |
| `temperature` | `0.2` |
| `top_p` | `0.9` |

### Returned result structure

`RAGPipeline.answer(query)` returns:

```python
{
    "answer": str,
    "retrieved_chunks": [
        {
            "rank": int,
            "doc_id": str,
            "chunk_index": int,
            "score": float,        # cosine similarity
            "preview": str,        # first 200 chars
            "text": str,
        },
        ...
    ],
    "context_text": str,           # full CONTEXT block fed to the LLM
    "final_prompt": str,           # complete prompt string
}
```

### Stress-test queries

These queries deliberately span conflicting documents:

```bash
python scripts/query.py --query "How much jail time for financial fraud?" --debug
python scripts/query.py --query "What is the penalty for identity theft?" --debug
python scripts/query.py --query "What is the penalty for benefits fraud?" --debug
python scripts/query.py --query "What are the differences between federal and state financial fraud penalties?" --debug
```

---

## Stage 2 — Metadata-Enriched Policy Layer

### What changed from baseline

**`loaders.py`**

Added `load_manifest(metadata_path)` which reads `metadata_manifest.csv` and
returns a lookup table keyed by `doc_id`. `load_documents()` now accepts an
optional `metadata_path` argument; when provided, it stamps three extra fields
onto each loaded document before chunking:

```
metadata["jurisdiction"]    → e.g. "US-FED"
metadata["effective_date"]  → e.g. "2018-01-01"
metadata["authority_rank"]  → e.g. 3  (statute)
```

LangChain propagates parent document metadata to all child chunks automatically,
so no changes to the splitter logic were needed.

**`splitting.py`**

Added `validate_metadata=False` parameter to `split_documents()`. When
`True` (set automatically by `build_index.py` when a manifest is provided),
it asserts that every chunk carries all three policy keys after splitting.
This catches missing or mismatched manifest rows before they reach the index.

**`retriever.py`**

`format_retrieved_chunks()` now includes `jurisdiction`, `effective_date`,
and `authority_rank` in each returned dict alongside the existing fields.
These are surfaced to the application layer to enable downstream filtering
by jurisdiction, recency, and document authority.

**`config.py`**

Added `metadata_path` field. `persist_dir` defaults to
`.chroma_langchain_policy` (separate from the baseline index).

### Build the policy index

```bash
python scripts_policy_layer/build_index.py --data_dir synthetic_data_legal --metadata_path synthetic_data_legal/metadata_manifest.csv --persist_dir .chroma_langchain_policy --chunk_size 350 --chunk_overlap 60 --embedding_model BAAI/bge-small-en-v1.5
```

### Verify the index

```bash
python scripts_policy_layer/test_index.py
```

Checks: index loads, 65 chunks stored, every chunk has `doc_id` +
`chunk_index` + all three policy keys, a live similarity search returns
results with metadata populated.

### Query with policy index

```bash
python scripts_policy_layer/query.py --query "How much jail time for financial fraud?" --debug
```

The `--debug` output now shows `jurisdiction`, `effective_date`, and
`authority_rank` alongside each retrieved chunk, making the metadata/mixing
problem visible.

### Interactive chat

```bash
python scripts_policy_layer/chat.py
```

| Command | Effect |
|---------|--------|
| `:debug on` | Print chunks + metadata + context + prompt for each response |
| `:debug off` | Turn off debug output |
| `:help` | Show available commands |
| `:exit` | Quit |

### Returned result structure (policy layer)

`RAGPipeline.answer(query)` returns the same shape as the baseline, with three
extra fields per chunk:

```python
{
    "answer": str,
    "retrieved_chunks": [
        {
            "rank": int,
            "doc_id": str,
            "chunk_index": int,
            "score": float,             # cosine similarity
            # Policy metadata fields (present when index built with manifest)
            "jurisdiction": str,        # e.g. "US-FED" or "US-CA"
            "effective_date": str,      # e.g. "2024-07-01"
            "authority_rank": int,      # 3 = statute, 2 = guidance/FAQ
            "preview": str,             # first 200 chars
            "text": str,
        },
        ...
    ],
    "context_text": str,                # full CONTEXT block fed to the LLM
    "final_prompt": str,                # complete prompt string
}
```

### Stress-test queries

These queries deliberately span conflicting documents across jurisdictions
and years — useful for observing the metadata mixing problem:

```bash
python scripts_policy_layer/query.py --query "How much jail time for financial fraud?" --debug
python scripts_policy_layer/query.py --query "What is the penalty for identity theft?" --debug
python scripts_policy_layer/query.py --query "What is the penalty for benefits fraud?" --debug
python scripts_policy_layer/query.py --query "What are the differences between federal and state financial fraud penalties?" --debug
```

### Configuration defaults (policy layer)

Same as baseline, plus:

| Parameter | Default |
|-----------|---------|
| `metadata_path` | `synthetic_data_legal/metadata_manifest.csv` |
| `persist_dir` | `.chroma_langchain_policy` |

