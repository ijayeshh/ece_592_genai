"""LLM construction — Groq (default) or HuggingFace Transformers."""

import logging
import os
import warnings

from .config import RAGConfig

logger = logging.getLogger(__name__)


def build_llm(config: RAGConfig):
    """Return a LangChain LLM/chat-model object.

    Backend is selected by config.llm_backend:
      "groq"         → ChatGroq (fast cloud API, requires GROQ_API_KEY)
      "huggingface"  → HuggingFacePipeline (local, GPU/CPU)
    """
    if config.llm_backend == "groq":
        return _build_groq_llm(config)
    if config.llm_backend == "huggingface":
        return _build_hf_llm(config)
    raise ValueError(
        f"Unknown llm_backend: {config.llm_backend!r}. "
        "Choose 'groq' or 'huggingface'."
    )


# ── Groq ──────────────────────────────────────────────────────────────────────

def _build_groq_llm(config: RAGConfig):
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )

    logger.info("Building Groq LLM: %s", config.llm_model)
    llm = ChatGroq(
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        api_key=api_key,
    )
    return llm


# ── HuggingFace (local) ───────────────────────────────────────────────────────

def _build_hf_llm(config: RAGConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from langchain_huggingface import HuggingFacePipeline

    model_id = config.hf_model
    has_cuda = torch.cuda.is_available()

    logger.info("Loading HF tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = _load_hf_model(model_id, has_cuda)

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature if config.do_sample else 1.0,
        top_p=config.top_p,
        do_sample=config.do_sample,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    logger.info("HuggingFace LLM ready: %s", model_id)
    return llm


def _load_hf_model(model_id: str, has_cuda: bool):
    import torch
    from transformers import AutoModelForCausalLM

    if has_cuda:
        model = _try_load_4bit(model_id)
        if model is not None:
            return model

        logger.info("Loading %s in fp16/bf16 on GPU.", model_id)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    warnings.warn(
        "No CUDA GPU detected. Running LLM on CPU — inference will be very slow.",
        RuntimeWarning,
        stacklevel=4,
    )
    logger.warning("No CUDA GPU detected. Loading %s on CPU.", model_id)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )


def _try_load_4bit(model_id: str):
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        logger.info("Loading %s with bitsandbytes 4-bit quantisation.", model_id)
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("bitsandbytes 4-bit load failed (%s). Falling back.", exc)
        return None
