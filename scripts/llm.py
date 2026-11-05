"""Shared model setup, driven by .env (see .env.example).

One LLM_PROVIDER setting drives both Ragas generation (02, 03) and the DeepEval
judge (05), so switching provider never needs a code change.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# blank template lines (e.g. AZURE_OPENAI_ENDPOINT=) would fail DeepEval's settings validation
for _k in [k for k, v in os.environ.items() if v == ""]:
    del os.environ[_k]

PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _require(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)} (see .env.example)")


def _model_name(role: str) -> str:
    """GEN_MODEL / JUDGE_MODEL, with a per-provider default."""
    default = {"openrouter": "anthropic/claude-sonnet-5.5", "ollama": "qwen3:14b",
               "gemini": "gemini-2.5-flash"}.get(PROVIDER, "gpt-4.1")
    return os.getenv("GEN_MODEL" if role == "gen" else "JUDGE_MODEL", default)


def langchain_chat(role: str = "gen", temperature: float = 0.2):
    """LangChain chat model for Ragas generation (role='gen') or verification (role='judge')."""
    if PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI

        _require("OPENROUTER_API_KEY")
        return ChatOpenAI(model=_model_name(role), base_url=OPENROUTER_URL, temperature=temperature,
                          api_key=os.environ["OPENROUTER_API_KEY"], max_retries=4)
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=_model_name(role), base_url=OLLAMA_URL, temperature=temperature,
                          num_ctx=16384, reasoning=False)
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        _require("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=_model_name(role), temperature=temperature)
    if PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        dep = "AZURE_GEN_DEPLOYMENT" if role == "gen" else "AZURE_JUDGE_DEPLOYMENT"
        _require("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", dep)
        return AzureChatOpenAI(azure_deployment=os.environ[dep],
                               api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                               temperature=temperature)
    from langchain_openai import ChatOpenAI

    _require("OPENAI_API_KEY")
    return ChatOpenAI(model=_model_name(role), temperature=temperature)


def langchain_embeddings():
    if PROVIDER == "openrouter":
        from langchain_openai import OpenAIEmbeddings

        _require("OPENROUTER_API_KEY")
        # check_embedding_ctx_length=False: send raw text, not tiktoken ids (non-OpenAI models reject ids)
        return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "openai/text-embedding-3-small"),
                                base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"],
                                check_embedding_ctx_length=False)
    if PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"), base_url=OLLAMA_URL)
    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _require("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(model=os.getenv("EMBED_MODEL", "models/gemini-embedding-001"))
    if PROVIDER == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        _require("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_EMBED_DEPLOYMENT")
        return AzureOpenAIEmbeddings(azure_deployment=os.environ["AZURE_EMBED_DEPLOYMENT"],
                                     api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
    from langchain_openai import OpenAIEmbeddings

    _require("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


def deepeval_judge():
    """Judge model object for DeepEval metrics (passed explicitly, so `deepeval set-*` isn't needed)."""
    if PROVIDER == "openrouter":
        from deepeval.models import OpenRouterModel

        _require("OPENROUTER_API_KEY")
        return OpenRouterModel(model=_model_name("judge"), api_key=os.environ["OPENROUTER_API_KEY"],
                               base_url=OPENROUTER_URL, temperature=0)
    if PROVIDER == "ollama":
        from deepeval.models import OllamaModel

        return OllamaModel(model=_model_name("judge"), base_url=OLLAMA_URL, temperature=0)
    if PROVIDER == "gemini":
        from deepeval.models import GeminiModel

        _require("GOOGLE_API_KEY")
        return GeminiModel(model=_model_name("judge"), api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
    if PROVIDER == "azure":
        from deepeval.models import AzureOpenAIModel

        _require("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_JUDGE_DEPLOYMENT")
        return AzureOpenAIModel(model=os.environ["AZURE_JUDGE_DEPLOYMENT"],
                                deployment_name=os.environ["AZURE_JUDGE_DEPLOYMENT"],
                                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
                                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                                temperature=0)
    from deepeval.models import GPTModel

    _require("OPENAI_API_KEY")
    return GPTModel(model=_model_name("judge"), temperature=0)


def load_testsets(names: str) -> list[dict]:
    """Cases from one or more test sets, e.g. 'A' or 'A,SAFETY' -> data/testset_<NAME>.jsonl."""
    import json

    cases = []
    for name in (n.strip() for n in names.split(",")):
        path = ROOT / "data" / f"testset_{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found (run make_testset.py first?)")
        cases += [json.loads(l) for l in path.open(encoding="utf-8")]
    return cases
