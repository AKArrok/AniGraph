"""Configuration — all settings from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── DashScope 共享密钥（LLM + Embeddings 共用）──
DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ── LLM (Qwen-Max) ──
QWEN_LLM_MODEL = os.getenv("QWEN_LLM_MODEL", "qwen-max")

# ── Embeddings (Qwen text-embedding) ──
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")

# Vector DB (Pinecone)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "vector")

# Web Search (Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Database (Neon PostgreSQL) — deprecated, using MemorySaver
# DATABASE_URL = os.getenv("DATABASE_URL")


# Observability — LangSmith (optional, or use LangFuse below)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "langgraph-agent")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")

# Observability — LangFuse (open-source alternative to LangSmith)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Agent Tuning
MAX_ITERATIONS    = int(os.getenv("MAX_ITERATIONS", "3"))
RETRIEVER_K       = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "20"))

# Validation
def validate():
    missing = [k for k, v in {
        "DASHSCOPE_API_KEY":  DASHSCOPE_API_KEY,
        "PINECONE_API_KEY":   PINECONE_API_KEY,
        "TAVILY_API_KEY":     TAVILY_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy .env.example to .env")
