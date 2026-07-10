"""Configuration — all settings from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── DashScope 共享密钥（LLM + Embeddings 共用）──
DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ── LLM (Qwen-Max) ──
QWEN_LLM_MODEL = os.getenv("QWEN_LLM_MODEL", "qwen-max")

# ── 轻量 LLM（简单事实查询）──
SIMPLE_LLM_MODEL = os.getenv("SIMPLE_LLM_MODEL", "qwen-flash")

# ── Embeddings ──
# 后端选择: "dashscope" (API) | "local" (HuggingFace 本地模型，零配额)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
# 本地模型（HuggingFace sentence-transformers）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")  # cpu | cuda
# HuggingFace 加速
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 禁用 Xet，避免 401 错误
# DashScope 模型（仅 EMBEDDING_BACKEND=dashscope 时生效）
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
QWEN_EMBEDDING_FALLBACKS = os.getenv(
    "QWEN_EMBEDDING_FALLBACKS",
    "text-embedding-v2,text-embedding-v1,qwen3-vl-rerank"
)
# 完整模型列表: 主模型 + 备用模型（按优先级降序）
EMBEDDING_MODELS = (
    [QWEN_EMBEDDING_MODEL] +
    [m.strip() for m in QWEN_EMBEDDING_FALLBACKS.split(",") if m.strip()]
)

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

# ── RAG 检索优化 ──
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# 本地重排序模型（优先使用本地路径，避免 HuggingFace 缓存符号链接问题）
LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "")
FUSION_STRATEGY = os.getenv("FUSION_STRATEGY", "rrf")  # rrf | weighted | max
ENABLE_QUERY_OPTIMIZATION = os.getenv("ENABLE_QUERY_OPTIMIZATION", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "false").lower() == "true"
HYBRID_DENSE_K = int(os.getenv("HYBRID_DENSE_K", "10"))
HYBRID_SPARSE_K = int(os.getenv("HYBRID_SPARSE_K", "10"))
WHOOSH_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "whoosh_index")

# ── Metadata Index ──
METADATA_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metadata_index.json")
METADATA_CACHE_SIZE = int(os.getenv("METADATA_CACHE_SIZE", "2000"))

# ── Multi-Agent ──
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen-max")
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.3"))
EXPERT_TEMPERATURE = float(os.getenv("EXPERT_TEMPERATURE", "0.7"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.7"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # Web fallback 触发阈值

# ── Short-term Memory ──
MEMORY_MAX_ROUNDS = int(os.getenv("MEMORY_MAX_ROUNDS", "5"))

# Validation
def validate():
    missing = [k for k, v in {
        "DASHSCOPE_API_KEY":  DASHSCOPE_API_KEY,
        "PINECONE_API_KEY":   PINECONE_API_KEY,
        "TAVILY_API_KEY":     TAVILY_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy .env.example to .env")
