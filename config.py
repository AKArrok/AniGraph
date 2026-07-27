"""Configuration — all settings from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM 通用配置（OpenAI 兼容协议）──
# 新变量名优先，旧 DashScope 变量名作为 fallback
LLM_API_KEY  = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("DASHSCOPE_BASE_URL",
                       "https://api.deepseek.com/v1"))

# ── LLM (主模型) ──
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("QWEN_LLM_MODEL", "deepseek-v4-pro")

# ── 轻量 LLM（简单事实查询）──
SIMPLE_LLM_MODEL = os.getenv("SIMPLE_LLM_MODEL", "deepseek-v4-flash")

# 向后兼容别名（旧代码可能引用）
DASHSCOPE_API_KEY = LLM_API_KEY
DASHSCOPE_BASE_URL = LLM_BASE_URL
QWEN_LLM_MODEL = LLM_MODEL

# ── Embeddings ──
# 后端选择: "ark" (火山方舟 Coding Plan, doubao-embedding-vision) | "local" (HuggingFace 本地模型) | "dashscope" (阿里云)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ark")
# Ark Coding Plan Embedding (doubao-embedding-vision, OpenAI 兼容协议)
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_EMBEDDING_API_KEY = os.getenv("ARK_EMBEDDING_API_KEY") or ARK_API_KEY
ARK_EMBEDDING_BASE_URL = os.getenv("ARK_EMBEDDING_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
ARK_EMBEDDING_MODEL = os.getenv("ARK_EMBEDDING_MODEL", "doubao-embedding-vision")
ARK_EMBEDDING_DIMENSIONS = int(os.getenv("ARK_EMBEDDING_DIMENSIONS", "1024"))
ARK_CODING_EMBEDDING_DIMENSION = 1024
ARK_EMBEDDING_REQUEST_INTERVAL = float(
    os.getenv("ARK_EMBEDDING_REQUEST_INTERVAL", "3.0")
)
ARK_EMBEDDING_MAX_RETRIES = int(os.getenv("ARK_EMBEDDING_MAX_RETRIES", "6"))
ARK_EMBEDDING_MAX_BACKOFF = float(
    os.getenv("ARK_EMBEDDING_MAX_BACKOFF", "30")
)
# 本地模型（HuggingFace sentence-transformers）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# 运行设备: "cpu" | "cuda" | "auto" (自动检测 CUDA，不可用时回退 cpu)
_LOCAL_EMBEDDING_DEVICE_RAW = os.getenv("LOCAL_EMBEDDING_DEVICE", "auto")


def _resolve_embedding_device() -> str:
    """解析 embedding 运行设备，支持 auto 自动检测 CUDA。"""
    if _LOCAL_EMBEDDING_DEVICE_RAW != "auto":
        return _LOCAL_EMBEDDING_DEVICE_RAW
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


LOCAL_EMBEDDING_DEVICE = _resolve_embedding_device()
# HuggingFace 加速
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 禁用 Xet，避免 401 错误
# DashScope 模型（仅 EMBEDDING_BACKEND=dashscope 时生效）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_FALLBACKS = os.getenv(
    "EMBEDDING_FALLBACKS",
    "text-embedding-v2,text-embedding-v1,qwen3-vl-rerank"
)
# 完整模型列表: 主模型 + 备用模型（按优先级降序）
EMBEDDING_MODELS = (
    [EMBEDDING_MODEL] +
    [m.strip() for m in EMBEDDING_FALLBACKS.split(",") if m.strip()]
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
RETRIEVER_K       = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "50"))
PINECONE_SEARCH_TYPE = os.getenv("PINECONE_SEARCH_TYPE", "similarity").lower()

# ── RAG 检索优化 ──
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# 本地重排序模型（优先使用本地路径，避免 HuggingFace 缓存符号链接问题）
LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "")
FUSION_STRATEGY = os.getenv("FUSION_STRATEGY", "rrf")  # rrf | weighted | max
ENABLE_QUERY_OPTIMIZATION = os.getenv("ENABLE_QUERY_OPTIMIZATION", "true").lower() == "true"
# CPU 重排单次约需数秒，默认关闭以保证交互延迟；有 GPU 或离线评测时可显式开启。
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
HYBRID_DENSE_K = int(os.getenv("HYBRID_DENSE_K", "20"))
HYBRID_SPARSE_K = int(os.getenv("HYBRID_SPARSE_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
WHOOSH_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "whoosh_index")

# ── Metadata Index ──
METADATA_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metadata_index.json")
METADATA_CACHE_SIZE = int(os.getenv("METADATA_CACHE_SIZE", "2000"))

# ── Multi-Agent ──
EXPERT_TEMPERATURE = float(os.getenv("EXPERT_TEMPERATURE", "0.7"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.7"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # Web fallback 触发阈值
MAX_REPLANS = min(1, max(0, int(os.getenv("MAX_REPLANS", "0"))))

# ── Embedding 预检（零 LLM 成本拦截闲聊/简单查询）──
ENABLE_EMBEDDING_PREFILTER = os.getenv("ENABLE_EMBEDDING_PREFILTER", "true").lower() == "true"
EMBEDDING_PREFILTER_THRESHOLD = float(os.getenv("EMBEDDING_PREFILTER_THRESHOLD", "0.85"))
# embedding 粗筛排除间距: 最佳匹配与某类别得分差超过此值，该类别视为明显不相关
EMBEDDING_EXCLUDE_MARGIN = float(os.getenv("EMBEDDING_EXCLUDE_MARGIN", "0.15"))
# 复杂度分析: 是否用小模型判断是否需要多查询扩展（省去不必要的策略细化）
ENABLE_COMPLEXITY_CHECK = os.getenv("ENABLE_COMPLEXITY_CHECK", "true").lower() == "true"

# ── 按需节点开关 ──
# 别名/实体解析: 是否启用（关闭后所有查询跳过 alias_resolve 节点）
ENABLE_ALIAS_RESOLVE = os.getenv("ENABLE_ALIAS_RESOLVE", "true").lower() == "true"
# 联网搜索: 是否允许触发 Tavily（不影响 plan.need_web 标记，只影响实际调用）
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"

# ── 动漫识图（trace.moe + VLM fallback）──
ENABLE_IMAGE_RECOGNITION = os.getenv("ENABLE_IMAGE_RECOGNITION", "true").lower() == "true"
TRACE_MOE_API_URL = os.getenv("TRACE_MOE_API_URL", "https://api.trace.moe/search")
TRACE_MOE_TIMEOUT = int(os.getenv("TRACE_MOE_TIMEOUT", "15"))
IMAGE_CONFIDENCE_THRESHOLD = float(os.getenv("IMAGE_CONFIDENCE_THRESHOLD", "0.8"))
IMAGE_MAX_SIZE_MB = int(os.getenv("IMAGE_MAX_SIZE_MB", "10"))
# VLM fallback（OpenAI 兼容协议，如 DashScope qwen-vl-max；留空则禁用 VLM fallback）
VLM_API_KEY  = os.getenv("VLM_API_KEY", "")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VLM_MODEL    = os.getenv("VLM_MODEL", "qwen-vl-max")

# ── Short-term Memory ──
MEMORY_MAX_ROUNDS = int(os.getenv("MEMORY_MAX_ROUNDS", "5"))

# Validation
def validate_retrieval_settings() -> None:
    """Reject invalid candidate-pool sizes before the first retrieval."""
    values = {
        "RETRIEVER_K": RETRIEVER_K,
        "RETRIEVER_FETCH_K": RETRIEVER_FETCH_K,
        "HYBRID_DENSE_K": HYBRID_DENSE_K,
        "HYBRID_SPARSE_K": HYBRID_SPARSE_K,
        "RERANK_TOP_K": RERANK_TOP_K,
    }
    non_positive = [name for name, value in values.items() if value <= 0]
    if non_positive:
        raise ValueError(f"Retrieval settings must be positive: {non_positive}")
    if PINECONE_SEARCH_TYPE not in ("similarity", "mmr"):
        raise ValueError("PINECONE_SEARCH_TYPE must be 'similarity' or 'mmr'")
    if PINECONE_SEARCH_TYPE == "mmr" and RETRIEVER_FETCH_K < HYBRID_DENSE_K:
        raise ValueError(
            "RETRIEVER_FETCH_K must be >= HYBRID_DENSE_K for MMR retrieval"
        )
    if HYBRID_DENSE_K < RETRIEVER_K or HYBRID_SPARSE_K < RETRIEVER_K:
        raise ValueError(
            "HYBRID_DENSE_K and HYBRID_SPARSE_K must be >= RETRIEVER_K"
        )
    if RERANK_TOP_K < RETRIEVER_K:
        raise ValueError("RERANK_TOP_K must be >= RETRIEVER_K")


def validate():
    missing = [k for k, v in {
        "LLM_API_KEY":       LLM_API_KEY,
        "PINECONE_API_KEY":  PINECONE_API_KEY,
        "TAVILY_API_KEY":    TAVILY_API_KEY,
    }.items() if not v]
    if EMBEDDING_BACKEND == "ark" and not ARK_EMBEDDING_API_KEY:
        missing.append("ARK_EMBEDDING_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy .env.example to .env")
    if EMBEDDING_BACKEND == "ark":
        if ARK_EMBEDDING_MODEL != "doubao-embedding-vision":
            raise ValueError(
                "Coding Plan embedding model must be doubao-embedding-vision"
            )
        if ARK_EMBEDDING_DIMENSIONS != ARK_CODING_EMBEDDING_DIMENSION:
            raise ValueError(
                "doubao-embedding-vision on Coding Plan must use 1024 dimensions; "
                "recreate any index built with a different dimension"
            )
    validate_retrieval_settings()
