"""LLM and embedding instances — imported across all nodes."""
from typing import List
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import config

# Qwen-Max 作为主 LLM（DashScope OpenAI 兼容模式）
_base = dict(base_url=config.DASHSCOPE_BASE_URL, api_key=config.DASHSCOPE_API_KEY, model=config.QWEN_LLM_MODEL)

answer_LLM = ChatOpenAI(**_base, temperature=0.9)
router_LLM = ChatOpenAI(**_base, temperature=0, max_tokens=512)


# ── 通义千问 Embeddings（DashScope 自定义封装，绕过 LangChain OpenAIEmbeddings 兼容性问题）──

class DashScopeEmbeddings(Embeddings):
    """DashScope text-embedding-v4，使用原生 OpenAI 客户端调用"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v4"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # DashScope 限制单次最多 10 条，分批处理
        all_vecs = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            resp = self.client.embeddings.create(model=self.model, input=batch, dimensions=1024)
            all_vecs.extend(d.embedding for d in resp.data)
        return all_vecs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


embeddings = DashScopeEmbeddings(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.DASHSCOPE_BASE_URL,
    model=config.QWEN_EMBEDDING_MODEL,
)
