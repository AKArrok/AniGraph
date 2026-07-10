"""LLM and embedding instances — imported across all nodes."""
import logging
from typing import List
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import config

# Qwen-Max 作为主 LLM（DashScope OpenAI 兼容模式）
_base = dict(base_url=config.DASHSCOPE_BASE_URL, api_key=config.DASHSCOPE_API_KEY,
             model=config.QWEN_LLM_MODEL, request_timeout=60)

answer_LLM = ChatOpenAI(**_base, temperature=0.9)
router_LLM = ChatOpenAI(**_base, temperature=0, max_tokens=512)
tool_LLM    = ChatOpenAI(**_base, temperature=0.3)  # 低温度，确保稳定调工具

# 轻量 LLM — 简单事实查询（评分/声优/公司等）
_base_simple = dict(base_url=config.DASHSCOPE_BASE_URL, api_key=config.DASHSCOPE_API_KEY,
                    model=config.SIMPLE_LLM_MODEL, request_timeout=60)
simple_LLM = ChatOpenAI(**_base_simple, temperature=0.5)


# ── 本地 HuggingFace Embeddings（零 API 调用，零配额）──

class LocalEmbeddings(Embeddings):
    """基于 HuggingFace sentence-transformers 的本地 Embedding 模型。

    优势: 无配额限制、零延迟、完全离线
    模型: Qwen3-Embedding-0.6B (1024维, 通义千问家族, 与 DashScope 同源)
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        logging.info(f"  加载本地 Embedding 模型: {model_name} (device={device}) ...")
        self._model = SentenceTransformer(model_name, device=device)
        self.model = model_name
        self.device = device
        # 验证维度
        self._dim = self._model.get_embedding_dimension()
        if self._dim is None:
            self._dim = len(self._model.encode("test", prompt_name=None))
        logging.info(f"  本地模型就绪: {model_name} ({self._dim}维)")

    @property
    def active_model(self) -> str:
        return self.model

    def embed_documents(self, texts: List[str], target_dim: int = 1024) -> List[List[float]]:
        # Qwen3-Embedding 使用 query/passage 前缀提升质量
        embeddings = self._model.encode(
            texts,
            prompt_name=None,  # 文档嵌入用 passage 前缀
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = [e.tolist() for e in embeddings]

        # 维度适配（正常情况不需要，Qwen3-Embedding 就是 1024 维）
        if vectors and len(vectors[0]) != target_dim:
            logging.info(f"  [维度适配] {self.model}: {len(vectors[0])}维 -> {target_dim}维")
            vectors = [v[:target_dim] if len(v) >= target_dim
                       else v + [0.0] * (target_dim - len(v))
                       for v in vectors]
        return vectors

    def embed_query(self, text: str, target_dim: int = 1024) -> List[float]:
        # 查询使用 query 前缀以获得更好的检索效果
        embedding = self._model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vec = embedding.tolist()
        if len(vec) != target_dim:
            vec = vec[:target_dim] if len(vec) >= target_dim else vec + [0.0] * (target_dim - len(vec))
        return vec


# ── DashScope Embeddings（API 模式，支持多模型自动降级）──

class DashScopeEmbeddings(Embeddings):
    """Embedding 客户端，配额耗尽时自动切换备用模型。

    模型优先级: EMBEDDING_MODELS[0] → [1] → [2] → ...
    切换条件: DashScope 返回 AllocationQuota.FreeTierOnly 时自动降级
    """

    def __init__(self, api_key: str, base_url: str, models: List[str]):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._models = models
        self._active_idx = 0
        self._exhausted: set = set()
        self._switch_count = 0
        self._dim_warned: set = set()  # 维度适配仅警告一次

    @property
    def model(self) -> str:
        """当前活跃的模型名"""
        return self._models[self._active_idx]

    @property
    def active_model(self) -> str:
        """别名，当前活跃模型"""
        return self.model

    def _switch(self) -> bool:
        """切换到下一个可用模型，返回是否成功"""
        for i in range(self._active_idx + 1, len(self._models)):
            if self._models[i] not in self._exhausted:
                old = self.model
                self._active_idx = i
                self._switch_count += 1
                logging.info(f"\n  [模型降级] {old} -> {self.model} (第{self._switch_count}次切换)")
                return True
        return False

    @staticmethod
    def _is_quota_error(err_str: str) -> bool:
        return "AllocationQuota.FreeTierOnly" in err_str

    @staticmethod
    def _is_dimensions_error(err_str: str) -> bool:
        return "dimensions" in err_str.lower()

    def embed_documents(self, texts: List[str], target_dim: int = 1024) -> List[List[float]]:
        all_vecs: List[List[float]] = []

        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            tried: set = set()

            while len(tried) < len(self._models):
                tried.add(self.model)
                kwargs: dict = dict(model=self.model, input=batch)
                if "text-embedding-v4" in self.model or "text-embedding-v2" in self.model:
                    kwargs["dimensions"] = target_dim

                try:
                    resp = self.client.embeddings.create(**kwargs)
                    dim = len(resp.data[0].embedding)
                    vectors = [d.embedding for d in resp.data]

                    if dim != target_dim:
                        if self.model not in self._dim_warned:
                            self._dim_warned.add(self.model)
                            logging.info(f"\n  [维度适配] {self.model}: {dim}维 -> {target_dim}维 (截断)")
                        vectors = [v[:target_dim] if len(v) >= target_dim
                                   else v + [0.0] * (target_dim - len(v))
                                   for v in vectors]

                    all_vecs.extend(vectors)
                    break
                except Exception as e:
                    err_str = str(e)
                    if self._is_dimensions_error(err_str):
                        kwargs.pop("dimensions", None)
                        continue
                    elif self._is_quota_error(err_str):
                        self._exhausted.add(self.model)
                        if not self._switch():
                            raise RuntimeError(
                                f"所有 Embedding 模型配额均已耗尽: {self._models}"
                            )
                    else:
                            self._exhausted.add(self.model)
                            logging.error(f"\n  [模型错误] {self.model}: {err_str[:120]}")
                    if not self._switch():
                            raise RuntimeError(
                                f"所有 Embedding 模型均不可用: {self._models}"
                            )

        return all_vecs

    def embed_query(self, text: str, target_dim: int = 1024) -> List[float]:
        return self.embed_documents([text], target_dim=target_dim)[0]


# ── Embedding 实例（根据 EMBEDDING_BACKEND 自动选择）──

if config.EMBEDDING_BACKEND == "local":
    embeddings = LocalEmbeddings(
        model_name=config.LOCAL_EMBEDDING_MODEL,
        device=config.LOCAL_EMBEDDING_DEVICE,
    )
else:
    embeddings = DashScopeEmbeddings(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        models=config.EMBEDDING_MODELS,
    )
logging.info(f"  Embedding 后端: {config.EMBEDDING_BACKEND} | 模型: {embeddings.model}")
