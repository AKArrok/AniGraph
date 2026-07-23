"""LLM and embedding instances — imported across all nodes."""
import json
import logging
from typing import List, Type, TypeVar
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APITimeoutError, RateLimitError
import config

T = TypeVar("T", bound=BaseModel)

# ── 主 LLM（OpenAI 兼容协议）──
_base = dict(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
             model=config.LLM_MODEL, request_timeout=45, max_retries=2)

# temperature 用 config.ANSWER_TEMPERATURE（默认 0.7），避免 0.9 过高导致回答不稳定
answer_LLM = ChatOpenAI(**_base, temperature=config.ANSWER_TEMPERATURE)
# router_LLM / tool_LLM 未使用，已移除

# ── 轻量 LLM（简单事实查询：评分/声优/公司等）──
_base_simple = dict(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                    model=config.SIMPLE_LLM_MODEL, request_timeout=30, max_retries=2)
simple_LLM = ChatOpenAI(**_base_simple, temperature=0.5)


# ══════════════════════════════════════════════════════════════════════
# LLM 调用重试 - 指数退避（网络抖动/限流自动恢复）
# ══════════════════════════════════════════════════════════════════════

_RETRYABLE = (APIError, APITimeoutError, RateLimitError)


def _make_retry(max_attempts: int = 3):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


def llm_invoke_with_retry(llm: ChatOpenAI, messages: list[BaseMessage],
                           max_retries: int = 3) -> BaseMessage:
    """对 LLM 调用加指数退避重试（仅对可恢复错误重试: 网络/超时/限流）。

    所有节点的 llm.invoke(...) 都应走这里，保证瞬时故障自动恢复。
    """
    return _make_retry(max_retries)(llm.invoke)(messages)


async def llm_ainvoke_with_retry(llm: ChatOpenAI, messages: list[BaseMessage],
                                  max_retries: int = 3) -> BaseMessage:
    """异步版 LLM 调用 + 指数退避重试。

    async 节点应优先用此函数，避免同步阻塞事件循环。
    特别是 Expert 节点（metadata_reasoner / similar_expert）通过 Send API 并行时，
    必须用 ainvoke 才能真正并行执行（invoke 会阻塞事件循环导致串行）。
    """
    @_make_retry(max_retries)
    async def _call():
        return await llm.ainvoke(messages)
    return await _call()


# ══════════════════════════════════════════════════════════════════════
# 结构化输出 - 自动降级（with_structured_output -> JSON fallback）
# ══════════════════════════════════════════════════════════════════════

def invoke_structured(llm: ChatOpenAI, output_class: Type[T],
                      messages: list[BaseMessage],
                      max_retries: int = 3) -> T:
    """带降级+重试的结构化输出。

    1. 优先 with_structured_output（OpenAI/GPT 原生支持）
    2. 失败降级为 JSON prompt + 手动解析（deepseek 等不支持 response_format）
    3. 网络/限流错误自动指数退避重试（通过 llm_invoke_with_retry 统一包装）
    """
    try:
        structured_llm = llm.with_structured_output(output_class)
        # 用 llm_invoke_with_retry 统一包装，保证可重试异常（APIError/超时/限流）自动重试
        return llm_invoke_with_retry(structured_llm, messages, max_retries=max_retries)
    except Exception as e:
        err_str = str(e).lower()
        if "response_format" in err_str or "unavailable" in err_str:
            logging.info(f"  [降级] with_structured_output 不可用，改用 JSON 模式")
            return _json_fallback_invoke(llm, output_class, messages, max_retries)
        raise


def _json_fallback_invoke(llm: ChatOpenAI, output_class: Type[T],
                           messages: list[BaseMessage],
                           max_retries: int = 3) -> T:
    """JSON 模式降级: 在 prompt 中要求输出 JSON，手动解析。"""
    schema = output_class.model_json_schema()
    schema_json = json.dumps(schema, ensure_ascii=False)

    prompt_text = (
        f"请严格按照以下 JSON Schema 输出，不要包含额外文字，不要用 markdown 代码块包裹:\n"
        f"{schema_json}"
    )
    try:
        json_llm = llm.bind(response_format={"type": "json_object"})
        resp = llm_invoke_with_retry(
            json_llm, [*messages, HumanMessage(content=prompt_text)],
            max_retries=max_retries,
        )
    except Exception:
        # response_format 也不支持时，退到纯 prompt 模式
        resp = llm_invoke_with_retry(
            llm,
            [*messages, SystemMessage(
                content="你只输出 JSON，不要包含任何解释、markdown 标记或额外文字。"
            )],
            max_retries=max_retries,
        )

    text = resp.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return output_class.model_validate_json(text)


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
