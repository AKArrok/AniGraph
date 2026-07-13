"""trace 数据模型 — 前端通过 SSE 接收的事件结构。

分层设计:
    NodeInfo   — 节点静态标识（name, display）
    LLMTrace   — 单次 LLM 调用记录（model, tokens, cost, prompt）
    StateDiff  — 状态变化摘要（added key-value, changed keys）
    NodeRuntime — 节点运行时信息（start, end, state_diff, llm_calls）
    TraceEvent — 前端统一事件（type + node + runtime + answer + summary）

全部使用 TypedDict 以消除 Python 对象序列化开销。
"""

from typing import TypedDict

# ============================================================
# 节点名 → 中文显示名 映射表
# ============================================================
NODE_DISPLAY: dict[str, str] = {
    "alias_resolve":       "别名/实体解析",
    "history_extractor":   "历史提取",
    "context_builder":     "上下文构建",
    "planner":             "规划器",
    "query_processing":    "查询优化",
    "knowledge_retrieval": "知识检索",
    "metadata_reasoner":   "元数据推理专家",
    "similar_expert":      "相似推荐专家",
    "merge":               "结果合并",
    "simple_fact_answer":  "简单事实回答",
    "web_fallback":        "联网兜底",
    "answer_planner":      "回答结构规划",
    "answer":              "回答生成",
}


# ============================================================
# 数据模型
# ============================================================

class NodeInfo(TypedDict):
    """节点静态标识。"""
    name: str        # 注册名 e.g. "planner"
    display: str     # 显示名 e.g. "规划器"


class LLMTrace(TypedDict):
    """单次 LLM 调用记录。"""
    model: str              # 模型名 e.g. "deepseek-v4-pro"
    input_tokens: int
    output_tokens: int
    cost: str               # "$0.0012"
    elapsed: float          # 秒
    system_prompt: str      # Prompt Viewer 用（可为空串）
    user_prompt: str        # Prompt Viewer 用（可为空串）


class StateDiff(TypedDict):
    """状态变化摘要。只传输变化字段，不传全量 state。"""
    added: dict             # 新增/修改的 {field: value}
    changed: list[str]      # 变化的字段名列表


class NodeRuntime(TypedDict):
    """节点运行时信息。"""
    start: float            # time.time() 时间戳
    end: float              # 0 = 仍在运行
    state_diff: StateDiff   # None = 无变化 / 未捕获
    llm_calls: list[LLMTrace]
    error: str              # 非空 = 节点异常


class TraceEvent(TypedDict):
    """前端统一事件格式，通过 SSE 逐个发送。"""
    type: str               # "node_start" | "node_end" | "answer_chunk" | "done"
    node: NodeInfo
    runtime: NodeRuntime
    answer_text: str        # 打字机流增量文本
    graph_path: list[str]   # 最终执行路径（done 事件携带）
    summary: dict           # 总耗时/token/成本（done 事件携带）
