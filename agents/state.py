"""多 Agent 架构 State 定义 — AgentState + ExecutionPlan + ExpertResult"""
from typing import TypedDict, Annotated, List
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ConversationContext(TypedDict):
    """对话上下文 — 由 context_builder 生成，Planner/Answer 消费"""
    history: list[dict]             # 最近 N 轮: [{user: str, assistant: str}]
    recent_entities: list[dict]     # 最近讨论的实体: [{name: str, type: str}]
    current_topic: str              # 当前话题
    is_followup: bool               # 是否为追问
    resolved_query: str             # 指代解析后的查询
    previous_intent: str            # 上一轮意图: recommend | fact | compare | chat


class ExecutionPlan(BaseModel):
    """Planner 输出的完整执行计划，图根据此计划自动编排"""
    query_type: str = Field(
        description="查询类型: simple_fact | recommendation | comparison | chat"
    )
    alias_resolved: bool = Field(
        description="是否已解析别名"
    )
    rewrite_strategy: str = Field(
        description="查询优化策略: direct | rewrite | hyde | decompose"
    )
    experts: list[str] = Field(
        description="需要调用的 Expert: metadata_reasoner | similar_expert"
    )
    parallel: bool = Field(
        description="Experts 是否并行执行"
    )
    query_category: str = Field(
        description="检索分类: metadata（结构化过滤，走 Metadata Index）| semantic（语义检索，走 Pinecone）| mixed（两者融合）"
    )
    need_web: bool = Field(
        description="是否需要联网搜索"
    )
    reasoning: str = Field(
        description="Planner 推理过程简述"
    )


class ExpertResult(BaseModel):
    """Expert 统一输出格式"""
    answer: str = Field(description="Expert 的分析结论")
    confidence: float = Field(description="置信度 0.0-1.0", ge=0.0, le=1.0)
    evidence: list[str] = Field(description="支撑依据列表")


class AgentState(TypedDict):
    """多 Agent 协作全局状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    plan: dict                              # ExecutionPlan (dict 形式)
    metadata: list[dict]                    # Metadata Index 查询结果（结构化）
    shared_context: list[str]              # Dense + Sparse 语义文本
    expert_results: Annotated[list[dict], add]  # 并行 Expert 合并写入
    merged_results: str                    # Merge 后的综合结果
    original_query: str                    # 用户原始查询
    resolved_query: str                    # 别名解析后的查询
    search_keywords: list[str]             # Alias从长查询中提取的番剧名
    metadata_cache: dict                   # {name: metadata_dict}
    alias_cache: dict                      # {alias: full_name}
    answer_plan: dict                      # Answer Planner 输出的结构指引
    entity_type: str                       # 实体类型: "character" | "meme" | "alias" | ""
    entity_name: str                       # 解析出的实体名
    entity_anime: str                      # 对应番剧名
    entity_confidence: float               # 置信度 0.0-1.0
    entity_source: str                     # 解析来源: "dict" | "llm" | "web"
    # ── 对话上下文 (v1.1) ──
    context: ConversationContext            # 当前轮上下文（由 context_builder 生成）
    recent_entities: list[dict]             # 持久化: 最近讨论的实体 [{name, type}]
    previous_intent: str                    # 持久化: 上一轮意图
