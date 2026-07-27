"""多 Agent 架构 State 定义 — AgentState + ExecutionPlan + ExpertResult"""
from typing import TypedDict, Annotated, List, Literal
from operator import add
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ConversationContext(TypedDict):
    """对话上下文 — 由 context_builder 生成，Planner/Answer 消费"""
    history: list[dict]             # 最近 N 轮: [{user: str, assistant: str}]
    history_text: str               # 完整拼接文本（planner 用，做意图分类需全量上下文）
    history_text_recent: str        # 最近 3 轮截断版（answer 用，避免 token 膨胀）
    recent_entities: list[dict]     # 最近讨论的实体: [{name: str, type: str}]
    topic_entity: dict              # 当前讨论主题，不随推荐候选列表变化
    current_topic: str              # 当前话题
    is_followup: bool               # 是否为追问
    resolved_query: str             # 指代解析后的查询
    previous_intent: str            # 上一轮意图: recommend | fact | compare | chat
    constraints: dict               # 跨轮继承的约束 e.g. {"exclude_same_series": True, "excluded_series": ["命运石之门"]}


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


class EvaluationResult(BaseModel):
    """Evaluator 对当前 attempt 的证据质量判定。"""
    verdict: Literal["pass", "replan", "fallback"]
    score: float = Field(ge=0.0, le=1.0)
    issues: list[Literal[
        "no_evidence",
        "low_confidence",
        "missing_dimension",
        "expert_conflict",
        "query_mismatch",
    ]]
    missing_dimensions: list[str]
    feedback: str


class ReplanPatch(BaseModel):
    """只允许修正执行策略，不允许重写用户意图。"""
    rewrite_strategy: Literal["direct", "rewrite", "hyde", "decompose"]
    query_category: Literal["metadata", "semantic", "mixed"]
    experts: list[Literal["metadata_reasoner", "similar_expert"]]
    parallel: bool
    need_web: bool
    additional_queries: list[str]
    reasoning: str


class AgentState(TypedDict):
    """多 Agent 协作全局状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    plan: dict                              # ExecutionPlan (dict 形式)
    metadata: list[dict]                    # Metadata Index 查询结果（结构化）
    shared_context: list[str]              # Dense + Sparse 语义文本
    expert_results: Annotated[list[dict], add]  # 并行 Expert 合并写入
    merged_results: str                    # Merge 后的综合结果
    optimized_queries: list[str]           # 当前 attempt 的检索查询
    query_strategy: str                    # 当前 attempt 的查询优化策略
    execution_id: str                      # 隔离 checkpoint 中不同用户请求
    attempt: int                           # 当前执行轮次，首次为 0
    max_replans: int                       # 最大重规划次数
    current_expert_index: int              # 串行 Expert 当前下标
    evaluation: dict                       # EvaluationResult (dict 形式)
    replan_feedback: dict                  # Replan 补丁、查询和前后差异
    execution_mode: str                    # parallel | serial | single
    termination_reason: str                # quality_pass / replan_exhausted / web_fallback
    quality_trace: Annotated[list[dict], add]  # 可观测的质量闭环事件
    retrieval_errors: Annotated[list[dict], add]  # 检索失败事件（结构化，前端可展示）
    original_query: str                    # 用户原始查询
    resolved_query: str                    # 别名解析后的查询
    search_keywords: list[str]             # Alias从长查询中提取的番剧名
    metadata_cache: dict                   # {name: metadata_dict}
    alias_cache: dict                      # {alias: full_name}
    answer_plan: dict                      # Answer Planner 输出的结构指引
    recommendation_count: int              # 用户明确要求的推荐数量，0 表示未指定
    recommendation_candidates: list[dict]  # Similar Expert 确定性整理的推荐候选列表
    entity_type: str                       # 实体类型: "character" | "meme" | "alias" | ""
    entity_name: str                       # 解析出的实体名
    entity_anime: str                      # 对应番剧名
    entity_confidence: float               # 置信度 0.0-1.0
    entity_source: str                     # 解析来源: "dict" | "llm" | "web"
    # ── 对话上下文 (v1.1) ──
    context: ConversationContext            # 当前轮上下文（由 context_builder 生成）
    recent_entities: list[dict]             # 持久化: 最近讨论的实体 [{name, type}]
    topic_entity: dict                      # 持久化: 当前话题实体 {name, type}
    previous_intent: str                    # 持久化: 上一轮意图
    # ── 识图 (v1.2) ──
    image_data: str                        # base64 图片（仅识图节点消费，识别后清空）
    image_recognition_result: dict          # 识别结果详情（番剧名/集数/时间戳/预览）
