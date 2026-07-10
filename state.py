from typing import TypedDict, Literal, Annotated, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class DimensionDecision(BaseModel):
    """ACG 维度分类结果"""
    dimensions: list[str] = Field(description="问题涉及的维度列表，至少一个，取值: rating/genre/director/studio/writer/seiyuu/similar/general")
    primary:    str       = Field(description="优先处理的维度，必须是单个维度名: rating/genre/director/studio/writer/seiyuu/similar/general 之一")
    reasoning:  str       = Field(min_length=5, max_length=200)


class State(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    router_decision: str
    reasoning:       str
    iteration_count: int
    dimensions:      list[str]   # 问题涉及的 ACG 维度
    active_dimension: str        # 当前处理的维度
    processed_dims:  list[str]   # 已处理维度（防重复）
    tool_results:    dict        # 各维度检索结果
    query_strategy:  str         # RAG 优化策略: direct/rewrite/hyde/decompose
    is_simple:       bool        # 是否为简单事实查询（用轻量模型回答）
