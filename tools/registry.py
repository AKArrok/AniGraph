"""Tool Registry — 统一工具注册、开关控制、分类管理

设计原则:
  - 单一入口: 所有工具在此注册，全局一份元数据
  - 按需加载: 注册时只存 import_path，首次调用才 import（避免循环依赖）
  - 分类分组: llm_tool(暴露给LLM) / pipeline(检索流水线) / web(联网) / debug
  - 开关控制: enabled 字段 + 按 category 过滤

使用:
  from tools.registry import tool_registry, register_default_tools
  register_default_tools()   # 启动时调用一次
  tools = tool_registry.get_llm_tools()  # 拿 LLM 可调用的工具列表
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import importlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """工具元数据"""

    name: str                          # 唯一标识
    description: str = ""              # 功能描述
    category: str = "pipeline"         # llm_tool | pipeline | web | debug
    import_path: str = ""              # "tools.rag.RAG" 格式，首次调用时懒加载
    _callable: Callable | None = field(default=None, repr=False)
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    @property
    def callable(self) -> Callable | None:
        """懒加载: 首次访问时通过 import_path 导入"""
        if self._callable is None and self.import_path:
            try:
                mod_path, _, attr = self.import_path.rpartition(".")
                mod = importlib.import_module(mod_path)
                self._callable = getattr(mod, attr)
            except Exception as e:
                logger.warning(f"  [registry] 无法加载 {self.import_path}: {e}")
        return self._callable


class ToolRegistry:
    """单例工具注册表"""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._initialized = False

    # ── 注册 ──

    def register(self, spec: ToolSpec):
        """注册一个工具（同名覆盖）"""
        self._tools[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]):
        for s in specs:
            self.register(s)

    # ── 查询 ──

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def get_callable(self, name: str) -> Callable | None:
        """获取工具的可调用对象（带 enabled 检查）"""
        spec = self._tools.get(name)
        if spec and spec.enabled:
            return spec.callable
        return None

    def get_llm_tools(self) -> list:
        """返回所有启用且可暴露给 LLM 的工具（用于 bind_tools）"""
        return [
            spec.callable
            for spec in self._tools.values()
            if spec.category == "llm_tool" and spec.enabled
        ]

    def get_enabled(self, category: str | None = None) -> list[ToolSpec]:
        """按分类获取启用的工具"""
        specs = [s for s in self._tools.values() if s.enabled]
        if category:
            specs = [s for s in specs if s.category == category]
        return specs

    # ── 开关 ──

    def enable(self, name: str):
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str):
        if name in self._tools:
            self._tools[name].enabled = False

    def is_enabled(self, name: str) -> bool:
        spec = self._tools.get(name)
        return spec is not None and spec.enabled

    # ── 列表 ──

    def list_all(self) -> dict[str, dict]:
        """返回 {name: {enabled, category, description}}"""
        return {
            name: {
                "enabled": s.enabled,
                "category": s.category,
                "description": s.description,
            }
            for name, s in self._tools.items()
        }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# 全局单例
tool_registry = ToolRegistry()


# ══════════════════════════════════════════════════════════════════════
# 默认注册（启动时调用一次）
# ══════════════════════════════════════════════════════════════════════

def register_default_tools():
    """注册所有内置工具。在 app 启动时调用一次。

    使用 import_path 懒加载，避免启动时的循环导入问题。
    首次调用 tool_registry.get_llm_tools() 时才真正 import。
    """
    import config

    # ── LLM 可调用工具（暴露给 LLM 通过 bind_tools 使用）──
    tool_registry.register_many([
        ToolSpec(
            name="RAG",
            description="ACG番剧知识库检索：番剧推荐、评分/声优/制作公司查找、相似作品发现",
            category="llm_tool",
            import_path="tools.rag.RAG",
            tags=["retrieval", "anime"],
        ),
        ToolSpec(
            name="search_web",
            description="联网搜索：最新番剧、实时资讯、冷门作品",
            category="llm_tool",
            import_path="tools.web_search.search_web",
            enabled=config.ENABLE_WEB_SEARCH,
            tags=["web", "fallback"],
        ),
    ])

    # ── 检索流水线工具（内部使用，不暴露给 LLM）──
    tool_registry.register_many([
        ToolSpec(
            name="retrieve_optimized",
            description="全链路 RAG 优化：查询分类 → 多路检索 → 融合 → 精排",
            category="pipeline",
            import_path="tools.rag_optimizer.retrieve_with_optimization",
            tags=["retrieval", "rag"],
        ),
        ToolSpec(
            name="classify",
            description="查询策略分类：direct / rewrite / hyde / decompose",
            category="pipeline",
            import_path="tools.query_processing.classify",
            tags=["query"],
        ),
        ToolSpec(
            name="multi_query_rewrite",
            description="多角度查询扩展（最多4个变体）",
            category="pipeline",
            import_path="tools.query_processing.multi_query_rewrite",
            tags=["query"],
        ),
        ToolSpec(
            name="hyde_generate",
            description="HyDE假设性答案生成（深度分析类查询）",
            category="pipeline",
            import_path="tools.query_processing.hyde_generate",
            tags=["query"],
        ),
        ToolSpec(
            name="decompose",
            description="查询拆分为多个子问题",
            category="pipeline",
            import_path="tools.query_processing.decompose",
            tags=["query"],
        ),
    ])

    # ── 检索子步骤工具 ──
    tool_registry.register_many([
        ToolSpec(
            name="search_whoosh",
            description="Whoosh BM25F 稀疏检索",
            category="pipeline",
            import_path="tools.knowledge_retrieval.search_whoosh",
            tags=["retrieval", "sparse"],
        ),
        ToolSpec(
            name="fusion",
            description="多路检索结果融合（RRF/Weighted/Max）",
            category="pipeline",
            import_path="tools.knowledge_retrieval.fusion",
            tags=["retrieval"],
        ),
        ToolSpec(
            name="rerank",
            description="CrossEncoder 精排",
            category="pipeline",
            import_path="tools.knowledge_retrieval.rerank",
            tags=["retrieval", "ranking"],
        ),
        ToolSpec(
            name="compress_docs",
            description="文档去重压缩",
            category="pipeline",
            import_path="tools.knowledge_retrieval.compress_docs",
            tags=["retrieval"],
        ),
    ])

    # ── 调试工具 ──
    tool_registry.register(
        ToolSpec(
            name="get_rag_debug",
            description="获取最近一次 RAG 检索的调试信息",
            category="debug",
            import_path="tools.rag.get_rag_debug",
            tags=["debug"],
        )
    )

    tool_registry._initialized = True
    logger.info(
        f"  ToolRegistry 初始化完成: {len(tool_registry)} 个工具 "
        f"(LLM: {len(tool_registry.get_llm_tools())})"
    )
