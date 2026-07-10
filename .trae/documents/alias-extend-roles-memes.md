# Entity Resolver：从 Alias 升级到实体解析

> **状态**: ✅ 已实施 — 三层实体解析（字典 → LLM → Web）已实现，entity_type/entity_name 字段已纳入 AgentState 和 ConversationContext。

## 核心理念（来自优化建议）

> 角色、梗、作品简称、公司——它们不是同一种数据。应该输出结构化的实体信息，而不是简单地把所有东西映射成番剧名。

当前方案的问题：
- 把角色/梗/简称都当"别名"，丢失了类型信息
- 判定"是否需要联网"只有成功/失败二元，没有置信度
- Planner 不知道解析来源是字典、LLM 还是联网，无法做精细化决策

---

## 升级方案：四层实体解析 + 置信度驱动

```
用户查询
  ↓
Entity Resolver (agents/entity_resolver.py)
  ├── L0: 高频字典 (≈100 条，零成本)
  ├── L1: LLM 推理 (qwen-flash，覆盖常见实体)
  └── L2: 联网兜底 (Tavily，仅在 confidence < 0.5 时触发)
  ↓
结构化实体: { type, entity, anime, confidence, source }
  ↓
Planner: 根据实体类型和置信度决策
  - 角色实体 → 侧重 metadata_reasoner（查角色相关番剧信息）
  - 梗实体 → 侧重 web（知识库可能没有）
  - confidence 低 → need_web=True
  ↓
Retrieval + Answer
```

## 改动文件

| 文件 | 改动 |
|------|------|
| `agents/entity_resolver.py` | **新建** — 实体解析核心，含 L0 字典 + LLM fallback |
| `agents/state.py` | `AgentState` 新增 `entity_type`、`entity_name`、`entity_source`、`entity_confidence` |
| `agents/graph.py` | `_alias_resolve_node` → 改为调用 entity_resolver，返回结构化实体 |
| `agents/planner.py` | `planner_node` 根据实体信息决策 need_web / experts |

保留 `agents/alias.py` 不动（现有别名逻辑仍可用，entity_resolver 内部复用其核心函数）。

---

## 详细设计

### 1. 新建 `agents/entity_resolver.py`

```python
"""Entity Resolver: 将用户输入中的实体（角色/梗/作品简称）解析为结构化信息。

层级:
  L0: 高频字典 (零成本)
  L1: LLM 推理 (覆盖常见实体)
  L2: 联网兜底 (Tavily, 仅在 confidence < 0.5 时触发)

输出格式:
  {
    "type": "character" | "alias" | "meme",
    "entity": "原始输入中的实体名",
    "anime": "对应番剧正式中文名",
    "confidence": 0.0-1.0,
    "source": "dict" | "llm" | "web"
  }
"""

import json, re, functools
from llms import simple_LLM

# ── L0: 高频实体字典 ──

ENTITY_DICT = {
    # ── 角色 (character) ──
    "夏亚": ("character", "机动战士高达"),
    "阿姆罗": ("character", "机动战士高达"),
    "惠惠": ("character", "为美好的世界献上祝福！"),
    "阿库娅": ("character", "为美好的世界献上祝福！"),
    "达克妮丝": ("character", "为美好的世界献上祝福！"),
    "雷姆": ("character", "Re:从零开始的异世界生活"),
    "拉姆": ("character", "Re:从零开始的异世界生活"),
    "艾米莉亚": ("character", "Re:从零开始的异世界生活"),
    "艾伦": ("character", "进击的巨人"),
    "三笠": ("character", "进击的巨人"),
    "兵长": ("character", "进击的巨人"),
    "利威尔": ("character", "进击的巨人"),
    "炭治郎": ("character", "鬼灭之刃"),
    "祢豆子": ("character", "鬼灭之刃"),
    "善逸": ("character", "鬼灭之刃"),
    "承太郎": ("character", "JOJO的奇妙冒险"),
    "迪奥": ("character", "JOJO的奇妙冒险"),
    "dio": ("character", "JOJO的奇妙冒险"),
    "琦玉": ("character", "一拳超人"),
    "埼玉": ("character", "一拳超人"),
    "鲁路修": ("character", "Code Geass 反叛的鲁路修"),
    "cc": ("character", "Code Geass 反叛的鲁路修"),
    "雪之下雪乃": ("character", "我的青春恋爱物语果然有问题。"),
    "比企谷八幡": ("character", "我的青春恋爱物语果然有问题。"),
    "大老师": ("character", "我的青春恋爱物语果然有问题。"),
    "折木奉太郎": ("character", "冰菓"),
    "千反田爱瑠": ("character", "冰菓"),
    "saber": ("character", "Fate/stay night"),
    "阿尔托莉雅": ("character", "Fate/stay night"),
    "金闪闪": ("character", "Fate/stay night"),
    "远坂凛": ("character", "Fate/stay night"),
    "卫宫士郎": ("character", "Fate/stay night"),
    "间桐樱": ("character", "Fate/stay night"),

    # ── 梗 (meme) ──
    "典明粥": ("meme", "JOJO的奇妙冒险"),
    "我不做人了": ("meme", "JOJO的奇妙冒险"),
    "欧拉欧拉": ("meme", "JOJO的奇妙冒险"),
    "木大木大": ("meme", "JOJO的奇妙冒险"),
    "都是时臣的错": ("meme", "Fate/Zero"),
    "人被殺就會死": ("meme", "Fate/stay night"),
    "我变秃了也变强了": ("meme", "一拳超人"),
    "错的不是我，是这个世界": ("meme", "东京喰种"),
    "献出心脏": ("meme", "进击的巨人"),
    "教练我想打篮球": ("meme", "灌篮高手"),
    "真相只有一个": ("meme", "名侦探柯南"),
    "我的钻头是突破天际的": ("meme", "天元突破 红莲螺岩"),
}

ENTITY_DICT_LOWER = {k.lower(): v for k, v in ENTITY_DICT.items()}


# ── 实体检测（判断用户是否在问角色/梗）──

CHARACTER_PATTERNS = [
    r"(谁|是谁|什么人|哪个角色|出自|登场|出现).{0,8}$",
    r"^{1,6}是.{0,4}(角色|人物)",
    r"(介绍|说说|讲一下).{1,6}(这|那).{0,2}(角色|人物)",
]
MEME_PATTERNS = [
    r"(什么梗|啥梗|意思是|出处|为什么说|梗是|这个梗)",
    r"^(这|那).{0,2}(啥|什么)(意思|梗)",
]

def detect_entity_type(query: str) -> str | None:
    """检测用户输入是否在询问角色/梗，返回 type 或 None"""
    q = query.strip()
    for pat in MEME_PATTERNS:
        if re.search(pat, q):
            return "meme"
    for pat in CHARACTER_PATTERNS:
        if re.search(pat, q):
            return "character"
    # 短纯文本（2-6字纯中文），可能是角色名
    if 2 <= len(q) <= 6 and re.match(r'^[\u4e00-\u9fff·]+$', q):
        if q not in ["你好","谢谢","再见","请问","帮我"]:
            return "character"
    return None


# ── 解析入口 ──

def resolve_entity(query: str) -> dict | None:
    """
    解析用户查询中的实体，返回:
      { type, entity, anime, confidence, source } 或 None
    
    策略: L0 dict → (如有必要) L1 LLM → (如有必要) L2 web
    """
    q = query.strip()
    
    # ── Step 1: L0 字典匹配 ──
    # 先精确匹配
    ql = q.lower()
    if ql in ENTITY_DICT_LOWER:
        etype, anime = ENTITY_DICT_LOWER[ql]
        return _make_result(etype, q, anime, 0.95, "dict")
    
    # 包含匹配（最长优先）
    for key, (etype, anime) in sorted(ENTITY_DICT_LOWER.items(), key=lambda x: -len(x[0])):
        if key in ql:
            return _make_result(etype, key, anime, 0.90, "dict")
    
    # 复用现有番剧别名字典（alias.py）
    from agents.alias import resolve_alias_dict
    alias_result = resolve_alias_dict(q)
    if alias_result:
        return _make_result("alias", q, alias_result, 0.90, "dict")
    
    # ── Step 2: 检测是否可能是实体查询 ──
    etype = detect_entity_type(q)
    if not etype:
        return None  # 不是实体查询，不需要解析
    
    # ── Step 3: L1 LLM 推理 ──
    llm_result = _llm_resolve(q, etype)
    if llm_result and llm_result.get("confidence", 0) >= 0.5:
        return _make_result(
            etype, q, llm_result["anime"],
            llm_result.get("confidence", 0.6), "llm"
        )
    
    # ── Step 4: L2 联网兜底 (confidence < 0.5) ──
    # 不在 resolver 内触发，而是返回低置信度结果，由 planner 决定
    return _make_result(etype, q, "", 0.3, "llm")


def _make_result(etype: str, entity: str, anime: str, confidence: float, source: str) -> dict:
    return {
        "type": etype,
        "entity": entity,
        "anime": anime,
        "confidence": confidence,
        "source": source,
    }


# ── L1: LLM 推理 ──

_LLM_PROMPT = """你是 ACG 番剧专家。用户提到了一个{type_label}，请写出它对应番剧的正式中文名称及你的置信度。

{type_label}: {query}

输出严格 JSON: {{"anime": "番剧正式中文名", "confidence": 0.85}}
如果无法确认，输出: {{"anime": "", "confidence": 0.0}}"""

@functools.lru_cache(maxsize=128)
def _llm_resolve(query: str, etype: str) -> dict | None:
    type_label = "角色" if etype == "character" else "梗"
    prompt = _LLM_PROMPT.format(type_label=type_label, query=query[:100])
    
    try:
        resp = simple_LLM.invoke([HumanMessage(content=prompt)])
        text = resp.content.strip()
        # 提取 JSON
        match = re.search(r'\{[^}]+\}', text)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None
```

### 2. `agents/state.py` — 新增实体字段

替换原来的 `unresolved_type` 思路，用结构化字段：

```python
class AgentState(TypedDict):
    # ... 现有字段 ...
    entity_type: str                     # "character" | "meme" | "alias" | ""
    entity_name: str                     # 解析出的实体名
    entity_anime: str                    # 对应番剧名
    entity_confidence: float             # 置信度 0-1
    entity_source: str                   # "dict" | "llm" | "web"
```

### 3. `agents/graph.py` — 更新 `_alias_resolve_node`

```python
async def _alias_resolve_node(state: AgentState) -> dict:
    from agents.entity_resolver import resolve_entity

    query = _get_query(state)
    
    # 先用原有 alias 系统解析（保留兼容）
    from agents.alias import resolve_alias
    resolved, was_resolved = resolve_alias(query, use_llm=False)
    
    if not was_resolved and _might_be_alias(query):
        resolved, was_resolved = resolve_alias(query, use_llm=True)
    
    # 再用 entity_resolver 解析角色/梗
    entity = resolve_entity(query)
    
    if was_resolved:
        # 番剧别名命中 → 正常流程
        result = {
            "resolved_query": resolved if len(query) <= 15 else query,
            "entity_type": "alias",
            "entity_name": resolved,
            "entity_anime": resolved,
            "entity_confidence": 0.90,
            "entity_source": "dict",
        }
        ...
    elif entity and entity["confidence"] >= 0.5:
        # 角色/梗命中 → 填充番剧名到 search_keywords
        result = {
            "resolved_query": query,  # 保持原查询
            "search_keywords": [entity["anime"]],
            "entity_type": entity["type"],
            "entity_name": entity["entity"],
            "entity_anime": entity["anime"],
            "entity_confidence": entity["confidence"],
            "entity_source": entity["source"],
        }
        ...
    elif entity:
        # 角色/梗低置信度 → 标记需要联网
        result = {
            "resolved_query": query,
            "entity_type": entity["type"],
            "entity_name": entity["entity"],
            "entity_confidence": entity["confidence"],
            "entity_source": entity["source"],
        }
        ...
    else:
        # 无实体 → 原有逻辑
        result = {"resolved_query": query}
    
    return result
```

### 4. `agents/planner.py` — 根据实体信息决策

```python
# 在 planner_node 中，在生成 plan 后增加:
entity_confidence = state.get("entity_confidence", 1.0)
entity_type = state.get("entity_type", "")
entity_source = state.get("entity_source", "")

# 低置信度实体 → 联网
if entity_confidence < 0.5:
    plan_dict["need_web"] = True

# 梗实体 → 知识库可能没有 → 同时走 web
if entity_type == "meme" and entity_source != "dict":
    plan_dict["need_web"] = True
```

---

## 与上一版方案对比

| 维度 | 上一版 | 升级版 |
|------|--------|--------|
| 输出格式 | 字符串（番剧名） | 结构化 `{type, anime, confidence, source}` |
| 联网触发 | 二元 "解析失败" | 置信度 < 0.5 触发 |
| 检测方式 | 长度 + 关键词 | 正则模式 + 上下文 |
| Planner 信息 | 不知道实体类型 | 知道类型/来源/置信度 |
| 架构 | alias.py 膨胀 | 独立 entity_resolver.py |
| 知识库依赖 | 无 | 无（不需要重建） |

## 验证

```
1. "夏亚是谁" → L0 dict命中 → {type:character, anime:机动战士高达, confidence:0.95} → 正常检索
2. "典明粥什么梗" → L0 dict命中 → {type:meme, anime:JOJO的奇妙冒险, confidence:0.95}
3. "芙莉莲哪个番" → L1 LLM推理 → {type:character, anime:葬送的芙莉莲, confidence:0.8}
4. "某冷门新角色是谁" → L1 LLM返回低置信度 → entity_confidence=0.3 → planner设need_web → 联网搜
```
