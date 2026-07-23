"""共享 Prompt 组件 - answer 和 simple_fact_answer 共用的禁止事项和上下文构建

提取原则:
  - 只提取真正重复的部分（禁止套话清单、对话上下文构建逻辑）
  - 各节点保留自己的角色定位和特色 prompt，不强行统一
"""

# 禁止使用的 AI 套话清单（answer / simple_fact_answer 共用）
BANNED_PHRASES = "、".join([
    "推荐理由", "综合分析", "值得注意的是", "综上所述", "根据分析", "笔者认为"
])

# 禁止暴露的内部术语清单
INTERNAL_TERMS = "、".join([
    "元数据", "数据库", "资料库", "检索结果", "数据源", "Expert"
])


def build_context_section(history_text: str, *, is_followup: bool = True) -> str:
    """构建对话上下文段落（answer / simple_fact_answer 共用）

    参数:
      history_text: 预拼接的截断对话历史（context_builder.history_text_recent）
      is_followup: 是否为追问场景（影响提示语）
    """
    if not history_text:
        return ""
    if is_followup:
        return (
            f"## 对话上下文（这是追问，别从头介绍）\n{history_text}"
        )
    return (
        f"## 对话上下文（请自然衔接）\n{history_text}\n\n"
        f"注意: 自然衔接上一轮话题，不要像第一次对话那样重新开场。"
        f"如果用户追问'还有吗'，不要重复上一轮已经推荐过的作品。"
    )
