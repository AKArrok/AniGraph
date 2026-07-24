"""Image Recognition Node - 动漫截图识别节点

职责:
  1. 读取 state["image_data"]（base64 图片）
  2. 调用 trace.moe API 识别番剧
  3. 将识别结果合成为中文自然语言查询，追加到 messages
  4. 写入与 alias_resolve 一致的 entity_* 字段（entity_type="alias"）

下游节点（context_builder / planner / knowledge_retrieval）零改动复用。
"""
import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


async def image_recognition_node(state: dict) -> dict:
    """识图节点: 图片 -> 番剧名 -> 合成查询 -> 注入 state

    输出:
      - messages: 追加一条 HumanMessage(synth) 使 messages[-1] 为合成查询
      - search_keywords: [中文化番剧名]（metadata 检索的可靠通道）
      - entity_*: 与 alias_resolve 一致格式（entity_type="alias"）
      - image_data: None（清空，避免 base64 原图进 checkpoint/trace）
      - image_recognition_result: 完整识别结果
    """
    image_b64 = state.get("image_data", "")
    if not image_b64:
        logger.warning("  [识图] image_data 为空，跳过识别")
        return {"image_data": None}

    # 提取用户原始文本（可能为空）
    user_text = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # 调用识图工具
    from tools.image_search import search_anime_by_image
    result = await search_anime_by_image(image_b64)

    title_cn = result.get("title_cn", "")
    episode = result.get("episode")
    timestamp = result.get("timestamp", "")
    similarity = result.get("similarity", 0.0)
    source = result.get("source", "failed")

    # 合成中文自然语言查询（让 context_builder/planner 自然读到番剧名）
    if result.get("matched") and title_cn:
        # trace.moe 高置信度命中
        ep_str = f"第{episode}话" if episode else ""
        ts_str = f" {timestamp}" if timestamp else ""
        if user_text and len(user_text.strip()) > 0:
            synth = f"{user_text}（截图识别：《{title_cn}》{ep_str}{ts_str}）"
        else:
            synth = f"这是《{title_cn}》的截图，介绍一下这部番"
    elif source == "vlm_fallback" and title_cn:
        # VLM fallback 返回了描述
        synth = f"这张截图识别结果: {title_cn}。请帮忙分析一下"
    else:
        # 完全失败
        synth = "这张动漫截图没能识别出来，请补充文字描述"
        title_cn = ""

    logger.info(
        f"  [识图] source={source} title={title_cn} "
        f"sim={similarity:.2f} ep={episode}"
    )

    return {
        "messages": [HumanMessage(content=synth)],
        "original_query": user_text or synth,
        "resolved_query": synth,
        "search_keywords": [title_cn] if title_cn else [],
        "entity_type": "alias",
        "entity_name": title_cn,
        "entity_anime": title_cn,
        "entity_confidence": similarity,
        "entity_source": source,
        "image_recognition_result": result,
        "image_data": None,  # 清空，不进 checkpoint/trace
    }
