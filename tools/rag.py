"""RAG tool — ACG 番剧知识库检索（全链路优化版）"""
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from llms import embeddings
import config


def _get_retriever():
    """创建 Pinecone retriever"""
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY,
    ).as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.HYBRID_DENSE_K,
            "fetch_k": config.RETRIEVER_FETCH_K,
            "lambda_mult": 0.7,
        },
    )


def get_rag_debug() -> dict:
    """获取最近一次 RAG 检索的调试信息"""
    from tools.rag_optimizer import get_last_debug
    return get_last_debug()


@tool
def RAG(query: str) -> str:
    """检索 ACG 番剧知识库。用于番剧推荐、类型筛选、导演/制作公司/编剧/声优查找、相似作品发现。
    输入应为描述性的中文搜索查询，包含相关关键词。"""
    try:
        from tools.rag_optimizer import retrieve_with_optimization, _last_debug

        retriever = _get_retriever()
        docs, strategy = retrieve_with_optimization(query, retriever, k_final=config.RETRIEVER_K)

        if not docs:
            return "No relevant content found."

        header = f"[策略: {strategy}] " if strategy != "rewrite" else ""
        return header + "".join(
            f"Source {i+1}:\n{d.strip()}\n\n" for i, d in enumerate(docs)
        )
    except Exception as e:
        # 记录错误到 debug 信息中
        try:
            from tools.rag_optimizer import _last_debug
            _last_debug["error"] = str(e)[:200]
        except Exception:
            pass
        return f"RAG error: {e}"
