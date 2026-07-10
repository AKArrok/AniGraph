"""Program Merge — 去重 + 排序 + 合并，零 LLM 成本

职责:
  1. 合并多个 Expert 的结果
  2. 去重（基于内容相似度）
  3. 按置信度排序
  4. 过滤低置信度结果
  5. 生成统一的 merged_results 文本
"""
import config


def merge_expert_results(state: dict) -> dict:
    """合并 Expert 结果，去重排序，生成最终文本"""
    expert_results = state.get("expert_results", [])
    if not expert_results:
        return {"merged_results": ""}

    # 1. 去重（基于 answer 的 n-gram 相似度）
    def _dedup(results: list[dict]) -> list[dict]:
        if len(results) <= 1:
            return results

        def _sim(a: str, b: str) -> float:
            a_grams = set(a[i:i+5] for i in range(len(a)-5))
            b_grams = set(b[i:i+5] for i in range(len(b)-5))
            if not a_grams or not b_grams:
                return 0
            return len(a_grams & b_grams) / len(a_grams | b_grams)

        unique = []
        for r in results:
            ans = r.get("answer", "")
            if all(_sim(ans, u.get("answer", "")) < 0.5 for u in unique):
                unique.append(r)
        return unique

    # 2. 按置信度排序
    def _sort(results: list[dict]) -> list[dict]:
        return sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)

    # 3. 过滤低置信度
    def _filter(results: list[dict]) -> list[dict]:
        return [r for r in results if r.get("confidence", 0) >= 0.3]

    deduped = _dedup(expert_results)
    filtered = _filter(deduped)
    sorted_results = _sort(filtered)

    # 4. 生成合并文本
    parts = []
    for i, r in enumerate(sorted_results, 1):
        answer = r.get("answer", "")
        confidence = r.get("confidence", 0)
        evidence = r.get("evidence", [])
        if answer:
            header = f"[Expert {i} | 置信度: {confidence:.0%}]"
            ev_str = ""
            if evidence:
                ev_str = "\n依据: " + "; ".join(evidence[:3])
            parts.append(f"{header}\n{answer}{ev_str}")

    return {"merged_results": "\n\n".join(parts) if parts else "(所有 Expert 结果置信度过低)"}
