import json, sqlite3, sys
from pathlib import Path

ROOT = Path("tests")
DB = "data/anime_data.db"
OUT = ROOT / "eval_hard_final.json"

con = sqlite3.connect(DB); con.row_factory = sqlite3.Row

existing = json.loads(OUT.read_text(encoding="utf-8"))
existing_ids = {c["id"] for c in existing}
used_aid = set()
for c in existing:
    for tok in [c["id"]]:
        digits = "".join(ch for ch in tok if ch.isdigit())
        if digits:
            used_aid.add(int(digits))

def anime(aid):
    return con.execute(
        "SELECT anime_id, anime_title, score, score_count, release_date FROM Anime WHERE anime_id=?",
        (aid,)).fetchone()

def prods_of(aid):
    rows = con.execute(
        "SELECT p.production_name FROM Production p "
        "JOIN Anime_Production ap ON ap.production_id=p.production_id "
        "WHERE ap.anime_id=?", (aid,)).fetchall()
    return [r["production_name"] for r in rows]

new_cases = []

# ---------- A: cold_score_precise (6 条) ----------
# 冷门番的精确 Bangumi 评分 —— LLM 记不住一位小数的评分
A_ids = [1638, 4282, 5910, 95679]  # score_count 100-400
for aid in A_ids:
    r = anime(aid)
    if not r: continue
    score_str = f"{r['score']:.1f}"
    new_cases.append({
        "id": f"CS{aid:06d}",
        "type": "cold_score_precise",
        "difficulty": "medium",
        "query": f"《{r['anime_title']}》在 Bangumi 上的评分是多少？精确到小数点后一位。",
        "gold_entities": [r["anime_title"]],
        "gold_answer_keywords": [score_str],
        "gold_evidence": {
            "anime_id": aid, "score": r["score"],
            "score_count": r["score_count"], "release_date": r["release_date"],
        },
        "expects_rag": True,
        "scoring": {
            "score_exact": score_str,
            "rule": "answer must contain the exact one-decimal Bangumi score",
        },
    })

# ---------- B: cold_longtail_fact (6 条) ----------
# score_count 60-150：只在知识库里存在
B_ids = [36974, 84997, 6375, 66405, 12436]
for aid in B_ids:
    r = anime(aid)
    if not r: continue
    year = r["release_date"][:4]
    score_str = f"{r['score']:.1f}"
    new_cases.append({
        "id": f"CL{aid:06d}",
        "type": "cold_longtail_fact",
        "difficulty": "hard",
        "query": f"《{r['anime_title']}》的评分是多少？上映年份是哪一年？",
        "gold_entities": [r["anime_title"]],
        "gold_answer_keywords": [score_str, year],
        "gold_evidence": {
            "anime_id": aid, "score": r["score"],
            "score_count": r["score_count"], "release_date": r["release_date"],
        },
        "expects_rag": True,
        "scoring": {
            "score_exact": score_str, "year_exact": year,
            "rule": "answer must contain BOTH the exact one-decimal score AND the year",
        },
    })

# ---------- C: release_date_precise (5 条) ----------
# 精确到 YYYY-MM。挑非季播月份，让 LLM 靠"番剧四月/十月开播"惯例猜不中
C_ids = [105278, 196053, 7106, 56118, 231888]
for aid in C_ids:
    r = anime(aid)
    if not r: continue
    ym = r["release_date"][:7]  # 2014-06
    year, month = ym.split("-")
    new_cases.append({
        "id": f"D{aid:06d}",
        "type": "release_date_precise",
        "difficulty": "medium",
        "query": f"《{r['anime_title']}》的首播日期精确到年月是哪一个月？",
        "gold_entities": [r["anime_title"]],
        "gold_answer_keywords": [ym, f"{year}年{int(month)}月"],
        "gold_evidence": {
            "anime_id": aid, "release_date": r["release_date"],
            "score_count": r["score_count"],
        },
        "expects_rag": True,
        "scoring": {
            "year": year, "month": month,
            "rule": "answer must contain BOTH the exact year AND month",
        },
    })

# ---------- D: cold_studio_year (5 条) ----------
# 小工作室 + 特定年份的作品列表。手工挑选，避免热门 studio
D_pairs = [
    ("Hal Film Maker", "2001", [16436, 4]),
    ("アクタス", "2007", [14538, 3054]),
    ("童夢", "2004", [10509, 3033]),
]

# We need actual titles from DB.  Query dynamically to avoid mistakes.
D_final = []
for studio, year, _ in D_pairs:
    rows = con.execute(
        "SELECT a.anime_title, a.anime_id FROM Anime a "
        "JOIN Anime_Production ap ON ap.anime_id=a.anime_id "
        "JOIN Production p ON p.production_id=ap.production_id "
        "WHERE p.production_name=? AND SUBSTR(a.release_date,1,4)=? ",
        (studio, year)).fetchall()
    titles = [r["anime_title"] for r in rows]
    if not titles: continue
    D_final.append((studio, year, titles, rows[0]["anime_id"]))
    new_cases.append({
        "id": f"CY_{studio.replace(' ','')}_{year}",
        "type": "cold_studio_year",
        "difficulty": "medium",
        "query": f"{studio} 在 {year} 年制作了哪些动画？",
        "gold_entities": titles,
        "gold_answer_keywords": titles,
        "gold_evidence": {
            "studio": studio, "year": year, "titles": titles,
        },
        "expects_rag": True,
        "scoring": {
            "titles": titles,
            "rule": "answer must mention ALL gold titles",
        },
    })

# ---------- E: tag_top_score (4 条) ----------
# 稀有 Bangumi 标签下评分最高的动画
E_tags = ["浪客剑心", "机动警察", "摇滚", "中二病"]
E_gold = {
    "浪客剑心":   ("浪客剑心 追忆篇", ["浪客剑心"]),
    "机动警察":   ("机动警察剧场版2 和平保卫战", ["机动警察 NEW OVA"]),
    "摇滚":       ("孤独摇滚！", ["摇滚新乐团"]),
    "中二病":     ("命运石之门", ["命运石之门 0"]),
}
for tag in E_tags:
    gold, distractors = E_gold[tag]
    new_cases.append({
        "id": f"TAG_{tag}",
        "type": "tag_top_score",
        "difficulty": "medium",
        "query": f"在 Bangumi 上被打了「{tag}」标签的动画里，评分最高的是哪一部？",
        "gold_entities": [gold],
        "gold_answer_keywords": [gold],
        "gold_evidence": {
            "tag": tag, "gold_title": gold,
            "distractor_titles": distractors,
        },
        "expects_rag": True,
        "scoring": {
            "gold_title": gold,
            "distractor_titles": distractors,
            "rule": "answer must name the top-scored anime carrying this tag",
        },
    })

# ---------- F: numeric_comparison (4 条) ----------
# 两部冷番/次冷番评分谁高，分差 >= 0.4
F_pairs = [
    # (higher, lower, higher_score, lower_score, year)
    ("百变之星 不死鸟传说 ～蕾拉・汉密尔顿物语～", "苍之瞳的少女", 8.1, 6.6, "2006"),
    ("嘟嘟猫观察日记", "希望宅邸 3D", 7.5, 6.5, "2012"),
    ("植木的法则", "ToHeart2", 6.9, 6.2, "2005"),
    ("拥抱！光之美少女", "鹿枫堂", 7.7, 6.8, "2018"),
]
for hi, lo, hs, ls, yr in F_pairs:
    new_cases.append({
        "id": f"CMP_{yr}_{hi[:6]}_{lo[:6]}",
        "type": "numeric_comparison",
        "difficulty": "medium",
        "query": f"《{hi}》和《{lo}》相比，哪一部在 Bangumi 上的评分更高？",
        "gold_entities": [hi, lo],
        "gold_answer_keywords": [hi],
        "gold_evidence": {
            "higher_title": hi, "higher_score": hs,
            "lower_title": lo,  "lower_score": ls,
            "year": yr, "score_gap": round(hs - ls, 2),
        },
        "expects_rag": True,
        "scoring": {
            "higher_title": hi, "lower_title": lo,
            "rule": "answer must identify the higher-scored title as the winner",
        },
    })

# ---------- G: seiyuu_cold_works (5 条) ----------
# 只出现在 3-5 部作品的声优的作品表
G_seiyuu = [
    "礒部花凜",
    "戸田めぐみ",
    "谷井あすか",
    "山田美鈴",
]
for name in G_seiyuu:
    rows = con.execute(
        "SELECT a.anime_title, a.score_count FROM Seiyuu s "
        "JOIN Anime_Seiyuu asy ON asy.seiyuu_id=s.seiyuu_id "
        "JOIN Anime a ON a.anime_id=asy.anime_id "
        "WHERE s.seiyuu_name=? ORDER BY a.score_count DESC", (name,)).fetchall()
    titles = [r["anime_title"] for r in rows]
    if len(titles) < 2: continue
    new_cases.append({
        "id": f"SY_{name}",
        "type": "seiyuu_cold_works",
        "difficulty": "hard",
        "query": f"声优 {name} 在 Bangumi 记录里参与配音过哪些动画？",
        "gold_entities": titles,
        "gold_answer_keywords": titles,
        "gold_evidence": {
            "seiyuu_name": name, "titles": titles,
        },
        "expects_rag": True,
        "scoring": {
            "gold_titles": titles, "match_ratio_gte": 0.5,
            "rule": f"answer must mention >=50% of the {len(titles)} works",
        },
    })

# ---------- H: refusal_fabricated (4 条) ----------
# 虚构的 anime 名字。真的知识边界：如果模型敢答就是幻觉
H_fake = [
    "《星海残响 Requiem》",
    "《魔女的午后咖啡馆 第二季》",
    "《银河列车 EX》",
    "《苍穹之翼：无名之战》",
]
refusal_kw = [
    "没有", "查不到", "无法确认", "找不到", "无相关", "未记录", "不清楚",
    "手上", "手头", "资料里", "知识库", "no such", "not found",
]
for name in H_fake:
    new_cases.append({
        "id": f"FAKE_{H_fake.index(name)+1}",
        "type": "refusal_fabricated",
        "difficulty": "hard",
        "query": f"{name}这部动画的评分和播出年份分别是多少？请只根据知识库回答。",
        "gold_entities": [name.strip("《》")],
        "gold_answer_keywords": refusal_kw,
        "gold_evidence": {
            "note": "Anime 不存在于数据库，正确行为是拒答/说未记录。",
            "correct_behavior": "acknowledge absence; do not fabricate score/year",
        },
        "expects_rag": True,
        "expects_refusal": True,
        "scoring": {
            "must_contain_refusal": True,
            "rule": "answer must admit missing info; MUST NOT invent score/year",
        },
    })

# ---------- 写入 ----------
final = existing + new_cases
Path("tests/eval_hard_final.json").write_text(
    json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"total now: {len(final)}  (added {len(new_cases)} new)")
by_type = {}
for c in final:
    by_type[c["type"]] = by_type.get(c["type"], 0) + 1
for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}")
print("\nNew case IDs:")
for c in new_cases:
    print(f"  {c['id']:32} [{c['type']}]  {c['query']}")
