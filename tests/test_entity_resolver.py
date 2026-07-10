"""快速验证 entity_resolver 功能"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.entity_resolver import resolve_entity, detect_entity_type

tests = [
    ("夏亚是谁", "character", "机动战士高达"),
    ("典明粥什么梗", "meme", "JOJO的奇妙冒险"),
    ("惠惠的配音是谁", "character", "为美好的世界献上祝福！"),
    ("我不做人了是什么梗", "meme", "JOJO的奇妙冒险"),
    ("素晴好看吗", "alias", "为美好的世界献上祝福！"),
    ("巨人怎么样", "alias", "进击的巨人"),
    ("兵长砍猴是哪部番", "character", "进击的巨人"),
    ("都是时臣的错什么梗", "meme", "Fate/Zero"),
]

all_ok = True
for query, exp_type, exp_anime in tests:
    r = resolve_entity(query)
    if r:
        ok = r["type"] == exp_type and exp_anime in r.get("anime", "")
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"[{status}] {query:25s} type={r['type']:10s} anime={r.get('anime','?')} (conf={r['confidence']:.0%} src={r['source']})")
    else:
        print(f"[FAIL] {query:25s} -> None (expected {exp_type}/{exp_anime})")
        all_ok = False

print()
print("全部通过!" if all_ok else "有失败!")
