import sys
import json
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "eval_hard.json"
cases = json.loads((Path(__file__).parent / name).read_text(encoding="utf-8"))
for c in cases:
    print(f"[{c['id']}] {c['type']} / {c['difficulty']}")
    print(f"  Q: {c['query']}")
    print(f"  gold_kw: {c['gold_answer_keywords']}")
    ev = c.get("gold_evidence", {})
    if "titles" in ev:
        print(f"  titles: {ev['titles'][:3]}{' ...' if len(ev.get('titles',[]))>3 else ''}")
    if "comment" in ev:
        print(f"  comment: {ev['comment'][:80]}...")
    if "score" in ev:
        print(f"  score: {ev.get('score')}  release_date: {ev.get('release_date')}")
    print()
