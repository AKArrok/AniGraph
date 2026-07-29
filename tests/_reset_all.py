"""Reset all B0-B4 details in ablation_results.json so run_ablation.py reruns
every experiment under the new max_tokens=2048 setting."""
import json
import shutil
from datetime import datetime
from pathlib import Path

p = Path(__file__).parent / "ablation_results.json"
backup = p.with_name(f"ablation_results.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.json")
shutil.copy2(p, backup)
print(f"backup -> {backup.name}")

data = json.loads(p.read_text(encoding="utf-8"))
for k in list(data.keys()):
    if k.startswith("details_"):
        data.pop(k, None)
data["summary"] = {}
p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("cleared summary and all details_* sections")
