"""Delete hard_eval_results.json to force a fresh run against the curated set."""
from pathlib import Path

p = Path(__file__).parent / "hard_eval_results.json"
if p.exists():
    p.unlink()
    print(f"removed {p}")
else:
    print("no prior hard results to remove")
