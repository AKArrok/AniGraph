"""AniGraph ablation experiment framework.

Design rules:
- Change exactly one variable per experiment; hold everything else fixed.
- Reuse the same test set (eval_dataset.json).
- Record four layers: routing, retrieval, generation, system.
- All numeric results in the table are SIMULATED. Do not cite them as real.

Usage:
  python tests/ablation.py --dry-run         Print the experiment matrix.
  python tests/ablation.py --config full      Run the full pipeline (once wired).
"""

import argparse
import json
import sys
from collections import Counter
from typing import Any

# ==================================================================
# Experiment definitions
# ==================================================================

EXPERIMENTS: list[dict[str, Any]] = [
    # -- Baselines --
    dict(id="B0_direct_llm", label="LLM-only (no retrieval)",
         retrieval="none", multi_expert=False, evaluator=False, replan=False,
         reranker=False, hyde=False, sparse=False, desc="No RAG, single LLM"),
    dict(id="B1_dense_rag", label="Dense RAG + single LLM",
         retrieval="dense", multi_expert=False, evaluator=False, replan=False,
         reranker=False, hyde=False, sparse=False, desc="Dense-only + single LLM"),
    dict(id="B2_hybrid_rag", label="Hybrid RAG + single LLM",
         retrieval="hybrid", multi_expert=False, evaluator=False, replan=False,
         reranker=False, hyde=False, sparse=True, desc="Dense+Sparse + single LLM"),
    dict(id="B3_hybrid_rerank", label="Hybrid + Reranker + single LLM",
         retrieval="hybrid", multi_expert=False, evaluator=False, replan=False,
         reranker=True, hyde=False, sparse=True, desc="+Reranker"),
    dict(id="B4_multi_agent", label="Hybrid + Reranker + Multi-Agent",
         retrieval="hybrid", multi_expert=True, evaluator=False, replan=False,
         reranker=True, hyde=False, sparse=True, desc="+Multi-Agent"),
    dict(id="B5_full", label="AniGraph full system",
         retrieval="hybrid", multi_expert=True, evaluator=True, replan=True,
         reranker=True, hyde=True, sparse=True, desc="+Evaluator+Replan+HyDE"),

    # -- Ablations (remove ONE module from B5_full) --
    dict(id="AB_sparse", label="Ablation: remove Sparse",
         retrieval="dense", multi_expert=True, evaluator=True, replan=True,
         reranker=True, hyde=True, sparse=False, desc="Dense-only retention"),
    dict(id="AB_reranker", label="Ablation: remove Reranker",
         retrieval="hybrid", multi_expert=True, evaluator=True, replan=True,
         reranker=False, hyde=True, sparse=True, desc="No reranking"),
    dict(id="AB_hyde", label="Ablation: remove HyDE",
         retrieval="hybrid", multi_expert=True, evaluator=True, replan=True,
         reranker=True, hyde=False, sparse=True, desc="No HyDE"),
    dict(id="AB_multiexpert", label="Ablation: single Expert",
         retrieval="hybrid", multi_expert=False, evaluator=True, replan=True,
         reranker=True, hyde=True, sparse=True, desc="Single expert only"),
    dict(id="AB_replan", label="Ablation: remove Replan",
         retrieval="hybrid", multi_expert=True, evaluator=True, replan=False,
         reranker=True, hyde=True, sparse=True, desc="No replanning"),
    dict(id="AB_evaluator", label="Ablation: remove Evaluator",
         retrieval="hybrid", multi_expert=True, evaluator=False, replan=False,
         reranker=True, hyde=True, sparse=True, desc="No quality gate"),
]

# Simulated results -- explicitly labeled, NOT from real experiments
_SIM = {
    "B0_direct_llm":   (0.00, 0.00, 0.00, 0.52, 0.48,  1.2,   400,  1.0),
    "B1_dense_rag":    (0.82, 0.71, 0.58, 0.69, 0.71,  2.1,  1600,  3.5),
    "B2_hybrid_rag":   (0.82, 0.79, 0.65, 0.73, 0.75,  2.5,  1800,  4.0),
    "B3_hybrid_rerank":(0.82, 0.82, 0.71, 0.76, 0.79,  3.8,  2000,  4.8),
    "B4_multi_agent":  (0.86, 0.82, 0.71, 0.78, 0.81,  6.4,  3600,  9.0),
    "B5_full":         (0.88, 0.82, 0.71, 0.80, 0.85,  7.8,  4800, 12.5),
    "AB_sparse":       (0.88, 0.71, 0.58, 0.75, 0.78,  7.0,  4400, 11.0),
    "AB_reranker":     (0.88, 0.79, 0.65, 0.78, 0.82,  6.2,  4200, 10.2),
    "AB_hyde":         (0.88, 0.82, 0.71, 0.81, 0.86,  6.1,  3800,  8.8),
    "AB_multiexpert":  (0.82, 0.82, 0.71, 0.78, 0.83,  4.4,  2600,  6.2),
    "AB_replan":       (0.88, 0.82, 0.71, 0.78, 0.83,  5.9,  3800,  9.0),
    "AB_evaluator":    (0.88, 0.82, 0.71, 0.78, 0.82,  6.3,  4000, 10.0),
}


def load_dataset(path: str = "tests/eval_dataset.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def print_matrix() -> None:
    header = f"{'ID':<18} {'Route':>5} {'R@5':>6} {'MRR':>6} {'Acc':>5} {'Faith':>5} {'P95':>7} {'Tok':>5} {'Cost':>5}  Notes"
    sep = "=" * 140
    print(sep)
    print(header)
    print("-" * 140)
    for exp in EXPERIMENTS:
        r = _SIM.get(exp["id"], (0, 0, 0, 0, 0, 0, 0, 0))
        row = (
            f"{exp['id']:<18} "
            f"{r[0]:>4.0%} {r[1]:>5.2f} {r[2]:>5.2f} "
            f"{r[3]:>4.0%} {r[4]:>5.2f} "
            f"{r[5]:>5.1f}s {r[6]:>5d} {r[7]:>5.1f}x  "
            f"{exp['desc']}"
        )
        print(row)
    print(sep)
    print()
    print("WARNING: All numbers above are SIMULATED.")
    print("Run real experiments to replace them before any report or interview.")
    print()


def print_interpretation() -> None:
    print("=== Interpretation guide ===")
    print()
    print("1. HyDE removed -> accuracy goes UP (0.80 -> 0.81):")
    print("   HyDE may hallucinate entities for ACG queries. Investigate per query type.")
    print()
    print("2. Multi-expert only adds ~2pp accuracy but doubles P95 latency:")
    print("   Is this gain statistically significant? Check with bootstrap CI.")
    print()
    print("3. Evaluator+Replan combined add ~2pp:")
    print("   Most gains come from evidence-missing recovery, not hallucination detection.")
    print()
    print("4. Hybrid retrieval vs dense-only: +4pp accuracy, +0.4s latency.")
    print("   Good ROI. But confirm the sparse channel uses proper Chinese tokenization.")
    print()
    print("5. Reranker: +3pp accuracy, +1.3s latency.")
    print("   Worth it only if the reranker model is domain-appropriate.")


def run_experiment(exp_id: str) -> None:
    experiment = next((e for e in EXPERIMENTS if e["id"] == exp_id), None)
    if experiment is None:
        print(f"Unknown experiment: {exp_id}")
        sys.exit(1)

    print(f"Experiment: {experiment['label']}")
    cfg = {k: v for k, v in experiment.items() if k not in ("id", "label", "desc")}
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    cases = load_dataset()
    print(f"Dataset: {len(cases)} cases")
    by_type = Counter(c["type"] for c in cases)
    print(f"Type distribution: {dict(by_type)}")
    by_diff = Counter(c["difficulty"] for c in cases)
    print(f"Difficulty distribution: {dict(by_diff)}")

    print()
    print("Skeleton ready. To produce real results:")
    for idx, metric in enumerate(
        ["Route accuracy", "Recall@5", "MRR", "Answer correctness",
         "Faithfulness", "P95 latency (s)", "Avg tokens", "Relative cost"], start=1
    ):
        print(f"  {idx}. {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AniGraph ablation experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment matrix without execution")
    parser.add_argument("--config", type=str, default="",
                        help="Experiment config ID (e.g. B5_full)")
    parser.add_argument("--dataset", type=str, default="tests/eval_dataset.json")
    args = parser.parse_args()

    if args.dry_run:
        print_matrix()
        print_interpretation()
        cases = load_dataset(args.dataset)
        print(f"\nLoaded {len(cases)} cases from {args.dataset}")
    elif args.config:
        run_experiment(args.config)
    else:
        print_matrix()
        print()
        print("Usage:")
        print("  python tests/ablation.py --dry-run        Print matrix + interpretation")
        print("  python tests/ablation.py --config B5_full   Run full system (wiring needed)")


if __name__ == "__main__":
    main()
