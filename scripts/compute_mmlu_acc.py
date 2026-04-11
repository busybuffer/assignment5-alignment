"""
Compute MMLU accuracy from an existing output JSONL file.

Usage:
    python scripts/compute_mmlu_acc.py outputs/mmlu_dpo_results.jsonl
    python scripts/compute_mmlu_acc.py outputs/mmlu_sft_results.jsonl outputs/mmlu_dpo_results.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def compute_acc(path: Path):
    results = [json.loads(l) for l in open(path) if l.strip()]
    total = len(results)
    correct = sum(r["correct"] for r in results)
    unparseable = sum(1 for r in results if r["predicted"] is None)

    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        subj = r["subject"]
        by_subject[subj]["total"] += 1
        by_subject[subj]["correct"] += int(r["correct"])

    print(f"\n=== {path} ===")
    print(f"Total:       {total}")
    print(f"Correct:     {correct}")
    print(f"Accuracy:    {correct/total:.4f} ({correct}/{total})")
    print(f"Unparseable: {unparseable}")
    print("\n--- Accuracy by subject ---")
    for subj, stats in sorted(by_subject.items()):
        acc = stats["correct"] / stats["total"]
        print(f"  {subj:<45} {acc:.4f} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_mmlu_acc.py <results.jsonl> [...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        compute_acc(Path(path))
