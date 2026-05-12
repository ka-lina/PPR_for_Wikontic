"""
Aggregate Exact Match and token-level F1 from qa_eval_hotpot.py jsonl logs vs Hotpot JSON.

Usage (from Wikontic repo root):
  python inference_and_eval/aggregate_hotpot_qa_metrics.py \\
    --predictions qa_logs/<your>_hotpot_test_run_1.jsonl \\
    --dataset_path datasets/hotpotqa200.json

Missing sample_ids (e.g. qa_eval errors / continue) are scored as empty predictions.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

from unidecode import unidecode


def normalize(input_string: str) -> str:
    """Same normalization as qa_eval_hotpot.py."""
    input_string = unidecode(input_string)
    input_string = input_string.lower()
    input_string = re.sub(r"(?<=\d)[,\.](?=\d)", "", input_string)
    input_string = re.sub(f"[{re.escape(string.punctuation)}]", " ", input_string)
    input_string = re.sub(r"\s+", " ", input_string)
    return input_string.strip()


def exact_match(pred: str, gold: str) -> int:
    return 1 if normalize(pred) == normalize(gold) else 0


def token_f1_multiset(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 (HotpotQA-style), after normalize."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    prec = overlap / len(pred_tokens)
    rec = overlap / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions_jsonl(path: Path) -> dict[str, str]:
    """Last line wins per sample_id (jsonl opened with append may duplicate ids)."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj["sample_id"])
            ans = obj.get("answer", "")
            out[sid] = "" if ans is None else str(ans)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="EM / F1 for qa_eval_hotpot jsonl vs Hotpot gold")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to qa_logs/..._hotpot_test_run_<n>.jsonl",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/hotpotqa200.json",
        help="Hotpot subset JSON with _id and answer fields",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to write metrics summary JSON",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    ds_path = Path(args.dataset_path)
    if not pred_path.is_file():
        raise SystemExit(f"Predictions file not found: {pred_path}")
    if not ds_path.is_file():
        raise SystemExit(f"Dataset file not found: {ds_path}")

    ds = load_dataset(ds_path)
    preds = load_predictions_jsonl(pred_path)

    gold_ids_in_ds = {str(x["_id"]) for x in ds}
    pred_ids_unknown = sorted(preds.keys() - gold_ids_in_ds)

    ems: list[int] = []
    f1s: list[float] = []
    missing_ids: list[str] = []

    for row in ds:
        sid = str(row["_id"])
        gold = row.get("answer", "")
        gold = "" if gold is None else str(gold)
        if sid not in preds:
            missing_ids.append(sid)
            pred = ""
        else:
            pred = preds[sid]
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1_multiset(pred, gold))

    n = len(ds)
    summary = {
        "predictions_file": str(pred_path.resolve()),
        "dataset_path": str(ds_path.resolve()),
        "num_dataset_examples": n,
        "num_predictions_in_file": len(preds),
        "num_missing_predictions": len(missing_ids),
        "exact_match": sum(ems) / n if n else 0.0,
        "token_f1": sum(f1s) / n if n else 0.0,
        "prediction_ids_not_in_dataset": pred_ids_unknown[:50],
        "num_prediction_ids_not_in_dataset": len(pred_ids_unknown),
        "missing_sample_ids": missing_ids,
    }

    print(f"Predictions: {pred_path}")
    print(f"Dataset:     {ds_path}")
    print(f"Examples:    {n}")
    print(f"Pred lines (unique ids): {len(preds)}")
    print(f"Missing preds (scored empty): {len(missing_ids)}")
    if pred_ids_unknown:
        print(f"Pred ids not in dataset (ignored in loop): {len(pred_ids_unknown)}")
    print(f"Exact match (normalized): {summary['exact_match']:.4f}")
    print(f"Token F1 (multiset, normalized): {summary['token_f1']:.4f}")

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
