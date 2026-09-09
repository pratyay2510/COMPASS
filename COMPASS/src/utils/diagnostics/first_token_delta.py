#!/usr/bin/env python3
"""First-token outcome contrast on a smoke collection -- the logit-lens check.

The outcome target (`elicitation_target --head_selection outcome`, what the
head ranking is built from) is

    delta_p = p(first token | correct rows) - p(first token | incorrect rows)

read at the last prompt token through the unembedding: one prompt-only
forward per row over the STORED prompts (records.jsonl, exactly what the
collection fed the model -- never rebuilt), no generation. This module
measures that quantity on the smoke collection, so whether an outcome target
exists is known BEFORE `collect` is paid for, and prints both sides of it:
the first-token distribution of the rows the model got right, of the rows it
got wrong, and their difference.

Aggregation is argmax only: the greedy first token per row, counted per
class. Under greedy decoding this IS the token the generation opened with,
and it is TARGET_AGG's default (what `target` builds delta_p from). The
class-mean softmax is deliberately not reported -- it weights tokens the
model merely considered, which greedy decoding never emits.

Total variation = sum|delta_p| / 2 (0 = the two classes open identically,
1 = disjoint). It is compared against MIN_TV, the same floor `check_target`
enforces later; here it is a verdict line, not an exit, because the smoke is
informational. Labels are the collection's stored `correct` (single-pass,
the family grader) -- run this BEFORE any formatter relabel so the number is
the one the llama fitting policy (FIT_LABELS=singlepass) will see.

Outputs, in the collection dir:
  first_token_delta.npz   delta_p / p_reason(=correct) / p_direct(=incorrect)
                          (the target's own key names, so
                          head_ranker could consume it) and the
                          class counts
  first_token_delta.json  the printed summary (tv per agg, top tokens)

Run (GPU, one model load, seconds):
  python3 -m src.utils.diagnostics.first_token_delta <smoke collection dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.core import common, io

NPZ_NAME = "first_token_delta.npz"
JSON_NAME = "first_token_delta.json"


def first_token_stats(records: List[Dict], tokenizer, model,
                      batch_size: int, max_prompt_tokens: int) -> Dict:
    """One prompt-only forward per row; class-conditional first-token stats.

    Returns a dict with, per class label in {0, 1}: `n`, `sum_prob`
    (vocab-sized sum of softmax rows) and `greedy` (Counter of argmax ids);
    plus `greedy_ids`, the greedy first token of every row in record order
    (for callers that audit the surface of the opening token).
    """
    import torch

    device = next(model.parameters()).device
    vocab = int(model.get_output_embeddings().weight.shape[0])
    stats = {lab: {"n": 0, "sum_prob": np.zeros(vocab, np.float64),
                   "greedy": collections.Counter()} for lab in (0, 1)}
    greedy_ids: List[int] = []

    prompts = [r["prompt"] for r in records]
    labels = [int(r.get("correct", 0)) for r in records]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_prompt_tokens).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for bi, tid in enumerate(logits.argmax(dim=-1).tolist()):
            s = stats[labels[i + bi]]
            s["n"] += 1
            s["sum_prob"] += probs[bi]
            s["greedy"][tid] += 1
            greedy_ids.append(tid)
    stats["greedy_ids"] = greedy_ids
    stats["vocab"] = vocab
    return stats


def class_dists(stats: Dict, agg: str) -> Dict[str, np.ndarray]:
    """p(first | correct), p(first | incorrect) and their difference, one agg."""
    vocab = stats["vocab"]
    out = {}
    for lab, name in ((1, "correct"), (0, "incorrect")):
        s = stats[lab]
        if agg == "argmax":
            d = np.zeros(vocab, np.float64)
            for tid, c in s["greedy"].items():
                d[tid] = c
        else:
            d = s["sum_prob"].copy()
        out[name] = d / max(d.sum(), 1e-12)
    out["delta"] = out["correct"] - out["incorrect"]
    out["tv"] = float(np.abs(out["delta"]).sum() / 2)
    return out


def _table(tokenizer, d: Dict[str, np.ndarray], top: int) -> List[Dict]:
    order = np.argsort(-np.abs(d["delta"]))[:top]
    rows = []
    for t in order:
        t = int(t)
        if abs(d["delta"][t]) < 1e-4 and d["correct"][t] < 1e-4 \
                and d["incorrect"][t] < 1e-4:
            continue
        rows.append({"token": tokenizer.decode([t]), "id": t,
                     "p_correct": float(d["correct"][t]),
                     "p_incorrect": float(d["incorrect"][t]),
                     "delta": float(d["delta"][t])})
    return rows


def report(tokenizer, stats: Dict, min_tv: float, top: int = 10,
           min_per_class: int = 10) -> Optional[Dict]:
    """Print both class distributions, delta_p and the MIN_TV verdict.

    Returns the summary dict (what the json holds), or None when a class is
    too small for the contrast to mean anything.
    """
    n_c, n_w = stats[1]["n"], stats[0]["n"]
    print("\n=== first-token outcome contrast: p(first | correct) - "
          "p(first | incorrect) ===")
    print("  logit lens at the last prompt token, stored prompts, stored "
          "(single-pass) labels")
    print(f"  rows: {n_c + n_w} untruncated = {n_c} correct + {n_w} incorrect")
    if min(n_c, n_w) < min_per_class:
        print(f"  contrast skipped: only {min(n_c, n_w)} rows in the smaller "
              f"class (need >= {min_per_class} here, >= 20 for `target`); "
              "rerun the smoke with a larger SMOKE_NUM")
        return None
    if min(n_c, n_w) < 20:
        print("  NOTE: `target` refuses a class under 20 rows; this estimate "
              "is noisier than the one the pipeline will build")

    summary: Dict = {"n_correct": n_c, "n_incorrect": n_w, "min_tv": min_tv}
    # Greedy decoding only: the argmax token IS the token the generation
    # opened with, and it is what TARGET_AGG=argmax builds delta_p from. The
    # class-mean softmax is not reported (it weights tokens the model merely
    # considered, which greedy decoding never emits).
    d = class_dists(stats, "argmax")
    rows = _table(tokenizer, d, top)
    print("\n  greedy first token, share of rows per class (TARGET_AGG=argmax)")
    print(f"    {'token':18} {'p(correct)':>10} {'p(incorrect)':>12} "
          f"{'delta':>8}")
    for r in rows:
        print(f"    {r['token']!r:18} {r['p_correct']:10.3f} "
              f"{r['p_incorrect']:12.3f} {r['delta']:+8.3f}")
    promoted = [r for r in rows if r["delta"] > 0][:4]
    suppressed = [r for r in rows if r["delta"] < 0][:4]
    print("    promoted by correctness:   " + ", ".join(
        f"{r['token']!r} {r['delta']:+.3f}" for r in promoted))
    print("    suppressed by correctness: " + ", ".join(
        f"{r['token']!r} {r['delta']:+.3f}" for r in suppressed))
    print(f"    total variation = {d['tv']:.3f}")
    summary["argmax"] = {"tv": d["tv"], "top": rows}

    tv = d["tv"]
    verdict = "PASS" if tv >= min_tv else "FAIL"
    print(f"\n  outcome target gate: TV = {tv:.3f} vs MIN_TV = "
          f"{min_tv:g}  -> {verdict}")
    if verdict == "FAIL":
        print("    correct and incorrect rows open on the same token(s): "
              "`target` (HEAD_SELECTION=outcome) would be rejected by "
              "check_target and any head ranking from it is noise. Either "
              "the prompt does not separate outcomes at the first token on "
              "this dataset, or the classes are too few -- read the tables "
              "above before spending GPU on `collect`.")
    else:
        print("    an outcome target exists at the first token; `target` will "
              "build it from the full fitting pool.")
    summary["verdict"] = verdict
    return summary


def save(out_dir: str, tokenizer, stats: Dict, summary: Optional[Dict]) -> None:
    if summary is None:
        return
    a = class_dists(stats, "argmax")
    np.savez(os.path.join(out_dir, NPZ_NAME),
             delta_p=a["delta"].astype(np.float32),
             p_reason=a["correct"].astype(np.float32),
             p_direct=a["incorrect"].astype(np.float32),
             n_correct=stats[1]["n"], n_incorrect=stats[0]["n"],
             agg="argmax", head_selection="outcome")
    with open(os.path.join(out_dir, JSON_NAME), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {os.path.join(out_dir, NPZ_NAME)} and {JSON_NAME}")


def load_untruncated(collection: str) -> List[Dict]:
    records = common.load_records(os.path.join(collection, "records.jsonl"))
    return [r for r in records
            if str(r.get("truncated", "0")) in ("0", "False", "false")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("collection", help="smoke collection dir (records.jsonl)")
    p.add_argument("--model_name", default=None,
                   help="default: the collection's own args.json")
    p.add_argument("--dtype", default="bf16", choices=list(common.DTYPES))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--min_tv", type=float, default=0.2,
                   help="the target gate's floor (run_compass MIN_TV)")
    p.add_argument("--top", type=int, default=10, help="tokens per table")
    args = p.parse_args()

    with open(os.path.join(args.collection, "args.json"), encoding="utf-8") as f:
        coll_args = json.load(f)
    if args.model_name is None:
        args.model_name = coll_args["model_name"]
    io.assert_model_matches(args.collection, args.model_name,
                            "the first-token contrast")

    kept = load_untruncated(args.collection)
    n_all = len(common.load_records(os.path.join(args.collection, "records.jsonl")))
    # the one convention: correct / ALL rows, truncated = incorrect
    acc = (sum(int(r.get("correct", 0)) for r in kept) / n_all) if n_all else 0.0
    print(f"{args.collection}\n  {n_all} rows ({len(kept)} untruncated), accuracy "
          f"{acc:.1%} over all rows, truncated = incorrect (stored single-pass labels)")
    if not kept:
        raise SystemExit("no untruncated rows to report on")

    tokenizer, model = common.load_model_and_tokenizer(args.model_name, args.dtype)
    model.eval()
    tokenizer.padding_side = "left"       # so position -1 is the last real token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stats = first_token_stats(kept, tokenizer, model, args.batch_size,
                              args.max_prompt_tokens)
    summary = report(tokenizer, stats, args.min_tv, top=args.top)
    save(args.collection, tokenizer, stats, summary)


if __name__ == "__main__":
    main()
