#!/usr/bin/env python3
"""The outcome target: what token does a CORRECT answer open with, vs a wrong one?

head_ranker ranks a head by how far its injected vector moves the
logits toward a target first-token distribution. This module measures that
target from the model itself -- no generation, no sweep -- as the
correctness contrast under the ONE baseline prompt the collection was run
with (--direct_mode, the prompt mode every path is keyed on):

    p_correct   = first-token distribution at the end of the prompt, over the
                  rows the model answered correctly (the collection's `correct`)
    p_incorrect = the same over the rows it answered incorrectly
    target      = p_correct - p_incorrect

No reasoning prompt, no mode label, no word count enters the construction --
only the answer key. Whatever token mass separates correct from incorrect
openings is discovered, not imposed. Under greedy decoding (agg=argmax) the
argmax of each row's distribution IS the first token the generation opened
with, so the target is exactly the contrast the smoke's logit-lens check
(src.utils.diagnostics.first_token_delta) previews on a few dozen rows.

HISTORY. Until 2026-08-28 this module also offered `--head_selection elicit`,
a prompt-pair contrast (cot prompt minus qa prompt). With the single
per-family prompt its two sides were the same prompt, its target identically
zero, and it cost two redundant forward passes to find that out; it was
removed. Head selection is the outcome contrast only. The npz keeps the
key names its consumers read: delta_p, p_reason (= correct), p_direct
(= incorrect).

Cost: one prompt-only forward per sampled row, a few hundred rows, well
under a minute.

Run:
  python3 -m src.steering.elicitation_target \\
      --exp1_collection <.../train_standard> --out <.../elicit_target.npz> \\
      --model_name meta-llama/Llama-3.1-8B-Instruct --n_prompts 256
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.core import common, io
from src.steering import resume

AGGS = ("argmax", "mean_prob")


def build_prompts(records: List[Dict], tokenizer, mode: str,
                  dataset: str = "") -> List[str]:
    """Templated prompts for one mode, from the collection's stored problems.

    `dataset` (from the collection's args.json) selects any (family, dataset)
    prompt override, so a row is rebuilt under the same builder collection
    used."""
    return [common.make_chat_prompt(tokenizer, r["problem"], mode, dataset=dataset)
            for r in records]


def next_token_dist(model, tokenizer, prompts: List[str], agg: str,
                    batch_size: int, max_prompt_tokens: int) -> Tuple[np.ndarray, Dict]:
    """Distribution over the token the model would emit next, averaged over prompts.

    agg="argmax"    counts the greedy choice per prompt -- literally the first
                    token of the generation the collection would have produced.
    agg="mean_prob" averages the full softmax; smoother, uses more of the signal,
                    but weights tokens the model was merely considering.
    """
    import torch

    vocab = int(model.get_output_embeddings().weight.shape[0])
    total = np.zeros(vocab, dtype=np.float64)
    counts: collections.Counter = collections.Counter()
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_prompt_tokens).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        if agg == "argmax":
            for tid in logits.argmax(dim=-1).tolist():
                counts[tid] += 1
        else:
            total += torch.softmax(logits, dim=-1).sum(dim=0).cpu().numpy()

    if agg == "argmax":
        for tid, n in counts.items():
            total[tid] = n
    dist = total / total.sum()
    top = [(int(t), float(dist[t])) for t in np.argsort(-dist)[:8]]
    return dist.astype(np.float32), {"top": top}


def prompt_fingerprint(model_name: str, direct_mode: str, dataset: str = "") -> str:
    """Hash of the prompt this target is measured under.

    The prompts are per model family and edited by hand
    (src.core.common.STANDARD_PROMPT_BUILDERS). Neither the mode name nor any
    input file changes when a builder is rewritten, so without this a target
    measured under an earlier wording would be silently reused -- the exact
    failure this pipeline hit when the qwen prompt was replaced. Cheap:
    tokenizer only, no weights.

    `dataset` (the collection's own, from args.json) also hashes the
    (family, dataset) prompt override, so editing an override invalidates
    only that dataset's targets.
    """
    tokenizer = io.load_tokenizer(model_name)
    probe = "__PROMPT_FINGERPRINT__"
    text = common.make_chat_prompt(tokenizer, probe, direct_mode, dataset=dataset)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def run(args: argparse.Namespace) -> None:
    io.assert_model_matches(args.exp1_collection, args.model_name,
                            "the outcome target")
    # Loaded before the stamp check because the fingerprint depends on the
    # collection's dataset (the (family, dataset) prompt overrides).
    records = common.load_records(os.path.join(args.exp1_collection, "records.jsonl"))
    records = [r for r in records if str(r.get("truncated", "0")) in ("0", "False", "false")]
    coll_dataset = str(io.load_source_args(args.exp1_collection).get("dataset") or "")
    sig = resume.signature(
        [os.path.join(args.exp1_collection, "records.jsonl")],
        model_name=args.model_name, head_selection="outcome",
        direct_mode=args.direct_mode, agg=args.agg, n_prompts=args.n_prompts,
        seed=args.seed,
        prompts=prompt_fingerprint(args.model_name, args.direct_mode,
                                   dataset=coll_dataset))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    if not args.force and resume.is_current(out_dir, sig, [os.path.basename(args.out)],
                                            name="target_stamp.json"):
        print(f"target is current for this collection and prompt — skipping "
              f"({args.out})")
        return
    rng = np.random.default_rng(args.seed)

    def sample(rows: List[Dict]) -> List[Dict]:
        if args.n_prompts < len(rows):
            return [rows[i] for i in
                    rng.choice(len(rows), args.n_prompts, replace=False)]
        return rows

    by_label = {1: [r for r in records if int(r.get("correct", 0)) == 1],
                0: [r for r in records if int(r.get("correct", 0)) == 0]}
    for lab, rows in by_label.items():
        if len(rows) < 20:
            raise SystemExit(f"only {len(rows)} rows with correct={lab} in "
                             f"{args.exp1_collection} -- the outcome contrast "
                             "needs both populations")
    groups = (("reason", "correct", sample(by_label[1])),
              ("direct", "incorrect", sample(by_label[0])))
    print(f"outcome target from {len(groups[0][2])} correct vs "
          f"{len(groups[1][2])} incorrect prompts of {args.exp1_collection} "
          f"(both under mode={args.direct_mode})")

    tokenizer, model = common.load_model_and_tokenizer(args.model_name, args.dtype)
    model.eval()
    tokenizer.padding_side = "left"       # so position -1 is the last real token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dists = {}
    for key, shown, rows in groups:
        prompts = build_prompts(rows, tokenizer, args.direct_mode, dataset=coll_dataset)
        dist, info = next_token_dist(model, tokenizer, prompts, args.agg,
                                     args.batch_size, args.max_prompt_tokens)
        dists[key] = dist
        print(f"  {shown:9} opens with: " + ", ".join(
            f"{tokenizer.decode([t])!r} {p:.3f}" for t, p in info["top"][:6]))

    delta = dists["reason"] - dists["direct"]
    moved = float(np.abs(delta).sum() / 2)
    print(f"  total variation between the two = {moved:.3f}   "
          "(0 = correct and incorrect rows open identically, 1 = disjoint)")
    if moved < 0.1:
        print("  WARNING: correct and incorrect rows barely differ in what the model "
              "says next. The target is near zero and the ranking it produces will "
              "be noise.")

    np.savez(args.out, delta_p=delta, p_reason=dists["reason"], p_direct=dists["direct"],
             n_prompts=sum(len(g[2]) for g in groups), agg=args.agg,
             head_selection="outcome", direct_mode=args.direct_mode)
    resume.write_stamp(out_dir, sig, name="target_stamp.json")
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp1_collection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bf16", choices=list(common.DTYPES))
    p.add_argument("--direct_mode", default="standard",
                   help="the baseline mode the collection was run with; both "
                        "sides of the contrast are measured under it")
    p.add_argument("--agg", default="argmax", choices=list(AGGS))
    p.add_argument("--n_prompts", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="recompute even when the stamp says it is current")
    run(p.parse_args())


if __name__ == "__main__":
    main()
