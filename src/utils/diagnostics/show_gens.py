#!/usr/bin/env python3
"""Pretty-print raw generations from a collection or a steering config.

Reads either kind of jsonl the suite writes and renders one block per row:

  <collection>/records.jsonl   the baseline generations (problem, trace, label)
  <steering_dir>/gen_<config>.jsonl   the steered generations, plus the
                                      baseline label they are compared against

Filters pick out exactly the rows worth eyeballing — --only flip shows just the
problems steering changed the answer on, which is the fastest way to see what a
config actually did.

Run (CPU):
  python3 -m src.utils.diagnostics.show_gens <file.jsonl> [-n 5] [--only flip]
  python3 -m src.utils.diagnostics.show_gens <steering_dir>/gen_K16_alpha5_com.jsonl \
      --only w2r -n 3 --chars 2000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

from src.core import common
from src.core.io import load_ground_truth

FILTERS = {
    "all": lambda r: True,
    "correct": lambda r: int(r.get("correct", 0)) == 1,
    "wrong": lambda r: int(r.get("correct", 0)) == 0,
    "truncated": lambda r: int(r.get("truncated", 0)) == 1,
    # steering-only: rows whose label changed, in either direction
    "flip": lambda r: "baseline_correct" in r
    and int(r["baseline_correct"]) != int(r["correct"]),
    "w2r": lambda r: "baseline_correct" in r
    and not int(r["baseline_correct"]) and int(r["correct"]),
    "r2w": lambda r: "baseline_correct" in r
    and int(r["baseline_correct"]) and not int(r["correct"]),
    # steering-only: rows the gate declined (never generated, baseline reused)
    "declined": lambda r: int(r.get("steered", 1)) == 0,
}


def questions_for(path: str) -> Dict[str, str]:
    """(subject, idx) -> problem text for a gen_*.jsonl, from the collection the
    sweep ran on (recorded in the steering dir's args.json). Empty if unknown:
    gen rows do not carry the problem statement, only its key."""
    import json

    args_path = os.path.join(os.path.dirname(os.path.abspath(path)), "args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path) as f:
        collection = json.load(f).get("exp1_test_collection")
    if not collection or not os.path.isdir(collection):
        return {}
    try:
        rows = load_ground_truth(collection)
    except FileNotFoundError:
        return {}
    return {f"{r['subject']}__{r['idx']}": r.get("problem", "") for r in rows}


def render(r: Dict, questions: Dict[str, str], chars: int, header: str) -> str:
    key = f"{r.get('subject')}__{r.get('idx')}"
    problem = r.get("problem") or questions.get(key, "")
    text = r.get("generated_text")
    if text is None:
        text = "<gate declined: never generated, the stored baseline answer stands>"
    elif chars > 0:
        text = text[:chars] + (" ... [truncated for display]" if len(text) > chars else "")

    label = f"correct={r.get('correct')}"
    if "baseline_correct" in r:
        label = (f"baseline={r['baseline_correct']} -> steered={r['correct']}"
                 f"  (gate {'fired' if int(r.get('steered', 1)) else 'declined'})")
    lines = [
        "=" * 78,
        f"{header}  {key}   {label}   truncated={r.get('truncated')}",
        "-" * 78,
    ]
    if problem:
        lines.append(f"PROBLEM : {problem[:600]}")
    lines += [
        f"GOLD    : {r.get('gold_answer')!r}",
        f"PRED    : {r.get('pred_answer')!r}",
        "GENERATION:",
        text,
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    recs: List[Dict] = common.load_records(args.path)
    if not recs:
        raise SystemExit(f"{args.path} is empty")
    keep = [r for r in recs if FILTERS[args.only](r)]
    print(f"{args.path}\n{len(recs)} rows, {len(keep)} matching --only {args.only}\n")
    if not keep:
        return
    if args.subject:
        keep = [r for r in keep if r.get("subject") == args.subject]
        print(f"{len(keep)} after --subject {args.subject}\n")
    chosen = (keep[: args.n] if args.head
              else random.Random(args.seed).sample(keep, min(args.n, len(keep))))
    questions = questions_for(args.path)
    for i, r in enumerate(chosen, 1):
        print(render(r, questions, args.chars, f"[{i}/{len(chosen)}]"))
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="records.jsonl or gen_<config>.jsonl")
    p.add_argument("-n", type=int, default=5, help="how many rows to show")
    p.add_argument("--only", default="all", choices=sorted(FILTERS),
                   help="row filter; flip/w2r/r2w/declined apply to steering files")
    p.add_argument("--subject", default=None, help="restrict to one MATH subject")
    p.add_argument("--chars", type=int, default=1200,
                   help="generation characters to print (0 = the whole trace)")
    p.add_argument("--head", action="store_true",
                   help="take the FIRST n matching rows instead of a random sample")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
