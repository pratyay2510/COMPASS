#!/usr/bin/env python3
"""The collector: run the model once over a split and write the collection.

For every example of --dataset / --split (in a fixed, seed-determined order):
1. Build the family's `standard` prompt for the problem.
2. One prompt-only forward: the residual state at every layer at the last
   prompt token (hidden/), and on the SAME pass the per-head o_proj-input
   activations (head/) that head ranking and steering read.
3. Greedy generation up to --max_new_tokens; the frozen family grader
   (src.core.grading, single-pass) labels the row correct / incorrect; a generation that hit the cap is flagged
   `truncated` (it is still graded on the partial text).

Collection is incremental and resumable: each row is appended to
ground_truth.csv and records.jsonl and its arrays written as it is processed,
so an interrupted run resumes on re-run (same command) instead of restarting.

Outputs (in --out_dir):
    ground_truth.csv    one row per problem: subject, idx, gold/pred answer,
                        correct, truncated, problem
    records.jsonl       per-example prompt, generation, label, gold aliases
    hidden/             <subject>__<idx>.npy, [layers+1, dim] float32
    head/               <subject>__<idx>.npy, [layers, heads*head_dim] float16
                        (--collect_heads, default on)
    args.json           run configuration

Modes:
    collect   one (dataset, split, prompt_mode) into --out_dir   (default)
    sweep     every (split, prompt_mode) combination under --out_dir, one
              model load, each into <out_dir>/<split>_<tag>

    cd COMPASS
    python3 -m src.utils.collect.collection --mode collect \\
        --model_name Qwen/Qwen3-4B --dataset gsm8k --subject gsm8k \\
        --split test --num_samples 0 --out_dir <collection>
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.core import common, grading
from src.core.common import (
    DATASETS,
    HeadCatcher,
    dataset_subjects,
    extract_prompt_hidden_states_batch,
    generate_batch,
    load_examples,
    load_model_and_tokenizer,
    make_chat_prompt,
    set_seed,
)
from src.core.io import (
    GROUND_TRUTH_COLUMNS,
    hidden_path,
    mode_tag,
    verify_collection,
)


@dataclass
class SampleRecord:
    """One collected example: problem, generation, and correctness label.

    `truncated` is 1 when the generation hit the token budget without
    emitting EOS; the row is still graded on the partial text.
    """

    idx: int
    subject: str
    problem: str
    gold_solution: str
    gold_answer: str
    generated_text: str
    pred_answer: str
    correct: int
    truncated: int
    prompt: str
    # Accepted alternative surface forms of the gold answer, for datasets that
    # ship them (HARP); empty everywhere else. Not in GROUND_TRUTH_COLUMNS,
    # so the CSV schema is unchanged and only records.jsonl carries it -- which
    # is where anything regrading a collection (steering_driver, regrade) reads
    # from, so a re-grade sees the same aliases collection was scored on.
    gold_aliases: List[str] = field(default_factory=list)


def _load_done_keys(csv_path: str, hidden_dir: str) -> set:
    """Return the set of completed (subject, idx) keys so an interrupted run can
    resume, and atomically repair the CSV in the process.

    Only rows that are fully written, unique, and backed by an existing
    per-example hidden-state file count as done; anything else (e.g. a partially
    written trailing row from a mid-write interruption) is dropped. Every
    column already on the CSV is preserved (other stages annotate it, e.g.
    the formatter's `correct_singlepass`). The CSV is rewritten via a temp
    file + os.replace so the repair itself is crash-safe.
    """
    if not os.path.exists(csv_path):
        return set()

    valid: List[Dict[str, str]] = []
    seen: set = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        extra = [c for c in (reader.fieldnames or []) if c not in GROUND_TRUTH_COLUMNS]
        fieldnames = GROUND_TRUTH_COLUMNS + extra
        for row in reader:
            subject, idx = row.get("subject"), row.get("idx")
            if not subject or not idx:
                continue
            if row.get("correct") not in ("0", "1") or row.get("truncated") not in ("0", "1"):
                continue
            if any(row.get(c) is None for c in GROUND_TRUTH_COLUMNS):
                continue
            key = (subject, idx)
            if key in seen or not os.path.exists(hidden_path(hidden_dir, subject, idx)):
                continue
            seen.add(key)
            valid.append(row)

    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in valid:
            writer.writerow({c: row.get(c, "") for c in fieldnames})
    os.replace(tmp, csv_path)
    return seen


def _print_collection_summary(csv_path: str, max_new_tokens: int) -> None:
    """Print totals and model accuracy over the full ground-truth CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    correct = np.array([int(r["correct"]) for r in rows])
    trunc = np.array([int(r["truncated"]) for r in rows], dtype=bool)
    print(f"Total collected: {len(rows)} | truncated at max_new_tokens={max_new_tokens}: "
          f"{int(trunc.sum())} ({trunc.mean():.1%})")
    # the one convention: correct / ALL rows; truncated rows count by their
    # label (2026-09-01 -- graded on partial text, formatter OR may lift them)
    print(f"Model accuracy (all rows; truncated graded on partial text): "
          f"{correct.mean():.4f}")


def collect(args: argparse.Namespace, bundle: Optional[Tuple[Any, Any]] = None) -> None:
    """Run the LLM over the examples, writing the ground-truth CSV, records and
    per-example arrays incrementally so an interrupted run resumes on re-run.

    For each example (in a fixed, seed-determined order) the hidden state is
    saved to ``hidden/<subject>__<idx>.npy`` and a row is appended to
    ``ground_truth.csv`` and ``records.jsonl``, each flushed immediately. On
    startup, already-completed (subject, idx) keys are skipped, so re-running the
    same command continues where it left off instead of overwriting.

    `bundle` is an optional preloaded (tokenizer, model); --mode sweep passes one
    in so several (split, prompt_mode) collections share a single model load.
    """
    set_seed(args.seed)
    examples = load_examples(args.dataset, [args.subject], args.split,
                             args.num_samples, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    hidden_dir = os.path.join(args.out_dir, "hidden")
    os.makedirs(hidden_dir, exist_ok=True)
    # per-head (o_proj input) activations, captured on the same forward pass as
    # the residual-stream states -- the head tensor ranking and steering read,
    # one <head_dir>/<subject>__<idx>.npy per row, no second GPU pass.
    head_dir = None
    if args.collect_heads:
        head_dir = os.path.join(args.head_out_dir or args.out_dir, "head")
        os.makedirs(head_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "ground_truth.csv")
    jsonl_path = os.path.join(args.out_dir, "records.jsonl")

    with open(os.path.join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    done = _load_done_keys(csv_path, hidden_dir)
    if head_dir is not None:
        # Done-ness stays defined by the CSV row + hidden state: re-processing a
        # finished row would append a DUPLICATE CSV/JSONL entry. So heads are
        # only written for rows generated from here on.
        stale = [k for k in done if not os.path.exists(os.path.join(head_dir, f"{k[0]}__{k[1]}.npy"))]
        if stale:
            print(f"NOTE: {len(stale)} of {len(done)} already-collected rows have no head "
                  f"activations. They will NOT be backfilled here (that would duplicate "
                  "their CSV rows); re-collect into a fresh --out_dir with --collect_heads "
                  "if those rows are needed.")
    todo = [ex for ex in examples if (ex["subject"], str(ex["idx"])) not in done]
    print(f"Loaded {len(examples)} {args.dataset} examples from subject={args.subject}, "
          f"split={args.split}")
    print(f"Resuming: {len(done)} already collected, {len(todo)} remaining")
    if not todo:
        print("All examples already collected; nothing to do.")
        _print_collection_summary(csv_path, args.max_new_tokens)
        return

    tokenizer, model = bundle if bundle is not None else load_model_and_tokenizer(
        args.model_name, args.dtype
    )
    # PER-FAMILY, like the prompt: qwen's answer surface (markdown emphasis,
    # unit tails, think blocks) needs its own parser; other families keep the
    # llama grader byte-for-byte. See grading.GRADERS.
    grader = grading.grader_for(args.model_name)
    # hooks live for the whole loop; removed in the finally below so a shared
    # model (--mode sweep reuses one) is never left with stale hooks attached
    catcher = HeadCatcher(model) if head_dir is not None else None
    if catcher is not None:
        print(f"Collecting per-head activations -> {head_dir} "
              f"({len(catcher.layers)} layers, same forward pass, no extra cost)")

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=GROUND_TRUTH_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_f.flush()
    jsonl_f = open(jsonl_path, "a", encoding="utf-8")

    n_new = 0
    n_truncated = 0
    try:
        bs = max(1, int(args.batch_size))
        batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        for batch in tqdm(batches, desc="Collecting", unit="batch"):
            prompts = [
                make_chat_prompt(
                    tokenizer, ex["problem"], args.prompt_mode,
                    use_chat_template=not args.no_chat_template,
                    dataset=args.dataset,
                )
                for ex in batch
            ]
            try:
                # One prompt-only forward for the whole batch. Per-head
                # activations ride along on that SAME pass, so they still cost
                # nothing; they must be collected here, before the generate
                # forwards overwrite the hooks' captured states.
                hs_batch = extract_prompt_hidden_states_batch(
                    tokenizer,
                    model,
                    prompts,
                    representation="last_token",
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                head_batch = catcher.collect() if catcher is not None else None
                gen_batch = generate_batch(
                    tokenizer,
                    model,
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
            except RuntimeError as e:
                keys = ", ".join(f"{ex['subject']}:{ex['idx']}" for ex in batch)
                print(f"RuntimeError on batch [{keys}]: {e}", file=sys.stderr)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for bi, ex in enumerate(batch):
                generated, truncated = gen_batch[bi]

                # Persist the features BEFORE the CSV row, so that every row in
                # the CSV is guaranteed to have an aligned feature file on resume.
                np.save(hidden_path(hidden_dir, ex["subject"], ex["idx"]),
                        hs_batch[bi].astype(np.float32))
                if head_batch is not None:
                    # atomic; the per-head layout every reader expects:
                    # <head_dir>/<subject>__<idx>.npy, (num_layers, num_heads*head_dim) float16
                    hp = os.path.join(head_dir, f"{ex['subject']}__{ex['idx']}.npy")
                    np.save(hp + ".tmp.npy", head_batch[bi])
                    os.replace(hp + ".tmp.npy", hp)

                pred_answer = grader.extract(generated)
                rec = SampleRecord(
                    idx=int(ex["idx"]),
                    subject=ex["subject"],
                    problem=ex["problem"],
                    gold_solution=ex["gold_solution"],
                    gold_answer=ex["gold_answer"],
                    generated_text=generated,
                    pred_answer=pred_answer,
                    correct=int(grader.is_correct(pred_answer, ex)),
                    truncated=int(truncated),
                    prompt=prompts[bi],
                    gold_aliases=list(ex.get("gold_aliases") or []),
                )
                # Append + flush per example: an interruption loses nothing
                # already written, and re-running resumes from here.
                writer.writerow({c: getattr(rec, c) for c in GROUND_TRUTH_COLUMNS})
                csv_f.flush()
                jsonl_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                jsonl_f.flush()
                n_new += 1
                n_truncated += int(truncated)
    finally:
        csv_f.close()
        jsonl_f.close()
        if catcher is not None:
            catcher.remove()

    print(f"Collected {n_new} new examples this run ({len(done) + n_new} total) -> {csv_path}")
    _print_collection_summary(csv_path, args.max_new_tokens)
    have = ["hidden/ (per-layer residual states)",
            "ground_truth.csv + records.jsonl (labels, prompts)"]
    if head_dir is not None:
        have.append("head/ (per-head o_proj activations)")
    print("\nThis collection now has:")
    for h in have:
        print(f"  [x] {h}")
    if head_dir is None:
        print("  [ ] head/ (needed for ranking and steering) -> re-run with --collect_heads")


def run_sweep(args: argparse.Namespace) -> None:
    """Collect every (split, prompt_mode) combination in ONE process.

    Each combination is written into its own subdirectory
    ``<--out_dir>/<split>_<tag>`` (e.g. ``test_standard``) so states never
    mix. The model is loaded once and reused across combinations.

    Every collection is individually resumable, so re-running the same sweep
    command skips finished work: point an existing collection at its expected
    subdirectory (move or symlink it) and the sweep will reuse it instead of
    regenerating.
    """
    if args.head_out_dir is not None:
        # every collection would write head/ into the SAME directory, and
        # (subject, idx) keys repeat across splits (algebra__0 exists in both
        # train and test), so the files would silently overwrite each other.
        raise SystemExit(
            "--head_out_dir cannot be used with --mode sweep: all collections would "
            "share one head/ directory and their (subject, idx) files would collide. "
            "Leave it unset (each collection then holds its own head/), or run "
            "--mode collect once per split."
        )
    combos = [(s, p) for p in args.sweep_prompt_modes for s in args.sweep_splits]
    root = args.out_dir
    print(f"Sweep: {len(combos)} collection(s) under {root}, in this order:")
    for i, (split, prompt_mode) in enumerate(combos, 1):
        tag = mode_tag(prompt_mode)
        print(f"  {i}. {split}/{tag} -> {os.path.join(root, f'{split}_{tag}')}")

    # Load once, share across every combination (weights are the expensive part).
    bundle = load_model_and_tokenizer(args.model_name, args.dtype)

    for split, prompt_mode in combos:
        sub = argparse.Namespace(**vars(args))
        sub.split, sub.prompt_mode = split, prompt_mode
        sub.out_dir = os.path.join(root, f"{split}_{mode_tag(prompt_mode)}")
        print(f"\n{'=' * 70}\n[sweep] collect split={split} prompt_mode={prompt_mode}\n{'=' * 70}")
        collect(sub, bundle=bundle)

    print(f"\n{'=' * 70}\n[sweep] done. Verifying collections under {root}:")
    for split, prompt_mode in combos:
        name = f"{split}_{mode_tag(prompt_mode)}"
        print(f"  {name:<12} {verify_collection(os.path.join(root, name))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--mode", choices=["collect", "sweep"], default="collect",
        help="collect = one (dataset, split, prompt_mode) into --out_dir; "
        "sweep = every (split, prompt_mode) combo under --out_dir in one run.",
    )
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--prompt_mode", choices=list(common.PROMPT_MODES), default="standard",
        help="'standard' is the ONE prompt, resolved per model family "
        "(common.STANDARD_PROMPT_BUILDERS). Tagged _standard in filenames; "
        "existing collections keep their historical tags (_qa, _qabrief, ...).",
    )
    parser.add_argument(
        "--collect_heads", action=argparse.BooleanOptionalAction, default=True,
        help="also save per-head o_proj-input activations from the SAME "
        "prompt-only forward pass that produces the hidden states. Free in "
        "compute, ~256 KB/example. --no-collect_heads for a labels-only collection.",
    )
    parser.add_argument(
        "--head_out_dir", default=None,
        help="where to write head/ (default: alongside hidden/ in --out_dir)",
    )
    # Defaults run the full collection: all subjects, entire test split.
    parser.add_argument(
        "--dataset", choices=list(DATASETS), default="math",
        help="math = EleutherAI/hendrycks_math (7 subjects, boxed LaTeX "
        "answers); gsm8k = openai/gsm8k main (one pseudo-subject 'gsm8k', "
        "numeric answers taken from the '####' marker). Recorded in "
        "args.json.",
    )
    # Validated against the dataset's own vocabulary in main(), since the legal
    # values depend on --dataset and argparse cannot express that.
    parser.add_argument(
        "--subject", default="all",
        help="A subject of --dataset, or 'all'. MATH: %s. GSM8K: gsm8k."
             % ", ".join(dataset_subjects("math")),
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "test", "calibration", *common.CALIB_SPLITS],
        help="'calibration' is MATH-only: MATH test minus the 500 MATH-500 "
        "rows (4500), the part of test that is provably disjoint from the "
        "steering eval set. The calib names are the fixed fitting pools cut "
        "from train (MATH, GSM8K; common.CALIB_SPLITS).",
    )
    parser.add_argument("--num_samples", type=int, default=0, help="0 = use the entire split.")
    parser.add_argument("--out_dir", required=True)

    # --mode sweep: which combinations to build under --out_dir (as a root).
    parser.add_argument(
        "--sweep_splits", nargs="+", default=["train", "test"],
        choices=["train", "test", "calibration", *common.CALIB_SPLITS],
        help="Splits to collect in --mode sweep.",
    )
    parser.add_argument(
        "--sweep_prompt_modes", nargs="+", default=["standard"],
        choices=list(common.PROMPT_MODES),
        help="Prompt modes to collect in --mode sweep, outermost loop.",
    )

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Prompts per forward/generate pass. Decoding one sequence at a time "
        "is memory-bandwidth bound, so batching is close to free throughput. A "
        "batch runs until its LONGEST member finishes, so the gain shrinks as the "
        "generation-length tail grows. 1 restores the exact pre-batching path "
        "(batched and unbatched greedy decoding agree mathematically but not "
        "bitwise -- a different batch size selects different reduction kernels).",
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Generation budget. Every example is recorded with a `truncated` "
        "flag when it hits the cap; the row is still graded on the partial text.",
    )
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legal = dataset_subjects(args.dataset) + ["all"]
    if args.subject not in legal:
        raise SystemExit(
            f"--subject {args.subject!r} is not a subject of --dataset {args.dataset}; "
            f"expected one of {', '.join(legal)}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "sweep":
        run_sweep(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
