"""Collection IO and naming conventions — the single home of the on-disk contract.

A "collection" is one directory produced by the collector
(src/utils/collect/collection.py):

    ground_truth.csv    one row per problem (GROUND_TRUTH_COLUMNS, plus any
                        columns later stages annotate, e.g. correct_singlepass)
    records.jsonl       per-example prompts, generations, labels
    hidden/<subject>__<idx>.npy   float32 [layers+1, dim] residual states
    head/<subject>__<idx>.npy     float16 (layers, heads*head_dim) o_proj inputs
    args.json           run configuration

Every consumer resolves paths and reads rows through this module, so the
contract cannot drift between the collector, the steering driver and the
diagnostics.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core import common

# -----------------------------------------------------------------------------
# Naming conventions
# -----------------------------------------------------------------------------

GROUND_TRUTH_COLUMNS = [
    "subject",
    "idx",
    "gold_answer",
    "pred_answer",
    "correct",
    "truncated",
    "problem",
]

# Legacy column some older collections carry ("1" = in a class-balanced
# subset). Nothing writes it any more; the loaders below only look at it when
# a caller asks for balanced_only=True, which the live pipeline never does.
BALANCED_COLUMN = "balanced"

# Prompt mode (as accepted by common.make_chat_prompt) -> short tag used in
# collection directory names. Collectable today: "standard" (the baseline)
# and "cot" (the vendor CoT reference ceiling, 2026-08-25 -- NOTE the old
# 2026-08-20 grid also had a `cot` mode; no `_cot` tree from it survives on
# disk, so the tag is unambiguous). The remaining legacy entries stay so tools
# reading an EXISTING collection's stored args.json can still resolve its
# directory tag (_qa, _qabrief, _reg, and the gemma FR baselines'
# _standarddirect trees, whose builder was removed on 2026-09-03).
MODE_TAG = {"standard": "standard", "cot": "cot",
            "standard-direct": "standarddirect", "direct": "reg",
            "qa": "qa", "qa_brief": "qabrief"}


def mode_tag(prompt_mode: str) -> str:
    """Directory tag for a prompt mode."""
    return MODE_TAG[prompt_mode]


def hidden_path(hidden_dir: str, subject: str, idx: Any) -> str:
    """Per-example hidden-state file, keyed by the unique (subject, idx) pair."""
    return os.path.join(hidden_dir, f"{subject}__{idx}.npy")


def head_path(head_dir: str, subject: str, idx: Any) -> str:
    """Per-example head-activation file, keyed like hidden_path."""
    return os.path.join(head_dir, f"{subject}__{idx}.npy")


def collection_root(collection: str) -> str:
    """Return the sweep/report root for a split collection path."""
    collection = os.path.normpath(collection)
    leaf = os.path.basename(collection)
    if leaf.startswith(("train_", "test_")):
        return os.path.dirname(collection)
    return collection


def infer_model_tag(collection: str) -> str:
    """Default report tag for a collection: exp1_<tag>_sweep/train_reg -> tag_reg."""
    root = os.path.basename(collection_root(collection))
    tag = root
    if tag.startswith("exp1_"):
        tag = tag[len("exp1_"):]
    if tag.endswith("_sweep"):
        tag = tag[:-len("_sweep")]
    leaf = os.path.basename(os.path.normpath(collection))
    if "_" in leaf:
        tag = f"{tag}_{leaf.split('_')[-1]}"
    return tag


# -----------------------------------------------------------------------------
# Row-level readers
# -----------------------------------------------------------------------------


def load_ground_truth(collection: str) -> List[Dict[str, str]]:
    csv_path = os.path.join(collection, "ground_truth.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_prompts(collection: str) -> Dict[Tuple[str, str], str]:
    """(subject, idx) -> stored prompt text from records.jsonl.

    This is the fully templated string the source collection fed the model, so
    replaying it reproduces that prompt mode exactly — no rebuilding via
    make_chat_prompt, which could drift if a tokenizer's chat template changed.
    """
    records = common.load_records(os.path.join(collection, "records.jsonl"))
    return {(str(r["subject"]), str(r["idx"])): r["prompt"] for r in records}


def attach_gold_aliases(collection: str, rows: List[Dict[str, Any]]) -> None:
    """Copy each row's `gold_aliases` from records.jsonl onto `rows`, in place.

    ground_truth.csv carries only the single canonical gold answer (its schema is
    frozen), so anything regrading a collection from the CSV would score an
    aliased row (HARP's raw `$...$` form) against one surface form while
    collection scored it against the whole alias list — the same generation
    labelled two ways. Rows of a dataset without aliases, and collections
    predating the field, are left untouched.
    """
    records = common.load_records(os.path.join(collection, "records.jsonl"))
    aliases = {(str(r["subject"]), str(r["idx"])): r.get("gold_aliases")
               for r in records}
    for row in rows:
        got = aliases.get((str(row["subject"]), str(row["idx"])))
        if got:
            row["gold_aliases"] = got


def load_source_args(collection: str) -> Dict[str, Any]:
    path = os.path.join(collection, "args.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def assert_model_matches(collection: str, model_name: str, what: str = "") -> str:
    """Fail unless `model_name` is the model that WROTE this collection.

    Activations, head directions and layer indices are only meaningful together
    with the network they came from, and nothing in a .npy says which that was.
    Every --model_name in this repo carries a default, so a forgotten flag reads
    one model's activations through another model's weights -- silently when the
    dimensions happen to agree, and with a shape error that names neither model
    when they do not.

    The collection records its own author in args.json, so that is the authority
    here. A collection predating the field (no args.json) cannot be checked and
    is allowed through; the returned name is the one to use.
    """
    owner = load_source_args(collection).get("model_name")
    if owner and model_name and owner != model_name:
        raise SystemExit(
            f"model mismatch{' in ' + what if what else ''}:\n"
            f"  {collection}\n"
            f"  was collected with {owner}\n"
            f"  but --model_name says {model_name}\n"
            "These are different networks; their activations and layer indices "
            "are not interchangeable. Fix --model_name (or MODEL_TAG, if the "
            "path is what is wrong).")
    return model_name or owner


def load_tokenizer(model_name: str):
    """Tokenizer from the shared model cache (no model weights loaded)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name, cache_dir=common.model_cache_dir(model_name))


# -----------------------------------------------------------------------------
# CSV <-> .npy alignment (one scan core, three views)
# -----------------------------------------------------------------------------


def scan_npy_dir(
    rows: List[Dict[str, str]], npy_dir: str, ndim: Optional[int] = None,
) -> Tuple[List[str], List[str], set, int, int]:
    """Check every CSV row against its <subject>__<idx>.npy file.

    Returns (missing paths, unreadable/bad-shape paths, shapes seen,
    orphan-file count, files on disk). Shapes are read from the .npy headers
    via mmap, so this stays cheap on thousands of examples.
    """
    referenced, missing, unreadable, shapes = set(), [], [], set()
    for row in rows:
        fname = f"{row['subject']}__{row['idx']}.npy"
        referenced.add(fname)
        path = os.path.join(npy_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            shape = tuple(np.load(path, mmap_mode="r").shape)
        except Exception:
            unreadable.append(path)
            continue
        if ndim is not None and len(shape) != ndim:
            unreadable.append(path)
            continue
        shapes.add(shape)
    on_disk = ({f for f in os.listdir(npy_dir) if f.endswith(".npy")}
               if os.path.isdir(npy_dir) else set())
    return missing, unreadable, shapes, len(on_disk - referenced), len(on_disk)


def verify_collection(out_dir: str, csv_name: str = "ground_truth.csv") -> str:
    """One-line status: are the hidden states complete and aligned with the CSV?"""
    csv_path = os.path.join(out_dir, csv_name)
    if not os.path.exists(csv_path):
        return f"MISSING {csv_path}"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing, unreadable, shapes, orphans, on_disk = scan_npy_dir(
        rows, os.path.join(out_dir, "hidden"))

    status = "OK" if (not missing and not unreadable and len(shapes) <= 1) else "PROBLEM"
    shape_str = (next(iter(shapes)) if len(shapes) == 1
                 else (f"{len(shapes)} DIFFERENT shapes" if shapes else "none"))
    detail = f"{len(rows)} rows, {on_disk} embeddings, shape {shape_str}"
    if missing:
        detail += f", MISSING {len(missing)}"
    if unreadable:
        detail += f", UNREADABLE {len(unreadable)}"
    if orphans:
        detail += f", {orphans} orphaned .npy"
    return f"[{status}] {detail}"


def verify_head_collection(
    exp1_collection: str,
    exp4_collection: str,
) -> Tuple[List[Dict[str, str]], str]:
    """Verify that every CSV row has a corresponding per-head activation file.

    New collections write `head/` next to `hidden/` by default, so consumers
    treat a partial head collection as a hard error instead of silently fitting
    on whatever files happen to exist.
    """
    csv_path = os.path.join(exp1_collection, "ground_truth.csv")
    head_dir = os.path.join(exp4_collection, "head")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    if not os.path.isdir(head_dir):
        raise FileNotFoundError(
            f"Missing {head_dir}. Collections carry head/ next to hidden/ when collected "
            "with --collect_heads; a collection without it has to be re-collected "
            "(the per-head backfill pass was retired)."
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing, unreadable, shapes, orphans, on_disk = scan_npy_dir(rows, head_dir, ndim=2)
    if missing or unreadable or len(shapes) != 1:
        shape_msg = sorted(shapes) if shapes else "none"
        first = missing[0] if missing else (unreadable[0] if unreadable else shape_msg)
        raise FileNotFoundError(
            f"Head activation collection is incomplete or inconsistent: "
            f"{len(rows)} csv rows, {len(missing)} missing, {len(unreadable)} unreadable/bad-shape, "
            f"shapes={shape_msg}, {orphans} orphan files (first issue: {first})."
        )
    return rows, (
        f"head verify: OK — {len(rows)} csv rows, {on_disk} head activations, "
        f"shape {next(iter(shapes))}, {orphans} orphan files"
    )


# -----------------------------------------------------------------------------
# Feature loaders (rows aligned to CSV order)
# -----------------------------------------------------------------------------


def load_head_collection(
    exp1_collection: str,
    exp4_collection: str,
    balanced_only: bool = True,
    drop_truncated: bool = True,
    balanced_column: str = BALANCED_COLUMN,
    exclude_keys: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X (N, L, H*D) float32, y (N,), subjects (N,)) for the rows of the
    ground-truth CSV that have head activations, in CSV order — the same
    alignment contract as the collector (records + hidden/ + head/). `exclude_keys` drops the named
    (subject, idx) rows (the scan hold-out, so com is never fitted on rows
    band and alpha are tuned on)."""
    rows, status = verify_head_collection(exp1_collection, exp4_collection)
    print(status)
    head_dir = os.path.join(exp4_collection, "head")
    X, y, subjects = [], [], []
    expected = 0
    missing = []
    bad = []
    skipped_truncated = 0
    skipped_unbalanced = 0
    skipped_excluded = 0
    for row in rows:
        if exclude_keys and (row["subject"], str(row["idx"])) in exclude_keys:
            skipped_excluded += 1
            continue
        if balanced_only and row.get(balanced_column) != "1":
            skipped_unbalanced += 1
            continue
        if drop_truncated and row["truncated"] != "0":
            skipped_truncated += 1
            continue
        expected += 1
        path = head_path(head_dir, row["subject"], row["idx"])
        if not os.path.exists(path):
            missing.append(path)
            continue
        try:
            X.append(np.load(path))
        except Exception as exc:
            bad.append((path, exc))
            continue
        y.append(int(row["correct"]))
        subjects.append(row["subject"])
    if missing or bad:
        first = missing[0] if missing else f"{bad[0][0]} ({bad[0][1]})"
        raise FileNotFoundError(
            f"Head activation verification failed for {exp4_collection}: "
            f"{len(missing)} missing and {len(bad)} unreadable files among {expected} "
            f"selected rows (first: {first}). Collections are expected to carry head/ "
            "next to hidden/ by default (--collect_heads); a collection missing them "
            "has to be re-collected -- the per-head backfill pass was retired."
        )
    if skipped_unbalanced or skipped_truncated or skipped_excluded:
        print(f"Selected {expected} rows "
              f"(skipped {skipped_unbalanced} unbalanced, {skipped_truncated} truncated, "
              f"{skipped_excluded} excluded scan rows)")
    if not X:
        raise ValueError(
            f"No selected head activations found under {head_dir}. Run --mode balance "
            f"on {exp1_collection} first, and ensure head/ is present."
        )
    return np.stack(X).astype(np.float32), np.array(y, dtype=int), np.array(subjects)


def load_layer_collection(
    exp1_collection: str,
    balanced_only: bool = True,
    drop_truncated: bool = True,
    balanced_column: str = BALANCED_COLUMN,
    drop_embedding: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The per-LAYER counterpart of load_head_collection, read from hidden/.

    Returns (X (N, L, d_model) float32, y (N,), subjects (N,)) for the rows of
    the ground-truth CSV that have a hidden-state file, in CSV order — the same
    alignment contract, the same filters, and the same row ORDER as
    load_head_collection, so a split_val_idxs.npy written against one indexes
    the same problems in the other.

    ``drop_embedding`` removes row 0 of the stored (num_layers + 1, d) stack.
    That row is the embedding output, not a decoder layer, so keeping it would
    put an un-steerable "layer -1" in the ranking and shift every layer index by
    one relative to model.model.layers. With it dropped, X[:, l] is the output
    of decoder layer l, which is exactly the tensor the layer-mode steering hook
    writes into.
    """
    csv_path = os.path.join(exp1_collection, "ground_truth.csv")
    hidden_dir = os.path.join(exp1_collection, "hidden")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    if not os.path.isdir(hidden_dir):
        raise FileNotFoundError(f"Missing {hidden_dir}")

    X: List[np.ndarray] = []
    y: List[int] = []
    subjects: List[str] = []
    missing: List[str] = []
    expected = 0
    skipped_unbalanced = skipped_truncated = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if balanced_only and not (reader.fieldnames and balanced_column in reader.fieldnames):
            raise ValueError(
                f"{csv_path} has no '{balanced_column}' column. Run --mode balance "
                f"first with the matching --majority_frac (or pass --all_rows).")
        for row in reader:
            if balanced_only and row.get(balanced_column) != "1":
                skipped_unbalanced += 1
                continue
            if drop_truncated and row["truncated"] != "0":
                skipped_truncated += 1
                continue
            expected += 1
            path = hidden_path(hidden_dir, row["subject"], row["idx"])
            if not os.path.exists(path):
                missing.append(path)
                continue
            arr = np.load(path)
            X.append(arr[1:] if drop_embedding else arr)
            y.append(int(row["correct"]))
            subjects.append(row["subject"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {expected} selected rows have no hidden state under "
            f"{hidden_dir} (first: {missing[0]}). Re-run --mode collect.")
    if not X:
        raise ValueError(f"No selected hidden states found under {hidden_dir}.")
    if skipped_unbalanced or skipped_truncated:
        print(f"Selected {expected} rows for layer probes "
              f"(skipped {skipped_unbalanced} unbalanced, {skipped_truncated} truncated)")
    return np.stack(X).astype(np.float32), np.array(y, dtype=int), np.array(subjects)


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------


def stratified_sample(rows: Sequence[Dict[str, str]], n: int, seed: int
                      ) -> List[Dict[str, str]]:
    """A seeded sample of `n` rows, proportional within (subject, correct).

    Same stratification as steering_driver.select_rows: the scan rows keep the
    subject mix and the baseline accuracy of the pool they came from, since
    both move the measured effect of steering. n <= 0 returns nothing; n >=
    len(rows) returns every row.
    """
    if n <= 0:
        return []
    if n >= len(rows):
        return list(rows)
    strata: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        strata.setdefault((r["subject"], r["correct"]), []).append(r)
    rng = np.random.default_rng(seed)
    out: List[Dict[str, str]] = []
    for key in sorted(strata):
        group = strata[key]
        k = int(round(n * len(group) / len(rows)))
        pick = rng.choice(len(group), size=min(k, len(group)), replace=False)
        out.extend(group[i] for i in sorted(pick))
    return out


def write_csv(path: str, rows: List[Dict[str, Any]],
              fieldnames: Optional[List[str]] = None) -> None:
    """Atomic CSV write (tmp + os.replace). With ``fieldnames=None`` the header
    is the union of the row keys in first-seen order."""
    if fieldnames is None:
        if not rows:
            return
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    """Atomic JSON write (tmp + os.replace).

    Every stage of run_compass.sh decides it has already run by testing that its
    output file EXISTS. A plain json.dump interrupted halfway leaves a truncated
    file that passes that test, so the crashed stage is skipped on the next run
    and every later stage reads its corrupt output. Renaming a fully written
    temp file makes the artifact appear whole or not at all, which is what makes
    "re-run after a crash" safe rather than merely convenient.
    """
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)
