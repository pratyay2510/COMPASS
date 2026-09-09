#!/usr/bin/env python3
"""Regenerate info/scores.json -- the reproducibility ledger -- from disk.

Every accuracy in the ledger is recomputed from the row files under $EMBED_ROOT,
never copied from a log or a summary. Every provenance field (band, alpha,
heads, schedule, fitting pool, caps, seeds) is read from the artifacts the
run itself wrote (summary_*.json, args.json, heads.sh, the fit stamps).

Grading rule (user decision 2026-09-01): a row is correct if the frozen
family grader accepts the raw generation OR the formatter's restatement of
it; truncated rows are graded on their partial text; the denominator is
EVERY row of the split; a steered row the steerer skipped (its baseline
generation was truncated, so it was never steered) counts INCORRECT.

    cd COMPASS
    python3 -m src.utils.diagnostics.score_cells info/scores.json          # rewrite in place
    python3 -m src.utils.diagnostics.score_cells info/scores.json --check  # recompute, diff, write nothing

The file is its own manifest: only `model`, `dataset`, `baseline.collection`
and `steered.gen_file` (plus free-text `job_card` / `notes`) are inputs; all
other fields under `baseline` / `steered` are overwritten on every run.
Hand-written sections outside `cells` are preserved verbatim.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from src.core import common, grading
from src.core.grading import singlepass_label, truncated_row, verdict

Key = tuple


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _key(r: Dict[str, Any]) -> Key:
    return (r["subject"], str(r["idx"]))


def _json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt_map(path: str) -> Optional[Dict[Key, str]]:
    if not os.path.exists(path):
        return None
    return {_key(f): f["formatted_text"] for f in _load(path)}


def _skipped(r: Dict[str, Any]) -> bool:
    return bool(int(r.get("skipped", 0) or 0)) or r.get("generated_text") is None \
        or r.get("steered") is False


def _rel(path: Optional[str]) -> Optional[str]:
    """Paths are stored relative to the embeddings root so the ledger reads."""
    if path is None:
        return None
    root = common.paths.require("EMBED_ROOT")
    return path.replace(root.rstrip("/") + "/", "")


def tally(model: str, coll: str, gen: Optional[str]) -> Dict[str, Any]:
    grader = grading.grader_for(model)
    recs = _load(os.path.join(coll, "records.jsonl"))
    meta = {_key(r): r for r in recs}
    if gen is None:
        rows, n_all = recs, len(recs)
        fm = _fmt_map(os.path.join(coll, "llm_formatted_answers.jsonl"))
    else:
        rows = _load(gen)
        n_all = len(rows)
        fm = _fmt_map(gen[:-len(".jsonl")] + "_llmfmt.jsonl")
    n_ok = n_sp = n_trunc = n_skip = n_fmt = 0
    for r in rows:
        if gen is not None and _skipped(r):
            n_skip += 1
            continue
        m = meta.get(_key(r), r)
        sp = bool(singlepass_label(r))
        ok = sp
        if fm is not None and _key(r) in fm:
            n_fmt += 1
            ok = ok or verdict(grader, fm[_key(r)], str(m["gold_answer"]),
                               m.get("gold_aliases", []))
        n_sp += sp
        n_ok += ok
        n_trunc += truncated_row(r)
    out = {
        "n_rows": n_all,
        "accuracy": round(100.0 * n_ok / n_all, 2),
        "accuracy_singlepass_only": round(100.0 * n_sp / n_all, 2),
        "n_correct": n_ok,
        "n_truncated": n_trunc,
        "n_skipped_counted_wrong": n_skip,
        "n_rows_with_formatter_verdict": n_fmt,
        "formatter_pass": "complete" if fm is not None and n_fmt == n_all - n_skip
                          else ("partial" if fm is not None else "MISSING"),
    }
    return out


def describe_baseline(model: str, coll: str) -> Dict[str, Any]:
    a = _json(os.path.join(coll, "args.json")) or {}
    d = {
        "collection": _rel(coll),
        "prompt_mode": a.get("prompt_mode"),
        "max_new_tokens": a.get("max_new_tokens"),
        "seed": a.get("seed"),
        "dtype": a.get("dtype"),
        "batch_size": a.get("batch_size"),
        "decoding": "greedy",
    }
    d.update(tally(model, coll, None))
    return d


def fit_label_evidence(fit_coll: Optional[str]) -> Optional[str]:
    """What the fitting pool's labels are TODAY: 'singlepass' if `correct`
    equals `correct_singlepass` on every row (no formatter relabel applied),
    'formatter' if they differ, None if the pool has no ground_truth.csv."""
    if not fit_coll:
        return None
    gt = os.path.join(fit_coll, "ground_truth.csv")
    if not os.path.exists(gt):
        return None
    import csv
    csv.field_size_limit(sys.maxsize)
    with open(gt, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "correct_singlepass" not in rows[0]:
        return "singlepass"
    diff = sum(1 for r in rows if str(r.get("correct", "")).strip() != str(r.get("correct_singlepass", "")).strip()
               and str(r.get("correct_singlepass", "")).strip() != "")
    return "formatter" if diff else "singlepass"


def describe_steered(model: str, coll: str, gen: str) -> Dict[str, Any]:
    gdir = os.path.dirname(gen)
    cfg = os.path.basename(gen)[len("gen_"):-len(".jsonl")]
    summ = _json(os.path.join(gdir, f"summary_{cfg}.json")) or {}
    args = _json(os.path.join(gdir, "args.json")) or {}
    fit_dir = args.get("probe_report")
    heads_sh = os.path.join(fit_dir, "heads.sh") if fit_dir else None
    elicit = _json(os.path.join(fit_dir, "elicit_stamp.json")) if fit_dir else None
    target = _json(os.path.join(fit_dir, "target_stamp.json")) if fit_dir else None
    m = re.match(r"HE(\d+)-(\d+)_alpha([0-9.]+)_", cfg)
    band = f"{m.group(1)}-{m.group(2)}" if m else summ.get("heads_tag")
    heads = summ.get("top_heads") or []
    d = {
        "gen_file": _rel(gen),
        "summary_file": _rel(os.path.join(gdir, f"summary_{cfg}.json")),
        "config": cfg,
        "band": band,
        "alpha": summ.get("alpha", float(m.group(3)) if m else None),
        "K": summ.get("K"),
        "heads": [f"L{l}H{h}" for l, h in heads],
        "direction": summ.get("direction"),
        "schedule": summ.get("schedule") or "constant",
        "rampdown_tokens": summ.get("rampdown"),
        "ramp_floor": summ.get("ramp_floor"),
        "site": args.get("site"),
        "steer_steps": args.get("steer_steps"),
        "gate": "none" if args.get("no_gate", True) else args.get("gate_agg"),
        "steer_max_new_tokens": args.get("max_new_tokens"),
        "max_prompt_tokens": args.get("max_prompt_tokens"),
        "seed": args.get("seed"),
        "dtype": args.get("dtype"),
        "fit": {
            "fit_collection": _rel(args.get("exp1_train_collection")),
            "fit_collection_labels_now": fit_label_evidence(args.get("exp1_train_collection")),
            "fit_dir": _rel(fit_dir),
            "heads_file": _rel(heads_sh),
            "head_selection": (target or {}).get("params", {}).get("head_selection"),
            "target_prompts": (target or {}).get("params", {}).get("n_prompts"),
            "target_seed": (target or {}).get("params", {}).get("seed"),
            "target_agg": (target or {}).get("params", {}).get("agg"),
            "readout": (elicit or {}).get("params", {}).get("readout"),
            "elbow_frac": (elicit or {}).get("params", {}).get("elbow_frac"),
            "bands": (elicit or {}).get("params", {}).get("bands"),
            "direction_mode": (elicit or {}).get("params", {}).get("direction_mode"),
        },
    }
    d.update(tally(model, coll, gen))
    return d


def regenerate(ledger: Dict[str, Any]) -> Dict[str, Any]:
    root = common.paths.require("EMBED_ROOT")
    out = copy.deepcopy(ledger)
    out["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    for cell in out["cells"]:
        model = cell["model"]
        coll = os.path.join(root, cell["baseline"]["collection"])
        keep_b = {k: cell["baseline"].get(k) for k in ("collection",)}
        cell["baseline"] = {**describe_baseline(model, coll), **{k: v for k, v in keep_b.items() if v}}
        st = cell.get("steered")
        if st and st.get("gen_file"):
            gen = os.path.join(root, st["gen_file"])
            cell["steered"] = describe_steered(model, coll, gen)
            cell["delta"] = round(cell["steered"]["accuracy"] - cell["baseline"]["accuracy"], 2)
        else:
            cell["delta"] = None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ledger", help="info/scores.json")
    p.add_argument("--check", action="store_true", help="recompute and report differences, write nothing")
    args = p.parse_args()
    with open(args.ledger, encoding="utf-8") as f:
        old = json.load(f)
    new = regenerate(old)
    print(f"{'model':<22} {'dataset':<9} {'base':>6} {'steer':>6} {'delta':>6}  band   alpha  fmt(base/steer)")
    for c in new["cells"]:
        b, s = c["baseline"], c.get("steered") or {}
        print(f"{c['model'].split('/')[-1]:<22} {c['dataset']:<9} {b['accuracy']:>6.2f} "
              f"{s.get('accuracy', float('nan')):>6.2f} {c['delta'] if c['delta'] is not None else float('nan'):>6.2f}  "
              f"{s.get('band', '-'):<6} {s.get('alpha', '-'):<6} {b['formatter_pass']}/{s.get('formatter_pass', '-')}")
    if args.check:
        diffs = []
        for o, n in zip(old["cells"], new["cells"]):
            for side in ("baseline", "steered"):
                if o.get(side) and n.get(side) and o[side].get("accuracy") != n[side].get("accuracy"):
                    diffs.append((n["model"], n["dataset"], side, o[side].get("accuracy"), n[side].get("accuracy")))
        print("\nno differences" if not diffs else "\nDIFFERENCES: " + "; ".join(map(str, diffs)))
        return
    tmp = args.ledger + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, args.ledger)
    print(f"\nwrote {args.ledger}")


if __name__ == "__main__":
    main()
