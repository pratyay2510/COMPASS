#!/usr/bin/env python3
"""The operating point: which (band, alpha) to steer with, and how it was found.

Three CPU modes, seconds each, around the one GPU stage (the scan) that sits
between them:

  plan    elicitation score -> the band grid and the dose-matched alpha grid
          the scan is handed (writes plan.sh; prints the scan commands)
  report  the scan/best table: what every (band, alpha) did, at what dose,
          with the label-free failure signals and the row-set check
  pick    the operating point: the scan config with the highest dev accuracy,
          ties to the smaller alpha; prints "<band> <alpha>" and nothing else

`report` and `pick` read the same gen_*.jsonl files through the same loader,
so the ranking a reader sees in the table is the ranking `pick` applies. The
dose schedule is cosine, always (2026-09-03); files from the retired constant
schedule (no `_cos` in the tag) are ignored by both, and counted once.

Nothing here recomputes a label or touches the test split: `pick` reads the
dev-row scan, and `best` is what runs the pick on test.

THE BAND GRID (plan). The scan is the expensive stage, and what it costs is
decided entirely by which (band, alpha) pairs it is handed. Choosing those by
hand does not port across models: a band is a list of LAYER INDICES, so
"28-31" is the last four layers of a 32-layer llama and does not exist on a
28-layer qwen, and an alpha is a multiplier on a per-model vector norm, so the
same number is a different physical perturbation on every model and every
band. The elicit stage scores every head by the logit shift its injection
delivers toward the elicitation target; summing the positive scores in a layer
gives that layer's elicitation mass. The mass is negligible early and climbs
late, so the useful question is where it lifts off -- the ELBOW -- and the grid
is every nested band from the elbow up: elbow-last, (elbow+1)-last, ...,
last-last. Nested, varying only the floor, because the score is a DIRECT-PATH
measure that credits a head as if nothing downstream touched its output, so it
is trustworthy at the final layer and progressively optimistic earlier;
sweeping the floor finds where that optimism overtakes the extra mass. The
elbow is the lowest layer whose mass, and the mass of every layer above it,
stays at or above `elbow_frac` of the peak layer's mass -- a contiguous
suffix, not a global threshold, because an isolated early spike is exactly the
direct-path artifact this is meant to discount.

THE ALPHA GRID (plan). dose = alpha * sum||r|| over the band's heads is the
size of the perturbation actually added to the residual stream. Comparing
bands at matched ALPHA compares different perturbations; the grid is built at
matched DOSE. The anchor band (the widest, hence largest-dose) fixes a
geometric ladder of doses from its own alpha range, and every other band gets
the alphas that land it on the same doses. Bands then differ in WHICH heads
are steered and in nothing else.

THE TABLE (report).
  dose    sum||r|| for the band's heads (from heads.sh) times alpha -- the
          physical size of the perturbation. Compare bands at matched DOSE.
  delta   accuracy minus the baseline accuracy of the same rows.
  h:h     helped / harmed. A config can gain accuracy while breaking more rows
          than it fixes only if the base rate is lopsided; this shows it.
  <=5w    share of generations still five words or fewer -- elicitation that
          did not happen.
  loop    share gone degenerate (a bigram repeated >= 10 times).
  a*      <=5w + loop, the label-free alpha criterion: unspent elicitation
          plus damage. Its minimum landed one grid step from the accuracy
          optimum on llama/gsm8k, so it is a CHECK on the pick, not the pick.
The row-set check: every config inside one scan must cover the same problems
or the comparison is between different tests. Asserted loudly, because the
failure it catches is silent -- a resumed run that lost rows would otherwise
just look like a better band.

THE PICK. Among scan configs whose row count equals the dev slice (a partial
rung never wins by covering easier rows), the highest accuracy; ties go to the
smaller alpha, then the band name. On roughly 300 dev rows the accuracy carries
two to three points of noise: read the table's a* column beside the pick.

Run (from COMPASS/):
  python3 -m src.steering.operating_point plan   --scores_npz <elicit_scores.npz> \
      --heads_file <heads.sh> --collection <train dir> --emit <plan.sh>
  python3 -m src.steering.operating_point report --scan_root <scan_*> \
      --best_root <best_*> --heads_file <heads.sh>
  python3 -m src.steering.operating_point pick   --scan_root <scan_*> \
      --rows_csv <scan_rows.csv>
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ plan knobs
ELBOW_FRAC = 0.35             # share of the peak layer's mass a band floor must hold
ANCHOR_ALPHAS = (8.0, 32.0)   # alpha range on the anchor band
N_ALPHA = 5                   # rungs on the geometric dose ladder

# --------------------------------------------------------------- table knobs
LOOP_MIN_TOKENS = 10          # a generation shorter than this cannot be judged
LOOP_MIN_REPEATS = 10         # a bigram this frequent is degenerate, not emphasis
SHORT_WORDS = 5               # "still answering directly"

DOSE_RE = re.compile(r"DOSE_(\S+)=([0-9.]+)")
# gen_<tag>.jsonl, tag = H<band>_alpha<a>_<dir>[_prefill]_cos[_skipN][_randheads][_rN]
# (the rule is steering_driver.config_tag; only alpha and the two flags are read here)
ALPHA_RE = re.compile(r"alpha([0-9.]+)_")
COSINE_MARK = "_cos"


# =============================================================================
# plan
# =============================================================================


def layer_mass(score: np.ndarray) -> np.ndarray:
    """Positive elicitation mass per layer: sum of the positive head scores."""
    return np.where(score > 0, score, 0.0).sum(axis=1)


def elbow_layer(mass: np.ndarray, frac: float = ELBOW_FRAC) -> int:
    """Lowest layer of the contiguous top suffix holding >= frac of peak mass.

    Walking down from the last layer and stopping at the first layer that falls
    below the bar -- rather than taking every layer above the bar wherever it
    sits -- is what keeps an isolated early spike out of the grid.
    """
    if mass.size == 0:
        raise SystemExit("empty score array")
    bar = frac * float(mass.max())
    floor = len(mass) - 1
    while floor - 1 >= 0 and mass[floor - 1] >= bar:
        floor -= 1
    return floor


def derive_bands(score: np.ndarray, frac: float = ELBOW_FRAC) -> List[Tuple[int, int]]:
    """Nested bands from the elbow to the last layer, one layer removed each time."""
    mass = layer_mass(score)
    last = score.shape[0] - 1
    return [(lo, last) for lo in range(elbow_layer(mass, frac), last + 1)]


def band_tag(lo: int, hi: int) -> str:
    return f"E{lo}_{hi}"


def read_doses(heads_file: str) -> Dict[str, float]:
    """{band tag -> sum||r|| per unit alpha} from the elicit stage's heads.sh."""
    doses: Dict[str, float] = {}
    if os.path.exists(heads_file):
        with open(heads_file) as f:
            for line in f:
                m = DOSE_RE.match(line.strip())
                if m:
                    doses[m.group(1)] = float(m.group(2))
    return doses


def residual_norm(collection: str, n_files: int = 128) -> Optional[float]:
    """Mean ||x|| of the final-layer residual stream, from the stored hiddens.

    Only for interpretation: it turns a dose into a FRACTION of the state being
    perturbed. Sampled, because the full collection is gigabytes.
    """
    paths = sorted(glob.glob(os.path.join(collection, "hidden", "*.npy")))[:n_files]
    if not paths:
        return None
    norms = []
    for p in paths:
        try:
            norms.append(float(np.linalg.norm(np.load(p)[-1])))
        except Exception:
            continue
    return float(np.mean(norms)) if norms else None


def dose_ladder(anchor_dose: float, alphas: Sequence[float] = ANCHOR_ALPHAS,
                n: int = N_ALPHA) -> List[float]:
    """Geometric ladder of DOSES, from the anchor band's own alpha range."""
    lo, hi = anchor_dose * alphas[0], anchor_dose * alphas[1]
    return list(np.geomspace(lo, hi, n))


def alphas_for(dose_per_alpha: float, doses: Sequence[float]) -> List[float]:
    """The alphas that put this band on each dose of the ladder."""
    return [round(d / dose_per_alpha, 2) for d in doses]


def run_plan(args: argparse.Namespace) -> None:
    z = np.load(args.scores_npz)
    score = np.asarray(z["score"])
    num_layers, num_heads = score.shape
    mass = layer_mass(score)
    bands = derive_bands(score, args.elbow_frac)
    elbow = bands[0][0]
    last = num_layers - 1

    print(f"\nBAND PLAN  ({num_layers} layers x {num_heads} heads)")
    print(f"\nper-layer elicitation mass    bar = {args.elbow_frac:g} x peak "
          f"= {args.elbow_frac * mass.max():.4f}")
    width = 44
    scale = width / max(float(mass.max()), 1e-12)
    for layer in range(num_layers):
        bar = "#" * int(round(mass[layer] * scale))
        mark = " <- elbow" if layer == elbow else ""
        inside = "*" if layer >= elbow else " "
        print(f"  L{layer:<3}{inside} {mass[layer]:8.4f}  {bar}{mark}")
    print(f"\nelbow = L{elbow}: every layer from here to L{last} holds at least "
          f"{args.elbow_frac:g} of the peak.")

    doses = read_doses(args.heads_file)
    missing = [band_tag(lo, hi) for lo, hi in bands if band_tag(lo, hi) not in doses]
    if missing:
        raise SystemExit(
            f"{args.heads_file} has no dose for {', '.join(missing)}.\n"
            "The elicit stage must be run over THESE bands before they can be "
            "dose-matched. With BANDS=auto the plan stage does that for you.")

    # The anchor is the widest band, which carries the most heads and so the
    # largest dose; taking the ladder from it keeps every other band's alphas
    # inside it rather than extrapolating past what any band was measured at.
    anchor = bands[0]
    ladder = dose_ladder(doses[band_tag(*anchor)], args.anchor_alphas, args.n_alpha)
    resid = residual_norm(args.collection) if args.collection else None

    print(f"\ndose-matched alpha grid   anchor = band {anchor[0]}-{anchor[1]}, "
          f"alpha {args.anchor_alphas[0]:g}-{args.anchor_alphas[1]:g}")
    print(f"{'band':>8} {'heads':>6} {'dose/alpha':>11}   alphas at each dose rung")
    for lo, hi in bands:
        dpa = doses[band_tag(lo, hi)]
        alphas = alphas_for(dpa, ladder)
        print(f"{f'{lo}-{hi}':>8} {(hi - lo + 1) * num_heads:>6} {dpa:>11.4f}   "
              + " ".join(f"{a:>7g}" for a in alphas))
    print(f"{'dose':>8} {'':>6} {'':>11}   " + " ".join(f"{d:>7.0f}" for d in ladder))
    if resid:
        print(f"\nmean ||x_final|| = {resid:.1f}; the ladder spans "
              f"{ladder[0] / resid:.2f}-{ladder[-1] / resid:.2f} of it. Above ~1 the "
              f"injection is replacing the residual stream rather than nudging it.")

    print("\n" + "=" * 78)
    print("RUN THESE. One command per band -- the alphas differ per band BY DESIGN,")
    print("so that every band is compared at the same dose.")
    print("=" * 78)
    for lo, hi in bands:
        alphas = " ".join(f"{a:g}" for a in alphas_for(doses[band_tag(lo, hi)], ladder))
        print(f'  BANDS="{lo}-{hi}" ALPHAS="{alphas}" ./run_compass.sh scan --nohup')
    print("\nthen:")
    print("  ./run_compass.sh report     # the table")
    print("  ./run_compass.sh best       # steers the pick on the full test split")
    print("\nThe pick is the highest dev accuracy, ties to the smaller alpha; `best`")
    print("reads it unless BEST_BAND/BEST_ALPHA are set. Nothing above touches test.")

    if args.emit:
        # The same grid, as a sourceable file, so `automate-all` can run the
        # scans without a human re-typing the printed commands. Content is
        # identical to the printout above -- a transcription, not a second
        # derivation.
        with open(args.emit, "w") as fh:
            fh.write("# written by src.steering.operating_point plan -- the "
                     "dose-matched grid, sourceable\n")
            fh.write(f'PLAN_BANDS="{" ".join(f"{lo}-{hi}" for lo, hi in bands)}"\n')
            for lo, hi in bands:
                alphas = " ".join(f"{a:g}" for a in
                                  alphas_for(doses[band_tag(lo, hi)], ladder))
                fh.write(f'PLAN_ALPHAS_{lo}_{hi}="{alphas}"\n')
        print(f"\nwrote {args.emit}")


# =============================================================================
# the shared loader (report + pick)
# =============================================================================


def is_loop(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < LOOP_MIN_TOKENS:
        return False
    top = collections.Counter(zip(tokens, tokens[1:])).most_common(1)
    return bool(top) and top[0][1] >= LOOP_MIN_REPEATS


def summary_n(gen_path: str) -> int:
    """The addressed-row count the steering run recorded beside this gen file
    (summary_<tag>.json `n`), or 0 when there is none."""
    tag = os.path.basename(gen_path)[len("gen_"):-len(".jsonl")]
    path = os.path.join(os.path.dirname(gen_path), f"summary_{tag}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return int(json.load(f).get("n") or 0)
    except (OSError, ValueError):
        return 0


def load_configs(root: str, kind: str) -> List[Dict]:
    """One entry per cosine, non-control gen_*.jsonl under root/<band>/.

    Constant-schedule files (no `_cos` in the tag: every rung before
    2026-08-29) are counted and skipped -- the driver no longer produces
    them, and a constant rung and a cosine rung at one alpha are different
    interventions.
    """
    out = []
    n_constant = 0
    for band_dir in sorted(glob.glob(os.path.join(root, "*"))):
        band = os.path.basename(band_dir)
        # gen_<tag>_llmfmt.jsonl is the LLM formatter's per-row cache
        # (subject/idx/formatted_text only), not a steered config -- the same
        # exclusion grading.grade_gen_dir applies.
        for path in sorted(p for p in glob.glob(os.path.join(band_dir, "gen_*.jsonl"))
                           if not p.endswith("_llmfmt.jsonl")):
            name = os.path.basename(path)
            m = ALPHA_RE.search(name)
            if not m or "rand" in name:      # random-head / random-direction controls
                continue
            if COSINE_MARK not in name:
                n_constant += 1
                continue
            # Same reading rule as steering_driver.load_done / common.load_records:
            # a torn line (interrupted or concurrent append) is skipped, and a
            # (subject, idx) written twice keeps its LAST copy, so this counts
            # exactly the rows a resumed run would. Both are reported: a file
            # that needed either may also be MISSING rows, which the row-set
            # check shows.
            rows, bad = [], 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        bad += 1
            by_key = {(str(r.get("subject", "")), str(r.get("idx", ""))): r for r in rows}
            if bad or len(by_key) != len(rows):
                print(f"!! {path}: {bad} unparsable line(s) skipped, "
                      f"{len(rows) - len(by_key)} duplicate row(s) collapsed "
                      f"(torn/concurrent writes) -- {len(by_key)} rows kept",
                      file=sys.stderr)
            rows = list(by_key.values())
            if rows:
                out.append({"kind": kind, "band": band, "alpha": float(m.group(1)),
                            "rows": rows, "path": path})
    if n_constant:
        print(f"({kind}: {n_constant} constant-schedule file(s) under {root} ignored)",
              file=sys.stderr)
    return out


def summarise(cfg: Dict, dose_per_alpha: Optional[float]) -> Dict:
    rows = cfg["rows"]
    # the one convention: correct / ALL addressed rows; a truncated row counts
    # by its label (2026-09-01 -- graded on partial text, formatter OR on top).
    # Files written since 2026-08-29 hold every addressed row (skipped ones as
    # incorrect records); the summary's `n` covers older whole-split files.
    n = max(len(rows), summary_n(cfg["path"]))
    base = sum(int(r["baseline_correct"]) for r in rows) / n
    acc = sum(int(r["correct"]) for r in rows) / n
    helped = sum(1 for r in rows if not r["baseline_correct"] and r["correct"])
    harmed = sum(1 for r in rows if r["baseline_correct"] and not r["correct"])
    short = sum(1 for r in rows if len(r["generated_text"].split()) <= SHORT_WORDS) / n
    loop = sum(1 for r in rows if is_loop(r["generated_text"])) / n
    return {**{k: cfg[k] for k in ("kind", "band", "alpha", "path")},
            "n": n, "base": base, "acc": acc, "delta": acc - base,
            "helped": helped, "harmed": harmed,
            "trunc": sum(int(r["truncated"]) for r in rows) / n,
            "short": short, "loop": loop, "crit": short + loop,
            "dose": dose_per_alpha * cfg["alpha"] if dose_per_alpha else None}


def row_ids(cfg: Dict) -> frozenset:
    return frozenset((str(r.get("subject", "")), str(r.get("idx", ""))) for r in cfg["rows"])


def check_row_sets(configs: Sequence[Dict], raw: Sequence[Dict]) -> List[str]:
    """Every scan config must cover the same problems. Returns warning lines."""
    by_kind: Dict[str, Dict[frozenset, List[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for summary, cfg in zip(configs, raw):
        by_kind[summary["kind"]][row_ids(cfg)].append(
            f"{summary['band']} alpha{summary['alpha']:g}")
    warnings = []
    for kind, groups in by_kind.items():
        if len(groups) > 1:
            warnings.append(
                f"!! {kind}: {len(groups)} DIFFERENT row sets across configs -- these "
                f"numbers are not comparable to each other.")
            for ids, tags in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                warnings.append(f"     {len(ids):>5} rows: {', '.join(tags)}")
    return warnings


# =============================================================================
# pick
# =============================================================================


def count_rows(rows_csv: str) -> int:
    with open(rows_csv, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def pick(scan_root: str, n_rows: int) -> Tuple[str, float, Dict]:
    """The operating point: (band, alpha, summary) of the best dev-row config.

    Only configs covering exactly the dev slice compete (n == n_rows), so a
    rung that stopped early cannot win on the rows it happened to finish.
    Highest accuracy; ties to the smaller alpha, then the band name.
    """
    raw = load_configs(scan_root, "scan")
    summaries = [summarise(c, None) for c in raw]
    eligible = [s for s in summaries if s["n"] == n_rows]
    if not eligible:
        seen = sorted({s["n"] for s in summaries})
        raise SystemExit(
            f"no scan config covers the {n_rows} dev rows under {scan_root}"
            + (f" (row counts on disk: {seen})" if seen else " (no cosine gen files)"))
    best = min(eligible, key=lambda s: (-s["acc"], s["alpha"], s["band"]))
    return best["band"][1:], best["alpha"], best


def run_pick(args: argparse.Namespace) -> None:
    band, alpha, _ = pick(args.scan_root, count_rows(args.rows_csv))
    print(f"{band} {alpha:g}")


# =============================================================================
# report
# =============================================================================


HEADER = (f"{'':4} {'band':9} {'alpha':>6} {'dose':>8} {'n':>5} {'base':>6} "
          f"{'acc':>7} {'delta':>7} {'help':>5} {'harm':>5} {'h:h':>6} "
          f"{'trunc':>6} {'loop':>6} {'<=5w':>6} {'a*':>6}")


def format_row(s: Dict, mark: str) -> str:
    ratio = f"{s['helped'] / s['harmed']:6.2f}" if s["harmed"] else "   inf"
    dose = f"{s['dose']:8.1f}" if s["dose"] is not None else "       -"
    return (f"{s['kind'][:4]:4} {s['band']:9} {s['alpha']:>6g} {dose} {s['n']:>5} "
            f"{s['base']:>6.3f} {s['acc']:>7.4f} {s['delta']:>+7.4f} "
            f"{s['helped']:>5} {s['harmed']:>5} {ratio} {s['trunc']:>6.3f} "
            f"{s['loop']:>6.3f} {s['short']:>6.3f} {s['crit']:>6.3f}{mark}")


def run_report(args: argparse.Namespace) -> None:
    dose = read_doses(args.heads_file)
    raw = (load_configs(args.scan_root, "scan")
           + load_configs(args.best_root, "best"))
    if not raw:
        print("nothing to report yet -- run scan")
        return
    raw.sort(key=lambda c: (c["kind"], c["band"], c["alpha"]))
    summaries = [summarise(c, dose.get(c["band"].replace("-", "_"))) for c in raw]

    # a* minimum per band, marked in the table
    best_crit: Dict[Tuple[str, str], float] = {}
    for s in summaries:
        key = (s["kind"], s["band"])
        best_crit[key] = min(best_crit.get(key, float("inf")), s["crit"])

    print(HEADER)
    last = None
    for s in summaries:
        if last is not None and (s["kind"], s["band"]) != last:
            print()
        last = (s["kind"], s["band"])
        mark = " *" if s["crit"] == best_crit[last] else ""
        print(format_row(s, mark))

    for line in check_row_sets(summaries, raw):
        print(line)

    scans = [s for s in summaries if s["kind"] == "scan"]
    if scans:
        # the pick, by the same rule `pick` applies: among configs on the
        # dev slice (the most common row count, when no --rows_csv is given)
        n_rows = (count_rows(args.rows_csv)
                  if args.rows_csv and os.path.exists(args.rows_csv) else
                  collections.Counter(s["n"] for s in scans).most_common(1)[0][0])
        eligible = sorted((s for s in scans if s["n"] == n_rows),
                          key=lambda s: (-s["acc"], s["alpha"], s["band"]))
        print(f"\noperating point on the {n_rows} dev rows (TRAIN rows held out of the")
        print("fit; nothing here has touched the test split):")
        for rank, s in enumerate(eligible[:2], 1):
            print(f"  {rank}. band {s['band'][1:]} alpha {s['alpha']:g}  "
                  f"acc {s['acc']:.4f} ({s['delta']:+.4f})  a* {s['crit']:.3f}")
        if eligible:
            s = eligible[0]
            print(f"\n`./run_compass.sh best` steers this pick on the full test split;"
                  f" to override:")
            print(f"  BEST_BAND={s['band'][1:]} BEST_ALPHA={s['alpha']:g} "
                  f"./run_compass.sh best --nohup")
    print("\ndose = sum||r|| x alpha: compare BANDS at matched dose, not matched alpha.")
    print("a* (marked *) is the label-free alpha criterion, <=5w + loop. It is a check")
    print("on the pick, not the pick itself.")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    q = sub.add_parser("plan", help="band grid + dose-matched alpha grid for the scan")
    q.add_argument("--scores_npz", required=True, help="elicit_scores.npz")
    q.add_argument("--heads_file", required=True, help="heads.sh, for the per-band dose")
    q.add_argument("--collection", default=None,
                   help="train collection, for the residual-norm reference")
    q.add_argument("--elbow_frac", type=float, default=ELBOW_FRAC)
    q.add_argument("--anchor_alphas", type=float, nargs=2, default=list(ANCHOR_ALPHAS))
    q.add_argument("--n_alpha", type=int, default=N_ALPHA)
    q.add_argument("--emit", default=None,
                   help="also write the grid as a sourceable shell file "
                        "(PLAN_BANDS / PLAN_ALPHAS_lo_hi), for automate-all")
    q.set_defaults(func=run_plan)

    r = sub.add_parser("report", help="the scan/best table")
    r.add_argument("--scan_root", required=True)
    r.add_argument("--best_root", required=True)
    r.add_argument("--heads_file", required=True,
                   help="elicit stage's heads.sh, for the per-band dose")
    r.add_argument("--rows_csv", default=None,
                   help="the dev slice (scan_rows.csv); names the pick exactly as "
                        "`pick` does. Default: the scan's most common row count")
    r.set_defaults(func=run_report)

    k = sub.add_parser("pick", help='the operating point: prints "<band> <alpha>"')
    k.add_argument("--scan_root", required=True)
    k.add_argument("--rows_csv", required=True,
                   help="the dev slice (scan_rows.csv); only configs covering "
                        "exactly these rows compete")
    k.set_defaults(func=run_pick)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
