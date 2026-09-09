#!/usr/bin/env python3
"""Inspect a collection or a live steering sweep.

Two targets, auto-detected from the path:

  collection dir      what split/prompt it used, accuracy, and what a token cap costs
  steering dir/file   per-config steered vs baseline accuracy, safe to run mid-sweep

The steering view reads gen_<config>.jsonl, which steering_driver.py appends to as
it goes, so it reports on however many rows exist right now. Partial files are
labelled as such: subject strata are interleaved in the row order, so a partial
read is roughly representative, but the paired test is what says whether a delta
has separated from noise yet.
"""
import csv, json, os, sys, glob
import numpy as np

from src.core.io import load_tokenizer

CAPS = (32, 48, 64, 96, 128, 192, 256, 384, 512, 1024)


def cap_table(gen, ok, max_new):
    """What each candidate --max_new_tokens would cost in kept correct answers."""
    print(f"\n=== cost of a cap (correct answers you would lose) ===")
    print(f"  {'cap':>6}{'rows finishing':>16}{'correct kept':>14}{'decode saved':>14}")
    tot = max(gen.sum(), 1)
    for c in [c for c in CAPS if c < max_new] + [max_new]:
        keep = gen <= c
        print(f"  {c:>6}{keep.mean():>15.1%}{(ok & keep).sum()/max(ok.sum(),1):>13.1%}"
              f"{1-np.minimum(gen,c).sum()/tot:>13.0%}")
    print("\nPick the smallest cap that keeps ~100% of correct answers.")


def token_stats(gen):
    print(f"\n=== generated tokens ===\n  mean {gen.mean():.0f}  median {np.median(gen):.0f}  "
          f"p90 {np.percentile(gen,90):.0f}  p99 {np.percentile(gen,99):.0f}  max {gen.max()}")


def word_stats(w):
    """Length of the answer in words, the unit the CoT-switch read is stated in.

    Whitespace tokens of the generation, so it is comparable across models and
    is the same count `frac(output > 10 words)` thresholds.
    """
    w = np.asarray(w, float)
    print(f"\n=== generated words ===\n  mean {w.mean():.1f}  median {np.median(w):.0f}  "
          f"p90 {np.percentile(w,90):.0f}  max {int(w.max())}  "
          f"frac(>10 words) {(w > 10).mean():.1%}")


# ---------------------------------------------------------------- collection

def report_collection(d):
    a = json.load(open(f"{d}/args.json"))
    print("=== what was actually run ===")
    for k in ("model_name","prompt_mode","word_budget","split","num_samples","subject",
              "max_new_tokens","keep_truncated","temperature","collect_heads"):
        if k in a: print(f"  {k:<18} {a[k]}")

    rows = list(csv.DictReader(open(f"{d}/ground_truth.csv")))
    trunc = np.array([int(r["truncated"]) for r in rows])
    # the one convention: correct / ALL rows; a truncated row counts by its
    # stored label (graded on partial text; formatter OR may lift it later)
    corr = np.array([int(r["correct"]) for r in rows])
    print(f"\n=== result ===\n  n={len(rows)}  accuracy={corr.mean():.4f}  "
          f"(over all rows; {trunc.sum()} truncated, graded on partial text)")
    se = (corr.mean()*(1-corr.mean())/len(rows))**.5
    print(f"  95% CI +/- {1.96*se:.3f}   (base Llama direct on full test = 0.1401)")

    tk = load_tokenizer(a["model_name"])
    key = {(r["subject"], r["idx"]): int(r["correct"]) for r in rows}
    gen, ok, words = [], [], []
    for line in open(f"{d}/records.jsonl"):
        r = json.loads(line)
        gen.append(len(tk(r["generated_text"]).input_ids))
        ok.append(key.get((r["subject"], str(r["idx"])), 0))
        words.append(len(r["generated_text"].split()))
    gen, ok = np.array(gen), np.array(ok, bool)
    token_stats(gen)
    print(f"  words/token {sum(words)/max(gen.sum(),1):.2f}")
    word_stats(words)
    if np.median(gen) > 40:
        print("\n  *** The model is REASONING despite the direct prompt. This collection is")
        print("      effectively CoT, and a low --max_new_tokens would decapitate it. ***")
    cap_table(gen, ok, a.get("max_new_tokens", 2048))


# ------------------------------------------------------------------ steering

def report_steering_file(path, a, tk, verbose):
    """One gen_<config>.jsonl. Returns a summary dict for the roll-up table."""
    cfg = os.path.basename(path)[4:-6]  # gen_<cfg>.jsonl
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    if not n:
        print(f"\n### {cfg}: empty"); return None

    steered = np.array([r["correct"] for r in rows], bool)
    base = np.array([r["baseline_correct"] for r in rows], bool)
    trunc = np.array([r["truncated"] for r in rows], bool)
    reused = np.array([r.get("generated_text") is None for r in rows], bool)
    w2r = int((steered & ~base).sum())   # steering fixed it
    r2w = int((~steered & base).sum())   # steering broke it

    # --subset is an upper bound (it caps a stratified draw), so rows never
    # reaching it does not mean unfinished. run_config writes summary_<cfg>.json
    # only after the config completes, so that file is the real signal.
    target = a.get("subset") or 0
    finished = os.path.exists(os.path.join(os.path.dirname(path), f"summary_{cfg}.json"))
    done = f"{n}" + (f"/{target}" if target and not finished else "")
    state = "COMPLETE" if finished else "in progress"
    delta = steered.mean() - base.mean()
    # paired binary test: only the disagreements carry information
    z = (w2r - r2w) / (w2r + r2w) ** .5 if (w2r + r2w) else 0.0

    print(f"\n### {cfg}   rows {done}   [{state}]")
    print(f"  steered  {steered.sum():>5}/{n} = {steered.mean():.4f}")
    print(f"  baseline {base.sum():>5}/{n} = {base.mean():.4f}")
    print(f"  delta    {delta:+.4f}   wrong->right {w2r}   right->wrong {r2w}   z={z:+.2f}"
          f"{'  (n.s.)' if abs(z) < 1.96 else '  *'}")
    print(f"  truncated {trunc.sum()} ({trunc.mean():.1%})")

    # Average answer length of THIS config's own generations (rows with no
    # text, from older runs, are excluded rather than counted as zero-length).
    words = [len(r["generated_text"].split()) for r in rows
             if r["generated_text"] is not None]
    mean_words = float(np.mean(words)) if words else float("nan")
    if words:
        print(f"  words/sample {mean_words:.1f}   frac(>10 words) "
              f"{np.mean(np.array(words) > 10):.1%}   (over {len(words)} generated rows)")

    if verbose and (~reused).any():
        texts = [r["generated_text"] for r in rows if r["generated_text"] is not None]
        gen = np.array([len(x) for x in tk(texts, add_special_tokens=False)["input_ids"]])
        gen[trunc[~reused]] = a.get("max_new_tokens", 2048)
        token_stats(gen)
        word_stats(words)
        cap_table(gen, steered[~reused], a.get("max_new_tokens", 2048))

    return dict(cfg=cfg, n=n, state=state, steered=steered.mean(), base=base.mean(),
                delta=delta, w2r=w2r, r2w=r2w, z=z, trunc=trunc.mean(),
                words=mean_words)


def report_steering(target):
    if not os.path.exists(target):
        sys.exit(f"nothing at {target} yet — the sweep writes args.json on launch")
    d = target if os.path.isdir(target) else os.path.dirname(target)
    if not os.path.exists(f"{d}/args.json"):
        sys.exit(f"no args.json in {d} — is this a steering dir?")
    a = json.load(open(f"{d}/args.json"))
    print("=== what is running ===")
    for k in ("model_name","direction_name","K","alpha","subset","max_new_tokens",
              "control_repeats"):
        if k in a: print(f"  {k:<18} {a[k]}")
    print(f"  {'collections':<18} {a.get('exp1_test_collection')}")

    files = sorted(glob.glob(f"{d}/gen_*.jsonl")) if os.path.isdir(target) else [target]
    if not files:
        print(f"\nno gen_*.jsonl in {d} yet"); return
    expect = len(a.get("K", [])) * len(a.get("alpha", []))
    print(f"\n{len(files)} config file(s) present" + (f" of {expect} in the grid" if expect else ""))

    tk = load_tokenizer(a["model_name"])
    out = [report_steering_file(f, a, tk, len(files) == 1) for f in files]
    out = [o for o in out if o]
    if len(out) > 1:
        print(f"\n=== roll-up ===")
        print(f"  {'config':<22}{'rows':>7}{'steered':>9}{'base':>8}{'delta':>9}"
              f"{'w->r':>6}{'r->w':>6}{'z':>7}{'trunc':>8}{'words':>8}")
        for o in sorted(out, key=lambda x: -x["delta"]):
            print(f"  {o['cfg']:<22}{o['n']:>7}{o['steered']:>9.4f}{o['base']:>8.4f}"
                  f"{o['delta']:>+9.4f}{o['w2r']:>6}{o['r2w']:>6}{o['z']:>+7.2f}{o['trunc']:>7.1%}"
                  f"{o['words']:>8.1f}")
    print("\nz is a paired test on the disagreements; |z|<1.96 means the delta has not\n"
          "separated from noise yet. Partial files are interleaved across subjects.")


USAGE = """usage: python3 -m src.utils.diagnostics.check_collection <path>

  <path>  a collection dir      -> settings, accuracy, and what a token cap costs
          a steering dir/file   -> per-config steered vs baseline

The target kind is auto-detected. Both views are read-only and safe to run
while a sweep is still generating."""


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    t = sys.argv[1].rstrip("/")
    if not os.path.exists(t):
        sys.exit(f"no such path: {t}")
    if os.path.isfile(t) or glob.glob(f"{t}/gen_*.jsonl") or t.endswith("steering"):
        report_steering(t)
    else:
        report_collection(t)
