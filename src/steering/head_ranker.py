#!/usr/bin/env python3
"""Rank heads by how much their steering vector promotes reasoning, in token space.

Motivation. Probe validation accuracy has failed twice as a selection criterion:
inside layers 20-27 the top-8 by val_acc produced no more helps than eight
RANDOM heads (136 vs 141), and two draws of the same collection picked top-8
sets overlapping in 3 heads while differing by 7.5 pp of steering effect. The
ranking is noise-dominated -- 1024 heads spread over a range narrower than one
standard error of the estimate. It needs replacing, not tuning.

This module ranks by causal effect on the output distribution instead, which is
possible because the mechanism turned out to be legible. Reading the injected
vector through the unembedding shows the winning head (L31H26, 28.8% of the
set's dose) promotes ' To', 'To', ' Since', ' Rather' and suppresses ' $', ' £'
-- the opening tokens of a worked solution against those of a bare numeric
answer. The steered generations begin "To find the total cost, we need to..."
verbatim. So the intervention works by promoting one token at one position.

THE SCORE. Let p_cot and p_direct be the first-token distributions of the
generations that reasoned (>5 words) and those that answered directly, measured
from sweeps already on disk -- no hand-picked token list. Their difference is
the logit shift we want, so pull it back into the residual stream through the
unembedding,

    u = (p_cot - p_direct) @ U            (d_model,)

and score head h by the logit shift its injection delivers through the head's
own output projection,

    delta_h = W_O^h (sigma_h theta_h)     the vector added at alpha=1
    score_h = <u, delta_h>                signed, dose-weighted
    cos_h   = score_h / (||u|| ||delta_h||)     dose-free

score is what to rank by for steering (a head that points the right way but
transmits nothing is useless); cos isolates direction from magnitude.

THE FINAL-NORM GAIN IS DELIBERATELY LEFT OUT (2026-08-27). The exact
first-order readout is <u, g * delta_h> with g the final RMSNorm gain -- a
diagonal reweighting of the residual stream. Measured on the four selections
the pipeline uses (llama gsm8k E30-31, llama MATH-train E29-31, qwen3 gsm8k
E34-35, qwen3 MATH-calib E34-35), dropping g left three of the four top-8 sets
identical and moved two rank-7/8 heads on the fourth -- slots a bootstrap over
the training rows could not pin down with g in place either (P(top-8) =
0.62/0.95), between heads mutually orthogonal in residual space. The g-free
score is the simpler statement of the same criterion -- "project the head's
write onto the unembedded target" -- so it is the one used. Selections made
before this date were ranked with g; the measurement is docs/gain-ablation.md.

CAVEAT -- this is a DIRECT-PATH measure. It credits a head for the logit shift
its output produces if nothing downstream touches it. That is nearly exact at
layer 31 and increasingly optimistic earlier, where the delta still has many
layers to pass through. Expect the ranking to over-credit early layers, and
keep the band restriction rather than trusting a raw global top-K. Layers 20-27
underperformed by a factor of six empirically, which no direct-path score can
see.

Run (CPU, a few minutes; needs W_O + lm_head from the local model cache):
  python3 -m src.steering.head_ranker \
      --probe_report <.../outcome_standard> \
      --exp1_train_collection <.../train_standard> \
      --gen_dir <.../steering_headsL28_31> \
      --bands 28-31 26-31 --K 8
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core import common, io
from src.steering import operating_point, resume, steering_driver

COT_MIN_WORDS = 5          # "reasoned" = more than this many words

# --------------------------------------------------------------------------
# Reading W_O straight out of the checkpoint (no model materialised)
# --------------------------------------------------------------------------

# The text decoder's key prefix inside a checkpoint. Dense causal LMs store
# the decoder at "model."; a multimodal wrapper nests it one level deeper --
# gemma-4 stores "model.language_model.layers.N...", older conditional-
# generation exports used "language_model.model.layers.N...". The safetensors
# key namespace and the live module tree (common.decoder_layers) name the SAME
# layers in the same order, whichever prefix is in use.
_TEXT_KEY_PREFIXES = ("model.", "model.language_model.", "language_model.model.")


def text_key_prefix(keys) -> str:
    """The prefix under which `keys` stores the decoder layers."""
    for prefix in _TEXT_KEY_PREFIXES:
        if f"{prefix}layers.0.self_attn.o_proj.weight" in keys:
            return prefix
    raise KeyError(
        "cannot find the decoder layers in this checkpoint under any of "
        f"{_TEXT_KEY_PREFIXES}; add its prefix to head_ranker._TEXT_KEY_PREFIXES")


def checkpoint_keys(model_name: str, snap: str, weight_map) -> "set":
    """Every tensor key of the checkpoint at `snap`.

    A single-file model has no index, and the *.json-only snapshot the callers
    start from does not hold model.safetensors -- resolve it through the cache
    the same way the tensor loads below do.
    """
    if weight_map is not None:
        return set(weight_map)
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    cache_dir = common.model_cache_dir(model_name)
    path = os.path.join(snap, "model.safetensors")
    if not os.path.exists(path):
        path = os.path.join(snapshot_download(
            model_name, cache_dir=cache_dir, local_files_only=True,
            allow_patterns=["model.safetensors"]), "model.safetensors")
    with safe_open(path, framework="pt") as f:
        return set(f.keys())


def load_o_proj(model_name: str, layers: Sequence[int]) -> Dict[int, np.ndarray]:
    """{layer: W_O (d_model, layer_width) float32} for the named layers only.

    Read straight out of the safetensors shards in the repo's per-model cache
    (common.model_cache_dir), so this never materialises the
    full model -- eight (4096, 4096) blocks instead of 16 GB of weights. On a
    width-heterogeneous model (gemma-4) the blocks differ per layer, which is
    exactly what the pad-slot guard in score_heads keys on.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cache_dir = common.model_cache_dir(model_name)
    snap = snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True,
                             allow_patterns=["*.json"])
    index_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = None
    prefix = text_key_prefix(checkpoint_keys(model_name, snap, weight_map))

    out: Dict[int, np.ndarray] = {}
    for layer in sorted(set(int(l) for l in layers)):
        key = f"{prefix}layers.{layer}.self_attn.o_proj.weight"
        shard = weight_map[key] if weight_map else "model.safetensors"
        path = os.path.join(snap, shard)
        if not os.path.exists(path):
            # weights were not part of the *.json allow_patterns fetch above;
            # pull just this shard from the cache.
            path = snapshot_download(model_name, cache_dir=cache_dir,
                                     local_files_only=True,
                                     allow_patterns=[shard])
            path = os.path.join(path, shard)
        with safe_open(path, framework="pt") as f:
            out[layer] = f.get_tensor(key).float().numpy()
    return out


def heads_str(heads: Sequence[Tuple[int, int]]) -> str:
    return " ".join(f"L{l}H{h}" for l, h in heads)




def load_unembedding(model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """(U (vocab, d_model), final_norm_gain (d_model,)) as float32.

    Read straight from the safetensors shards so this never materialises the
    model -- two tensors instead of 16 GB.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cache_dir = common.model_cache_dir(model_name)
    snap = snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True,
                             allow_patterns=["*.json"])
    # A model small enough to fit one file has no index; its weights are all in
    # model.safetensors. Same fallback load_o_proj uses.
    index_path = os.path.join(snap, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

    def shard_path(shard: str) -> str:
        path = os.path.join(snap, shard)
        if not os.path.exists(path):
            path = os.path.join(snapshot_download(
                model_name, cache_dir=cache_dir, local_files_only=True,
                allow_patterns=[shard]), shard)
        return path

    def tensor(key: str) -> np.ndarray:
        shard = weight_map[key] if weight_map else "model.safetensors"
        with safe_open(shard_path(shard), framework="pt") as f:
            return f.get_tensor(key).float().numpy()

    # A tied-embedding model has no lm_head; the input embedding IS the decoder.
    # Which keys exist comes from the index when there is one and from the file
    # itself when there is not -- guessing "lm_head.weight" and letting
    # get_tensor raise would report a missing-key error for a model that is
    # simply tied.
    if weight_map is not None:
        keys = set(weight_map)
    else:
        with safe_open(shard_path("model.safetensors"), framework="pt") as f:
            keys = set(f.keys())
    # The decoder's key prefix: "model." on dense LMs, one level deeper on
    # multimodal wrappers (gemma-4: "model.language_model.") -- same
    # resolution as load_o_proj, so both loaders read the same tree.
    prefix = text_key_prefix(keys)
    key = "lm_head.weight" if "lm_head.weight" in keys else f"{prefix}embed_tokens.weight"
    return tensor(key), tensor(f"{prefix}norm.weight")


def elicitation_target(gen_dir: str, tokenizer, vocab: int) -> Tuple[np.ndarray, Dict]:
    """u-space target: first-token distribution of reasoned minus direct answers.

    Measured from every gen_*.jsonl in gen_dir except the control runs -- those
    steer random heads or random directions, so their generations do not
    describe what successful elicitation looks like.
    """
    cot, direct = collections.Counter(), collections.Counter()
    for path in sorted(glob.glob(os.path.join(gen_dir, "gen_*.jsonl"))):
        if "rand" in os.path.basename(path):
            continue
        with open(path) as f:
            for line in f:
                text = json.loads(line)["generated_text"]
                if not text.strip():
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if not ids:
                    continue
                bucket = cot if len(text.split()) > COT_MIN_WORDS else direct
                bucket[ids[0]] += 1
    if not cot or not direct:
        raise SystemExit(f"{gen_dir} has no {'reasoned' if not cot else 'direct'} "
                         "generations -- point --gen_dir at a finished alpha sweep")

    def dist(counter: collections.Counter) -> np.ndarray:
        v = np.zeros(vocab, dtype=np.float32)
        total = sum(counter.values())
        for tid, n in counter.items():
            v[tid] = n / total
        return v

    p_cot, p_direct = dist(cot), dist(direct)
    info = {
        "n_cot": sum(cot.values()), "n_direct": sum(direct.values()),
        "top_cot": [(tokenizer.decode([i]), float(p_cot[i]))
                    for i in np.argsort(-p_cot)[:6]],
        "top_direct": [(tokenizer.decode([i]), float(p_direct[i]))
                       for i in np.argsort(-p_direct)[:6]],
    }
    return p_cot - p_direct, info


def score_heads(u: np.ndarray, com: np.ndarray, tuning: np.ndarray,
                model_name: str, num_layers: int, num_heads: int,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(score, cos, resid_norm), each (num_layers, num_heads).

    sigma is recomputed exactly as steering_driver.build_intervention_vectors does, so
    delta_h is the vector the steering run really adds at alpha=1.
    """
    head_dim = tuning.shape[-1]
    u_norm = float(np.linalg.norm(u))
    score = np.zeros((num_layers, num_heads), dtype=np.float32)
    cos = np.zeros_like(score)
    norm = np.zeros_like(score)
    valid = np.ones((num_layers, num_heads), dtype=bool)
    # One pass over the weight shards for all L layers, not one snapshot lookup
    # and shard open per layer.
    W_O = load_o_proj(model_name, range(num_layers))
    for layer in range(num_layers):
        W = W_O[layer]
        for head in range(num_heads):
            # A grid slot past this layer's true o_proj width is HeadCatcher
            # padding on a width-heterogeneous model (gemma-4): its
            # activations are identically zero, so com is the zero vector
            # (normalizing it would be 0/0) and there is nothing there to
            # steer. Score/cos/norm stay 0 and the slot is marked invalid so
            # band_top_k can never select it.
            if (head + 1) * head_dim > W.shape[1]:
                valid[layer, head] = False
                continue
            theta = com[layer, head].astype(np.float32)
            theta = theta / np.linalg.norm(theta)
            sigma = float(np.std(tuning[:, layer, head, :].astype(np.float32) @ theta, ddof=1))
            delta = W[:, head * head_dim:(head + 1) * head_dim] @ (theta * sigma)
            n = float(np.linalg.norm(delta))
            score[layer, head] = float(u @ delta)
            cos[layer, head] = score[layer, head] / (n * u_norm) if n else 0.0
            norm[layer, head] = n
    return score, cos, norm, valid


def band_top_k(score: np.ndarray, num_heads: int, K: int, lo: int, hi: int,
               valid: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """The K highest-scoring heads inside layers [lo, hi].

    A band that names layers this model does not have is an ERROR, not an empty
    selection. Masking everything to -inf and taking an argsort still returns K
    indices -- in whatever arbitrary order the (unstable) sort left the ties --
    so a band of 28-31 on a 28-layer model silently produced a head set led by
    L0H0. That looks like a result and is not one.
    """
    num_layers = score.shape[0]
    if lo > hi:
        raise SystemExit(f"band {lo}-{hi}: lo > hi")
    if lo >= num_layers:
        raise SystemExit(
            f"band {lo}-{hi} names no layer of this model, which has "
            f"{num_layers} layers (L0-L{num_layers - 1}). The default BANDS are "
            f"written for a 32-layer model; set BANDS for this one.")
    if hi >= num_layers:
        print(f"  note: band {lo}-{hi} clipped to {lo}-{num_layers - 1} "
              f"(model has {num_layers} layers)")
        hi = num_layers - 1
    flat = score.reshape(-1).astype(np.float64).copy()
    layers = np.arange(flat.size) // num_heads
    flat[(layers < lo) | (layers > hi)] = -np.inf
    if valid is not None:
        # Pad pseudo-heads of a width-heterogeneous model (see score_heads):
        # nothing lives there, so they are not candidates, not even at score 0.
        flat[~valid.reshape(-1)] = -np.inf
    in_band = int(np.isfinite(flat).sum())
    if in_band < K:
        raise SystemExit(f"band {lo}-{hi} holds {in_band} heads, fewer than K={K}")
    # stable, so a tie between two heads resolves the same way on every run
    order = np.argsort(flat, kind="stable")[::-1][:K]
    return [steering_driver.flattened_idx_to_layer_head(int(i), num_heads) for i in order]


def run(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer

    io.assert_model_matches(args.exp1_train_collection, args.model_name,
                            "the elicitation ranking")

    # Skip only when these exact inputs already produced these exact outputs.
    # File existence alone is not evidence: a directions.npz says nothing about
    # which target, which probe fit or which direction built it.
    outputs = [os.path.basename(p) for p in
               (args.directions_out, args.heads_out, args.out_npz) if p]
    out_dir = (os.path.dirname(os.path.abspath(args.directions_out))
               if args.directions_out else None)
    # `readout` names the score formula. Stamps written before 2026-08-27 lack
    # it, so a selection ranked with the final-norm gain is never mistaken for
    # a current one and gets re-ranked on the next run.
    sig = resume.signature(
        [p for p in (args.target_npz, args.exclude_rows_file,
                     os.path.join(args.exp1_train_collection, "ground_truth.csv")) if p],
        model_name=args.model_name, direction_mode="com",
        K=args.K, bands=list(args.bands), elbow_frac=args.elbow_frac,
        gen_dir=args.gen_dir, readout="u.delta")
    if out_dir and not args.force and resume.is_current(out_dir, sig, outputs,
                                                        name="elicit_stamp.json"):
        print(f"elicit is current for this target, probe fit and direction — "
              f"skipping ({out_dir})")
        return

    # The gain is read (layer_transmission's lens needs it) but not used here.
    U, _gain = load_unembedding(args.model_name)
    cache_dir = common.model_cache_dir(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                              cache_dir=cache_dir)

    # --- the target: either a prompt-pair npz, or counted from generations ---
    if args.target_npz:
        z = np.load(args.target_npz, allow_pickle=True)
        delta_p = z["delta_p"]
        print(f"outcome target from {args.target_npz}")
        print(f"  {int(z['n_prompts'])} prompts, agg={str(z['agg'])}, "
              f"correct vs incorrect under {str(z['direct_mode'])}")
        for key, label in (("p_reason", "correct"), ("p_direct", "incorrect")):
            p = z[key]
            print(f"  {label:6} opens with: " + ", ".join(
                f"{tokenizer.decode([int(t)])!r} {p[t]:.3f}" for t in np.argsort(-p)[:6]))
    elif args.gen_dir:
        delta_p, info = elicitation_target(args.gen_dir, tokenizer, U.shape[0])
        print(f"elicitation target from {args.gen_dir}")
        print(f"  {info['n_cot']} reasoned / {info['n_direct']} direct generations")
        print("  reasoned opens with:", [(repr(t), round(p, 3)) for t, p in info["top_cot"]])
        print("  direct   opens with:", [(repr(t), round(p, 3)) for t, p in info["top_direct"]])
    else:
        raise SystemExit("need --target_npz (from src.steering.elicitation_target) "
                         "or --gen_dir (a finished steering sweep)")
    u = delta_p @ U
    print(f"  ||u|| = {np.linalg.norm(u):.4f}")

    # --- head activations, and the direction to inject at each head ---
    exp4 = args.exp4_train_collection or args.exp1_train_collection
    # The scan hold-out: rows band and alpha are tuned on are never part of
    # the com fit (run_compass writes them before elicit runs). On MATH the
    # scan rows live in another split, so the file names none of this pool.
    exclude = None
    if args.exclude_rows_file:
        exclude = {(r["subject"], str(r["idx"]))
                   for r in csv.DictReader(open(args.exclude_rows_file, encoding="utf-8"))}
    X, y, _ = io.load_head_collection(args.exp1_train_collection, exp4,
                                      balanced_only=False, drop_truncated=True,
                                      exclude_keys=exclude)
    # The one direction this run ranks and writes: com, keyed by the name
    # steering_driver's --direction looks up. com is a difference of two class
    # means over every labelled row of the pool -- it needs labels, not a
    # fitted probe, and no class balancing (each class mean is its own
    # average). The per-head logistic probes were removed on 2026-08-28.
    bank: Dict[str, np.ndarray] = {}
    cfg = common.text_config(
        AutoConfig.from_pretrained(args.model_name, cache_dir=cache_dir))
    # The GRID, not the architecture's head count: on a width-heterogeneous
    # model (gemma-4) the stored rows are padded to the widest layer and split
    # into pseudo-heads of the config head_dim -- see common.head_grid_num_heads.
    # Homogeneous models get exactly num_attention_heads, as before.
    num_heads = common.head_grid_num_heads(cfg, int(X.shape[-1]))
    num_layers = int(cfg.num_hidden_layers)
    if num_heads != int(cfg.num_attention_heads):
        print(f"head grid: {num_heads} pseudo-heads of "
              f"{int(X.shape[-1]) // num_heads} dims per layer "
              f"(config: {int(cfg.num_attention_heads)} heads; widths differ "
              f"per layer, pad slots are masked from selection)")
    print(f"\ncom directions from {len(y)} labelled rows ({int(y.sum())} correct)")
    bank["com"] = steering_driver.get_com_directions(
        steering_driver.split_heads(X, num_heads), y).reshape(num_layers, num_heads, -1)
    tuning = steering_driver.split_heads(X, num_heads)
    head_dim = bank["com"].shape[-1]

    # RANK by the vector that will actually be INJECTED: com. The score is a
    # property of (head, direction) jointly, so ranking one vector and
    # steering another would select heads for a shift the run never applies.
    theta = bank["com"]
    score, cos, norm, valid = score_heads(u, theta, tuning, args.model_name,
                                          num_layers, num_heads)
    if not valid.all():
        print(f"masked {int((~valid).sum())} pad pseudo-head slots "
              f"(beyond their layer's true o_proj width)")

    # "auto" derives the grid from THIS model's own mass profile, which is the
    # only way a band list ports across models: a layer index means nothing
    # without the depth it indexes into.
    if len(args.bands) == 1 and args.bands[0] == "auto":
        args.bands = [f"{lo}-{hi}"
                      for lo, hi in operating_point.derive_bands(score, args.elbow_frac)]
        print(f"\nbands=auto -> {' '.join(args.bands)}   "
              f"(elbow at L{args.bands[0].split('-')[0]}, "
              f"elbow_frac={args.elbow_frac:g})")

    print("\npositive elicitation mass per layer")
    for layer in range(num_layers):
        pos = score[layer][score[layer] > 0]
        if pos.size:
            print(f"  L{layer:<2} sum+={pos.sum():7.4f}  max={score[layer].max():+7.4f}  "
                  f"heads>0={int((score[layer] > 0).sum()):3}")

    for spec in args.bands:
        lo, _, hi = spec.partition("-")
        lo, hi = int(lo), int(hi or lo)
        heads = band_top_k(score, num_heads, args.K, lo, hi, valid)
        s = [float(score[l, h]) for l, h in heads]
        dose = float(sum(norm[l, h] for l, h in heads))
        print(f"\n# layers {lo}-{hi}: top-{args.K} by elicitation score")
        print(f"  HEADS=\"{heads_str(heads)}\" HEADS_TAG=E{lo}_{hi}")
        c = [float(cos[l, h]) for l, h in heads]
        print(f"  score {min(s):+.4f}..{max(s):+.4f}  (all positive: {all(x > 0 for x in s)})")
        # cos is the fraction of the injected vector that points at the target.
        # Below ~0.1 the steering is mostly off-target push: it will degrade the
        # model before it elicits anything, whatever the alpha.
        print(f"  cos(injection,u) {min(c):.4f}..{max(c):.4f}")
        print(f"  sum||r|| per alpha = {dose:.4f}   "
              f"alpha for dose D = D/{dose:.4f}")

    if args.directions_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.directions_out)), exist_ok=True)
        # com under its own key, named exactly as steering_driver's --direction
        # spells it (get_directions). Written unconditionally: the elicit
        # stage is gated on this file's existence.
        #
        # val_accs is an all-NaN PLACEHOLDER kept only because
        # build_layer_vectors takes num_layers from its shape; this pipeline
        # always names its heads explicitly.
        payload = {f"{name}_directions": vec.reshape(num_layers * num_heads, -1)
                   for name, vec in bank.items()}
        np.savez(args.directions_out,
                 val_accs=np.full((num_layers, num_heads), np.nan, dtype=np.float32),
                 num_heads=np.array(num_heads), head_dim=np.array(head_dim),
                 **payload)
        print(f"\nwrote {args.directions_out} "
              f"({', '.join(sorted(payload))}; ranked and steered on "
              f"{"com"})")

    if args.heads_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.heads_out)), exist_ok=True)
        with open(args.heads_out, "w") as f:
            # Stamp WHICH model, depth and direction these heads belong to. The
            # file is sourced by bash, so without this a head list selected for
            # one model is indistinguishable from one selected for another --
            # and layer indices do not mean the same thing across depths.
            f.write(f"# {args.model_name}  L0-L{num_layers - 1}  "
                    f"direction={"com"}  K={args.K}\n")
            f.write(f'HEADS_MODEL="{args.model_name}"\n')
            f.write(f"HEADS_NUM_LAYERS={num_layers}\n")
            f.write(f'HEADS_BANDS="{" ".join(args.bands)}"\n')
            for spec in args.bands:
                lo, _, hi = spec.partition("-")
                lo, hi = int(lo), int(hi or lo)
                hs = band_top_k(score, num_heads, args.K, lo, hi, valid)
                f.write(f'HEADS_E{lo}_{hi}="{heads_str(hs)}"\n')
                f.write(f'DOSE_E{lo}_{hi}={sum(norm[l, h] for l, h in hs):.4f}\n')
        print(f"\nwrote {args.heads_out}")

    if args.out_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
        np.savez(args.out_npz, score=score, cos=cos, resid_norm=norm, u=u)
        print(f"\nwrote {args.out_npz}")

    if out_dir:
        resume.write_stamp(out_dir, sig, name="elicit_stamp.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exclude_rows_file", default=None,
                   help="(subject, idx) CSV of rows to leave OUT of the com fit: "
                        "the scan hold-out, when the scan rows come from the "
                        "fitting pool itself (run_compass SCAN_ROWS_CSV)")
    p.add_argument("--exp1_train_collection", required=True)
    p.add_argument("--exp4_train_collection", default=None)
    p.add_argument("--target_npz", default=None,
                   help="from src.steering.elicitation_target -- the prompt-pair "
                        "target. Preferred: needs no generations and no labels.")
    p.add_argument("--gen_dir", default=None,
                   help="alternative to --target_npz: a finished steering sweep, "
                        "whose generations are split into reasoned/direct by length")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--bands", nargs="+", default=["auto"],
                   help="layer bands as lo-hi, or the single word 'auto' to derive "
                        "the nested grid from this model's own elicitation-mass "
                        "profile (src.steering.operating_point plan). A literal band list is "
                        "model-specific and does not port across depths.")
    p.add_argument("--elbow_frac", type=float, default=operating_point.ELBOW_FRAC,
                   help="bands=auto: a band floor must hold this share of the peak "
                        "layer's mass, and so must every layer above it")
    p.add_argument("--directions_out", default=None,
                   help="write a directions.npz holding the com directions under "
                        "the key steering_driver's --direction looks up, so the steering "
                        "run needs nothing else.")
    p.add_argument("--heads_out", default=None,
                   help="write one shell-sourceable HEADS_<tag>=... line per band, "
                        "so run_compass.sh can consume the selection directly")
    p.add_argument("--out_npz", default=None)
    p.add_argument("--force", action="store_true",
                   help="recompute even when the stamp says it is current")
    run(p.parse_args())


if __name__ == "__main__":
    main()
