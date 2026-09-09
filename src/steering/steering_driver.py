#!/usr/bin/env python3
"""ITI-style inference-time intervention on MATH correctness.

Implements Eq. 2 of the ITI paper (Li et al., NeurIPS 2023, p.5),
which is the entire intervention:

    x_{l+1} = x_l + sum_{h=1..H} Q_l^h ( Att_l^h(P_l^h x_l) + alpha * sigma_l^h * theta_l^h )

"For not-selected attention heads, theta is a zero vector." The sum runs over
EVERY head of EVERY layer; head selection is expressed purely as theta = 0, so
there is no per-head, per-layer or per-question branching anywhere — the same
vector is added at every step for every question. Non-selected heads contribute
alpha * sigma * 0 = 0 exactly.

This file implements exactly that and nothing else:

- theta_l^h  = unit com direction of head (l,h) if (l,h) is in the selected
  set (--heads, the band's top-K by elicitation score from head_ranker), else
  the zero vector. Directions are read from the --probe_report directory (the
  elicit stage's directions.npz, fitted on the fitting pool only) — the test
  collection never touches direction finding.
  --direction is com (the default), or one of the two controls: anticom (the
  same vector reversed) and random (see DIRECTIONS below).
- sigma_l^h  = std of the tuning activations of head (l,h) projected on
  theta_l^h (ddof=1, matching torch.std at validate_2fold.py:174). Tuning set =
  all non-truncated train rows, labels unused (the reference uses a separate
  unlabeled set; ours matches the steering-time prompt distribution exactly,
  since probing is prompt-only).
- The per-layer vector [sigma_l^0 theta_l^0, ..., sigma_l^{H-1} theta_l^{H-1}]
  is assembled by build_intervention_vectors and added, times alpha,
  to the input of self_attn.o_proj — the point where the concatenated per-head
  Att values sit just before the Q_l^h projection, so adding there is Eq. 2
  term by term (honest_llama/validation/validate_2fold.py:160-182).
- Position: the LAST position of the o_proj input. With KV caching that is the
  current token at every decode step, and only the final prompt token during
  prefill — the reference's 'lt' behavior (interveners.py:40-45).

Generation replays the stored test prompts (records.jsonl) with the same greedy
/ max_new_tokens=2048 settings as the source collection; answers are graded with
grading.answers_equal against the stored gold answers. The stored source
generations ARE the alpha=0 baseline; --sanity N re-generates N examples with
hooks attached and alpha=0 and asserts the text is identical.

Modes:
  sweep:    grid over --K and --alpha on a stratified subset (--subset, by
            subject x baseline-correctness) of the test collection.
  full:     a single (K, alpha) on every test row.
  controls: both ablations at ONE (K, alpha) — K random heads with the real
            direction, and the real top-K heads with random directions. See
            config_grid().

Outputs under --out_dir: per-config resumable gen_*.jsonl + summary_*.json
and a cumulative sweep_summary.csv.

Run (GPU):
  python3 -m src.steering.steering_driver --mode sweep \
    --exp1_train_collection $EMBED_ROOT/zs_llama_gsm8k_all/train_standard \
    --exp1_test_collection  $EMBED_ROOT/zs_llama_gsm8k_all/test_standard \
    --probe_report $EMBED_ROOT/zs_llama_gsm8k_all/outcome_standard \
    --heads L31H26 L31H24 L31H3 --heads_tag E31-31 --K 3 --alpha 8 16 --direction com
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.core import common, grading, io
from src.core.io import (attach_gold_aliases, load_ground_truth,
                         load_head_collection, load_layer_collection,
                         load_prompts, write_csv, write_json)
# -----------------------------------------------------------------------------
# The ITI primitives (likenneth/honest_llama @ 2c6b217), transcribed
# -----------------------------------------------------------------------------
#
# Each function below mirrors one piece of the reference implementation and
# its docstring cites the exact source lines. Adaptations, all label-preserving:
# activations are flat (N, L, H, D) arrays instead of per-question lists (one
# generation per problem makes the two identical); num_layers / num_heads /
# head_dim are parameters (the reference hardcodes 32 heads and head_dim 128);
# sigma uses ddof=1 to match torch.std on the reference's pyvene path
# (validate_2fold.py:174). These were asserted bit-identical to the reference
# on synthetic data while they lived in a vendor module (removed 2026-09-03);
# do not edit them to fix a bug elsewhere.

# Index conventions (honest_llama/utils.py:676-680, verbatim)


def flattened_idx_to_layer_head(flattened_idx: int, num_heads: int) -> Tuple[int, int]:
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer: int, head: int, num_heads: int) -> int:
    return layer * num_heads + head


def split_heads(activations: np.ndarray, num_heads: int) -> np.ndarray:
    """(N, L, H*D) -> (N, L, H, D), the load-time head split of
    honest_llama/validation/validate_2fold.py:124 (einops 'b l (h d) -> b l h d')."""
    n, layers, hidden = activations.shape
    head_dim = hidden // num_heads
    return activations.reshape(n, layers, num_heads, head_dim)


def get_top_heads(
    val_accs: np.ndarray,
    num_to_intervene: int,
    num_heads: int,
    use_random_dir: bool = False,
) -> List[Tuple[int, int]]:
    """Global top-K heads by a flat (L*H,) score; here only the random-heads
    control path uses it (select_heads with use_random_dir).

    Mirrors honest_llama/utils.py:707-721: plain descending argsort over the
    flat (L*H,) accuracy vector, no per-layer quota. use_random_dir replaces
    the selection with a uniform random draw (their random-head baseline),
    consuming the global numpy RNG exactly like the reference.
    """
    top_accs = np.argsort(val_accs.reshape(-1))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(int(idx), num_heads) for idx in top_accs]
    if use_random_dir:
        random_idxs = np.random.choice(len(val_accs), len(val_accs), replace=False)
        top_heads = [flattened_idx_to_layer_head(int(idx), num_heads) for idx in random_idxs[:num_to_intervene]]
    return top_heads


def get_com_directions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mass-mean-shift directions, one per (layer, head), flat-indexed.

    Mirrors honest_llama/utils.py:768-782 (get_com_directions): for each head,
    mean activation of correct rows minus mean of incorrect rows, computed on
    all supplied rows (the reference uses train+val). Unnormalized — callers
    normalize at use site, as in validate_2fold.py:171.
    Returns (num_layers * num_heads, head_dim).
    """
    num_layers, num_heads = X.shape[1], X.shape[2]
    com_directions = []
    for layer in tqdm(range(num_layers), desc="get_com_directions"):
        for head in range(num_heads):
            acts = X[:, layer, head, :]
            com_directions.append(np.mean(acts[y == 1], axis=0) - np.mean(acts[y == 0], axis=0))
    return np.array(com_directions)


def build_intervention_vectors(
    top_heads: Sequence[Tuple[int, int]],
    directions: np.ndarray,
    tuning_activations: np.ndarray,
    num_heads: int,
) -> Dict[int, np.ndarray]:
    """Per-layer ITI shift vectors: {layer: (num_heads*head_dim,) float32}.

    Mirrors the inline logic of honest_llama/validation/validate_2fold.py:160-182:
    the layer vector is zero except in the head_dim slice of each selected head,
    which holds unit_direction * sigma, where sigma is the std (ddof=1, matching
    torch.std's Bessel correction at line 174) of the tuning activations
    projected onto the unit direction. The runtime intervention is then
    x[:, -1, :] += alpha * vector at every decode step (ITI_Intervener,
    interveners.py:40-45); alpha is applied by the caller, not baked in here.

    directions: (L*H, head_dim) unnormalized; tuning_activations: (N, L, H, D).
    """
    head_dim = tuning_activations.shape[-1]
    vectors: Dict[int, np.ndarray] = {}
    for layer, head in top_heads:
        vec = vectors.setdefault(layer, np.zeros(num_heads * head_dim, dtype=np.float32))
        direction = directions[layer_head_to_flattened_idx(layer, head, num_heads)].astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        proj_vals = tuning_activations[:, layer, head, :].astype(np.float32) @ direction
        proj_val_std = np.std(proj_vals, ddof=1)
        vec[head * head_dim : (head + 1) * head_dim] = direction * proj_val_std
    return vectors


# -----------------------------------------------------------------------------
# Shared vocabulary: directions, head specs, the config-tag rule, head selection,
# flip / per-subject bookkeeping. core.grading imports these lazily (never at
# module level) when it rewrites a summary, so the dependency stays acyclic.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# THE INTERVENTION AND ITS CONTROLS: the three directions theta can be
# -----------------------------------------------------------------------------
#
#   com      THE intervention (default): mean(correct) - mean(incorrect) of each
#            head's activations over the fitting pool (honest_llama utils.py:731,
#            "mass mean shift", the ITI paper's best direction).
#   anticom  control 1, suppression: the same vector with its sign flipped, at
#            the same (positive) alpha -- the promoted cell run backwards.
#   random   control 2, ITI's own: one iid N(0, I) vector per head (utils.py:733).
#
# The third control, random HEADS with the real direction (utils.py:716-719), is
# not a direction: it is `--use_random_dir` / an explicit random --heads draw
# (select_heads below), tagged _randheads. The direction name is part of the
# config tag, so the three directions never share a file or a resume.
DIRECTIONS = {"com": "com", "anticom": "anticom", "random": "random"}


def get_directions(args: argparse.Namespace, data: Any) -> np.ndarray:
    """theta for every (layer, head), flat-indexed (L*H, head_dim), unnormalized.

    com      mean(activations | correct) - mean(activations | incorrect), computed
             by head_ranker over the fitting pool exactly as the reference's
             get_com_directions does (utils.py:768-782). Points from wrong toward
             correct, so a positive alpha pushes toward correct.
    anticom  -com: the same shift reversed. Eq. 2 is linear in theta, so this is
             the promoted intervention at the same dose, pushing toward wrong.
    random   one iid N(0, I) vector per head (utils.py:733). The reference draws
             these inside its per-head loop from the global RNG; we draw the
             whole (L*H, head_dim) matrix at once from a Generator seeded with
             --seed, which is the same distribution but makes a head's direction
             independent of K and of loop order, so the same head gets the same
             random direction across an alpha/K sweep.

    All three are normalized to unit length and scaled by the same sigma
    downstream (build_intervention_vectors), so --direction changes
    theta and nothing else.
    """
    if args.direction == "com":
        return data["com_directions"]
    if args.direction == "anticom":
        return -data["com_directions"]
    shape = data["com_directions"].shape  # (L*H, head_dim)
    return np.random.default_rng(args.seed).normal(size=shape).astype(np.float32)


# -----------------------------------------------------------------------------
# The config-tag rule (the ONE naming rule for gen_/summary_ files)
# -----------------------------------------------------------------------------


HEAD_RE = re.compile(r"^[Ll](\d+)[Hh](\d+)$")
ALT_HEAD_RE = re.compile(r"^(\d+)[:_,](\d+)$")


def parse_head(spec: str) -> Tuple[int, int]:
    """'L45H9' (or '45:9') -> (45, 9). Same spelling query_head_alignment and
    --print_heads use, so a head list can be copied between them verbatim."""
    m = HEAD_RE.match(spec.strip()) or ALT_HEAD_RE.match(spec.strip())
    if not m:
        raise ValueError(f"bad head spec {spec!r}; use L45H9 or 45:9")
    return int(m.group(1)), int(m.group(2))


def parse_heads(specs: Sequence[str]) -> List[Tuple[int, int]]:
    """A head list, order preserved, duplicates rejected.

    Order is kept because it is part of the configuration (the tag hashes it);
    a silently deduplicated or re-sorted list would name a different run.
    """
    heads = [parse_head(s) for s in specs]
    if len(set(heads)) != len(heads):
        dupes = sorted({h for h in heads if heads.count(h) > 1})
        raise ValueError(f"duplicate head(s) in --heads: "
                         f"{', '.join(f'L{l}H{h}' for l, h in dupes)}")
    return heads


def heads_tag(heads: Sequence[Tuple[int, int]], label: Optional[str] = None) -> str:
    """The tag field that identifies an explicit head set.

    A label (--heads_tag) is used verbatim, because a run the user named is one
    they can find again. Without one, the set is identified by its size and a
    hash of its members: unbounded head lists cannot go in a filename, and a
    hash is at least collision-free and order-insensitive... except it is NOT
    order-insensitive here on purpose — the order is part of the configuration
    (see parse_heads), so reordering the same heads names a different run.
    """
    if label:
        return label
    digest = hashlib.sha1(
        ",".join(f"{l}:{h}" for l, h in heads).encode()).hexdigest()[:6]
    return f"sel{len(heads)}_{digest}"


def config_tag(
    K: int,
    alpha: float,
    direction: str,
    random_heads: bool = False,
    rep: Optional[int] = None,
    skip_top: int = 0,
    heads_tag: Optional[str] = None,
    steer_steps: str = "all",
) -> str:
    """gen_<tag>.jsonl / summary_<tag>.json naming for one steering config.

    ``direction`` is the CANONICAL name (com/probe/random — resolve aliases via
    DIRECTIONS first). ``rep`` marks a best-config control repeat (_r<rep>).
    ``skip_top`` marks a rank WINDOW rather than a prefix (_skip<S>): the S best
    heads are dropped and the next K taken, so K=7 skip=1 steers ranks 2-8.
    skip_top=0 adds nothing, so every pre-existing tag is unchanged.

    The rule is byte-identical to the one every existing gen_/summary_ file on
    disk was named with (retired run kinds carried extra suffixes that no live
    run produces).

    ``steer_steps`` is "all" (the shift is applied on the prompt and on every
    generated token, ITI's behaviour) or "prefill" (prompt only, tagged
    ``_prefill``). These are different interventions producing different
    generations, so they must not share a filename — or a resume.

    The dose schedule is cosine, always, and the tag carries ``_cos`` for it:
    files without it are the retired constant schedule (one alpha throughout,
    every cell before 2026-08-29) -- same alpha, different dose per token,
    different generations -- and nothing live reads or resumes into them.
    """
    # `heads_tag` replaces the K field: an explicit head list is not "the top K
    # of the ranking", so naming it K<n> would collide with the rank-based run
    # of the same size and claim something false about how the heads were found.
    size = f"H{heads_tag}" if heads_tag else f"K{K}"
    tag = f"{size}_alpha{alpha:g}_{direction}"
    if steer_steps == "prefill":
        tag += "_prefill"
    tag += "_cos"
    if skip_top:
        tag += f"_skip{skip_top}"
    if random_heads:
        tag += "_randheads"
    if rep is not None:
        tag += f"_r{rep}"
    return tag


# -----------------------------------------------------------------------------
# Head selection
# -----------------------------------------------------------------------------


def select_heads(
    val_accs: np.ndarray, K: int, num_heads: int, skip_top: int = 0,
    use_random_dir: bool = False,
) -> List[Tuple[int, int]]:
    """The K heads to steer: the global accuracy ranking, offset by skip_top.

    skip_top=0 is the reference's plain top-K prefix (get_top_heads,
    verbatim). skip_top=S drops the S best heads and takes the next K, which
    turns the ranking into a sliding WINDOW — the ablation that asks whether the
    effect is carried by a few paramount heads or spread across the ranking.
    """
    heads = get_top_heads(val_accs, K + skip_top, num_heads,
                                     use_random_dir=use_random_dir)
    return heads[skip_top:]


# -----------------------------------------------------------------------------
# Outcome bookkeeping: flips and per-subject accuracy
# -----------------------------------------------------------------------------


def flip_counts(base: np.ndarray, corr: np.ndarray) -> Tuple[int, int]:
    """(wrong->right, right->wrong) counts between baseline and outcome labels."""
    base = np.asarray(base).astype(bool)
    corr = np.asarray(corr).astype(bool)
    return int((corr & ~base).sum()), int((~corr & base).sum())


def per_subject_table(
    subjects: np.ndarray,
    base: np.ndarray,
    corr: np.ndarray,
    flips: bool = False,
) -> Dict[str, Dict]:
    """Per-subject before/after accuracy (+ optional flips)."""
    subjects = np.asarray(subjects)
    base = np.asarray(base)
    corr = np.asarray(corr)
    out: Dict[str, Dict] = {}
    for s in sorted(set(subjects)):
        m = subjects == s
        entry = {
            "n": int(m.sum()),
            "acc": float(corr[m].mean()),
            "baseline_acc": float(base[m].mean()),
        }
        if flips:
            w2r, r2w = flip_counts(base[m], corr[m])
            entry["wrong_to_right"] = w2r
            entry["right_to_wrong"] = r2w
        out[s] = entry
    return out

# -----------------------------------------------------------------------------
# theta: see DIRECTIONS / get_directions above
# -----------------------------------------------------------------------------
#
# The reference selects theta in get_interventions_dict (honest_llama
# utils.py:729-736) and normalizes it to unit length (:736) before scaling by
# sigma. We keep its com branch (:731) and its random branch (:733); anticom is
# -com. Every branch is normalized and sigma-scaled identically, so the ONLY
# thing --direction changes is theta.


# WHERE the shift is added. Both sites take the same alpha * sigma * theta and
# apply it at the last position; they differ in which tensor that is, and so in
# what theta has to be a direction IN.
#
#   o_proj_in   the concatenated per-head attention output, dim H*D. ITI's own
#               site (interveners.py:19-24), and the one head probes live in.
#   layer_out   the decoder layer's output residual stream, dim d_model. The
#               layer-mode site: one shift per layer rather than per head, added
#               after that layer's attention and MLP, which is identically the
#               residual stream layer l+1 reads.
STEER_SITES = ("o_proj_in", "layer_out")

# WHEN the shift is applied, across the passes one generation makes.
#
#   all      every forward pass: the prefill (writing the last PROMPT token) and
#            then each decode step (writing that step's NEW token). This is
#            ITI's own behaviour (interveners.py:40-45) and the default, so
#            every existing run keeps its meaning.
#   prefill  the prefill pass only. The shift lands on the last prompt token,
#            enters the KV cache once, and generation proceeds from a perturbed
#            state with no further intervention.
#
# `prefill` is the setting that matches how theta and sigma were MEASURED: the
# probes, the com direction and sigma all come from the last prompt token of a
# prompt-only forward. Under `all`, that prompt-derived direction is also added
# to generated tokens, whose activation distribution it was never fitted on, and
# the perturbation compounds once per token rather than being applied once.
STEER_STEPS = ("all", "prefill")

# HOW THE DOSE VARIES OVER THE GENERATION. The cosine schedule, always
# (2026-09-03: the constant Eq. 2 schedule -- one alpha at the prompt and at
# every generated token, every cell before 2026-08-29 -- was removed; its files
# carry no `_cos` in the tag and are ignored by the report and the pick). It is
# Fractional Reasoning's own schedule, ported verbatim from their
# utils/llm_layers.py:9-12,34 (the same code baselines/fr.py runs):
#
#     rate = max(0.5 * (cos(pi * step / RAMPDOWN) + 1), RAMP_FLOOR)
#
# with `step` a PER-LAYER counter of decode passes that resets to 0 on any
# pass carrying more than one position (the prefill). So the multiplier on
# alpha is a half-cosine falling from 1.0 to 0 over the first RAMPDOWN=50
# generated tokens, clipped at RAMP_FLOOR=0.5 -- full dose while the model
# commits to an approach, then a permanent half dose for the rest of the
# generation. It never turns the intervention off. On llama/SVAMP it lifted the
# best cell past the CoT ceiling at the same heads and alpha (2026-08-29).
#
# FR injects one normalized ICV into the MLP output of every layer; we add
# alpha * sigma * theta at the o_proj input of selected heads. Only the ramp is
# shared.
RAMPDOWN = 50.0      # FR utils/llm_layers.py:34
RAMP_FLOOR = 0.5     # FR utils/llm_layers.py:34


def cosine_rampdown(current: float, length: float = RAMPDOWN) -> float:
    """FR utils/llm_layers.py:9-12 -- 1.0 at step 0, 0.0 at step `length`."""
    current = float(np.clip(current, 0.0, length))
    return float(0.5 * (np.cos(np.pi * current / length) + 1))


class Steerer:
    """Adds `alpha * layer_vector` to the o_proj input at position -1, in every
    layer — the runtime half of Eq. 2.

    One forward pre-hook per layer, unconditional: the only thing that decides
    whether a head moves is whether its slice of the vector is zero. Layers with
    no selected head carry an all-zero vector and are therefore exact no-ops
    (x + 0 == x in any float format), which is Eq. 2's "theta is a zero vector
    for not-selected heads" taken literally. The reference skips those layers as
    an optimization (validate_2fold.py:176-180); attaching them anyway is
    numerically identical and keeps the code branch-free.

    alpha == 0 makes every layer a no-op, which is what --sanity checks
    against the stored source generations.
    """

    def __init__(self, model: Any, layer_vectors: Dict[int, np.ndarray], alpha: float,
                 site: str = "o_proj_in", steer_steps: str = "all",
                 rampdown: float = RAMPDOWN, ramp_floor: float = RAMP_FLOOR):
        self.alpha = float(alpha)
        self.site = site
        self.steer_steps = steer_steps
        self.rampdown = float(rampdown)
        self.ramp_floor = float(ramp_floor)
        # FR's per-layer decode-step counter (utils/llm_layers.py:34). Reset by
        # any pass with more than one position, which is the prefill of the
        # next generate() call -- so nothing has to reset it between batches.
        self.steps: Dict[int, int] = {}
        self.handles = []
        dtype = next(model.parameters()).dtype
        if site not in STEER_SITES:
            raise ValueError(f"unknown site {site!r}; use one of {tuple(STEER_SITES)}")
        if steer_steps not in STEER_STEPS:
            raise ValueError(f"unknown steer_steps {steer_steps!r}; "
                             f"use one of {tuple(STEER_STEPS)}")
        for layer_idx in sorted(layer_vectors):
            shift = torch.tensor(layer_vectors[layer_idx], dtype=dtype)

            def add(x, layer=layer_idx, shift=shift):
                """The one place the shift is applied, shared by both sites."""
                # A decode step feeds exactly one position; the prefill feeds the
                # whole prompt. So seq_len == 1 identifies a decode pass, and
                # skipping those is what makes `prefill` mean "steer the prompt,
                # then let the model generate untouched". (A one-token prompt
                # would be indistinguishable, which no chat-templated prompt is.)
                decode = x.shape[1] == 1
                if not decode:
                    self.steps[layer] = 0      # FR: the prefill restarts the ramp
                if self.steer_steps == "prefill" and decode:
                    return
                rate = self._rate(layer)
                if decode:
                    self.steps[layer] = self.steps.get(layer, 0) + 1
                # Vectors are built on the STORED grid, which on a
                # width-heterogeneous model (gemma-4) is padded to the widest
                # layer's o_proj width; this layer's true input can be
                # narrower. The pad columns are zero by construction (pad
                # pseudo-heads are masked out of selection), so truncating to
                # the live width drops nothing -- and on a homogeneous model
                # the slice is the whole vector, byte-identical to before.
                width = x.shape[-1]
                if self.alpha != 0.0:
                    x[:, -1, :] = x[:, -1, :] + (self.alpha * rate) * shift[:width].to(x.device)

            if site == "o_proj_in":
                module = common.decoder_layers(model)[layer_idx].self_attn.o_proj

                def hook(module, args, add=add):
                    add(args[0])
                    return None

                self.handles.append(module.register_forward_pre_hook(hook))
            else:  # layer_out — the residual stream this layer emits
                module = common.decoder_layers(model)[layer_idx]

                def hook(module, args, output, add=add):
                    # a decoder layer returns either the hidden state or a tuple
                    # whose FIRST element is it; the rest (attn weights, cache)
                    # are untouched and passed straight back
                    hs = output[0] if isinstance(output, tuple) else output
                    add(hs)
                    return output

                self.handles.append(module.register_forward_hook(hook))

    def _rate(self, layer: int) -> float:
        """The multiplier on alpha at this layer's current decode step
        (1.0 on the prefill, where the counter is reset)."""
        return max(cosine_rampdown(self.steps.get(layer, 0), self.rampdown),
                   self.ramp_floor)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


# -----------------------------------------------------------------------------
# Eq. 2: sigma * theta for every head of every layer, zero where not selected
# -----------------------------------------------------------------------------


_TUNING_CACHE: Dict[str, np.ndarray] = {}


def load_tuning_activations(args: argparse.Namespace, num_heads: int) -> np.ndarray:
    """Tuning activations for sigma: every non-truncated train row, labels
    unused. Cached so a sweep loads the ~7.5k per-example files only once.

    Read from the SAME tensor the shift is added to — head activations for the
    o_proj_in site, residual states for layer_out. sigma is the std of these
    projected on theta, so reading the other one would dose the injection in the
    units of a space it never touches.
    """
    key = (args.exp4_train_collection, args.site)
    if key not in _TUNING_CACHE:
        if args.site == "layer_out":
            X3, _, _ = load_layer_collection(
                args.exp1_train_collection, balanced_only=False, drop_truncated=True)
            _TUNING_CACHE[key] = X3[:, :, None, :]          # (N, L, 1, d_model)
        else:
            X_flat, _, _ = load_head_collection(
                args.exp1_train_collection, args.exp4_train_collection,
                balanced_only=False, drop_truncated=True,
            )
            _TUNING_CACHE[key] = split_heads(X_flat, num_heads)
    return _TUNING_CACHE[key]


def assert_eq2_structure(
    vectors: Dict[int, np.ndarray],
    top_heads: Sequence[Tuple[int, int]],
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> None:
    """Verify the built vectors ARE Eq. 2 before a single token is generated.

    Checks, for every layer 0..L-1 and every head 0..H-1:
      - the layer is present (Eq. 2 sums over all layers; unoccupied ones are
        all-zero vectors, not missing entries),
      - the slice of a selected head is a non-zero unit direction scaled by
        sigma > 0,
      - the slice of a NOT-selected head is exactly zero (theta = 0),
      - each selected head appears exactly once.
    """
    selected = set(map(tuple, top_heads))
    assert len(selected) == len(top_heads), "duplicate head in top_heads"
    assert set(vectors) == set(range(num_layers)), (
        f"Eq. 2 covers all {num_layers} layers; got {sorted(vectors)}")
    n_nonzero = 0
    for layer in range(num_layers):
        vec = vectors[layer]
        assert vec.shape == (num_heads * head_dim,), f"layer {layer}: {vec.shape}"
        for head in range(num_heads):
            block = vec[head * head_dim: (head + 1) * head_dim]
            if (layer, head) in selected:
                norm = float(np.linalg.norm(block))
                assert norm > 0, f"selected head ({layer},{head}) has a zero shift"
                n_nonzero += 1
            else:
                assert not block.any(), (
                    f"not-selected head ({layer},{head}) has a non-zero theta")
    assert n_nonzero == len(selected), "selected-head count mismatch"


def explicit_heads(args: argparse.Namespace) -> Optional[List[Tuple[int, int]]]:
    """The head list --heads names, or None when selection is by val_accs."""
    return parse_heads(args.heads) if args.heads else None


def selected_heads(args: argparse.Namespace) -> List[Tuple[int, int]]:
    """The (layer, head) pairs this config steers.

    --heads names them outright: the ranking is not consulted at all, so the
    run can test a hypothesis about specific heads (say, the ones whose
    reasoning direction aligns with the steering direction) rather than the ones
    the correctness probes happen to rank highest. Everything downstream — the
    ablation, the report — follows from this function, so overriding it here
    is enough.

    The random-head CONTROL still means "the same number of heads, chosen at
    random", which is the comparison the control exists to make either way.

    Without --heads it is val_accs alone:

    Reading the ranking costs one small npz, while building the shift vectors
    costs a pass over every train head activation. A finished config still has
    to report WHICH heads it steered, so that is split out here and the
    expensive half is skipped when there is nothing left to generate.
    """
    data = np.load(os.path.join(args.probe_report, "directions.npz"))
    named = explicit_heads(args)
    if named is not None and not args.use_random_dir:
        return named
    if args.use_random_dir:  # reference's random-head baseline; seeded per config
        np.random.seed(args.seed + args.current_K)
    # global argsort over all L*H probe accuracies — no per-layer quota
    # (utils.py:707-721), so layer coverage is sparse and uneven by design.
    # --skip_top turns that prefix into a window (see select_heads).
    return select_heads(
        data["val_accs"].reshape(-1), args.current_K, int(data["num_heads"]),
        skip_top=args.skip_top, use_random_dir=args.use_random_dir,
    )


def build_layer_vectors(args: argparse.Namespace) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int]]]:
    """{layer: (num_heads*head_dim,)} for EVERY layer — the sigma*theta terms of
    Eq. 2, zero on every head outside the global top-K."""
    data = np.load(os.path.join(args.probe_report, "directions.npz"))
    num_heads = int(data["num_heads"])
    val_accs = data["val_accs"]
    num_layers = int(val_accs.shape[0])
    directions = get_directions(args, data)
    top_heads = selected_heads(args)
    tuning = load_tuning_activations(args, num_heads)
    head_dim = tuning.shape[-1]
    occupied = build_intervention_vectors(top_heads, directions, tuning, num_heads)
    # theta = 0 for every head of a layer that got no selected head
    vectors = {layer: occupied.get(layer, np.zeros(num_heads * head_dim, dtype=np.float32))
               for layer in range(num_layers)}
    assert_eq2_structure(vectors, top_heads, num_layers, num_heads, head_dim)

    per_layer = {l: sum(1 for a, _ in top_heads if a == l)
                 for l in sorted({a for a, _ in top_heads})}
    norms = [float(np.linalg.norm(vectors[l])) for l in per_layer]
    window = (f" (ranks {args.skip_top + 1}-{args.skip_top + args.current_K}, "
              f"skipping the top {args.skip_top})") if args.skip_top else ""
    how = (f"heads={heads_tag_of(args)}" if args.heads else f"K={args.current_K}{window}")
    print(f"{how} {args.direction_name}: {len(top_heads)} heads selected across "
          f"{len(per_layer)}/{num_layers} layers (heads per layer {per_layer}); "
          f"the other {num_layers - len(per_layer)} layers get theta=0 (exact no-op); "
          f"non-zero vector norms min/max = {min(norms):.3f}/{max(norms):.3f}")
    return vectors, top_heads


class Heartbeat:
    """Whole-line progress for a log file, for when the tqdm bar is disabled.

    Prints at most once per `step_frac` of the work (and never twice for the
    same batch), so a config contributes a bounded, greppable number of lines
    instead of a few thousand carriage returns.
    """

    def __init__(self, tag: str, total: int, step_frac: float = 0.1):
        self.tag, self.total = tag, int(total or 0)
        self.step = max(1, int(self.total * step_frac))
        self.done = 0
        self.next_at = self.step
        self.start = time.time()
        self.enabled = bool(os.environ.get("TQDM_DISABLE")) and self.total > 0
        if self.enabled:
            print(f"[{tag}] generating {self.total} rows", flush=True)

    def update(self, n: int) -> None:
        self.done += int(n)
        if not self.enabled or self.done < self.next_at:
            return
        self.next_at = self.done + self.step
        elapsed = time.time() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        print(f"[{self.tag}] {self.done}/{self.total} rows "
              f"({100.0 * self.done / self.total:.0f}%)  "
              f"{rate * 60:.1f} rows/min  elapsed {elapsed / 60:.1f}m  "
              f"eta {eta / 60:.1f}m", flush=True)


def heads_tag_of(args: argparse.Namespace) -> Optional[str]:
    """The tag field naming this run's explicit head set, or None for top-K."""
    named = explicit_heads(args)
    return None if named is None else heads_tag(named, args.heads_tag)


# -----------------------------------------------------------------------------
# Test rows
# -----------------------------------------------------------------------------


def config_grid(args: argparse.Namespace) -> List[Tuple[int, float, str, str, bool]]:
    """The (K, alpha, direction, direction_name, random_heads) tuples to run.

    sweep/full: the --K x --alpha grid at the single requested direction.

    controls: the two ablations of the paper's claim that BOTH the head ranking
    and the direction carry the effect, at one (K, alpha) each --

      1. random heads, real direction   -> does head SELECTION matter?
         (utils.py:716-719, the reference's random-head baseline)
      2. real top-K heads, random direction -> does the DIRECTION matter?
         (utils.py:733, ITI's "random direction" control)

    Together with the main run (real heads, real direction) that is a clean 2x2
    minus one cell: a real effect has to beat both. Everything else is held
    fixed -- same K, same alpha, same rows (--subset is seeded and stratified).
    So the only thing that differs between the three arms is the intervention
    itself.
    """
    if args.mode == "controls":
        K, alpha = args.K[0], args.alpha[0]
        return [
            (K, alpha, args.direction, args.direction_name, True),
            (K, alpha, "random", "random", False),
        ]
    return [(K, alpha, args.direction, args.direction_name, args.use_random_dir)
            for K in args.K for alpha in args.alpha]


def select_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    all_rows = load_ground_truth(args.exp1_test_collection)
    # Every reported accuracy is correct/ALL rows of the split: rows the steerer
    # skips (baseline-truncated) count as incorrect rather than vanishing from
    # the denominator, so a table cell is comparable across runs that skip
    # different numbers of rows. Kept-row accuracy stays as provenance.
    args.split_n = len(all_rows)
    rows = [r for r in all_rows if r["truncated"] == "0"]
    # THE ONE ACCURACY CONVENTION (2026-08-29): correct / ALL addressed rows,
    # truncated = incorrect. A whole-split run addresses every row of the
    # split, so the baseline-truncated rows it cannot steer are still in the
    # denominator: they are written into gen_*.jsonl as skipped records
    # (correct = baseline_correct = 0) rather than dropped, and every reader
    # of the gen file then divides by all rows with no special case. A row
    # list or --subset addresses exactly the rows it names.
    args.skipped_rows = []
    # An explicit row list names the rows outright, so it overrides both the
    # whole-split default and --subset. This is how the scan steers its dev
    # rows: they are not a random sample of the split, they are the fixed
    # hold-out the direction was NOT fitted on, and a seeded stratified
    # subsample cannot express that.
    if getattr(args, "row_ids_file", None):
        wanted = load_row_ids(args.row_ids_file)
        by_key = {(r["subject"], str(r["idx"])): r for r in rows}
        missing = [k for k in wanted if k not in by_key]
        if missing:
            raise SystemExit(
                f"{args.row_ids_file}: {len(missing)} of {len(wanted)} named rows are not "
                f"in {args.exp1_test_collection} (first {missing[0]}). The row list and "
                "the collection disagree — regenerate the row list against this collection.")
        print(f"row list: {len(wanted)} rows from {args.row_ids_file}")
        args.covers_split = False
        return [by_key[k] for k in wanted]
    # subset <= 0 (or None) means the whole split, so "all rows" is expressible
    # without knowing the split size in advance -- which is the only way one
    # setting can mean the same thing across datasets of different sizes.
    if args.mode == "full" or not args.subset or args.subset >= len(rows):
        args.covers_split = True
        args.skipped_rows = [r for r in all_rows if r["truncated"] != "0"]
        return rows
    # proportional stratification by (subject, baseline correctness)
    strata: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        strata[(r["subject"], r["correct"])].append(r)
    args.covers_split = False
    rng = np.random.default_rng(args.seed)
    picked: List[Dict[str, str]] = []
    for key in sorted(strata):
        group = strata[key]
        k = max(1, round(args.subset * len(group) / len(rows)))
        idx = rng.choice(len(group), size=min(k, len(group)), replace=False)
        picked.extend(group[i] for i in idx)
    return picked


def load_row_ids(path: str) -> List[Tuple[str, str]]:
    """(subject, idx) pairs from a CSV, order preserved, duplicates rejected.

    Order is kept because it is the generation order, and a duplicate would make
    the run's row count disagree with the file it was driven by — which is how a
    fold silently overlapping another would go unnoticed.
    """
    import csv as _csv

    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows or "subject" not in rows[0] or "idx" not in rows[0]:
        raise SystemExit(f"{path}: expected a CSV with 'subject' and 'idx' columns")
    keys = [(str(r["subject"]), str(r["idx"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{path}: duplicate (subject, idx) rows")
    return keys


def load_done(path: str) -> Dict[Tuple[str, str], Dict]:
    if not os.path.exists(path):
        return {}
    return {(str(r["subject"]), str(r["idx"])): r for r in common.load_records(path)}


# -----------------------------------------------------------------------------
# One configuration
# -----------------------------------------------------------------------------


def run_config(
    args: argparse.Namespace,
    get_model: Callable[[], Tuple[Any, Any]],
    rows: List[Dict[str, str]],
    prompts: Dict[Tuple[str, str], str],
) -> Dict:
    tag = config_tag(
        args.current_K, args.current_alpha, args.direction,
        random_heads=args.use_random_dir,
        rep=getattr(args, "control_rep", None),
        skip_top=args.skip_top,
        heads_tag=heads_tag_of(args),
        steer_steps=args.steer_steps,
    )
    gen_path = os.path.join(args.out_dir, f"gen_{tag}.jsonl")
    done = load_done(gen_path)
    os.makedirs(args.out_dir, exist_ok=True)
    added = ensure_skipped_records(gen_path, done, getattr(args, "skipped_rows", []))
    if added:
        print(f"{tag}: {added} baseline-truncated rows recorded as skipped "
              "(incorrect on both sides; in the denominator)")
    pending = [(i, r) for i, r in enumerate(rows) if (r["subject"], r["idx"]) not in done]

    # Nothing left to generate: rebuild the summary from the stored rows and
    # return. This is the resume path, so it must not load the model, must not
    # read the train activations, and must not re-open a single generation.
    if not pending:
        print(f"{tag}: {len(done)}/{len(rows) + len(getattr(args, 'skipped_rows', []))} "
              "rows already on disk — "
              "nothing to do, summary rebuilt from gen_" + tag + ".jsonl")
        return summarize_config(args, tag, rows, done, selected_heads(args))

    vectors, top_heads = build_layer_vectors(args)

    # PER-FAMILY, matching what collection graded the baseline with -- a
    # steered qwen row must not be held to the llama parser while its baseline
    # was graded by the qwen one. See grading.GRADERS.
    grader = grading.grader_for(args.model_name)

    def steered_record(r: Dict[str, str], text: str, truncated: bool) -> Dict:
        pred = grader.extract(text)
        return {
            "subject": r["subject"], "idx": r["idx"],
            "gold_answer": r["gold_answer"],
            "baseline_correct": int(r["correct"]),
            "pred_answer": pred,
            # `r` is the stored record, so a row is regraded against the same
            # gold the collection used.
            # A generation that hit the cap is graded on its partial text like
            # any other row (2026-09-01; the formatter OR can lift it later).
            "correct": int(grader.is_correct(pred, r)),
            "truncated": int(truncated),
            "steered": 1,
            "generated_text": text,
        }

    bs = max(1, int(args.batch_size))
    tokenizer, model = get_model()
    steerer = Steerer(model, vectors, args.current_alpha, site=args.site,
                      steer_steps=args.steer_steps,
                      rampdown=args.rampdown, ramp_floor=args.ramp_floor)
    try:
        with open(gen_path, "a", encoding="utf-8") as f:
            bar = tqdm(total=len(pending), desc=tag, unit="gen")
            # tqdm redraws with \r, which turns a nohup log into one unreadable
            # line per config. With TQDM_DISABLE=1 (what the shell drivers export)
            # the bar is silent, so progress is printed as whole lines instead —
            # ~10 per config, enough to follow a multi-hour stage with tail -f.
            beat = Heartbeat(tag, bar.total)
            for start in range(0, len(pending), bs):
                chunk = pending[start:start + bs]
                out = common.generate_batch(
                    tokenizer, model, [prompts[(r["subject"], r["idx"])] for _, r in chunk],
                    max_new_tokens=args.max_new_tokens, temperature=0.0,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                bar.update(len(chunk))
                beat.update(len(chunk))
                for (_, r), res in zip(chunk, out):
                    rec = steered_record(r, *res)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[(r["subject"], r["idx"])] = rec
            bar.close()
    finally:
        steerer.remove()

    return summarize_config(args, tag, rows, done, top_heads)


def skipped_record(r: Dict[str, str]) -> Dict:
    """A baseline-truncated row the steerer never generates for: in the
    denominator, incorrect on both sides (the convention), marked `skipped`
    so no reader re-grades a fragment for it."""
    return {
        "subject": r["subject"], "idx": r["idx"],
        "gold_answer": r["gold_answer"],
        "baseline_correct": 0, "pred_answer": "", "correct": 0,
        "truncated": 1, "steered": 0, "skipped": 1,
        "generated_text": "",
    }


def ensure_skipped_records(gen_path: str, done: Dict[Tuple[str, str], Dict],
                           skipped_rows: List[Dict[str, str]]) -> int:
    """Append a skipped record for every baseline-truncated row of a
    whole-split run that the gen file does not yet hold (older files predate
    the convention and are completed on resume). Returns the count added."""
    missing = [r for r in skipped_rows if (r["subject"], r["idx"]) not in done]
    if missing:
        with open(gen_path, "a", encoding="utf-8") as f:
            for r in missing:
                rec = skipped_record(r)
                f.write(json.dumps(rec) + "\n")
                done[(r["subject"], r["idx"])] = rec
    return len(missing)


def summarize_config(
    args: argparse.Namespace,
    tag: str,
    rows: List[Dict[str, str]],
    done: Dict[Tuple[str, str], Dict],
    top_heads: List[Tuple[int, int]],
) -> Dict:
    """summary_<tag>.json from the stored generations — no GPU, no activations.

    Derived purely from gen_<tag>.jsonl, so a finished config and a config that
    just finished produce the same file.
    """
    # Every addressed row: the generated ones plus the skipped (baseline-
    # truncated) ones of a whole-split run, which sit in the gen file as
    # incorrect-on-both-sides records. Truncated = incorrect throughout.
    addressed = list(rows) + list(getattr(args, "skipped_rows", []))
    recs = [done[(r["subject"], r["idx"])] for r in addressed
            if (r["subject"], r["idx"]) in done]
    # 2026-09-01: a truncated row counts by its label (graded partial text,
    # formatter OR may lift it); only skipped rows are structurally incorrect.
    trunc = np.array([int(x["truncated"]) for x in recs], dtype=bool)
    corr = np.array([int(x["correct"]) for x in recs])
    base = np.array([int(x["baseline_correct"]) for x in recs]) * (~np.array(
        [int(x.get("skipped", 0)) for x in recs], dtype=bool))
    skipped = np.array([int(x.get("skipped", 0)) for x in recs], dtype=bool)
    subjects = np.array([x["subject"] for x in recs])
    per_subject = per_subject_table(subjects, base, corr)
    w2r, r2w = flip_counts(base, corr)
    summary = {
        "config": tag,
        "K": args.current_K,
        "skip_top": args.skip_top,
        "alpha": args.current_alpha,
        # the DOSE SCHEDULE: FR's per-decode-step cosine ramp, always
        "schedule": "cosine",
        "rampdown": args.rampdown,
        "ramp_floor": args.ramp_floor,
        "direction": args.direction,
        "direction_name": args.direction_name,
        "random_heads": int(args.use_random_dir),
        # THE ONE CONVENTION: correct / all addressed rows, truncated (and
        # skipped) = incorrect, on both sides. `n` is the denominator; there is
        # no kept-row or non-truncated variant anywhere.
        "n": len(recs),
        "n_generated": int((~skipped).sum()),
        "n_skipped": int(skipped.sum()),
        "accuracy": float(corr.mean()),
        "baseline_accuracy": float(base.mean()),
        "delta": float(corr.mean() - base.mean()),
        "truncation_rate": float(trunc[~skipped].mean()) if (~skipped).any() else None,
        "flipped_to_correct": w2r,
        "flipped_to_wrong": r2w,
        "per_subject": per_subject,
        # how the heads were CHOSEN, not just which they are
        "explicit_heads": int(bool(args.heads)),
        "heads_tag": heads_tag_of(args),
        "top_heads": [[int(l), int(h)] for l, h in top_heads],
        "heads_per_layer": {str(l): sum(1 for a, _ in top_heads if a == l)
                            for l in sorted({a for a, _ in top_heads})},
        "layers_touched": len({l for l, _ in top_heads}),
    }
    # Atomic: a config counts as done when its summary exists (the shell
    # drivers' done-tests, and the sweep's own skip), so a half-written one would
    # retire a config that never finished.
    write_json(os.path.join(args.out_dir, f"summary_{tag}.json"), summary)
    print({k: summary[k] for k in ("config", "n", "n_generated", "n_skipped",
                                   "accuracy", "baseline_accuracy", "delta",
                                   "truncation_rate", "flipped_to_correct", "flipped_to_wrong")})
    return summary


def _check_stored_identity(args, tokenizer, model, rows, prompts, label) -> None:
    """Generate each row with the CURRENT hook state (unbatched, greedy, same
    budgets) and assert the text equals the stored source generation."""
    records = {(str(r["subject"]), str(r["idx"])): r
               for r in common.load_records(os.path.join(args.exp1_test_collection, "records.jsonl"))}
    for r in rows:
        key = (r["subject"], r["idx"])
        text = common.generate_one(
            tokenizer, model, prompts[key],
            max_new_tokens=args.max_new_tokens, temperature=0.0,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        stored = records[key]["generated_text"]
        status = "IDENTICAL" if text == stored else "MISMATCH"
        print(f"{label} {key}: {status}")
        if status == "MISMATCH":
            print(f"  stored : {stored[:160]!r}\n  regen  : {text[:160]!r}")


def sanity_alpha0(args, tokenizer, model, rows, prompts) -> None:
    """With hooks attached on every layer and alpha=0, generations must equal
    the stored source generations token-for-token. This is the end-to-end proof
    that the hook plumbing itself is inert and that only alpha * sigma * theta
    moves the model."""
    args.current_K = max(args.K)
    vectors, _ = build_layer_vectors(args)
    steerer = Steerer(model, vectors, alpha=0.0, site=args.site,
                      steer_steps=args.steer_steps,
                      rampdown=args.rampdown, ramp_floor=args.ramp_floor)
    try:
        _check_stored_identity(args, tokenizer, model, rows[: args.sanity], prompts,
                               "sanity alpha=0")
    finally:
        steerer.remove()


def run(args: argparse.Namespace) -> None:
    # Both collections must belong to the model about to be steered: the train
    # side supplies sigma and the directions, the test side the prompts and the
    # baseline labels. A mismatch here is hours of GPU spent steering one model
    # with another's head geometry.
    for collection, what in ((args.exp1_train_collection, "the steering directions"),
                             (args.exp1_test_collection, "the prompts being steered")):
        if collection:
            io.assert_model_matches(collection, args.model_name, what)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    rows = select_rows(args)
    prompts = load_prompts(args.exp1_test_collection)
    rows = [r for r in rows if (r["subject"], r["idx"]) in prompts]
    # The CSV holds one gold answer; any accepted alternatives a dataset ships
    # live in records.jsonl. Without this the steered rows
    # would be graded on a stricter rule than the baseline labels they are
    # compared against, and every delta would read negative.
    attach_gold_aliases(args.exp1_test_collection, rows)
    n_skip = len(getattr(args, "skipped_rows", []))
    n_addr = len(rows) + n_skip
    print(f"{n_addr} rows addressed: {len(rows)} to steer + {n_skip} baseline-truncated "
          f"(skipped, incorrect on both sides); baseline acc "
          f"{sum(int(r['correct']) for r in rows) / max(n_addr, 1):.4f} over all {n_addr}")

    # Loaded on first use, not on entry: a fully resumed sweep has no forward
    # pass to make, and loading 8B weights to write summaries it already has
    # would be the one unavoidable cost of re-running the command.
    loaded: List[Tuple[Any, Any]] = []

    def get_model() -> Tuple[Any, Any]:
        if not loaded:
            loaded.append(common.load_model_and_tokenizer(args.model_name, args.dtype))
        return loaded[0]

    if args.sanity > 0:
        sanity_alpha0(args, *get_model(), rows, prompts)

    summaries = []
    for K, alpha, direction, direction_name, random_heads in config_grid(args):
        # per-config, exactly like current_K/current_alpha: in controls mode the
        # direction and head selection vary between the two arms
        args.current_K, args.current_alpha = K, alpha
        args.direction, args.direction_name = direction, direction_name
        args.use_random_dir = random_heads
        summaries.append(run_config(args, get_model, rows, prompts))

    rebuild_sweep_summary(args.out_dir)

    if args.run_best_controls:
        control_summaries = run_best_controls(args, *get_model(), rows, prompts,
                                              summaries)
        rebuild_sweep_summary(args.out_dir)
        best_controls_table(args, summaries, control_summaries)


SWEEP_SUMMARY_FIELDS = [
    "config", "K", "alpha",
    "schedule", "rampdown", "ramp_floor",
    "direction", "direction_name", "random_heads", "n", "n_generated", "n_skipped",
    "accuracy", "baseline_accuracy", "delta", "truncation_rate",
    "flipped_to_correct", "flipped_to_wrong", "layers_touched"]


def rebuild_sweep_summary(out_dir: str) -> None:
    """Rebuild sweep_summary.csv from every summary_*.json in the directory.

    Regenerate-from-source, written atomically: reruns and resumed sweeps
    always produce the same file instead of appending duplicate rows.
    """
    summaries = []
    for path in glob.glob(os.path.join(out_dir, "summary_*.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    if not summaries:
        return
    # `or 0`: older summaries may store alpha=null
    summaries.sort(key=lambda s: (int(s.get("K", 0)), float(s.get("alpha") or 0),
                                  str(s.get("config", ""))))
    sweep_path = os.path.join(out_dir, "sweep_summary.csv")
    write_csv(sweep_path, summaries, SWEEP_SUMMARY_FIELDS)
    print(f"sweep_summary.csv rebuilt from {len(summaries)} summaries -> {sweep_path}")


# -----------------------------------------------------------------------------
# One-command best-config controls + per-subject table
# -----------------------------------------------------------------------------


def load_summary(out_dir: str, tag: str) -> Dict:
    with open(os.path.join(out_dir, f"summary_{tag}.json")) as f:
        return json.load(f)


def select_best_summary(args: argparse.Namespace, summaries: List[Dict]) -> Dict:
    """Select the main config whose full-test accuracy rose the most."""
    loaded = [load_summary(args.out_dir, s["config"]) for s in summaries]
    candidates = [(float(s["delta"]), float(s["accuracy"]), int(s["K"]),
                   float(s["alpha"] or 0), s) for s in loaded]
    best = max(candidates, key=lambda x: (x[0], x[1], -x[2], -x[3]))[-1]
    print(f"best config by delta: {best['config']} (K={best['K']}, {alpha_label(best)}, "
          f"delta={best['delta']:+.4f})")
    return best


def alpha_label(summary: Dict) -> str:
    """How this config set alpha, for a log line."""
    return f"alpha={summary['alpha']:g}"


def run_best_controls(
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    rows: List[Dict[str, str]],
    prompts: Dict[Tuple[str, str], str],
    main_summaries: List[Dict],
) -> List[Dict]:
    """Run random-head and random-direction controls at the best main K/alpha.

    Repeats use different seeds and unique tags so the table can report mean + std.
    """
    best = select_best_summary(args, main_summaries)
    control_summaries: List[Dict] = []
    original = {
        "current_K": getattr(args, "current_K", None),
        "current_alpha": getattr(args, "current_alpha", None),
        "direction": args.direction,
        "direction_name": args.direction_name,
        "use_random_dir": args.use_random_dir,
        "seed": args.seed,
    }
    try:
        for label in ("random_heads", "random_directions"):
            for rep in range(args.control_repeats):
                args.current_K = int(best["K"])
                args.current_alpha = float(best["alpha"] or 0.0)
                args.seed = int(original["seed"]) + args.control_seed_offset + rep
                if label == "random_heads":
                    args.direction = best["direction"]
                    args.direction_name = best.get("direction_name", best["direction"])
                    args.use_random_dir = True
                else:
                    args.direction = "random"
                    args.direction_name = "random"
                    args.use_random_dir = False
                # config_tag adds _randheads from use_random_dir and _r<rep> from
                # control_rep, so the arms come out K<k>_alpha<a>_<dir>_randheads_r<n>
                # and K<k>_alpha<a>_random_r<n> — no doubled suffixes.
                args.control_rep = rep + 1
                print(f"\n[best-control] {label} repeat {rep + 1}/{args.control_repeats}: "
                      f"K={args.current_K}, {alpha_label(best)}, seed={args.seed}")
                control_summaries.append(
                    run_config(args, lambda: (tokenizer, model), rows, prompts))
    finally:
        args.current_K = original["current_K"]
        args.current_alpha = original["current_alpha"]
        args.direction = original["direction"]
        args.direction_name = original["direction_name"]
        args.use_random_dir = original["use_random_dir"]
        args.seed = original["seed"]
        args.control_rep = None
    return control_summaries


def _summary_acc(summary: Dict, group: str) -> Tuple[float, int]:
    if group == "__full__":
        return float(summary["accuracy"]), int(summary["n"])
    info = summary["per_subject"][group]
    return float(info["acc"]), int(info["n"])


def _baseline_acc(summary: Dict, group: str) -> Tuple[float, int]:
    if group == "__full__":
        return float(summary["baseline_accuracy"]), int(summary["n"])
    info = summary["per_subject"][group]
    return float(info["baseline_acc"]), int(info["n"])


def _control_mean_std(
    summaries: List[Dict], group: str, predicate,
) -> Tuple[float, float, int]:
    vals = []
    n = 0
    for s in summaries:
        if not predicate(s):
            continue
        acc, n = _summary_acc(s, group)
        vals.append(acc)
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.array(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, int(arr.size)


def best_controls_table(
    args: argparse.Namespace,
    main_summaries: List[Dict],
    control_summaries: List[Dict],
) -> None:
    """Per-subject / full-test accuracy of baseline, steered and the two
    random controls at the best config, as CSV + JSON."""
    best = select_best_summary(args, main_summaries)
    groups = ["__full__"] + sorted(best["per_subject"])
    labels = ["Full test"] + [g.replace("_", " ") for g in groups[1:]]

    rows = []
    for group in groups:
        baseline, n = _baseline_acc(best, group)
        steered, _ = _summary_acc(best, group)
        rand_heads_mean, rand_heads_std, rand_heads_repeats = _control_mean_std(
            control_summaries, group, lambda s: bool(s.get("random_heads"))
        )
        rand_dirs_mean, rand_dirs_std, rand_dirs_repeats = _control_mean_std(
            control_summaries, group,
            lambda s: (not bool(s.get("random_heads"))) and s.get("direction") == "random",
        )
        rows.append({
            "group": "full_test" if group == "__full__" else group,
            "n": n,
            "best_config": best["config"],
            "K": best["K"],
            "alpha": best["alpha"],
            "baseline": baseline,
            "steered": steered,
            "random_heads_mean": rand_heads_mean,
            "random_heads_std": rand_heads_std,
            "random_heads_repeats": rand_heads_repeats,
            "random_directions_mean": rand_dirs_mean,
            "random_directions_std": rand_dirs_std,
            "random_directions_repeats": rand_dirs_repeats,
        })

    csv_path = os.path.join(args.out_dir, "best_controls_subject_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out_dir, "best_controls_subject_accuracy.json"), "w") as f:
        json.dump({
            "best_config": best["config"],
            "selection_metric": "delta",
            "control_repeats": args.control_repeats,
            "csv": csv_path,
            "rows": rows,
        }, f, indent=2)
    print(f"best-controls table -> {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp1_train_collection", required=True)
    p.add_argument("--exp4_train_collection", default=None,
                   help="dir holding train head/ activations (default: --exp1_train_collection)")
    p.add_argument("--exp1_test_collection", required=True)
    p.add_argument("--probe_report", default=None,
                   help="the elicit directory holding directions.npz and heads.sh "
                        "(default: <exp4_train_collection>/probe_report, kept for legacy layouts)")
    p.add_argument("--out_dir", default=None, help="default: <exp4_train_collection>/steering")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bf16", choices=list(common.DTYPES))
    p.add_argument("--mode", default="sweep", choices=["sweep", "full", "controls"],
                   help="sweep: the --K x --alpha grid on --subset rows. full: one config on "
                        "every test row. controls: BOTH ablations at one (K, alpha) — random "
                        "heads with the real direction, and the real top-K heads with random "
                        "directions")
    p.add_argument("--K", type=int, nargs="+", default=[48])
    p.add_argument("--heads", nargs="+", default=None,
                   help="steer THESE heads (L45H9 L37H11 ...) instead of the top-K "
                        "by probe validation accuracy. The ranking is not consulted: "
                        "the named heads are what gets the shift. --K "
                        "and --skip_top no longer apply; the config is tagged "
                        "H<heads_tag> instead of K<k>, so it cannot collide with a "
                        "top-K run of the same size")
    p.add_argument("--heads_tag", default=None,
                   help="short label naming the --heads set in filenames "
                        "(default: sel<N>_<hash>). Use something you will recognise, "
                        "e.g. comaligned")
    p.add_argument("--alpha", type=float, nargs="+", default=[15.0])
    p.add_argument("--direction", default="com", choices=sorted(DIRECTIONS),
                   help="theta in Eq. 2. com (default): mean(correct)-mean(incorrect), "
                        "the intervention. anticom: the same vector reversed (the "
                        "suppression control, at a positive alpha). random: one "
                        "N(0,I) vector per head (ITI's control)")
    p.add_argument("--skip_top", type=int, default=0,
                   help="drop the S highest-ranked heads and steer the NEXT --K instead, "
                        "turning the top-K prefix into a rank window: --K 7 --skip_top 1 "
                        "steers ranks 2-8. The head ablation — does the effect come from a "
                        "few paramount heads, or is it spread down the ranking? Tagged "
                        "_skip<S>, so it never collides with a plain top-K run")
    p.add_argument("--use_random_dir", action="store_true",
                   help="control: intervene on K randomly chosen HEADS instead of the top-K "
                        "(utils.py:716-719). Orthogonal to --direction, which chooses theta; "
                        "the reference's single --use_random_dir flag does both at once, so "
                        "`--use_random_dir --direction random` reproduces it exactly")
    p.add_argument("--steer_steps", default="all", choices=list(STEER_STEPS),
                   help="WHEN the shift is applied. all (default) = every forward "
                        "pass, i.e. the prompt AND every generated token, which is "
                        "ITI's own behaviour. prefill = the prompt only: the shift "
                        "lands on the last prompt token, enters the KV cache once, and "
                        "generation runs untouched from there. prefill is the setting "
                        "that matches how theta and sigma were measured (both come "
                        "from the last prompt token), and it does not compound per "
                        "generated token")
    p.add_argument("--rampdown", type=float, default=RAMPDOWN,
                   help="the cosine dose schedule: decode steps over which the dose "
                        "falls from full to the floor (FR: 50)")
    p.add_argument("--ramp_floor", type=float, default=RAMP_FLOOR,
                   help="the cosine dose schedule: the dose the ramp clips at, as a "
                        "fraction of alpha (FR: 0.5)")
    p.add_argument("--site", default="o_proj_in", choices=list(STEER_SITES),
                   help="WHERE the shift is added, and therefore which activations "
                        "sigma is measured in. o_proj_in (default) is ITI's own site: "
                        "the concatenated per-head attention output, one shift per "
                        "selected HEAD. layer_out adds one shift per selected LAYER to "
                        "that layer's output residual stream. The probe report must "
                        "match: a layer report has num_heads=1 and head_dim=d_model")
    p.add_argument("--run_best_controls", action="store_true",
                   help="after the main ungated sweep, pick the config with the largest "
                        "overall accuracy gain, run random-head and random-direction controls "
                        "at the same K/alpha, and write a per-subject/full-test table (CSV + JSON)")
    p.add_argument("--control_repeats", type=int, default=3,
                   help="random-control repeats; the table reports mean +/- sample std")
    p.add_argument("--control_seed_offset", type=int, default=10000,
                   help="offset added to --seed for the random-control repeats")
    p.add_argument("--subset", type=int, default=0,
                   help="sweep-mode test rows: 0 (the default) is the ENTIRE test split; "
                        "N > 0 takes a seeded, stratified subsample of N rows. A fixed "
                        "positive default silently subsamples any split larger than it, "
                        "which is a different evaluation set per dataset rather than a "
                        "consistent one")
    p.add_argument("--row_ids_file", default=None,
                   help="CSV with subject,idx columns naming EXACTLY the rows to "
                        "generate, in place of the whole split. The scan stage passes "
                        "its dev rows this way, so every band and alpha is compared on "
                        "the identical problems; --subset is a seeded sample of a split, "
                        "which cannot express 'these particular rows'. Absent (the "
                        "default) leaves row selection bit-identical to before")
    p.add_argument("--sanity", type=int, default=0, help="check N alpha=0 generations against stored source text")
    p.add_argument("--batch_size", type=int, default=32,
                   help="rows generated per batched pass. 1 restores the pre-batching "
                        "path; --sanity always runs unbatched because it asserts "
                        "bit-identity against singly-generated stored text.")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.exp4_train_collection is None:
        args.exp4_train_collection = args.exp1_train_collection
    # keep the spelling the user typed for logs, but resolve aliases to the
    # the canonical direction name is what filenames carry
    args.direction_name = args.direction
    args.direction = DIRECTIONS[args.direction]
    if any(a < 0 for a in args.alpha):
        p.error("alpha must be >= 0; the reversed shift is --direction anticom")
    if args.heads:
        try:
            named = parse_heads(args.heads)
        except ValueError as exc:
            p.error(str(exc))
        if args.skip_top:
            p.error("--skip_top is a window over the RANKING; it means nothing "
                    "for an explicit --heads list. Drop one of the two.")
        if len(args.K) > 1 or args.K != [len(named)]:
            print(f"--heads: ignoring --K {args.K} — the head set is given, so K is "
                  f"{len(named)} by definition")
        # K stops being a grid axis; it is still SET because the summary and
        # the tag rule record it, and here it is the size of the named set.
        args.K = [len(named)]
    if args.run_best_controls:
        if args.control_repeats < 1:
            p.error("--control_repeats must be >= 1")
    if args.mode == "controls":
        # the controls are an ablation at one operating point, not a search
        if len(args.K) != 1 or len(args.alpha) != 1:
            p.error("--mode controls runs both ablations at ONE (K, alpha); pass a single "
                    f"--K and a single --alpha (got K={args.K}, alpha={args.alpha})")
        if args.use_random_dir:
            p.error("--use_random_dir is implied by --mode controls (it IS the first control)")
        if args.direction == "random":
            p.error("--direction random is implied by --mode controls (it IS the second "
                    "control); --direction should name the REAL direction being ablated")
    if args.probe_report is None:
        args.probe_report = os.path.join(args.exp4_train_collection, "probe_report")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.exp4_train_collection, "steering")
    return args


if __name__ == "__main__":
    run(parse_args())
