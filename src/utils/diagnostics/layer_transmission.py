#!/usr/bin/env python3
"""Which layers can be steered at all: does an injection at layer l reach the output?

Motivation. head_ranker scores a head by the logit shift its injected
vector would produce IF nothing downstream touched it -- a DIRECT-PATH score.
That assumption is why the score failed at layer 26: L26 carried the third
highest positive elicitation mass (0.215, behind L31's 0.269 and L30's 0.233),
yet the E26_31 head set lost to E28_31 at every matched dose. Five intervening
layers undid the injection. The band is not a hyperparameter to grid-search; it
is the region where the direct path survives, and that is measurable.

TWO MODES, cheap first.

  lens      CPU, zero GPU, reads the hidden/ collection already on disk. For
            each layer, read the stored residual stream through the final norm
            and the unembedding (the classic logit lens) and ask how well it
            already reproduces the model's own final prediction. Where the lens
            is faithful, the map from that layer to the output is close to the
            identity in the directions that matter, so an injection there should
            transmit. This is a PROXY: it measures whether layer l already
            encodes the answer, not whether a perturbation survives.

  transmit  GPU, but prompts only -- prefill forwards, no generation, ~1 minute.
            The direct measurement. Inject a fixed delta at layer l's output on
            the last prompt token, run forward, and see how much of it is still
            in the final residual stream:

                t_l   = <x'_final - x_final, delta> / ||delta||^2
                dist_l= ||(x'_final - x_final) - delta|| / ||delta||

            t=1, dist=0 means the injection arrived untouched. t>1 means
            downstream layers amplified it; dist large means they rewrote it
            into something else. Layer 31 has nothing downstream, so it must
            score t=1, dist=0 -- a built-in correctness check on the whole
            measurement.

            It also reports the causal version of the elicitation score:
            <delta_logits, p_cot - p_direct>, i.e. how far an injection at this
            layer actually moves the model toward opening with " To" rather than
            with " $". That is the quantity the direct-path score estimates, so
            comparing the two columns shows exactly where the estimate is
            trustworthy.

The injection reuses steering_driver.Steerer at site="layer_out", so what is
measured here is the same mechanism a steering run applies -- not a
re-implementation that could drift from it.

Run:
  # free, no GPU
  python3 -m src.utils.diagnostics.layer_transmission lens \
      --exp1_collection <.../train_qa> --model_name meta-llama/Llama-3.1-8B-Instruct

  # the causal check, prompts only
  python3 -m src.utils.diagnostics.layer_transmission transmit \
      --exp1_collection <.../test_qa> --gen_dir <.../steering_headsL28_31> \
      --model_name meta-llama/Llama-3.1-8B-Instruct --n_prompts 64
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core import common, io
from src.steering.head_ranker import COT_MIN_WORDS, load_unembedding

RMS_EPS = 1e-5          # fallback only; read the model's own value, see below


def rms_norm_eps(model_name: str) -> float:
    """The model's own RMSNorm epsilon.

    It is NOT a constant across families -- llama-3.1 uses 1e-5 and qwen3 uses
    1e-6 -- and this function's whole job is to reproduce the model's norm
    exactly, so it is read from the config rather than assumed. (The term sits
    under a sqrt beside mean(x^2), so the numerical difference is small; using
    the wrong one is still a silent divergence from the model being measured.)
    """
    from pathlib import Path

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        model_name, cache_dir=common.model_cache_dir(model_name))
    return float(getattr(cfg, "rms_norm_eps", RMS_EPS))


def rms_norm(X: np.ndarray, gain: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """RMSNorm as the model applies it: g * x / sqrt(mean(x^2) + eps)."""
    scale = np.sqrt((X.astype(np.float32) ** 2).mean(axis=-1, keepdims=True) + eps)
    return (X / scale) * gain


def elicitation_delta_p(gen_dir: str, tokenizer, vocab: int) -> np.ndarray:
    """p_cot - p_direct over first tokens, as in head_ranker."""
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
                if ids:
                    (cot if len(text.split()) > COT_MIN_WORDS else direct)[ids[0]] += 1
    if not cot or not direct:
        raise SystemExit(f"{gen_dir}: need both reasoned and direct generations")
    out = np.zeros(vocab, dtype=np.float32)
    for counter, sign in ((cot, +1.0), (direct, -1.0)):
        total = sum(counter.values())
        for tid, n in counter.items():
            out[tid] += sign * n / total
    return out


def target_delta_p(args, tokenizer, vocab: int):
    """p_reason - p_direct, from whichever source was given.

    --target_npz is the prompt-pair target src.steering.elicitation_target
    writes: two prefill passes per problem, available BEFORE any steering has
    run, which is when the band is actually being chosen. --gen_dir is the older
    source and needs a finished sweep to classify by length. Neither: the
    transmission columns still print, without the elicitation one.
    """
    if getattr(args, "target_npz", None):
        return np.load(args.target_npz, allow_pickle=True)["delta_p"]
    if args.gen_dir:
        return elicitation_delta_p(args.gen_dir, tokenizer, vocab)
    return None


# --------------------------------------------------------------------------
# lens mode
# --------------------------------------------------------------------------

def run_lens(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    io.assert_model_matches(args.exp1_collection, args.model_name,
                            "layer transmission")

    U, gain = load_unembedding(args.model_name)
    cache_dir = common.model_cache_dir(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                              cache_dir=cache_dir)

    X, _, _ = io.load_layer_collection(args.exp1_collection, balanced_only=False,
                                       drop_truncated=True)
    if args.n_rows and args.n_rows < X.shape[0]:
        rng = np.random.default_rng(args.seed)
        X = X[rng.choice(X.shape[0], args.n_rows, replace=False)]
    N, L, _ = X.shape
    print(f"lens: {N} rows x {L} layers from {args.exp1_collection}")

    dp = target_delta_p(args, tokenizer, U.shape[0])

    final = X[:, L - 1]
    eps = rms_norm_eps(args.model_name)
    final_logits = rms_norm(final, gain, eps) @ U.T
    final_top1 = final_logits.argmax(axis=-1)
    final_norm = np.linalg.norm(final, axis=-1).mean()
    final_elicit = (final_logits @ dp) if dp is not None else None

    print(f"\n{'layer':>5} {'top1 agree':>11} {'top1 in lens top5':>18} "
          f"{'||x_l||/||x_L||':>16} {'cos(x_l,x_L)':>13}"
          + (f" {'elicit corr':>12}" if dp is not None else ""))
    for l in range(L):
        h = X[:, l]
        logits = rms_norm(h, gain, eps) @ U.T
        agree = float((logits.argmax(axis=-1) == final_top1).mean())
        top5 = np.argpartition(-logits, 5, axis=-1)[:, :5]
        in5 = float(np.mean([final_top1[i] in top5[i] for i in range(N)]))
        nrm = float(np.linalg.norm(h, axis=-1).mean() / final_norm)
        cos = float(np.mean((h * final).sum(-1)
                            / (np.linalg.norm(h, axis=-1) * np.linalg.norm(final, axis=-1))))
        line = f"{l:>5} {agree:>11.3f} {in5:>18.3f} {nrm:>16.3f} {cos:>13.3f}"
        if dp is not None:
            e = logits @ dp
            line += f" {float(np.corrcoef(e, final_elicit)[0, 1]):>12.3f}"
        print(line)
    print("\nRead: the band is where top1-agree stops being near zero. That is where the\n"
          "residual stream already carries the model's answer, so an injection is not\n"
          "going to be rewritten by what follows. Confirm with `transmit`.")


# --------------------------------------------------------------------------
# transmit mode
# --------------------------------------------------------------------------

def run_transmit(args: argparse.Namespace) -> None:
    import torch

    io.assert_model_matches(args.exp1_collection, args.model_name,
                            "layer transmission")
    from transformers import AutoTokenizer
    from src.steering import steering_driver

    U, gain = load_unembedding(args.model_name)
    # returns (tokenizer, model), in that order
    tokenizer, model = common.load_model_and_tokenizer(args.model_name, args.dtype)
    model.eval()
    layers = common.decoder_layers(model)
    L = len(layers)
    device = next(model.parameters()).device

    prompts = list(io.load_prompts(args.exp1_collection).values())
    rng = np.random.default_rng(args.seed)
    if args.n_prompts < len(prompts):
        prompts = [prompts[i] for i in rng.choice(len(prompts), args.n_prompts, replace=False)]
    print(f"transmit: {len(prompts)} prompts, {L} layers, dtype {args.dtype}")

    dp = target_delta_p(args, tokenizer, U.shape[0])
    if dp is not None:
        dp = torch.tensor(dp, device=device, dtype=torch.float32)

    # Capture the FINAL decoder layer's output at the last position. Own hook
    # rather than output_hidden_states: HF has moved whether the last entry is
    # pre- or post-norm between versions, and this cannot be ambiguous.
    captured: List[torch.Tensor] = []

    def capture(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured.append(hs[:, -1, :].detach().float().clone())

    def forward_all() -> torch.Tensor:
        """Prefill every prompt and return the final layer's last-position state.

        Any Steerer is already attached to the model, and its hook was registered
        BEFORE this capture hook, so on layer L-1 the capture sees the injected
        value rather than the clean one -- which is what makes layer L-1 read
        t=1.000 and act as the correctness check.
        """
        captured.clear()
        handle = layers[L - 1].register_forward_hook(capture)
        try:
            for i in range(0, len(prompts), args.batch_size):
                batch = prompts[i:i + args.batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=args.max_prompt_tokens).to(device)
                with torch.no_grad():
                    model(**enc)
            outs = torch.cat(captured, dim=0)
        finally:
            handle.remove()
        return outs

    tokenizer.padding_side = "left"      # so position -1 is the last real token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = forward_all()
    base_scale = float(base.norm(dim=-1).mean())
    eps = args.eps_rel * base_scale
    print(f"  mean ||x_final|| = {base_scale:.2f}; injecting ||delta|| = {eps:.3f} "
          f"({100 * args.eps_rel:.1f}% of it) at each layer\n")

    u = None
    if dp is not None:
        u = torch.tensor((dp.cpu().numpy() @ U), device=device, dtype=torch.float32)
        u = u / u.norm()

    Ut = torch.tensor(U, device=device, dtype=torch.float32)
    gt = torch.tensor(gain, device=device, dtype=torch.float32)

    # norm_eps is the model's RMSNorm epsilon; `eps` above is the SIZE of the
    # injected delta. Two unrelated quantities, kept apart by name.
    norm_eps = rms_norm_eps(args.model_name)

    def logits_of(x: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt((x ** 2).mean(-1, keepdim=True) + norm_eps)
        return (x / scale * gt) @ Ut.T

    base_logit_elicit = (logits_of(base) @ dp).mean().item() if dp is not None else None

    dirs: Dict[str, torch.Tensor] = {}
    if u is not None:
        dirs["elicit"] = u
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    r = torch.randn(base.shape[-1], generator=g).to(device)
    dirs["random"] = r / r.norm()

    print(f"{'layer':>5} " + " ".join(
        f"{k+'_t':>10} {k+'_dist':>10}" for k in dirs)
        + (f" {'dlogit_elicit':>14}" if dp is not None else ""))
    for l in range(L):
        row = f"{l:>5} "
        cells = []
        dlogit = None
        for name, d in dirs.items():
            vec = (d * eps).cpu().numpy().astype(np.float32)
            st = steering_driver.Steerer(model, {l: vec}, alpha=1.0, site="layer_out")
            try:
                out = forward_all()
            finally:
                st.remove()
            delta = out - base
            dv = torch.tensor(vec, device=device, dtype=torch.float32)
            t = float((delta @ dv).mean() / (dv @ dv))
            dist = float(((delta - dv).norm(dim=-1) / dv.norm()).mean())
            cells.append(f"{t:>10.3f} {dist:>10.3f}")
            if name == "elicit" and dp is not None:
                dlogit = float(((logits_of(out) @ dp).mean().item() - base_logit_elicit))
        row += " ".join(cells)
        if dlogit is not None:
            row += f" {dlogit:>14.4f}"
        print(row, flush=True)
    print("\nSanity: layer", L - 1, "must read t=1.000 dist=0.000 -- nothing follows it.\n"
          "Read: t near 1 with small dist = the injection arrives intact, so this layer is\n"
          "steerable. dist blowing up = downstream layers rewrite it. dlogit_elicit is the\n"
          "causal elicitation effect the direct-path score was estimating.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["lens", "transmit"])
    p.add_argument("--exp1_collection", required=True)
    p.add_argument("--target_npz", default=None,
                   help="elicit_target.npz from src.steering.elicitation_target -- the "
                        "prompt-pair target. Preferred: needs no generations, so the "
                        "band can be checked before any steering has run.")
    p.add_argument("--gen_dir", default=None,
                   help="alternative to --target_npz: a finished steering sweep, whose "
                        "generations are split into reasoned/direct by length")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bf16", choices=list(common.DTYPES))
    p.add_argument("--n_rows", type=int, default=1000, help="lens mode")
    p.add_argument("--n_prompts", type=int, default=64, help="transmit mode")
    p.add_argument("--eps_rel", type=float, default=0.02,
                   help="injected norm as a fraction of the mean final residual norm")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    (run_lens if args.mode == "lens" else run_transmit)(args)


if __name__ == "__main__":
    main()
