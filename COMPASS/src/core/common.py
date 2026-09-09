#!/usr/bin/env python3
"""Shared utilities for COMPASS procedures.

Everything that is common to collection, probing, and steering lives here:

- Reproducibility (seeding).
- Loading Llama-style causal LMs and tokenizers from Hugging Face.
- Prompt construction for "direct" (answer-only) and "cot" (chain-of-thought)
  inference modes, with a plain-text fallback when no chat template exists.
- Loading examples from the EleutherAI/hendrycks_math (MATH), openai/gsm8k
  (GSM8K) and the eval-only transfer sets (MATH-500, GSM-Plus, HARP, SVAMP),
  behind one `load_examples(dataset, ...)` entry point.
- MATH answer extraction (\\boxed{...}, \\fbox{...}, textual fallbacks),
  LaTeX-aware normalization, and approximate answer equality (including
  percent handling and numeric/fraction comparison).
- Prompt-only hidden-state extraction (one vector per layer, including the
  embedding layer at index 0).
- Single-prompt greedy/sampled generation.
- JSONL record loading.

The answer matcher here is a heuristic exact/numeric matcher, good enough for
diagnostic probes but not a full MATH benchmark evaluator.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.grading import extract_final_answer  # gold-answer extraction for the loaders

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# SVAMP labels each word problem with the arithmetic operation it needs
# (`Type`); that is the subject taxonomy. Slugs of the dataset's own strings
# ("Common-Division" -> common_division; one train row is misspelled
# "Common-Divison" and is folded onto the same subject by the loader).
SVAMP_SUBJECTS = [
    "addition",
    "subtraction",
    "multiplication",
    "common_division",
]

# GSM-Plus perturbs each GSM8K seed question 8 ways; the perturbation type is
# the natural subject taxonomy (slugs of the dataset's `perturbation_type`).
GSM_PLUS_SUBJECTS = [
    "adding_operation",
    "digit_expansion",
    "distraction_insertion",
    "integer_decimal_fraction_conversion",
    "numerical_substitution",
    "problem_understanding",
    "reversing_operation",
]


# GSM8K has no subject taxonomy, but the whole on-disk contract (hidden/<subject>
# __<idx>.npy, the (subject, idx) key, every per-subject breakdown) is keyed by
# one, so it gets a single pseudo-subject. Per-subject tables then have exactly
# one group, which is the honest rendering of a dataset with one category.
GSM8K_SUBJECTS = ["gsm8k"]

# Perturbation types EXCLUDED from the pipeline entirely. `critical_thinking`
# rows are UNANSWERABLE by design (the perturbation deletes a premise and the
# gold answer is the literal "None"), so a `correct` label on them measures
# refusal style, not problem solving -- one polluted label class would leak
# into every probe fit and steering delta downstream. The loader stamps each
# row `use=0/1` from this set and DROPS the use=0 rows before returning, so an
# excluded row can never reach a collection, a probe, or a scan; they are not
# in GSM_PLUS_SUBJECTS, so --subject cannot name them back in either.
GSM_PLUS_EXCLUDED_SUBJECTS = {"critical_thinking"}


DATASETS = ("math", "gsm8k", "gsm_plus", "math500", "harp", "svamp")


# Root of the pre-downloaded MATH cache, one subdir per subject, matching the
# per-subject cache_dir convention used across the repo (see ICL/ICL-demo.py).
# Override with the HENDRYCKS_MATH_ROOT env var. If a subject is not cached
# there, load_dataset falls back to downloading it from the Hugging Face Hub.
# THE FIXED FITTING POOLS (2026-08-30). A `<family>calib` split is 1,600 rows
# of a dataset's train split -- 800 correct + 800 incorrect under THAT
# family's single-pass labels, each class restricted to rows whose first
# generated token is more likely under its own class (opener LLR), so the
# first-token outcome target is well separated (TV = 1.0 by construction).
# FROZEN as data/<dataset>_<split>.json ((subject, idx) into the dataset's
# *train*); the repo carries no code that (re)defines them. The loaders serve
# exactly those rows to EVERY model, so a qwen run on `llamacalib` fits on
# the identical problems.
#   llamacalib  chosen on Llama-3.1-8B's labels 
#   qwencalib   chosen on Qwen3-4B's labels
#   gemmacalib  chosen on gemma-4-12B's labels
CALIB_SPLITS = ("llamacalib", "qwencalib", "gemmacalib")
CALIB_DATASETS = ("math", "gsm8k")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")


def calib_manifest(dataset: str, split: str) -> str:
    """Path of the (subject, idx) manifest that defines `split` of `dataset`."""
    if split not in CALIB_SPLITS or dataset not in CALIB_DATASETS:
        raise ValueError(
            f"{dataset!r} has no {split!r} split: the fixed fitting pools are "
            f"{CALIB_SPLITS} on {CALIB_DATASETS}, frozen under data/)")
    return os.environ.get("CALIB_MANIFEST_DIR", DATA_DIR) + f"/{dataset}_{split}.json"


def calib_keys(dataset: str, split: str) -> Dict[str, set]:
    """{subject -> set(idx)} of the rows the manifest names."""
    with open(calib_manifest(dataset, split), encoding="utf-8") as f:
        out: Dict[str, set] = {}
        for subj, idx in json.load(f)["rows"]:
            out.setdefault(subj, set()).add(int(idx))
    return out


# -----------------------------------------------------------------------------
# Machine-specific paths: ALL of them live in data/paths.py, the registry
# (dataset caches, model weights, the embeddings root, venvs). It is loaded by
# file location so it works from any cwd, and nothing in src/ names an
# absolute path. The module attributes below are the RESOLVED values
# (environment overrides applied, "" when unconfigured); every point of use
# goes through paths.require() so an unconfigured path fails THERE, naming
# the registry, rather than quietly loading into the current directory.
# -----------------------------------------------------------------------------


def _load_paths():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compass_paths", os.path.join(DATA_DIR, "paths.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paths = _load_paths()
EMBED_ROOT = paths.get("EMBED_ROOT")
DATASETS_ROOT = paths.get("DATASETS_ROOT")
MODEL_CACHE_ROOT = paths.get("MODEL_CACHE_ROOT")


def cache_safe_name(name: str) -> str:
    """Filesystem-safe slug for a model name, used as the per-model cache dir
    (e.g. meta-llama__Llama-3.1-8B-Instruct)."""
    resolved = os.path.basename(os.path.abspath(name)) if os.path.exists(name) else name
    return re.sub(r"[^A-Za-z0-9._-]+", "__", resolved).strip("_")


def model_cache_dir(model_name: str) -> str:
    """<MODEL_CACHE_ROOT>/<cache_safe_name>: the one place a model's weights,
    config and tokenizer are read from. Refuses when the root is unconfigured."""
    return str(Path(paths.require("MODEL_CACHE_ROOT")) / cache_safe_name(model_name))

# Accepts both short (bf16) and long (bfloat16) spellings used by the CLIs.
DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model_and_tokenizer(model_name: str, dtype: str) -> Tuple[Any, Any]:
    """Load a causal LM in eval mode with device_map="auto" plus its tokenizer.

    `dtype` may be any key of DTYPES (e.g. "bf16" or "bfloat16"). Weights are
    read from the per-model cache under MODEL_CACHE_ROOT so a pre-downloaded
    model loads without re-downloading; a model absent there is fetched from the
    Hub into that same cache.
    """
    cache_dir = model_cache_dir(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPES[dtype],
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    model.eval()
    return tokenizer, model


# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------


PROMPT_MODES = ("standard", "cot")


# -----------------------------------------------------------------------------
# The per-family `standard` prompt
# -----------------------------------------------------------------------------
# ONE prompt per model family: the finalized baseline that every collection,
# probe and steering run uses. The old prompt-mode grid (direct / cot /
# cot_budget / qa / qa_brief) was removed on 2026-08-20 once each family's
# variant was settled; the winner was promoted to "standard" and the rest
# deleted. One string cannot serve every family -- families differ in how they
# read the same instruction (llama volunteers ~30 words of reasoning, qwen
# collapses to a bare number) -- so the prompt is per FAMILY, resolved from
# the tokenizer's model id.
#
# Where each family's string comes from, character for character (do not
# "tidy" any of them -- they must keep matching their stored collections):
#   llama    the original shared `qa` prompt. Every llama collection tagged
#            _qa was generated from exactly this string.
#   qwen     the finalized qwen `qa_brief` prompt (deliberately LEAKY terse
#            instruction). Existing _qabrief qwen collections match it.
#   gemma    the gemma-4 builder (2026-08-31), see _standard_prompt_gemma.
#   other    falls back to STANDARD_PROMPT_DEFAULT (llama's string).
#
# Adding a family: write a builder taking the problem text and returning
# (system, user), and register it in STANDARD_PROMPT_BUILDERS.
#
# WARNING: existing collection directories keep the historical tag of the mode
# their prompt was finalized under (_qa, _qabrief -- io.MODE_TAG); NEW
# collections are tagged _standard. A family whose builder changes must be
# re-collected, never resumed.

_MODEL_FAMILY_PATTERNS = (
    ("llama", ("llama",)),
    ("qwen", ("qwen",)),
    ("gemma", ("gemma",)),
)


def model_family(model_ref: Any) -> str:
    """Family slug for a model id, tokenizer or model ("qwen", "llama", ...).

    Matched on the HF repo id, lowercased, so "Qwen/Qwen3-4B" and a
    local snapshot path both resolve to "qwen". Anything unrecognised (including
    a tokenizer with no name_or_path) is "unknown", which is a valid answer
    here: it selects the default prompt rather than raising.
    """
    name = model_ref if isinstance(model_ref, str) else getattr(model_ref, "name_or_path", "")
    name = str(name or "").lower()
    for family, needles in _MODEL_FAMILY_PATTERNS:
        if any(needle in name for needle in needles):
            return family
    return "unknown"


# -----------------------------------------------------------------------------
# The `standard` prompt 
# -----------------------------------------------------------------------------

def _standard_prompt_default(problem: str) -> Tuple[str, str]:
    return "", (
        f"You are a helpful assitant. Question: {problem}. "
        f"Give me the answer directly. "
    )


def _standard_prompt_qwen(problem: str) -> Tuple[str, str]:
    return "", (
        f"Answer the following question with only the final answer. "
        f"Do not show any working or explanation.\n\n"
        f"Question: {problem}\n\n"
        f"Final answer:"
    )


def _standard_prompt_gemma(problem: str) -> Tuple[str, str]:
    return "", (
        f"Answer the following question directly.\n\n"
        # f"Reply in the form 'Final answer: <answer>'.\n\n"
        f"Question: {problem}"
    )


STANDARD_PROMPT_BUILDERS: Dict[str, Any] = {
    "llama": _standard_prompt_default,
    "qwen": _standard_prompt_qwen,
    "gemma": _standard_prompt_gemma,
}
STANDARD_PROMPT_DEFAULT = _standard_prompt_default


# -----------------------------------------------------------------------------
# The `cot` prompt : For the CoT ceiling. 
# -----------------------------------------------------------------------------


def _cot_prompt_llama(problem: str) -> Tuple[str, str]:
    return "", (
        "Solve the following math problem efficiently and clearly:\n\n"
        "- For simple problems (2 steps or fewer):\n"
        "Provide a concise solution with minimal explanation.\n\n"
        "- For complex problems (3 steps or more):\n"
        "Use this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "...\n\n"
        "Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem.\n\n"
        f"Problem: {problem}"
    )


def _cot_prompt_qwen(problem: str) -> Tuple[str, str]:
    return "", (
        f"{problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def _cot_prompt_gemma(problem: str) -> Tuple[str, str]:
    return _cot_prompt_qwen(problem)


COT_PROMPT_BUILDERS: Dict[str, Any] = {
    "llama": _cot_prompt_llama,
    "qwen": _cot_prompt_qwen,
    "gemma": _cot_prompt_gemma,
}


# -----------------------------------------------------------------------------
# Per-(family, dataset) prompt overrides
# -----------------------------------------------------------------------------
# The narrowest prompt knob: one builder for one family on one dataset,
# resolved BEFORE the family fallback chain. Exists because a family's
# standard prompt can be calibrated right on one dataset and wrong on
# another, and re-tuning the family string would invalidate every finished
# run.
#
# WARNING: registering an override obsoletes every existing collection of
# that (family, dataset) pair -- re-collect from an empty dir, never resume.


def _gemma_standard_math(problem: str) -> Tuple[str, str]:
    # gemma-standard-math (2026-09-01). The family baseline carries no
    # prohibition and on MATH gemma-4 reasons on every row, so there is no
    # terse mode for the com to contrast against; this override tightens the
    # prompt ON MATH ONLY (every other dataset keeps the family baseline).
    # Ladder (qwencalib smokes, 100 rows):
    #   m1 "directly without explanation" -- worked somewhat (user verdict).
    #   m2 (PROMOTED 2026-09-01) "with only the final answer" -- swaps the
    #      prohibition for a positive ask; the gemma MATH standard prompt.
    # Empty system for the usual gemma reason.
    return "", (
        f"Answer the following question with only the final answer.\n\n"
        f"Question: {problem}"
    )


# (family, dataset) -> builder. Missing key = no override, family chain rules.
DATASET_PROMPT_BUILDERS: Dict[Tuple[str, str], Any] = {
    ("gemma", "math"): _gemma_standard_math,
    # math500 is MATH test rows; the eval must run under the prompt the
    # gemmacalib fit was collected with.
    ("gemma", "math500"): _gemma_standard_math,
}


# -----------------------------------------------------------------------------
# Thinking mode
# -----------------------------------------------------------------------------
# Hybrid reasoning models (Qwen3) open EVERY generation with a `<think>` block.
# The block's contents are controllable -- Qwen3's `/no_think` soft switch
# empties it -- but the opening `<think>` token is emitted either way, and that
# breaks the elicitation target at its root:
#
#   The target is p(first token | cot prompt) - p(first token | qa prompt). On
#   Qwen3-4B/GSM8K both distributions are a point mass on '<think>' (measured:
#   1.000 and 1.000, total variation 0.000), because the model has no freedom at
#   that position. The whole prompt effect has moved one token later, to the
#   first token after `</think>`.
#
# Qwen3's chat template takes `enable_thinking=False`, which appends an EMPTY
# `<think></think>` pair to the generation prompt instead of leaving the model
# to open one. The next token the model produces is then its first content
# token, which is the position the target was always defined at, and generations
# stop carrying a `<think>\n\n</think>` prefix that inflates every word count.
#
# Applied to EVERY mode, deliberately. Enabling thinking for `cot` and disabling
# it for `qa` would give a large total variation instantly, but the target would
# then be "promote `<think>`" -- a template artifact -- and the head ranking
# would be selecting for one special token rather than for reasoning. It would
# also be steering a switch the template already exposes for free.
#
# Set THINKING = True to let such a model think (and expect the first-token
# target to go to zero on any model that gates reasoning behind a block).
THINKING = False


def chat_template_kwargs(tokenizer: Any) -> Dict[str, Any]:
    """Extra apply_chat_template kwargs for this tokenizer's template.

    Keyed on what the template actually references rather than on the model
    family, so it is provably a no-op everywhere the option does not exist: a
    template with no `enable_thinking` in its source gets an empty dict and
    renders byte-for-byte as before (verified against llama-3.1, whose
    stored collection prompts are unchanged by this).
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" in template:
        return {"enable_thinking": THINKING}
    return {}


def make_chat_prompt(
    tokenizer: Any,
    problem: str,
    mode: str = "standard",
    use_chat_template: bool = True,
    dataset: str = "",
) -> str:
    """Build THE prompt for a problem, PER MODEL FAMILY.

    There is one BASELINE prompt per family (STANDARD_PROMPT_BUILDERS,
    resolved from the tokenizer's model id). `mode` selects between it
    ("standard") and the
    vendor CoT reference prompt ("cot", COT_PROMPT_BUILDERS -- the ceiling
    collected as test_cot, never fitted on). Every family's baseline shares
    what defines it: no
    \\boxed{} instruction, so the answer is graded through
    extract_final_answer's fallback (marker phrase, then last number), and a
    single user turn.

    `dataset` (the collection's dataset slug, e.g. "math") is checked FIRST:
    a (family, dataset) entry in DATASET_PROMPT_BUILDERS overrides the whole
    family chain -- currently gemma/math and gemma/math500. Omitting it (or
    any pair with no entry) changes nothing, so every existing caller and
    collection stays byte-identical.

    Uses the tokenizer's chat template when available (and requested);
    otherwise falls back to a plain instruction-style prompt. A family whose
    builder returns an empty `system` is sent as a lone user turn, not as an
    empty system turn, since an empty system message is not the same prompt as
    no system message.
    """
    if mode not in PROMPT_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r} (the prompt-mode grid was removed on "
            f"2026-08-20; modes are {PROMPT_MODES} -- see "
            f"STANDARD_PROMPT_BUILDERS)")
    family = model_family(tokenizer)
    if mode == "cot":
        # The vendor's CoT eval prompt, as published: no dataset override
        # (see COT_PROMPT_BUILDERS).
        builder = COT_PROMPT_BUILDERS.get(family)
        if builder is None:
            raise ValueError(
                f"cot is registered only for {sorted(COT_PROMPT_BUILDERS)} "
                f"(got family {family!r}); add the family's vendor CoT prompt "
                f"to COT_PROMPT_BUILDERS")
    else:
        builder = DATASET_PROMPT_BUILDERS.get((family, dataset or ""))
        if builder is None:
            builder = STANDARD_PROMPT_BUILDERS.get(family) or STANDARD_PROMPT_DEFAULT
    system, user = builder(problem)

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        extra = chat_template_kwargs(tokenizer)
        try:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": user},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra
            )
        except Exception:
            # Some templates reject the system role outright (gemma:
            # "System role not supported"). Prepend the instructions to the
            # user turn instead so the prompt text is equivalent.
            messages = [{"role": "user", "content": f"{system}\n\n{user}" if system else user}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra
            )
    return f"{system}\n\n{user}\n\nAnswer:" if system else user


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def get_gold_answer(example: Dict[str, Any]) -> str:
    """Return the gold answer: explicit `answer` field if present, else the
    boxed (or otherwise extracted) answer from the gold solution text."""
    if example.get("answer") is not None:
        return str(example["answer"])
    return extract_final_answer(example.get("solution", ""))


def _problem_key(text: Any) -> str:
    """Whitespace-normalized problem text, the join key between MATH and
    MATH-500. MATH-500 was carved out of MATH test verbatim, so exact text
    (modulo whitespace) matches all 500 rows; MATH test carries no row id the
    two datasets share."""
    return re.sub(r"\s+", " ", str(text)).strip()


def math500_problem_keys() -> Dict[str, set]:
    """MATH-500's problems as {math_subject: {problem_key, ...}}.

    Subjects are slugified back onto MATH_SUBJECTS, the same fixup
    load_math500_examples applies, so the keys index the MATH subject a row
    actually lives under.
    """
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                      cache_dir=paths.require("MATH500_ROOT"))
    fixups = {"counting_probability": "counting_and_probability"}
    by_subject: Dict[str, set] = {}
    for ex in ds:
        slug = _subject_slug(ex.get("subject", ""))
        subject = fixups.get(slug, slug)
        by_subject.setdefault(subject, set()).add(_problem_key(ex["problem"]))
    return by_subject


def load_math_examples(
    subjects: Sequence[str],
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and shuffle MATH examples from EleutherAI/hendrycks_math.

    `subjects` may be ["all"] to use every subject. Each returned dict has:
    idx, subject, problem, gold_solution, gold_answer.

    Splits: MATH's own `train` (7500) and `test` (5000), plus a derived
    `calibration` split -- MATH test MINUS every MATH-500 row, 4500 rows,
    matched on problem text. MATH-500 is the steering evaluation set, so
    anything fit or tuned on MATH test leaks into it; `calibration` is the
    part of test that provably does not, and is what threshold/gate fitting
    should use. The exclusion is recomputed from the two datasets on every
    load rather than read from a frozen list, so it cannot go stale, and the
    per-subject drop count is ASSERTED against MATH-500's own subject
    histogram -- an under-match would silently leak eval rows into the
    calibration pool, which is the one failure this split exists to prevent.

    `idx` is the row's position in MATH's OWN split enumeration, so a
    calibration row keeps the idx it has in `test` (the kept indices are
    sparse). That makes (subject, idx) traceable straight back to MATH test
    instead of naming a different row in each split.

    `calib` (2026-08-28) is the FITTING POOL for every model on MATH: a
    fixed 1,600-row subset of `train` (800 correct + 800 incorrect under
    Llama-3.1-8B single-pass labels, each class restricted to rows whose
    first generated token is more likely under that class than the other,
    stratified by subject), listed by (subject, idx) in the frozen
    data/math_<split>.json manifest. The same rows are collected for
    every model, so fits are compared on identical problems; `idx` is the
    row's position in MATH `train`. Band and dose are tuned on `calibration`
    rows (MATH test minus MATH-500), never on `calib`.
    """
    if list(subjects) == ["all"]:
        subjects = MATH_SUBJECTS

    is_calib = split in CALIB_SPLITS
    hf_split = "train" if is_calib else {"calibration": "test"}.get(split, split)
    excluded = math500_problem_keys() if split == "calibration" else {}
    keep = calib_keys("math", split) if is_calib else {}

    examples: List[Dict[str, Any]] = []
    for subject in subjects:
        cache_dir = Path(paths.require("HENDRYCKS_MATH_ROOT")) / subject
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(
            "EleutherAI/hendrycks_math", subject, split=hf_split,
            cache_dir=str(cache_dir)
        )
        drop_keys = excluded.get(subject, set())
        dropped = 0
        keep_idx = keep.get(subject, set())
        for i, ex in enumerate(ds):
            if drop_keys and _problem_key(ex["problem"]) in drop_keys:
                dropped += 1
                continue
            if is_calib and i not in keep_idx:
                continue
            examples.append(
                {
                    "idx": i,
                    "subject": subject,
                    "problem": ex["problem"],
                    "gold_solution": ex.get("solution", ""),
                    "gold_answer": get_gold_answer(ex),
                }
            )
        if split == "calibration":
            if dropped != len(drop_keys):
                raise RuntimeError(
                    f"[math] calibration split for {subject!r}: dropped "
                    f"{dropped} rows but MATH-500 holds {len(drop_keys)} of "
                    "that subject -- the two datasets no longer agree on "
                    "problem text, so disjointness from MATH-500 is NOT "
                    "guaranteed. Re-check the MATH / MATH-500 downloads "
                    "before using this split.")
            print(f"[math] calibration {subject}: {len(ds) - dropped} of "
                  f"{len(ds)} test rows kept ({dropped} MATH-500 rows excluded)")
        if is_calib:
            got = sum(1 for e in examples if e["subject"] == subject)
            if got != len(keep_idx):
                raise RuntimeError(
                    f"[math] {split} split for {subject!r}: manifest names "
                    f"{len(keep_idx)} train rows but {got} were found -- "
                    f"{calib_manifest('math', split)} does not match this MATH download.")
            print(f"[math] {split} {subject}: {got} of {len(ds)} train rows "
                  f"(fixed manifest {os.path.basename(calib_manifest('math', split))})")

    random.Random(seed).shuffle(examples)
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    return examples


def gsm8k_gold_answer(answer: str) -> str:
    """Gold answer from a GSM8K solution: the value after the '####' marker.

    GSM8K stores the worked solution and the final answer in one `answer`
    field, terminated by "#### 18". Taking that marker is exact; falling back
    to extract_final_answer (last number) would silently pick a calculator
    annotation from the middle of the trace on a malformed row.
    """
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    return extract_final_answer(answer)


def load_gsm8k_examples(
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and shuffle GSM8K (openai/gsm8k, 'main') examples.

    Same dict shape as load_math_examples — idx, subject, problem,
    gold_solution, gold_answer — so every downstream stage is unchanged.
    `subject` is the constant "gsm8k" and `idx` is the row's position in the
    split, which is what makes the (subject, idx) key unique and every
    positional join on it exact.

    `llamacalib` / `qwencalib` are the fixed fitting pools cut from `train`
    (CALIB_SPLITS, frozen manifests under data/): the same rows for every
    model, each keeping its `train` idx.
    """
    cache_dir = Path(paths.require("GSM8K_ROOT"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    is_calib = split in CALIB_SPLITS
    hf_split = "train" if is_calib else split
    ds = load_dataset("openai/gsm8k", "main", split=hf_split, cache_dir=str(cache_dir))

    keep_idx: Optional[set] = None
    if is_calib:
        keep_idx = calib_keys("gsm8k", split).get(GSM8K_SUBJECTS[0], set())

    examples: List[Dict[str, Any]] = [
        {
            "idx": i,
            "subject": GSM8K_SUBJECTS[0],
            "problem": ex["question"],
            "gold_solution": ex.get("answer", ""),
            "gold_answer": gsm8k_gold_answer(ex.get("answer", "")),
        }
        for i, ex in enumerate(ds)
        if keep_idx is None or i in keep_idx
    ]
    if keep_idx is not None:
        if len(examples) != len(keep_idx):
            raise RuntimeError(
                f"[gsm8k] {split} split: manifest names {len(keep_idx)} train rows "
                f"but {len(examples)} were found -- {calib_manifest('gsm8k', split)} "
                "does not match this GSM8K download.")
        print(f"[gsm8k] {split}: {len(examples)} of {len(ds)} train rows "
              f"(fixed manifest {os.path.basename(calib_manifest('gsm8k', split))})")
    random.Random(seed).shuffle(examples)
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    return examples


def _subject_slug(name: str) -> str:
    """Lowercase [a-z0-9_] slug of a dataset's own category string."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _finish_examples(
    examples: List[Dict[str, Any]],
    subjects: Sequence[str],
    legal: Sequence[str],
    dataset: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Shared tail of the eval-only loaders: validate the subject request,
    filter, shuffle, cap. `examples` must already carry a `subject` key drawn
    from `legal`."""
    wanted = list(legal) if list(subjects) == ["all"] else list(subjects)
    unknown = [s for s in wanted if s not in legal]
    if unknown:
        raise ValueError(
            f"{dataset} has no subject(s) {unknown}; legal: {sorted(legal)}")
    observed = {e["subject"] for e in examples}
    stray = observed - set(legal)
    if stray:
        raise ValueError(
            f"{dataset} rows carry unlisted subject(s) {sorted(stray)}; update "
            f"the subject list in common.py to match the download")
    examples = [e for e in examples if e["subject"] in wanted]
    random.Random(seed).shuffle(examples)
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    return examples


def load_gsm_plus_examples(
    subjects: Sequence[str],
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and shuffle GSM-Plus (qintongli/GSM-Plus) examples.

    Splits: `test` (10552) and `testmini` (2400, a subset of test). There is no
    train split -- GSM-Plus is an evaluation-only perturbation of GSM8K's test
    questions, so a "train" request is an error rather than an alias: any
    aliasing target would overlap test. `subject` is the perturbation type;
    `idx` is the row's position in the FULL split, so keys stay stable no
    matter what is excluded.

    Every row is stamped `use` (0 for GSM_PLUS_EXCLUDED_SUBJECTS, 1 otherwise)
    and the use=0 rows are dropped here, before anything can consume them --
    see the constant for why critical_thinking is out.
    """
    if split not in ("test", "testmini"):
        raise ValueError(
            f"gsm_plus has no {split!r} split (eval-only: test | testmini)")
    ds = load_dataset("qintongli/GSM-Plus", split=split,
                      cache_dir=paths.require("GSM_PLUS_ROOT"))
    examples = []
    dropped = 0
    for i, ex in enumerate(ds):
        subject = _subject_slug(ex.get("perturbation_type", ""))
        use = int(subject not in GSM_PLUS_EXCLUDED_SUBJECTS)
        if not use:
            dropped += 1
            continue
        examples.append(
            {
                "idx": i,
                "subject": subject,
                "problem": ex["question"],
                "gold_solution": ex.get("solution") or "",
                "gold_answer": str(ex.get("answer", "")).strip(),
                "use": use,
            }
        )
    if dropped:
        print(f"[gsm_plus] dropped {dropped} row(s) of excluded subject(s) "
              f"{sorted(GSM_PLUS_EXCLUDED_SUBJECTS)} (unanswerable by design)")
    return _finish_examples(examples, subjects, GSM_PLUS_SUBJECTS, "gsm_plus",
                            max_examples, seed)


def load_math500_examples(
    subjects: Sequence[str],
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and shuffle MATH-500 (HuggingFaceH4/MATH-500) examples.

    `test` is the 500-row sample of MATH's test split; subjects are MATH's
    own, slugified back onto MATH_SUBJECTS. `answer` is shipped explicitly, so
    grading does not depend on re-extracting the boxed value.

    `train` is served from MATH's OWN train split (7.5K rows), which is
    disjoint from the 500 test rows by construction (they were sampled from
    MATH *test*). That makes math500 a full train/test pair: fit and calibrate
    on MATH train, evaluate on the untouched 500 -- the SEAL protocol -- in
    one run tree.
    """
    if split == "train":
        print("[math500] split 'train' -> MATH train "
              "(disjoint from the 500 test rows, which come from MATH test)")
        return load_math_examples(subjects, "train", max_examples, seed)
    if split != "test":
        raise ValueError(f"math500 has no {split!r} split (train | test)")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                      cache_dir=paths.require("MATH500_ROOT"))
    fixups = {"counting_probability": "counting_and_probability"}
    examples = []
    for i, ex in enumerate(ds):
        slug = _subject_slug(ex.get("subject", ""))
        examples.append(
            {
                "idx": i,
                "subject": fixups.get(slug, slug),
                "problem": ex["problem"],
                "gold_solution": ex.get("solution") or "",
                "gold_answer": str(ex.get("answer", "")).strip(),
            }
        )
    return _finish_examples(examples, subjects, MATH_SUBJECTS, "math500",
                            max_examples, seed)


def load_svamp_examples(
    subjects: Sequence[str],
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and shuffle SVAMP (ChilleD/SVAMP) examples.

    Splits: `train` (700) and `test` (300), disjoint by question text
    (verified at wiring time: zero overlap), so SVAMP is a full train/test
    pair and the default SPLITS="train test" pipeline applies. `subject` is
    the arithmetic operation the dataset labels each row with (`Type`);
    `idx` is the row's position in the split.

    `problem` is the dataset's own `question_concat` (Body + " " + Question);
    the Body often lacks terminal punctuation, which is how SVAMP is shipped
    and evaluated elsewhere, so it is left as is. `gold_solution` is the
    arithmetic `Equation`; `gold_answer` is an integer stored as a string
    (no decimals or fractions in either split), graded via the numeric path.
    """
    if split not in ("train", "test"):
        raise ValueError(f"svamp has no {split!r} split (train | test)")
    ds = load_dataset("ChilleD/SVAMP", split=split, cache_dir=paths.require("SVAMP_ROOT"))
    fixups = {"common_divison": "common_division"}  # one misspelled train row
    examples = []
    for i, ex in enumerate(ds):
        slug = _subject_slug(ex.get("Type", ""))
        problem = (ex.get("question_concat") or "").strip() or \
            f"{str(ex.get('Body', '')).strip()} {str(ex.get('Question', '')).strip()}"
        examples.append(
            {
                "idx": i,
                "subject": fixups.get(slug, slug),
                "problem": problem,
                "gold_solution": ex.get("Equation") or "",
                "gold_answer": str(ex.get("Answer", "")).strip(),
            }
        )
    return _finish_examples(examples, subjects, SVAMP_SUBJECTS, "svamp",
                            max_examples, seed)


# HARP's six subjects are MATH's own taxonomy minus intermediate_algebra.
HARP_SUBJECTS = ["algebra", "counting_and_probability", "geometry",
                 "number_theory", "prealgebra", "precalculus"]


def load_harp_examples(
    subjects: Sequence[str],
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load and shuffle HARP (github.com/aadityasingh/HARP) examples.

    The default 4,780-problem short-answer split: US competition problems
    (AMC/AIME/..., 1950-2024), difficulty levels 1-6, subjects on MATH's own
    taxonomy. Answers ship as `$...$`-wrapped LaTeX; the gold is the content
    with the wrapping dollars stripped, so it grades through the same
    answers_equal route as MATH, with the raw form kept as an alias.

    Eval-only (`test`): the steering protocol calibrates on MATH train and
    transfers here, so a train request is an error, not an alias.
    """
    if split != "test":
        raise ValueError(f"harp has no {split!r} split (test only; "
                         "calibrate on MATH train and transfer)")
    harp_jsonl = paths.require("HARP_JSONL")
    if not os.path.exists(harp_jsonl):
        raise FileNotFoundError(
            f"{harp_jsonl} not found -- download HARP.jsonl.zip from "
            "github.com/aadityasingh/HARP and unzip it there")
    examples = []
    with open(harp_jsonl) as fh:
        for i, line in enumerate(fh):
            ex = json.loads(line)
            raw = str(ex["answer"]).strip()
            gold = raw
            if gold.startswith("$") and gold.endswith("$") and len(gold) > 1:
                gold = gold[1:-1].strip()
            examples.append(
                {
                    "idx": i,
                    "subject": str(ex["subject"]),
                    "problem": str(ex["problem"]),
                    "gold_solution": ex.get("solution_1") or "",
                    "gold_answer": gold,
                    "gold_aliases": [raw] if raw != gold else [],
                }
            )
    return _finish_examples(examples, subjects, HARP_SUBJECTS, "harp",
                            max_examples, seed)


def dataset_subjects(dataset: str) -> List[str]:
    """The subject vocabulary of a dataset (what --subject may name)."""
    if dataset == "math":
        return list(MATH_SUBJECTS)
    if dataset == "gsm8k":
        return list(GSM8K_SUBJECTS)
    if dataset == "gsm_plus":
        return list(GSM_PLUS_SUBJECTS)
    if dataset == "math500":
        return list(MATH_SUBJECTS)
    if dataset == "harp":
        return list(HARP_SUBJECTS)
    if dataset == "svamp":
        return list(SVAMP_SUBJECTS)
    raise ValueError(f"Unknown dataset: {dataset!r} (expected one of {DATASETS})")


def load_examples(
    dataset: str,
    subjects: Sequence[str],
    split: str,
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Load examples from `dataset`, dispatching to its loader.

    The one entry point collection uses, so adding a dataset means adding a
    loader here rather than touching the collector. GSM8K has a single
    category and accepts ["all"] or its own pseudo-subject; every other
    dataset takes its subject vocabulary from dataset_subjects.
    """
    if split == "calibration" and dataset != "math":
        raise ValueError(
            f"{dataset!r} has no 'calibration' split -- it is MATH-only "
            "(MATH test minus the 500 MATH-500 rows). Use --dataset math.")
    if split in CALIB_SPLITS:
        calib_manifest(dataset, split)      # raises for a dataset without one
    if dataset == "math":
        return load_math_examples(subjects, split, max_examples, seed)
    if dataset == "gsm8k":
        wanted = list(subjects)
        if wanted not in (["all"], GSM8K_SUBJECTS):
            raise ValueError(
                f"gsm8k has no subjects; --subject must be 'all' or 'gsm8k', got {wanted}")
        return load_gsm8k_examples(split, max_examples, seed)
    if dataset == "gsm_plus":
        return load_gsm_plus_examples(subjects, split, max_examples, seed)
    if dataset == "math500":
        return load_math500_examples(subjects, split, max_examples, seed)
    if dataset == "harp":
        return load_harp_examples(subjects, split, max_examples, seed)
    if dataset == "svamp":
        return load_svamp_examples(subjects, split, max_examples, seed)
    raise ValueError(f"Unknown dataset: {dataset!r} (expected one of {DATASETS})")


# -----------------------------------------------------------------------------
# Hidden states and generation
# -----------------------------------------------------------------------------


def _tokenize(tokenizer: Any, model: Any, prompt: str, max_prompt_tokens: Optional[int]) -> Dict[str, torch.Tensor]:
    """Tokenize one prompt (with optional truncation) and move it to the model device."""
    kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if max_prompt_tokens is not None:
        kwargs.update(truncation=True, max_length=max_prompt_tokens)
    inputs = tokenizer(prompt, **kwargs)
    return {k: v.to(model.device) for k, v in inputs.items()}


def _tokenize_batch(
    tokenizer: Any, model: Any, prompts: Sequence[str], max_prompt_tokens: Optional[int]
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of prompts with LEFT padding and move them to the model device.

    Left padding is what makes batching safe for everything downstream here: the
    final position of every row is that row's real last prompt token, so
    `[:, -1]` means the same thing it meant at batch size 1. That is the position
    HeadCatcher captures, the one extract_prompt_hidden_states reads, and the one
    steering_driver's Steerer adds its shift to. load_model_and_tokenizer already
    sets padding_side="left"; assert it rather than trusting the caller.
    """
    assert tokenizer.padding_side == "left", (
        f"batching needs left padding, got padding_side={tokenizer.padding_side!r}"
    )
    kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
    if max_prompt_tokens is not None:
        kwargs.update(truncation=True, max_length=max_prompt_tokens)
    inputs = tokenizer(list(prompts), **kwargs)
    return {k: v.to(model.device) for k, v in inputs.items()}


def decoder_layers(model: Any) -> Any:
    """The text decoder's ModuleList of layers, for any supported architecture.

    Dense causal LMs (Llama, Qwen3) hold it at `model.model.layers`,
    which every hook site here used directly. A multimodal wrapper does not:
    gemma-4 loads as a ForConditionalGeneration wrapper whose `model` holds
    `vision_tower` / `multi_modal_projector` / `language_model`, so the
    decoder is one level deeper and the flat path raises AttributeError
    before a single activation is captured.

    Resolving it in one place keeps HeadCatcher, the head collector and the
    steering hooks pointed at the same modules — the probes and the intervention
    must index the SAME layers or the steering vectors land on the wrong ones.
    """
    for path in (("model", "layers"),
                 ("model", "language_model", "layers"),
                 ("language_model", "model", "layers"),
                 ("transformer", "h")):
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None:
            return node
    raise AttributeError(
        f"cannot locate the decoder layers of {type(model).__name__}; add its "
        f"path to common.decoder_layers")


def num_attention_heads(model: Any) -> int:
    """The text decoder's head count, for any supported architecture.

    The module-path counterpart of decoder_layers, and broken by the same thing:
    on a multimodal wrapper (gemma-4) the top-level config describes the
    WRAPPER and has no `num_attention_heads` at all — it sits on
    `config.text_config`, alongside the vision tower's own separate head count.
    Reading the flat attribute returns None there and callers cannot tell "no
    such field" from "zero heads".

    Order matters: get_text_config() is what transformers itself uses to
    disambiguate, so it is tried first, and the flat attribute last. Never infer
    the count by dividing the o_proj input width, which is num_heads * head_dim
    and NOT hidden_size on models where those differ (gemma-4's wide
    full-attention layers are the case in point), so division would silently
    give a wrong split into things that are not attention heads.
    """
    cfg = getattr(model, "config", None)
    getter = getattr(cfg, "get_text_config", None)
    candidates = [getter() if callable(getter) else None,
                  getattr(cfg, "text_config", None),
                  cfg]
    for c in candidates:
        n = getattr(c, "num_attention_heads", None)
        if n:
            return int(n)
    raise AttributeError(
        f"cannot determine num_attention_heads for {type(model).__name__} "
        f"(config {type(cfg).__name__}); add its path to "
        f"common.num_attention_heads")


def text_config(cfg: Any) -> Any:
    """The text decoder's config, for flat and multimodal-wrapper configs alike.

    On a wrapper (gemma-4) the flat config describes the WRAPPER and has no
    num_hidden_layers / num_attention_heads / head_dim at all; transformers'
    own get_text_config() is the disambiguation it uses internally.
    """
    getter = getattr(cfg, "get_text_config", None)
    inner = getter() if callable(getter) else getattr(cfg, "text_config", None)
    return inner if inner is not None else cfg


def head_grid_num_heads(cfg: Any, stored_width: int) -> int:
    """How many grid slots a stored head row of `stored_width` splits into.

    The head grid every fit/steering artifact is indexed by is
    (num_layers, N, D) with D = the CONFIG's head_dim and N = stored_width
    // D. On a homogeneous model that is exactly (num_attention_heads,
    head_dim) -- byte-identical to the old `cfg.num_attention_heads` read.
    On a width-heterogeneous model (gemma-4-12B: config head_dim 256, but 8
    full-attention layers run 512-dim heads, so HeadCatcher pads everything
    to 8192) it is a PSEUDO-HEAD grid: 32 slots of 256, where a wide layer's
    real 512-dim head occupies two slots and a narrow layer's pad columns
    are zero slots that can never be selected (zero activations -> zero com,
    sigma and score, and head_ranker masks them from every band).
    A slot is still a fixed slice of one layer's o_proj input, so Eq. 2 --
    inject sigma * theta into that slice through W_O -- is unchanged; only
    the granularity of "head" is finer on the wide layers.

    Falls back to num_attention_heads when the config carries no head_dim
    (then stored_width must be uniform, as it always was pre-gemma-4).
    """
    t = text_config(cfg)
    # transformers 5.x marks head_dim per-layer on a heterogeneous config and
    # RAISES on the global read (even through getattr-with-default, since the
    # refusal lives in __getattribute__). The per-layer configs are the honest
    # source anyway: the grid dim is the SMALLEST head_dim any layer runs, so
    # every real head is a whole number of grid slots. On a homogeneous config
    # per_layer_config is absent and the plain head_dim read works as before.
    head_dim = None
    per_layer = getattr(t, "per_layer_config", None)
    if per_layer:
        dims = set()
        for lc in (per_layer.values() if hasattr(per_layer, "values") else per_layer):
            d = getattr(lc, "head_dim", None)
            if d:
                dims.add(int(d))
        head_dim = min(dims) if dims else None
    if head_dim is None:
        try:
            head_dim = getattr(t, "head_dim", None)
        except Exception:
            head_dim = None
    if head_dim:
        n, rem = divmod(int(stored_width), int(head_dim))
        if rem:
            raise ValueError(
                f"stored head width {stored_width} is not a multiple of the "
                f"config head_dim {head_dim}; the collection and the model "
                f"disagree about the o_proj layout")
        return n
    return int(t.num_attention_heads)


class HeadCatcher:
    """Captures each layer's per-head attention output for the LAST token of the
    current forward pass: the input to `self_attn.o_proj`, which is the
    concatenated `(num_heads, head_dim)` block before the output projection.

    This is the tensor ITI intervenes on (honest_llama/interveners.py:19-24) and
    the one the head probes use. It is head-separable and GQA-safe, because the o_proj
    input is already post-KV-repeat.

    Attach it around a prompt-only forward pass and call collect() immediately
    after, before any generation overwrites the captured states.

    Batch-aware: the capture is `[:, -1]`, so with LEFT padding (which
    _tokenize_batch asserts) row b of the batch yields row b of collect(). At
    batch size 1 this is exactly the old `[0, -1]` capture.
    """

    def __init__(self, model: Any):
        self.layers = decoder_layers(model)
        self.states: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(i))
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, args):
            self.states[layer_idx] = args[0][:, -1].detach().to(torch.float16).cpu()
            return None

        return hook

    def collect(self) -> np.ndarray:
        """(batch, num_layers, max_width) float16 for the last pass.

        HETEROGENEOUS WIDTHS (gemma-4, 2026-08-31): not every layer's o_proj
        input has the same width -- gemma-4-12B's 8 full-attention layers
        (every 6th) run head_dim 512 (16 x 512 = 8192) while its 40 sliding
        layers run 256 (16 x 256 = 4096). Narrow layers are ZERO-PADDED on
        the right to the widest layer's width so the stack stays rectangular
        and every existing on-disk collection keeps its exact layout (on a
        homogeneous model the pad is zero columns and this is byte-identical
        to the old stack). Downstream, the grid is indexed in PSEUDO-HEADS of
        the model's config head_dim (head_grid_num_heads): a wide layer's
        512-dim head is two 256-dim pseudo-heads, and a narrow layer's pad
        columns are pseudo-heads whose activations are identically zero --
        their com, sigma and elicit score are all zero, and
        head_ranker masks them out of every band selection.
        """
        assert all(s is not None for s in self.states), "hook missed a layer"
        width = max(s.shape[-1] for s in self.states)
        states = [
            s if s.shape[-1] == width else
            torch.nn.functional.pad(s, (0, width - s.shape[-1]))
            for s in self.states
        ]
        out = torch.stack(states, dim=1).numpy()  # (B, L, max_width) float16
        self.states = [None] * len(self.layers)
        return out

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def extract_prompt_hidden_states(
    tokenizer: Any,
    model: Any,
    prompt: str,
    representation: str = "last_token",
    max_prompt_tokens: Optional[int] = None,
) -> np.ndarray:
    """Run the prompt through the model (no generation) and return per-layer
    representations, shape [num_layers + 1, hidden_dim].

    Index 0 is the embedding layer output. `representation` is "last_token"
    (final prompt token) or "mean" (mean over prompt tokens).
    """
    return extract_prompt_hidden_states_batch(
        tokenizer, model, [prompt], representation, max_prompt_tokens
    )[0]


@torch.no_grad()
def extract_prompt_hidden_states_batch(
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    representation: str = "last_token",
    max_prompt_tokens: Optional[int] = None,
) -> np.ndarray:
    """Batched extract_prompt_hidden_states: [batch, num_layers + 1, hidden_dim].

    One forward pass for the whole batch. Any HeadCatcher attached to the model
    fills from this same pass, so per-head activations still ride along free.
    """
    inputs = _tokenize_batch(tokenizer, model, prompts, max_prompt_tokens)
    # A bare forward defaults position_ids to arange(seq_len), which under LEFT
    # padding hands every real token of a short row the wrong RoPE position and
    # silently changes its activations. model.generate() derives these from the
    # mask for us; a plain forward does not, so do it here.
    mask = inputs["attention_mask"]
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 1)
    outputs = model(**inputs, position_ids=position_ids, output_hidden_states=True,
                    return_dict=True, use_cache=False)

    reps = []
    for hs in outputs.hidden_states:  # each: [B, seq, dim]
        if representation == "last_token":
            # left padding => index -1 is every row's real last prompt token
            vec = hs[:, -1, :]
        elif representation == "mean":
            # mask the pads out rather than averaging them in
            m = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)
            vec = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown representation: {representation}")
        reps.append(vec.detach().float().cpu().numpy())
    return np.stack(reps, axis=1)  # (B, L+1, D)


def stop_token_ids(tokenizer: Any, model: Any) -> List[int]:
    """Every id that ends a turn for this model, not just `tokenizer.eos_token`.

    Passing the tokenizer's single eos id to `generate` OVERRIDES the model's own
    stop set, and on some chat models that set is strictly larger:

        Llama-3.1   tok.eos = <|eot_id|>  gen = [<|end_of_text|>, <|eom_id|>, <|eot_id|>]
        Gemma       tok.eos = <eos>       gen = [<eos>, <end_of_turn>]

    Gemma is the case that breaks: it ends every turn with <end_of_turn> and
    never emits <eos>, so with the tokenizer id alone generation does not stop.
    It runs the FULL max_new_tokens budget emitting <end_of_turn> over and over
    (invisible afterwards, because decoding skips special tokens), which both
    costs ~200x the decode steps a finished answer needs and marks every row
    `truncated` -- and truncated rows are dropped, so the whole collection
    empties out. Llama is unaffected in practice: its turn-end
    token is already the tokenizer's eos, and the ids this adds for Llama are
    ones an instruct turn does not emit before <|eot_id|>.
    """
    ids = []
    for source in (getattr(tokenizer, "eos_token_id", None),
                   getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if source is None:
            continue
        ids.extend([source] if isinstance(source, int) else list(source))
    # dict.fromkeys, not set(): the first id stays first, so anything reading
    # ids[0] as "the" eos still sees the tokenizer's.
    return list(dict.fromkeys(int(i) for i in ids))


@torch.no_grad()
def generate_one(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    return_truncated: bool = False,
):
    """Generate a completion for one prompt and return only the new text.

    temperature <= 0 means greedy decoding (top_p ignored). `seed` reseeds
    torch before sampling so individual samples are reproducible.

    When `return_truncated` is True, returns (text, truncated) where truncated
    is True iff generation stopped by hitting max_new_tokens without emitting
    any of the model's stop tokens (see stop_token_ids) -- i.e. the reasoning
    trace was cut off, not finished.
    """
    out = generate_batch(
        tokenizer, model, [prompt], max_new_tokens, temperature, top_p, seed,
        max_prompt_tokens,
    )
    text, truncated = out[0]
    return (text, truncated) if return_truncated else text


@torch.no_grad()
def generate_batch(
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
) -> List[Tuple[str, bool]]:
    """Generate for a list of prompts in ONE batched pass: [(text, truncated)].

    Same semantics as generate_one per row, at a fraction of the wall clock:
    decoding an 8B model one sequence at a time is memory-bandwidth bound, so a
    batch costs barely more per step than a single sequence does.

    Two things make the per-row results well defined:

    * LEFT padding (asserted in _tokenize_batch) puts every row's prompt flush
      against the generated region, so `out[:, input_len:]` is exactly the new
      tokens for every row, with no per-row offset bookkeeping.
    * `generate` runs until EVERY row has emitted EOS or the budget is spent,
      padding finished rows meanwhile. So a row that never emits EOS is one the
      budget cut off -- the same definition of `truncated` generate_one uses,
      and it does not depend on what the other rows in the batch did.

    Caveat worth knowing: batched and unbatched greedy decoding are
    mathematically identical but not bitwise identical, because a different
    batch size selects different reduction kernels. On a long greedy generation
    a sub-ulp logit difference can flip one argmax and diverge from there. Rows
    are not reproducible across batch sizes; use batch_size=1 where bit-identity
    is the thing being asserted (steering_driver's --sanity and --gate_verify).
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    inputs = _tokenize_batch(tokenizer, model, prompts, max_prompt_tokens)
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    stops = stop_token_ids(tokenizer, model)
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=stops,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    out = model.generate(**inputs, **gen_kwargs)
    new_ids = out[:, input_len:]
    texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    # Judged against the SAME stop set generation used, or a model that ends its
    # turn on a non-tokenizer eos is recorded as truncated on every row.
    stop_set = set(stops)
    results = []
    for row, text in zip(new_ids, texts):
        ids = row.tolist()
        truncated = bool(len(ids) >= max_new_tokens and not stop_set.intersection(ids))
        results.append((text.strip(), truncated))
    return results


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    Blank lines are skipped, and a malformed final line (e.g. a partially
    written trailing row from an interrupted append) is ignored rather than
    raising, so live-appended files stay readable.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
