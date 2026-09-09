#!/usr/bin/env python3
"""EVERY machine-specific path the project reads, in one file.

Edit section 1 (and section 2 if your dataset tree is laid out differently),
nothing else. No other file in COMPASS/ holds an absolute path: Python
modules import this file, shell scripts source its `--sh` output, so a path
changed here changes everywhere at once.

    python3 data/paths.py            # the table: every name, its value, on disk?
    python3 data/paths.py --check    # same, exit 1 if a required path is unset/missing
    python3 data/paths.py --sh       # `export NAME=value` lines for bash
    python3 data/paths.py --json

From bash (run_compass.sh, ablations.sh, every srun/ card), after the cd
into COMPASS/:

    eval "$(python3 data/paths.py --sh)"

From Python (src/core/common.py): loaded by file location, never by package
name, so it works from any cwd.

OVERRIDES. Every name can be overridden for one command by exporting a
variable of the same name (MODEL_CACHE_ROOT answers to HF_MODEL_CACHE_ROOT,
the name the loaders always used): `EMBED_ROOT=/scratch/x ./run_compass.sh
best` redirects that run and nothing else. Exported values win over this
file, so a stale export in the shell silently redirects a tree -- the srun/
cards unset the pipeline knobs for exactly this reason.

BLANK VALUES. A name left "" is "not configured". Shell scripts refuse at the
point of use (`: "${EMBED_ROOT:?...}"`), Python raises through require(),
each naming this file. That is what a fresh clone sees until section 1 is
filled in, and what --check reports.

This file is stdlib-only and must stay so: the shell scripts run it with the
system python3 BEFORE any venv is activated (it is how they find the venv).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

# =============================================================================
# 1. EDIT THESE
# =============================================================================

# Collections, fits, scans, best/transfer sweeps: zs_<model><dataset>_<subject>/
# trees. Written by the pipeline; read by everything. 
EMBED_ROOT = "/p/vast1/dutta5/embeddings"

# The dataset caches (layout in section 2). Read-only inputs; the loaders
# create a per-dataset subdirectory on first use and download into it.
DATASETS_ROOT = "/p/vast1/dutta5/datasets"

# Hugging Face model weights, one <org>__<name>/ per model (common.cache_safe_name).
MODEL_CACHE_ROOT = "/p/vast1/dutta5/cache-dirs/models"

# One virtualenv per model family (three transformers generations that do not
# coexist): requirements.txt / requirements-qwen3.txt / requirements-gemma4.txt.
VENV_LLAMA = "/p/vast1/dutta5/envs/ICL-venv"
VENV_QWEN3 = "/p/vast1/dutta5/envs/RS-venv"
VENV_GEMMA4 = "/p/vast1/dutta5/envs/G4-venv"

# =============================================================================
# 2. Dataset layout under DATASETS_ROOT (edit only if yours differs; an
#    absolute path exported under the name wins, e.g. HARP_JSONL=/x/HARP.jsonl)
# =============================================================================

DATASET_LAYOUT = {
    "HENDRYCKS_MATH_ROOT": "EleutherAI__hendrycks_math",       # MATH, per-subject caches
    "GSM8K_ROOT": "openai__gsm8k/hf_cache",
    "GSM_PLUS_ROOT": "qintongli__GSM-Plus/hf_cache",
    "MATH500_ROOT": "HuggingFaceH4__MATH-500/hf_cache",
    "SVAMP_ROOT": "ChilleD__SVAMP/hf_cache",
    "HARP_JSONL": "HARP/HARP.jsonl",                           # the unzipped HARP.jsonl
}

# =============================================================================
# 3. Resolution (nothing below is machine-specific)
# =============================================================================

ROOTS = ("EMBED_ROOT", "DATASETS_ROOT", "MODEL_CACHE_ROOT",
         "VENV_LLAMA", "VENV_QWEN3", "VENV_GEMMA4")
# Environment variable that overrides a name, where it is not the name itself.
ENV_ALIAS = {"MODEL_CACHE_ROOT": "HF_MODEL_CACHE_ROOT"}
# Written by the pipeline (mkdir on first use); --check reports "new" rather
# than "MISSING" when it does not exist yet.
CREATED_ON_USE = {"EMBED_ROOT"}

WHAT = {
    "EMBED_ROOT": "collections + steering outputs (zs_* trees)",
    "DATASETS_ROOT": "dataset caches (section 2 layout)",
    "MODEL_CACHE_ROOT": "HF model weights",
    "VENV_LLAMA": "venv: llama (requirements.txt)",
    "VENV_QWEN3": "venv: qwen3 (requirements-qwen3.txt)",
    "VENV_GEMMA4": "venv: gemma-4 (requirements-gemma4.txt)",
    "HENDRYCKS_MATH_ROOT": "MATH (EleutherAI/hendrycks_math)",
    "GSM8K_ROOT": "GSM8K (openai/gsm8k)",
    "GSM_PLUS_ROOT": "GSM-Plus (qintongli/GSM-Plus)",
    "MATH500_ROOT": "MATH-500 (HuggingFaceH4/MATH-500)",
    "SVAMP_ROOT": "SVAMP (ChilleD/SVAMP)",
    "HARP_JSONL": "HARP short-answer test (aadityasingh/HARP)",
}

THIS_FILE = os.path.abspath(__file__)
_FILE_VALUES = {name: globals()[name] for name in ROOTS}


def env_name(name: str) -> str:
    return ENV_ALIAS.get(name, name)


def resolve() -> dict:
    """{name: value} with environment overrides applied; "" = not configured."""
    out = {}
    for name in ROOTS:
        out[name] = (os.environ.get(env_name(name)) or _FILE_VALUES[name] or "").rstrip("/")
    ds = out["DATASETS_ROOT"]
    for name, rel in DATASET_LAYOUT.items():
        out[name] = os.environ.get(name) or (os.path.join(ds, rel) if ds else "")
    return out


PATHS = resolve()
globals().update(PATHS)          # paths.EMBED_ROOT etc. are the RESOLVED values
NAMES = tuple(PATHS)


def get(name: str) -> str:
    """The resolved value, "" when not configured."""
    return PATHS[name]


def require(name: str) -> str:
    """The resolved value, or a clear exit naming this file."""
    value = PATHS[name]
    if not value:
        raise SystemExit(
            f"{name} is not configured: set it in {THIS_FILE} "
            f"(or export {env_name(name)}).")
    return value


def status(name: str) -> str:
    value = PATHS[name]
    if not value:
        return "UNSET"
    if os.path.exists(value):
        return "ok"
    if name in CREATED_ON_USE:
        return "new (created on first use)"
    return "MISSING"


def main() -> None:
    p = argparse.ArgumentParser(description="COMPASS path registry (see the module docstring)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sh", action="store_true", help="export lines for bash")
    g.add_argument("--json", action="store_true")
    g.add_argument("--check", action="store_true",
                   help="print the table; exit 1 if a required path is unset or missing")
    args = p.parse_args()
    if args.sh:
        for name in NAMES:
            print(f"export {name}={shlex.quote(PATHS[name])}")
        return
    if args.json:
        print(json.dumps(PATHS, indent=2))
        return
    width = max(len(n) for n in NAMES)
    bad = []
    print(f"paths from {THIS_FILE} (+ environment overrides)")
    for name in NAMES:
        st = status(name)
        src = "env" if os.environ.get(env_name(name)) else "file"
        print(f"  {name:<{width}}  {st:<27}  {src:<4}  {PATHS[name] or '-'}   # {WHAT[name]}")
        if st in ("UNSET", "MISSING"):
            bad.append(name)
    if args.check and bad:
        sys.exit(f"\n{len(bad)} path(s) need attention: {', '.join(bad)}  -> edit {THIS_FILE}")


if __name__ == "__main__":
    main()
