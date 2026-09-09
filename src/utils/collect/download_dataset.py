"""Download an eval dataset into the shared dataset tree ($DATASETS_ROOT,
data/paths.py).

    source "$VENV_LLAMA/bin/activate"
    python3 -m src.utils.collect.download_dataset gsm8k
    python3 -m src.utils.collect.download_dataset math500

Layout is the one data/paths.py DATASET_LAYOUT expects: one directory named
<author>__<name> under DATASETS_ROOT, holding the raw repo files under raw/ and
one arrow cache under hf_cache/ so downstream code can load_dataset offline.
"""

import argparse
import os
from pathlib import Path

from src.core.common import paths

REGISTRY = {
    # MATH: one config per subject; each subject is its own arrow cache
    # (<root>/<subject>), the layout src.core.common.load_math_examples reads.
    "math": ("EleutherAI/hendrycks_math", None),
    "gsm8k": ("openai/gsm8k", ["main", "socratic"]),
    # None = every config the repo declares (get_dataset_config_names).
    "gsm_plus": ("qintongli/GSM-Plus", None),
    "math500": ("HuggingFaceH4/MATH-500", None),
    # SVAMP (Patel et al. 2021), ChilleD's HF re-export: one 'default' config,
    # train 700 / test 300, columns ID/Body/Question/Equation/Answer/Type.
    "svamp": ("ChilleD/SVAMP", None),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=sorted(REGISTRY))
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--configs", nargs="+", default=None)
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    from datasets import load_dataset, get_dataset_config_names

    repo, configs = REGISTRY[args.name]
    configs = args.configs or configs or get_dataset_config_names(repo)
    root = args.root or Path(paths.require("DATASETS_ROOT")) / repo.replace("/", "__")

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, repo_type="dataset", local_dir=str(raw))
    print(f"raw -> {raw}")

    for cfg in configs:
        cache = root / (cfg if args.name == "math" else "hf_cache")
        ds = load_dataset(repo, cfg, cache_dir=str(cache))
        for split in ds:
            print(f"{cfg:16s} {split:6s} n={len(ds[split]):5d} "
                  f"cols={ds[split].column_names}")

    print(f"\narrow cache -> {cache}")
    print(f"load with: load_dataset('{repo}', '{configs[-1]}', "
          f"cache_dir='{cache}')")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    main()
