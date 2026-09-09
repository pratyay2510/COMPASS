#!/usr/bin/env python3
"""Skip a derivation step whose output is already current.

The steering scripts are chains of deterministic CPU steps in front of the GPU
work: refit the com, re-rank by PC1, print the geometry, report. Each one reads
a few large arrays, so re-running a command to add one more spec used to pay for
all of them again even though their outputs had not changed.

Each step writes a stamp beside its output recording what it was derived FROM:
the identity (size + mtime) of every input file plus the parameters that shaped
it. On the next run the step recomputes that signature and skips itself when it
matches and its outputs are all present. Any change to an input file, a
parameter or an output — including deleting the output to force a rebuild —
makes the signature miss and the step runs.

This is for derived artifacts only. Generations resume per row inside
steering_driver, which is a different mechanism (and the authoritative one: the
stamp never decides whether a problem was answered).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional

STAMP = "resume_stamp.json"


def file_id(path: str) -> Optional[Dict]:
    """Identity of an input file: size and mtime, not a hash.

    Every input here is a multi-hundred-MB array written once by an earlier
    step, so stat is the right granularity — hashing them would cost more than
    the step being skipped.
    """
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def signature(inputs: Iterable[str], **params) -> Dict:
    """What an output was derived from: its input files and its parameters."""
    return {"inputs": [file_id(p) for p in inputs],
            "params": {k: params[k] for k in sorted(params)}}


def is_current(out_dir: str, sig: Dict, outputs: Iterable[str],
               name: str = STAMP) -> bool:
    """True when the stamp matches AND every declared output still exists."""
    stamp = os.path.join(out_dir, name)
    if not os.path.exists(stamp):
        return False
    if any(not os.path.exists(os.path.join(out_dir, o)) for o in outputs):
        return False
    try:
        with open(stamp) as f:
            return json.load(f) == sig
    except (json.JSONDecodeError, OSError):
        return False


def write_stamp(out_dir: str, sig: Dict, name: str = STAMP) -> None:
    """Record the signature, last, so a crash mid-step never looks current.

    ``name`` lets several steps stamp the same directory without colliding.
    """
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(sig, f, indent=2)


def skip(step: str, out_dir: str) -> None:
    print(f"{step}: already current for these inputs — skipping ({out_dir})")
    print(f"  rebuild with --force, or by deleting {os.path.join(out_dir, STAMP)}")
