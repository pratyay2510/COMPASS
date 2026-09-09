# `src/` — structure

Every module is a CLI (`python3 -m src.<package>.<module> --help`) run from
the repo root, and every module imports its dependencies absolutely
(`from src.core import common`). There are no `sys.path` tricks and no relative
imports, so a module's location in this tree is exactly its import path.

```
src/
├── core/          shared foundations — no experiment logic
│   ├── common.py       model + tokenizer loading, dataset loading (MATH / GSM8K
│   │                   / MATH-500 / GSM-Plus / HARP / SVAMP + the fixed
│   │                   calibration pools), prompt building, stop_token_ids,
│   │                   hidden-state extraction, batched generation, HeadCatcher
│   ├── grading.py      THE GRADING PROTOCOL: the frozen per-family single-pass
│   │                   graders (answers_equal, grader_for), the LLM answer
│   │                   formatter, and the OR that yields the final label;
│   │                   `python3 -m src.core.grading --collection <dir>` runs
│   │                   all three and writes `correct` (GPU for the formatter)
│   └── io.py           the on-disk contract: collection layout, naming rules,
│                       row/feature loaders, verifiers, attach_gold_aliases,
│                       atomic write_csv / write_json
│
├── steering/      stages 2-4 — target, ranking, the intervention
│   ├── elicitation_target.py  the OUTCOME target: first-token distribution of
│   │                          the rows the model got right minus the rows it
│   │                          got wrong, prefill only; gated by MIN_TV
│   ├── head_ranker.py         THE head ranking: logit movement toward the
│   │                          target through each head's own W_O and the
│   │                          unembedding (reads W_O / lm_head straight from
│   │                          the safetensors shards); writes heads.sh,
│   │                          elicit_scores.npz, directions.npz (com)
│   ├── steering_driver.py     THE STEERING, one file: the ITI primitives
│   │                          (transcribed from honest_llama, file:line cited),
│   │                          the three directions (com / anticom / random),
│   │                          the config-tag rule, head specs and selection,
│   │                          the Eq. 2 hooks under the cosine dose schedule,
│   │                          the K x alpha sweep, controls, the driver (GPU,
│   │                          per-config resume), flip bookkeeping
│   ├── operating_point.py     THE OPERATING POINT, three CPU modes around the
│   │                          GPU scan: `plan` (elbow -> nested bands, dose-
│   │                          matched alpha ladder -> plan.sh), `report` (the
│   │                          scan/best table: accuracy, dose, label-free
│   │                          failure signals, row-set check) and `pick` (the
│   │                          winner: highest dev accuracy, ties to the smaller
│   │                          alpha). report and pick share one loader
│   └── resume.py              input-stamped skip for the deterministic CPU
│                              steps in front of the GPU work
│
└── utils/         the tooling around the pipeline
    ├── collect/       stage 1 — run the model, label, save activations (GPU)
    │   ├── collection.py                THE collector (collect / sweep), per-row
    │   │                                resumable, head/ next to hidden/
    │   └── download_dataset.py          fetch a benchmark into $DATASETS_ROOT (data/paths.py)
    │
    └── diagnostics/   read-only inspection
        ├── layer_transmission.py     how optimistic the direct-path score is per
        │                             layer (does an injection at l reach the output?)
        ├── first_token_delta.py      the logit-lens check on a smoke collection:
        │                             the outcome contrast and its TV vs MIN_TV
        ├── check_collection.py       collection health, token caps, live sweep table
        ├── show_gens.py              raw generations, filterable (flips, declined, ...)
        ├── score_cells.py            regenerates info/scores.json (the ledger) from
        │                             the row files under the OR grading rule
        └── test_grading.py           the grader's regression suite
```

## Dependency direction

```
core.common ── core.grading ── core.io   (common imports grading's extractor; grading
                                          imports common lazily; io uses common)
                  │
   utils/collect ─┤
                  │
   steering.elicitation_target ── steering.head_ranker ──> steering.operating_point
                                                            (derive_bands)

   steering.steering_driver ──> core.grading, core.io
        │        (grading imports steering_driver LAZILY for the summary rewrite)
        └──> steering.operating_point, utils/diagnostics/
```

Acyclic by construction, and two things are single-sourced on purpose:

- **`steering/steering_driver.py`** holds the config-tag rule, head selection
  and the flip counts next to the intervention itself; the grader imports them
  lazily, and nothing else builds a gen_/summary_ filename.
- **`steering/head_ranker.py`** is the only place heads are
  ranked and the com direction is written. `run_compass.sh elicit`,
  `ablations.sh kheads` and the `_k<K>` trees all go through it, so the
  heads a report names are the heads that were steered.

## The pipeline, and what each stage leaves on disk

```
smoke     -> <embed>/_zs_smoke/<tag><data>_<mode>/   a small collection + the
             cap table + first_token_delta.{npz,json}
collect   -> <root>/train_<mode>/ , test_<mode>/  (train = the fitting pool,
             e.g. llamacalib / qwencalib / calibration)
             ground_truth.csv, records.jsonl, hidden/, head/, args.json
format    -> relabels the fitting pool's ground_truth.csv under FIT_LABELS
             (single-pass label kept as correct_singlepass)
scanrows  -> <root>/outcome_<mode>/scan_rows.csv  (the dev hold-out, when no
             SCAN_SPLIT), else <root>/<scan_split>_<mode>/scan_rows*.csv
target    -> <root>/outcome_<mode>/elicit_target.npz
elicit    -> <root>/outcome_<mode>[_k<K>]/  heads.sh, elicit_scores.npz,
             directions.npz (com), plan.sh
scan      -> <root>/scan_<mode>[_...]/E<band>/  gen_<config>.jsonl,
             summary_<config>.json, sweep_summary.csv
best      -> <root>/best_<mode>[_...]/E<band>/  same layout, full test split
transfer  -> <root>/transfer_<dataset>_.../E<band>/  same layout, another split
cot       -> <root>/test_cot/  the vendor-CoT reference collection
.zs/      -> stage stamps
```

`<root>` is `<embed>/zs_<model_tag><data_slug>_<subject>`; `<data_slug>` is
empty for MATH, else `_gsm8k` / `_gsmplus` / `_math500` / `_harp` / `_svamp`;
`<mode>` is `standard` (the one per-family prompt; `qa` / `cot` / `qabrief`
only name collections that already exist). `run_compass.sh` derives all of it
from `MODEL_TAG`, `DATASET`, `SUBJECT`, `PROMPT_MODE`, `FIT_TAG`, `SCAN_SPLIT`
and `K`.

## Conventions worth knowing before editing

- **One grader.** `core.grading.answers_equal` (dispatched per family by
  `grader_for`) is the only labeling logic; the collector, the steering driver
  and the regression tests all call it. The graders are frozen and pinned by
  `src.utils.diagnostics.test_grading`; `python3 -m src.core.grading` is the one
  way a label is produced, and labels are never patched by hand.
- **One tag rule.** `steering.steering_driver.config_tag` names every
  `gen_`/`summary_` file; `operating_point` only parses what it wrote.
- **Resumability is per row.** The collector and the steering driver append and
  flush per example, so an interrupted run resumes on re-run. Report stages
  recompute from the gen files (seconds) and are skipped by artifact existence.
- **Regenerate, don't append.** `sweep_summary.csv` is rebuilt from the
  per-config JSONs on every run, so re-running never duplicates rows.
- **One chooser.** `steering.operating_point pick` (the scan's cosine gen
  files on the dev slice: highest accuracy, ties to the smaller alpha) is the
  only definition of "which (band, alpha) won". `best` steers it unless
  `BEST_BAND`/`BEST_ALPHA` are set, `automate-*` and the job cards read it
  through `run_compass.sh pick`, and `report` names it with the same loader.
  Do not add a second "pick the best row" anywhere.
- **One schedule.** The dose over a generation is always the cosine ramp
  (config tags carry `_cos`); files without it are pre-2026-08-29 constant
  rungs, which `report`/`pick` skip and nothing resumes into.
- **Stamped skips for the CPU prelude.** `steering.resume` records what each
  deterministic step was derived FROM (size + mtime of every input, plus the
  parameters), so re-running `elicit` to add one more band costs that band and
  not the com refit again.
- **Atomic artifacts.** `io.write_json` / `write_csv` write to a temp file and
  rename, because every stage decides it has already run by testing that its
  output EXISTS; a half-written file would retire a stage that never finished.
