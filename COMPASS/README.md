# COMPASS: Finding Where Reasoning Lives in Language Models

**CO**rrectness-**M**apped **P**er-head **A**ttribution for **S**teering. An
inference-time steering method that elicits reasoning from a frozen LLM using
nothing but the correctness of its own direct answers.

> Explicitly eliciting reasoning substantially improves LLM performance.
> Existing approaches require a predefined characterization of reasoning,
> whether through CoT prompt design, contrastive CoT directions, or via SAE
> derived reasoning features. For mathematical reasoning with verifiable
> answers, we show that a much simpler signal suffices, which is the
> correctness of the model's own direct answer attempts. This signal yields a
> latent direction that elicits reasoning. This direction is decodable within
> the activations of most attention heads, but only a small subset of them can
> be effectively intervened. We introduce COMPASS, an inference-time steering
> method that identifies these heads using a logit-space attribution score and
> steers their activations along the correctness direction, requiring only
> per-head activation statistics. Across three model families and multiple
> math benchmarks, COMPASS outperforms the activation-steering baselines we
> compare against, improves GSM8K accuracy by 16 percentage points on average,
> and approaches CoT accuracy with 20-70% fewer generated tokens. Interventions
> transfer without re-fitting to unseen benchmarks, and ablations show that
> both the correctness direction and the small set of heads carrying it are
> necessary, with the effect concentrated in remarkably few heads.

## How it works

1. **Fit.** Run the direct-answer prompt once over a fitting pool, grade each
   answer, and cache every attention head's activation at the last prompt
   token. Per head, the correctness direction is the unit vector from the
   mean incorrect activation to the mean correct one (Eq. 3), scaled by the
   pool's standard deviation along it (Eq. 4).
2. **Rank.** Score each head by how far its projected direction moves the
   first-token logits toward the tokens that precede correct answers
   (Eq. 9). Only a late band of layers carries mass; the top K=8 heads in
   that band are steered.
3. **Scan.** Choose the layer band and the strength α on 300 held-out dev
   rows, comparing bands at matched dose.
4. **Steer.** Add α σ θ to the selected heads at every decode step (Eq. 5)
   and generate once over the full test split.

## Results

Table 1 of the paper. Accuracy (%) with mean generated tokens in parentheses.

| | Llama-3.1-8B | | | Qwen3-4B | | | Gemma-4-12B | | |
|---|---|---|---|---|---|---|---|---|---|
| | GSM8K | SVAMP | MATH-500 | GSM8K | SVAMP | MATH-500 | GSM8K | SVAMP | MATH-500 |
| Standard prompt | 66.8 <sub>(93)</sub> | 81.0 <sub>(48)</sub> | 42.4 <sub>(345)</sub> | 76.2 <sub>(68)</sub> | 81.0 <sub>(22)</sub> | 55.2 <sub>(250)</sub> | 49.4 <sub>(67)</sub> | 74.7 <sub>(3)</sub> | 68.4 <sub>(347)</sub> |
| ITI | 73.8 <sub>(132)</sub> | 68.7 <sub>(45)</sub> | 23.6 <sub>(409)</sub> | 71.9 <sub>(45)</sub> | 44.3 <sub>(78)</sub> | 30.6 <sub>(56)</sub> | 22.3 <sub>(52)</sub> | 64.0 <sub>(46)</sub> | 30.3 <sub>(872)</sub> |
| Fractional Reasoning | 79.9 <sub>(214)</sub> | 81.0 <sub>(119)</sub> | 42.2 <sub>(524)</sub> | 88.3 <sub>(220)</sub> | 89.3 <sub>(118)</sub> | 77.8 <sub>(561)</sub> | 65.4 <sub>(55)</sub> | **85.6** <sub>(7)</sub> | 72.7 <sub>(371)</sub> |
| **COMPASS** | **83.7** <sub>(143)</sub> | **85.7** <sub>(81)</sub> | **45.2** <sub>(515)</sub> | **89.8** <sub>(138)</sub> | **90.7** <sub>(68)</sub> | **77.8** <sub>(404)</sub> | **88.0** <sub>(140)</sub> | 79.3 <sub>(51)</sub> | **79.8** <sub>(366)</sub> |
| CoT (oracle) | 83.4 <sub>(205)</sub> | 85.0 <sub>(157)</sub> | 47.0 <sub>(788)</sub> | 92.3 <sub>(303)</sub> | 91.7 <sub>(224)</sub> | 83.2 <sub>(924)</sub> | 96.1 <sub>(338)</sub> | 94.3 <sub>(261)</sub> | 91.4 <sub>(730)</sub> |
| GPT-5.4-mini (oracle) | 95.1 <sub>(273)</sub> | 94.7 <sub>(202)</sub> | 88.2 <sub>(786)</sub> | 95.1 <sub>(273)</sub> | 94.7 <sub>(202)</sub> | 88.2 <sub>(786)</sub> | 95.1 <sub>(273)</sub> | 94.7 <sub>(202)</sub> | 88.2 <sub>(786)</sub> |

## Setup

**1. Environment.** One venv per model family; the three transformers
generations do not coexist.

```bash
python3 -m venv .venv-llama  && . .venv-llama/bin/activate  && pip install -r requirements.txt         # Llama-3.1-8B
python3 -m venv .venv-qwen3  && . .venv-qwen3/bin/activate  && pip install -r requirements-qwen3.txt   # Qwen3-4B
python3 -m venv .venv-gemma4 && . .venv-gemma4/bin/activate && pip install -r requirements-gemma4.txt  # Gemma-4-12B (see its header for torch)
```

**2. Paths.** Every machine path lives in [`data/paths.py`](data/paths.py):
where collections are written, where datasets and model weights are cached,
and the three venvs. Fill in section 1, then check it.

```bash
python3 data/paths.py --check
```

**3. Models.** Weights are read from `MODEL_CACHE_ROOT` and fetched from the
Hub on first use. To pre-download:

```bash
python3 -c "from huggingface_hub import snapshot_download as d; from src.core.common import model_cache_dir as c; m='meta-llama/Llama-3.1-8B-Instruct'; d(m, cache_dir=c(m))"
```

**4. Datasets.** Into `DATASETS_ROOT`, in the layout `data/paths.py` expects:

```bash
python3 -m src.utils.collect.download_dataset math       # EleutherAI/hendrycks_math
python3 -m src.utils.collect.download_dataset gsm8k
python3 -m src.utils.collect.download_dataset svamp
python3 -m src.utils.collect.download_dataset math500
python3 -m src.utils.collect.download_dataset gsm_plus
```

HARP is a manual download: unzip `HARP.jsonl` from
[aadityasingh/HARP](https://github.com/aadityasingh/HARP) to
`$DATASETS_ROOT/HARP/HARP.jsonl`.

The fitting pools are frozen in [`data/`](data/): for MATH and GSM8K, 1,600
rows of the train split per model family (`<dataset>_<family>calib.json`,
800 correct + 800 incorrect under that family's own answers). SVAMP fits on
its 700-row train split. The right pool is picked from `MODEL_ID`.

## Run

Three commands per (model, dataset). Activate the family's venv, then:

```bash
export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MODEL_TAG=llama DATASET=gsm8k

./run_compass.sh all             # 1. collect the fitting pool -> target -> head ranking -> band/alpha grid
./run_compass.sh automate-scan   # 2. scan the grid on the dev rows -> prints the pick (band, alpha)
./run_compass.sh best            # 3. collect the test baseline, then steer it at the pick
```

For MATH the evaluation split is MATH-500, so step 3 is a transfer:

```bash
export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MODEL_TAG=llama DATASET=math MAX_NEW_TOKENS=2048 STEER_MAX_NEW_TOKENS=2048
./run_compass.sh all && ./run_compass.sh automate-scan
TRANSFER_DATASET=math500 ./run_compass.sh transfer
```

`MODEL_ID` is the Hub id, `MODEL_TAG` the directory slug (`llama`,
`qwen3_4b`, `gemma4_12b`), `DATASET` one of `math | gsm8k | svamp`. Every
stage resumes per row, so a killed run is re-run with the same command.
Append `--nohup` to detach into `logs/`.

What each step leaves under `$EMBED_ROOT/zs_<tag>[_<dataset>]_all/`:

| step | writes | contents |
|---|---|---|
| `all` | `<pool>_standard/` | the fitting pool: generations, labels, per-head activations |
| | `outcome_standard_fit<pool>/` | `elicit_target.npz` (Δp, u), `heads.sh` (ranked heads per band), `directions.npz` (θ, σ), `plan.sh` (the grid) |
| `automate-scan` | `scan_standard_outcome_fit<pool>/E<band>/` | one `gen_*.jsonl` + `summary_*.json` per (band, α) on the dev rows |
| `best` / `transfer` | `test_standard/`, then `best_…/E<band>/` or `transfer_<dataset>_…/E<band>/` | the unsteered baseline, then the steered full split |

`./run_compass.sh report` prints the scan table and the test rows;
`./run_compass.sh pick` prints the operating point alone.

Token caps default to 1,024. Set `MAX_NEW_TOKENS` (pool and baseline) and
`STEER_MAX_NEW_TOKENS` (steered rows) per dataset; the values used for the
paper are in the table below.

### Reproducing the paper's operating points

To skip the scan, pass the promoted band and α directly. Heads are still
ranked by the pipeline (`all` is required); only the search is bypassed.

| model | dataset | pool | band | α | `MAX_NEW_TOKENS` / `STEER_MAX_NEW_TOKENS` | command |
|---|---|---|---|---|---|---|
| llama | gsm8k | llamacalib | 31-31 | 16.88 | 512 / 512 | `BEST_BAND=31-31 BEST_ALPHA=16.88 ./run_compass.sh best` |
| llama | svamp | train | 31-31 | 24.06 | 512 / 512 | `BEST_BAND=31-31 BEST_ALPHA=24.06 ./run_compass.sh best` |
| llama | math → math500 | llamacalib | 30-31 | 14.56 | 2048 / 2048 | `TRANSFER_DATASET=math500 TRANSFER_BAND=30-31 TRANSFER_ALPHA=14.56 ./run_compass.sh transfer` |
| qwen3_4b | gsm8k | qwencalib | 32-35 | 11.05 | 512 / 512 | `BEST_BAND=32-35 BEST_ALPHA=11.05 ./run_compass.sh best` |
| qwen3_4b | svamp | train | 35-35 | 45.25 | 512 / 512 | `BEST_BAND=35-35 BEST_ALPHA=45.25 ./run_compass.sh best` |
| qwen3_4b | math → math500 | qwencalib | 35-35 | 20.44 | 2048 / 2048 | `TRANSFER_DATASET=math500 TRANSFER_BAND=35-35 TRANSFER_ALPHA=20.44 ./run_compass.sh transfer` |
| gemma4_12b | gsm8k | gemmacalib | 47-47 | 8 | 1024 / 1024 | `BEST_BAND=47-47 BEST_ALPHA=8 ./run_compass.sh best` |
| gemma4_12b | svamp | train | 46-47 | 21 | 512 / 1024 | `BEST_BAND=46-47 BEST_ALPHA=21 ./run_compass.sh best` |
| gemma4_12b | math → math500 | gemmacalib | 47-47 | 5 | 2048 / 2048 | `TRANSFER_DATASET=math500 TRANSFER_BAND=47-47 TRANSFER_ALPHA=5 ./run_compass.sh transfer` |

### Transfer, CoT reference, ablations

```bash
# Table 2: the GSM8K selection on GSM-Plus, the MATH selection on HARP (nothing re-fitted)
DATASET=gsm8k TRANSFER_DATASET=gsm_plus ./run_compass.sh transfer
DATASET=math  TRANSFER_DATASET=harp     ./run_compass.sh transfer

# CoT oracle: the model's own chain-of-thought prompt on the full test split
PROMPT_MODE=cot ./run_compass.sh cot

# Figure 5: number of heads, head/direction controls, reversed direction
./ablations.sh kheads  llama_gsm8k      # K in {4, 2, 1} at the promoted band and alpha
./ablations.sh heads   llama_math       # random heads in the band; random direction at the selected heads
./ablations.sh anticom llama_gsm8k      # -theta at the selected heads
```

## Grading

Every accuracy comes from one instrument (`src/core/grading.py`): a
deterministic grader on the raw generation, OR the same grader on an LLM
restatement of the answer as `\boxed{…}`. The restatement is requested only
for rows the deterministic pass marks wrong, so it can promote a row but never
demote one. Accuracy is over all rows of the split; a generation that hits
the token cap is graded on its partial text.

## Layout

```
COMPASS/
├── run_compass.sh      the pipeline: all -> automate-scan -> best | transfer (+ smoke, cot, report, pick)
├── ablations.sh        the ablations at a promoted operating point
├── data/               paths.py (every machine path) + the frozen fitting-pool manifests
├── requirements*.txt   one per model family
└── src/
    ├── core/           model + dataset loading, prompts, the on-disk contract, grading
    ├── steering/       elicitation_target (Δp, u), head_ranker (Eq. 9), steering_driver (Eq. 5), operating_point (band/alpha)
    └── utils/          collect (the collector, dataset download), diagnostics
```

## Citation

```bibtex
@inproceedings{dutta2026compass,
  title     = {{COMPASS}: Finding Where Reasoning Lives in Language Models},
  author    = {Dutta, Pratyay and Thopalli, Kowshik and Narayanaswamy, Vivek},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
