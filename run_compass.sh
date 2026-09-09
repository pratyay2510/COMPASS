#!/bin/bash
# =============================================================================
# Reasoning-elicitation steering, end to end.
#
#   export MODEL_ID=... MODEL_TAG=... DATASET=...
#   ./run_compass.sh all              1. fitting pool -> target -> head ranking -> band/alpha grid
#   ./run_compass.sh automate-scan    2. scan the grid on the dev rows -> the pick (band, alpha)
#   ./run_compass.sh best             3. test baseline, then steer it at the pick
#      (MATH: TRANSFER_DATASET=math500 ./run_compass.sh transfer)
#
#   ./run_compass.sh <stage>          smoke collect format scanrows target elicit
#                                      plan scan report pick best transfer cot
#                                      automate-scan automate-all automate-transfer
#   DRY_RUN=1 ./run_compass.sh all    print the commands, run nothing
#   ... --nohup                        detach, log under logs/
#
# THIS SCRIPT IS THE WHOLE PIPELINE AND NOTHING ELSE: the train collection, the
# outcome-based head selection + com direction, the dev-row scan, the test
# baseline collection, best/transfer on the full test split, and the CoT
# reference collection. Published baselines (ITI, Fractional Reasoning) are a
# separate, import-free package at ../baselines/; the ablations at the promoted
# operating point live in ablations.sh.
#
# WHAT THIS REPLACES. The retired probe suite followed ITI's recipe: fit 1024
# logistic probes on labelled activations, take the top-K by validation
# accuracy, steer them. That criterion failed twice on llama/gsm8k. Inside layers 20-27 its
# top-8 produced no more helped rows than eight RANDOM heads (136 vs 141). Two
# draws of one collection picked top-8 sets overlapping in 3 of 8 and differing
# by 7.5 pp of effect, because 1024 heads spread over a range narrower than one
# standard error of the accuracy estimate. And three of the eight heads it chose
# ranked 1010th, 1012th and 1016th of 1024 on the criterion below -- their
# injection pushes AWAY from reasoning.
#
# THE PIPELINE HERE.
#
#   smoke    a few dozen rows at a generous token cap, then the cap table: what
#            would a smaller --max_new_tokens have cost in correct answers?
#            Run this BEFORE collect; collect is the expensive step.
#   collect  generate under the baseline prompt, keeping per-head activations.
#            The only GPU-heavy step before the sweeps, and the only one that
#            needs the grader -- for measuring the baseline, not for choosing.
#   scanrows the rows band and alpha are tuned on: SCAN_ROWS of the SCAN_SPLIT
#            collection (MATH: `calibration`), or, without one, a stratified
#            hold-out of the fitting pool that the com fit then excludes.
#            (The per-head logistic probes and their 50/50 balance draw were
#            removed on 2026-08-28: nothing downstream used them -- com is a
#            difference of class means and needs neither.)
#   target   what token do the rows the model got RIGHT open with, minus the
#            rows it got WRONG, both under the one baseline prompt (the outcome
#            contrast)? One prefill pass per sampled row, no generation. On
#            llama/gsm8k this is "promote ' To', suppress ' $'" -- and the
#            steered generations do begin "To find the total cost...". Under
#            1 minute.
#   elicit   score every head by how far its injected vector moves the logits
#            toward that target, through its own W_O and the unembedding. Weight
#            math plus one pass over the cached activations. CPU, minutes.
#   plan     target + elicit + the grid: verify the target is worth steering
#            toward (total variation floor), find the ELBOW of the elicitation
#            mass profile, emit the nested bands from it, and print the exact
#            scan command per band with alphas matched on DOSE. This is the
#            whole chain up to the GPU scan, and it is model- and
#            dataset-agnostic: no layer index or alpha is ever typed in.
#   scan     steer the top-K of each candidate band across an alpha ladder, on
#            the dev rows. Every config sees the identical problems and none of
#            them is in the test split.
#   report   the scan table: accuracy, dose, and the label-free a* criterion.
#   best     collect the unsteered baseline on the FULL test split, then steer
#            it at the scan's pick (or BEST_BAND / BEST_ALPHA). Not part of
#            `all`, and deliberately: it is the only run that touches test.
#   transfer the same, with this run's heads + com direction on ANOTHER
#            dataset's full test split (MATH -> MATH-500; GSM8K -> GSM-Plus).
#   cot      the vendor CoT prompt on the full test split (PROMPT_MODE=cot):
#            the reference ceiling; collect-only, nothing is fitted on it.
#
# BAND AND ALPHA ARE SUGGESTED, NOT CHOSEN. `plan` derives the grid and prints
# the commands; you run them and pick the winner from `report`. Nothing between
# `plan` and `best` is automatic, and `best` is the only stage that touches the
# test split.
#
# WHY A BAND AT ALL. The elicit score is a direct-path measure: it credits a
# head as if nothing downstream touched its output. layer_transmission measured
# how false that is -- at layer 28 an injection drags 1.6 units of unintended
# change to the output per unit injected, at layer 31 only 0.09. So the score is
# trustworthy late and optimistic early, and `scan` finds where the line falls.
#
# ALPHA is a hyperparameter, chosen by `scan`. Two label-free signals bracket it
# in the report: `<=5w` (problems still answering directly -- elicitation left on
# the table) and `loop` (degenerate repetition -- pushed off-manifold). Their sum
# was minimised one grid step from the true accuracy optimum on llama/gsm8k.
# The ladder is walked upward and a band stops at the first dip below its
# peak (SCAN_EARLY_STOP): the rungs past the peak are the costliest and only
# ever confirm the fall.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------- MODEL / DATA
# NO DEFAULTS, deliberately. These three decide which experiment's directory
# tree every stage reads and writes, so a default is not a convenience -- it is
# a silent redirection. A shell that had lost these variables used to resolve to
# llama/math, find that run's finished artifacts and print "skip elicit (done)"
# while the intended run had nothing done at all. Unset is now an error.
: "${MODEL_ID:?set MODEL_ID (e.g. Qwen/Qwen3-4B)}"
: "${MODEL_TAG:?set MODEL_TAG, the directory slug (e.g. qwen3_4b)}"
: "${DATASET:?set DATASET (math | gsm8k | gsm_plus | math500 | harp | svamp)}"
SUBJECT=${SUBJECT:-all}
PROMPT_MODE=${PROMPT_MODE:-standard}   # 'standard' is the ONE prompt (per model family).
                                       # Legacy tags (qa | qa_brief | cot) remain valid only
                                       # for addressing collections that already exist on disk.
# THE GRADER (2026-09-01). `formatter` below no longer means "the formatter
# verdict REPLACES the label". It means the OR:
#
#     correct = single-pass grader  OR  formatter verdict
#
# and the formatter is only asked about rows the frozen grader marked wrong.
# One rule for every model; a formatter that fails as a generation can now only
# fail to add, never subtract. It is the only final label src.core.grading
# writes (single-pass OR formatter).
#
# Why: the formatter is a second generation and can fail as one. On
# gemma-4/MATH it re-derived the problem instead of restating it and never
# reached the box on 59% of rows, regrading a correct 64.4% down to 41.8%.
# Under the OR that cell reads 75.4%. On llama/qwen3 the rule moves the
# headline cells by at most a point (it lifts baselines slightly more than
# steered rows), so nothing about those results turns on it.
#
# Per-model differences live in the PROMPT, not the grader: gemma has no
# grading.GRADERS entry (it uses the frozen llama grader like everyone else) and
# only overrides the math formatting prompt, via
# grading.FORMATTING_PROMPT_OVERRIDES.
#
# TWO GRADING POLICIES, deliberately separate (2026-08-24, late).
#
# FIT_LABELS   which labels the FITTING POOL carries when the
#              outcome target / com are computed -- and therefore what the
#              com direction MEANS.
#     singlepass  the frozen family grader's labels. The `format` stage
#                 REVERTS any earlier formatter relabel (correct <-
#                 correct_singlepass). llama's policy: fitting on formatter
#                 labels rotated com off the reasoning axis (cos ~0.6 on the
#                 steered heads) and the math500 gain vanished; single-pass
#                 labels are the ones the +9.4 was fitted on.
#     formatter   the `format` stage relabels the pool with the LLM answer
#                 formatter (src.core.grading) before target/elicit -- under the OR
#                 above, so a pool label can only move incorrect -> correct.
#              Scans (selection on the fitting pool) grade their steered rows
#              by the SAME instrument as the pool, so a scan never compares
#              two gradings.
# EVAL_GRADING how the headline numbers are graded: `best` / `transfer`
#              steered rows AND the transfer baseline they are read against.
#     formatter   (default) baseline relabeled and steered rows re-graded
#                 under the OR -- one instrument, both sides. Both sides must
#                 actually be relabeled: a formatter-graded steered arm read
#                 against a single-pass baseline is a cross-instrument delta,
#                 which is how gemma/svamp came to report +3.6 for what is
#                 -0.7 under one rule.
#     singlepass  the frozen grader on both sides.
#
# Defaults by family, until the policy is made universal: llama fits on
# single-pass labels; every other family keeps the formatter-label fit its
# running streams were launched under (the qwen3 stream is NOT to be touched
# mid-run). Only datasets with a formatting prompt (math | math500 | harp |
# gsm8k | gsm_plus | svamp, the gsm pair added 2026-08-25, svamp 2026-08-27
# -- format_supported())
# have a formatter at all; elsewhere both policies reduce to single-pass.
EVAL_GRADING=${EVAL_GRADING:-formatter}
# FIT_LABELS defaults once TRAIN_SPLIT is known (below): a calib pool is
# DEFINED on single-pass labels, so it is fitted on them for every family.
# MATH (2026-08-28, renamed 2026-08-30): the fitting pool is a fixed calib split -- 1,600 rows
# of MATH train, 800 correct + 800 incorrect, chosen so the outcome target is
# well separated (frozen manifests data/math_<split>.json; nothing in the repo redefines them) -- for
# EVERY model, and band/alpha are tuned on 300 rows of `calibration` (MATH
# test minus MATH-500, SCAN_SPLIT below). MATH `train` and the 4,500-row
# `calibration` are no longer fitting pools; the fit artifacts carry _fitcalib
# so the older trees stay addressable but are never resumed into.
#
# 2026-08-30: the pool is named for the family whose openers chose it --
# `llamacalib` (the split formerly called `calib` / `tvcal`, identical rows)
# or `qwencalib` -- and GSM8K has both too (data/<dataset>_<split>.json,
# common.CALIB_SPLITS). The fit artifacts carry _fit<split>, so the two
# pools and the plain train fit never resume into one another.
is_calib_split() { case "$1" in llamacalib|qwencalib|gemmacalib) return 0 ;; *) return 1 ;; esac; }
if [ "$DATASET" = math ]; then
  # The pool is fixed. A stale `export SPLITS=train` from another dataset's
  # session once resumed qwen3's retired train_standard (768 rows, "6732
  # remaining") instead of collecting calib -- so on MATH any other value is
  # an error, not a redirection. MATH_SPLITS_OVERRIDE=1 lifts it for a
  # deliberate one-off (collecting `test` for a reference ceiling, or the
  # full `train` a new calib split is defined from).
  for v in SPLITS TRAIN_SPLIT; do
    if [ -n "${!v:-}" ] && ! is_calib_split "${!v}" && [ -z "${MATH_SPLITS_OVERRIDE:-}" ]; then
      echo "$v=${!v} on DATASET=math: the MATH fitting pool is a fixed calib split (llamacalib | qwencalib)." >&2
      echo "Unset $v (it is probably exported from an earlier session), or set MATH_SPLITS_OVERRIDE=1 on purpose." >&2
      exit 1
    fi
  done
fi
# THE FITTING POOL, by family (2026-09-08). On MATH and GSM8K it is the fixed
# calib split chosen on THIS family's openers (data/<dataset>_<family>calib.json,
# the promoted cells' pools); SVAMP fits on its train split. `all` collects the
# pool only -- the test baseline is collected by `best`, right before it is
# steered -- so SPLITS defaults to the pool alone.
case "$MODEL_ID" in
  *[Ll]lama*) FAMILY_CALIB=llamacalib ;;
  *[Qq]wen*)  FAMILY_CALIB=qwencalib ;;
  *[Gg]emma*) FAMILY_CALIB=gemmacalib ;;
  *)          FAMILY_CALIB=llamacalib ;;
esac
case "$DATASET" in
  math|gsm8k) TRAIN_SPLIT=${TRAIN_SPLIT:-$FAMILY_CALIB} ;;
  *)          TRAIN_SPLIT=${TRAIN_SPLIT:-train} ;;
esac
SPLITS=${SPLITS:-$TRAIN_SPLIT}
MATH_FIT_TAG=$(is_calib_split "$TRAIN_SPLIT" && echo "$TRAIN_SPLIT" || echo "")
# A calib pool is fitted on single-pass labels (it is defined on them, and a
# formatter relabel would move its rows between classes). A plain train pool
# keeps the per-family default the cells were run under: llama single-pass,
# qwen3 / gemma-4 formatter.
if is_calib_split "$TRAIN_SPLIT"; then FIT_LABELS=${FIT_LABELS:-singlepass}
else case "$MODEL_ID" in
  *[Ll]lama*) FIT_LABELS=${FIT_LABELS:-singlepass} ;;
  *)          FIT_LABELS=${FIT_LABELS:-formatter} ;;
esac; fi
case "$FIT_LABELS$EVAL_GRADING" in
  singlepasssinglepass|singlepassformatter|formattersinglepass|formatterformatter) ;;
  *) echo "FIT_LABELS=$FIT_LABELS / EVAL_GRADING=$EVAL_GRADING must each be singlepass | formatter" >&2; exit 1 ;;
esac
# WHICH COLLECTED SPLIT IS THE FITTING POOL. Everything that chooses -- the
# scan hold-out, the elicitation target, the
# com directions, the scan -- reads $TRAIN_DIR, and $TRAIN_DIR is this split.
# It is a variable because the fitting pool is not always literally "train"
# (MATH: `llamacalib` / `qwencalib`, above). Must name a split that $SPLITS actually collected.
NUM_SAMPLES=${NUM_SAMPLES:-0}          # 0 = the entire split
SMOKE_NUM=${SMOKE_NUM:-60}             # rows for the `smoke` calibration run
SMOKE_MAX_NEW_TOKENS=${SMOKE_MAX_NEW_TOKENS:-2048}

# ---------------------------------------------------------------- SELECTION
K=${K:-8}
# K NAMES ITS OWN TREES. The steered files are named by heads tag and alpha
# (gen_HE30-31_alpha16_com.jsonl), not by K, and heads.sh / plan.sh hold one
# selection each -- so a K=4 run into the K=8 directories would overwrite the
# recorded selection and RESUME into the K=8 generations as if they were its
# own. Every K-dependent artifact (selection tree, scan, best, transfer
# outputs) therefore carries _k<K> for any K other than 8, the value every
# existing tree was made under. The target and the probe fit (hence the dev
# slice the scan runs on) do not depend on K and are shared, so scans at
# different K are compared on the identical rows.
K_SUFFIX=${K_SUFFIX-$([ "$K" = 8 ] && echo "" || echo "_k$K")}
# Candidate bands for `scan`, as LAYER INDICES OF THIS MODEL.
#
# There is no model-independent default here and there must not be one: 28-31 is
# the last four layers of a 32-layer llama and does not exist on a 28-layer qwen.
# A fixed default was silently applied across models once already and produced a
# head set led by L0H0 that looked like a result. So the default is DERIVED from
# this model's depth (the last 2/3/4/6 layers), and any explicit BANDS is
# validated against it before a single stage runs.
#
# The bands are deliberately nested: they vary only the floor, which is the one
# question the scan can answer -- how far back the direct-path score stays
# trustworthy. Add an unrestricted band as a control only if it selects
# DIFFERENT heads; where the top scores already cluster late it does not, and
# the elicit stage's heads.sh will show that.
#
# auto (the default) derives them from THIS model's elicitation-mass profile:
# find the elbow -- the lowest layer whose mass, and every layer above it, holds
# ELBOW_FRAC of the peak -- and emit elbow-last, (elbow+1)-last, ... last-last.
# See src/steering/operating_point.py (plan). An explicit list is still accepted and is
# validated against the model's depth.
BANDS=${BANDS:-auto}
ELBOW_FRAC=${ELBOW_FRAC:-0.35}
# The alpha range the dose ladder is built from, on the widest band. Every other
# band gets the alphas that land it on the SAME doses.
ANCHOR_ALPHAS=${ANCHOR_ALPHAS:-"8 32"}
N_ALPHA=${N_ALPHA:-5}
# The target is the first-token contrast between correct and incorrect rows.
# If the two classes open alike there is nothing to steer toward and every
# ranking below is noise, so this is a hard gate, not a warning.
MIN_TV=${MIN_TV:-0.2}
TARGET_N_PROMPTS=${TARGET_N_PROMPTS:-256}
TARGET_AGG=${TARGET_AGG:-argmax}       # argmax = the token greedy decoding would emit
# THE INJECTED VECTOR IS COM, ONLY: ITI's mass-mean direction, mean(correct) -
# mean(incorrect), ranked (elicit) and injected (scan / best / transfer) at
# each selected head. There is no other direction in this pipeline; com was
# always slug-free, so every existing path keeps its name. Any DIRECTION other
# than com is refused.
if [ -n "${DIRECTION:-${DIRECTION_MODE:-}}" ] && [ "${DIRECTION:-$DIRECTION_MODE}" != com ]; then
  echo "DIRECTION=${DIRECTION:-$DIRECTION_MODE} is not supported: the injected vector is com only. Unset it." >&2
  exit 1
fi
DIRECTION=com

# ---------------------------------------------------------------- STEERING
ALPHAS=${ALPHAS:-"8 12 16 20 26"}
# EARLY STOP ON THE LADDER (2026-08-27). Every scan so far has had one shape:
# accuracy climbs with alpha, peaks, then falls off a cliff (qwen3/MATH E34-35:
# 76.3 79.3 79.3 80.0 71.7; E35-35 ends at 46.0). The rungs past the peak
# are the most expensive ones -- the highest doses, the longest and loopiest
# generations -- and they only ever confirm the fall. So the scan runs a
# band's alphas in ASCENDING order, one at a time, and stops the band the
# first time an alpha's accuracy comes in below the best seen so far (by
# more than SCAN_DIP_TOL, a fraction: 0.01 = one point). Ties and
# plateaus continue; the first rung can never stop. Which alphas were not
# run is recorded in the band's early_stop.txt, so a missing rung is a
# decision, not a crash. SCAN_EARLY_STOP=0 restores the whole ladder.
# THE DOSE SCHEDULE over a generation is cosine, always (2026-09-03; the
# constant Eq. 2 schedule was removed): Fractional Reasoning's own ramp (their
# utils/llm_layers.py:9-12,34, the code baselines/fr.py runs),
# alpha * max(0.5*(cos(pi*step/50)+1), 0.5), where step counts decode passes
# per layer and the prefill resets it -- full dose for the first tokens, then
# a permanent HALF dose. Every config tag carries `_cos`; files without it are
# pre-2026-08-29 constant rungs, which report/pick ignore.
SCAN_EARLY_STOP=${SCAN_EARLY_STOP:-1}
SCAN_DIP_TOL=${SCAN_DIP_TOL:-0}
SCAN_ROWS=${SCAN_ROWS:-300}            # rows the scan runs on; see `scanrows` / SCAN_SPLIT
BEST_BAND=${BEST_BAND:-}               # `best` takes the scan's pick unless both are set
BEST_ALPHA=${BEST_ALPHA:-}
# Measured, not guessed: on llama/gsm8k the longest of ~1290 non-degenerate
# generations was 370 tokens, so 512 clips nothing and kills runaway loops 4x
# sooner than 2048. MATH solutions are longer -- raise it after reading `collect`.
# Recorded BEFORE the defaults land: `transfer` runs on eval-only benchmarks
# whose solutions run long (MATH-500, HARP), so it defaults both caps to
# 2048 -- but only when the user did not set them explicitly.
MNT_WAS_SET=${MAX_NEW_TOKENS+x}
STEER_MNT_WAS_SET=${STEER_MAX_NEW_TOKENS+x}
STEER_MAX_NEW_TOKENS=${STEER_MAX_NEW_TOKENS:-1024}

# ---------------------------------------------------------------- RUNTIME
DTYPE=${DTYPE:-bf16}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
SEED=${SEED:-42}
# Every machine path comes from data/paths.py (EMBED_ROOT, the dataset and
# model caches, the venvs). An exported EMBED_ROOT still wins, for a one-off
# redirect; a blank registry refuses here rather than resolving to "".
eval "$(python3 data/paths.py --sh)"
: "${EMBED_ROOT:?set EMBED_ROOT in data/paths.py}"

# ---------------------------------------------------------------- PATHS
case "$PROMPT_MODE" in
  direct) echo "PROMPT_MODE=direct is deprecated; use qa" >&2; exit 1 ;;
  standard) MODE_TAG=standard ;;
  cot) MODE_TAG=cot ;;                          # vendor CoT reference ceiling (`cot` stage)
  qa) MODE_TAG=qa ;;  qa_brief) MODE_TAG=qabrief ;;
  *) echo "unknown PROMPT_MODE=$PROMPT_MODE (standard | qa | qa_brief | cot)" >&2; exit 1 ;;
esac
# One name -> one tree slug, used for $DATASET here and for $TRANSFER_DATASET
# inside stage_transfer. The eval-only benchmarks (no train split, or
# train==test) are fine for `smoke` and `transfer`; a full `collect` on them
# needs SPLITS=test.
data_slug() {
  case "$1" in
    math) echo "" ;;  gsm8k) echo _gsm8k ;;
    gsm_plus) echo _gsmplus ;;  math500) echo _math500 ;;
    harp) echo _harp ;;  svamp) echo _svamp ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}
DATA_SLUG=$(data_slug "$DATASET") || exit 1

ROOT=$EMBED_ROOT/zs_${MODEL_TAG}${DATA_SLUG}_${SUBJECT}
TRAIN_DIR=$ROOT/${TRAIN_SPLIT}_${MODE_TAG}
TEST_DIR=$ROOT/test_${MODE_TAG}
# The target is direction-independent, so it lives in the unsuffixed directory
# and is computed once. Everything downstream of it depends on DIRECTION and is
# kept in a separate tree, so a probe run never overwrites a com run's heads.
#
# HEAD SELECTION IS THE OUTCOME CONTRAST, ONLY (2026-08-28). The token-space
# target u is p(first token | correct rows) - p(first token | incorrect rows),
# both under the one baseline prompt -- no reasoning prompt anywhere in the
# construction (src.steering.elicitation_target). The former `elicit`
# procedure (a cot-vs-qa prompt-pair contrast) was REMOVED: with the single
# per-family prompt its two sides were the same prompt and its target
# identically zero, and it ran two redundant forward passes to find that out.
# Every selection artifact keeps the outcome_* tree names it always had.
if [ -n "${HEAD_SELECTION:-}" ] && [ "$HEAD_SELECTION" != outcome ]; then
  echo "HEAD_SELECTION=$HEAD_SELECTION was removed on 2026-08-28: head selection is the outcome contrast only. Unset it." >&2
  exit 1
fi
HEAD_SELECTION=outcome
SEL_STEM=outcome; SEL_SUFFIX=_outcome
# FIT NAMESPACE. The fit artifacts below (probes, target, heads, scans, best,
# transfer outputs) are keyed by prompt mode + selection + direction, NOT by
# which split they were fitted on. Two fits of the same root -- e.g. MATH
# `calibration` and MATH `train` -- would therefore overwrite and
# resume-collide. FIT_TAG=<slug> suffixes every fit artifact with _fit<slug>
# so fits coexist; empty keeps every existing path. MATH defaults to `llamacalib`
# (the fitting pool), so its trees are outcome_standard_fitllamacalib etc.
FIT_TAG=${FIT_TAG-$MATH_FIT_TAG}
FIT_SUFFIX=${FIT_TAG:+_fit$FIT_TAG}
TARGET_DIR=$ROOT/${SEL_STEM}_${MODE_TAG}${FIT_SUFFIX}
ELICIT_DIR=$ROOT/${SEL_STEM}_${MODE_TAG}${FIT_SUFFIX}${K_SUFFIX}
TARGET_NPZ=$TARGET_DIR/elicit_target.npz
HEADS_FILE=$ELICIT_DIR/heads.sh
SCORES_NPZ=$ELICIT_DIR/elicit_scores.npz
PLAN_FILE=$ELICIT_DIR/plan.sh

# ---------------------------------------------------------------- BANDS
# How deep THIS model is. Everything band-shaped is derived from it, so a layer
# index can never mean one thing on one model and something else on another.
# get_text_config() first: on a multimodal wrapper (gemma-4) the flat
# config describes the WRAPPER and has no num_hidden_layers at all -- same
# reason common.num_attention_heads resolves through it.
NUM_LAYERS=$(python3 -c "
from pathlib import Path
from transformers import AutoConfig
from src.core import common
cfg = AutoConfig.from_pretrained('$MODEL_ID', cache_dir=common.model_cache_dir('$MODEL_ID'))
cfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
print(cfg.num_hidden_layers)")
LAST=$((NUM_LAYERS - 1))
# Reject a band this model does not have, here, before any stage runs -- the
# python rejects it too, but by then a detached scan has already been launched.
# "auto" is resolved from the score profile inside the elicit stage, so there is
# nothing to validate here and nothing model-specific to get wrong.
if [ "$BANDS" != auto ]; then
  for _b in $BANDS; do
    _lo=${_b%%-*}; _hi=${_b##*-}
    case "$_lo$_hi" in *[!0-9]*)
      echo "bad band '$_b': use lo-hi, e.g. 25-$LAST, or BANDS=auto" >&2; exit 1 ;; esac
    if [ "$_lo" -gt "$_hi" ] || [ "$_hi" -gt "$LAST" ]; then
      echo "band $_b is outside $MODEL_ID, which has $NUM_LAYERS layers (L0-L$LAST)." >&2
      echo "Bands are layer indices of THIS model. Use BANDS=auto to derive them." >&2
      exit 1
    fi
  done
fi
# The rows band and alpha are tuned on: held out of the probe fit, and disjoint
# from the test split by construction (they are TRAIN rows).
# SCAN_SPLIT (2026-08-28) names a SEPARATE collected split as the scan set
# instead of a dev slice of the fitting pool, so the whole pool is fitted on
# and band/alpha are tuned on rows outside it. MATH defaults to
# `calibration` (MATH test minus MATH-500, disjoint from the eval set):
# run_plan_grid collects SCAN_ROWS of it (the first rows of the seeded
# shuffle) before the first scan if the split is not on disk, and `scanrows`
# writes scan_rows.csv into it from its untruncated rows. Without a
# SCAN_SPLIT the scan rows are a stratified hold-out of the fitting pool
# ($TARGET_DIR/scan_rows.csv, carved before `elicit`, which then EXCLUDES
# them from the com fit).
#
# A CALIB POOL IS FITTED WHOLE (2026-08-30): on GSM8K a llamacalib/qwencalib
# pool scans on `train` rows OUTSIDE the 1,600 -- the rest of the (already
# collected) train split -- so the pool is never thinned to 1,300. `scanrows`
# excludes the pool's rows from the draw, and the csv is fit-specific
# (scan_rows_fit<split>.csv) because the excluded rows differ per pool.
if [ -z "${SCAN_SPLIT+x}" ]; then
  if [ "$DATASET" = math ]; then SCAN_SPLIT=calibration
  elif is_calib_split "$TRAIN_SPLIT"; then SCAN_SPLIT=train
  else SCAN_SPLIT=; fi
fi
if [ -n "$SCAN_SPLIT" ]; then
  SCAN_DIR=$ROOT/${SCAN_SPLIT}_${MODE_TAG}
  if [ "$SCAN_SPLIT" = train ] && is_calib_split "$TRAIN_SPLIT"; then
    SCAN_ROWS_CSV=$SCAN_DIR/scan_rows${FIT_SUFFIX}.csv
  else
    SCAN_ROWS_CSV=$SCAN_DIR/scan_rows.csv
  fi
else
  SCAN_DIR=$TRAIN_DIR
  SCAN_ROWS_CSV=$TARGET_DIR/scan_rows.csv
fi
# steering_driver builds its intervention vectors from a directions.npz. `elicit`
# writes one holding the com directions under the key --direction looks up.
# The file is a vector store, not a probe report.
DIRECTIONS_NPZ=$ELICIT_DIR/directions.npz
# Deliberately OUTSIDE $ROOT: a partial collection sitting at $TEST_DIR would
# make `collect` skip itself and the real run would silently inherit 60 rows.
# Which split the smoke samples. `test` (the default) keeps every existing
# smoke directory name; any other split gets its own suffix so a train smoke
# can never resume into -- or be mistaken for -- the test smoke.
SMOKE_SPLIT=${SMOKE_SPLIT:-test}
SMOKE_DIR=$EMBED_ROOT/_zs_smoke/${MODEL_TAG}${DATA_SLUG}_${MODE_TAG}
[ "$SMOKE_SPLIT" = test ] || SMOKE_DIR=${SMOKE_DIR}_${SMOKE_SPLIT}
# SMOKE_TAG names a FRESH smoke tree (<dir>_<tag>). A smoke tree resumes per
# row, so after a prompt edit the old tree must not be reused; set a tag (or
# move the old tree aside) -- check_smoke_prompt below refuses to resume a
# tree whose stored prompts no longer match the current builder.
[ -z "${SMOKE_TAG:-}" ] || SMOKE_DIR=${SMOKE_DIR}_${SMOKE_TAG}
SCAN_ROOT=$ROOT/scan_${MODE_TAG}${SEL_SUFFIX}${FIT_SUFFIX}${K_SUFFIX}
BEST_ROOT=$ROOT/best_${MODE_TAG}${SEL_SUFFIX}${FIT_SUFFIX}${K_SUFFIX}
STAMP_DIR=$ROOT/.zs

PY="python3 -u"

# The collection records which model wrote it. If that disagrees with MODEL_ID,
# MODEL_TAG is pointing at another experiment's tree -- stop, rather than fit
# probes on one model's activations and steer a different model with them.
check_root() {
  local args_json=$TRAIN_DIR/args.json owner
  [ -f "$args_json" ] || return 0        # nothing collected here yet
  owner=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('model_name',''))" \
          "$args_json")
  [ "$owner" = "$MODEL_ID" ] || {
    echo "MODEL_TAG=$MODEL_TAG resolves to $ROOT," >&2
    echo "but that collection was written by $owner, not $MODEL_ID." >&2
    echo "Fix MODEL_TAG/MODEL_ID; they must name the same experiment." >&2
    exit 1; }
}
run() { echo "+ $*"; [ -n "${DRY_RUN:-}" ] || "$@"; }
banner() { echo; echo "=============== $* ==============="; }
# DRY_RUN must not stamp. Without this guard a dry run marks target and elicit
# done without computing either, and the next real run skips them -- which is
# how a pipeline ends up steering along a target it never measured.
stamp() {
  [ -n "${DRY_RUN:-}" ] && return 0
  mkdir -p "$STAMP_DIR"; : > "$STAMP_DIR/$1.done"
}
done_if() { [ -z "${FORCE:-}" ] && [ -e "$2" ]; }
band_var() { echo "HEADS_${1//-/_}"; }
# Does a CSV already carry this column? Pure bash: `head | tr | grep -q` looks
# equivalent but grep -q exits on the first match, SIGPIPEs tr, and `set -o
# pipefail` then reports the whole test as false -- the column would be redrawn
# on every run. The \r strip is for CRLF headers.
header_has() {
  local header
  header=$(head -1 "$1" 2>/dev/null | tr -d '\r') || return 0
  case ",$header," in *",$2,"*) echo 1 ;; esac
}

# Is every split of $SPLITS fully collected? "Done" is the collector's own
# resume rule -- a (subject, idx) key counts only when its CSV row AND its
# hidden .npy both exist -- compared against the loader's row count for that
# split. Existence of the CSV is NOT completion: a single-split run creates
# the file at row one, and an existence gate once skipped a 30%-collected
# pool straight into format/balance. Costs a dataset load (CPU, seconds),
# never a model load. Never skips under DRY_RUN, so the dry run always shows
# the collect command it would run.
collect_complete() {
  [ -n "${DRY_RUN:-}" ] && return 1
  [ -n "${FORCE:-}" ] && return 1
  $PY - "$ROOT" "$MODE_TAG" "$DATASET" "$SUBJECT" "$NUM_SAMPLES" "$SEED" $SPLITS <<'PYEOF'
import csv, os, sys
from src.core import common
from src.core.io import hidden_path
root, mode_tag, dataset, subject, num, seed, *splits = sys.argv[1:]
complete = True
for split in splits:
    d = os.path.join(root, f"{split}_{mode_tag}")
    expected = len(common.load_examples(dataset, [subject], split,
                                        int(num) or None, int(seed)))
    done = 0
    csv_path = os.path.join(d, "ground_truth.csv")
    if os.path.exists(csv_path):
        hidden_dir = os.path.join(d, "hidden")
        seen = set()
        for row in csv.DictReader(open(csv_path, newline="", encoding="utf-8")):
            key = (row.get("subject"), row.get("idx"))
            if None in key or key in seen:
                continue
            if os.path.exists(hidden_path(hidden_dir, *key)):
                seen.add(key)
                done += 1
    print(f"[collect-gate] {split}: {done}/{expected} rows backed on disk")
    complete &= done >= expected
sys.exit(0 if complete else 1)
PYEOF
}

# A smoke tree RESUMES per row and the collector never looks at the prompt, so
# after a prompt edit a re-smoke into the old tree silently reports the OLD
# prompt's generations (this bit on 2026-08-27: llama's standard prompt lost
# "without explanations" and every llama smoke on disk still carried it).
# Rebuild the first stored row's prompt with the CURRENT builder and refuse on
# any difference. Tokenizer only -- CPU, seconds. Legacy modes the builder no
# longer knows (qa, qa_brief) count as a mismatch. Under DRY_RUN it reports
# and continues, so a dry run still shows the commands.
check_smoke_prompt() {
  [ -f "$SMOKE_DIR/records.jsonl" ] || return 0
  $PY - "$SMOKE_DIR" "$MODEL_ID" "${DRY_RUN:-}" <<'PYEOF'
import difflib, json, sys
from pathlib import Path
from transformers import AutoTokenizer
from src.core import common
d, model_id, dry = sys.argv[1:]
args = json.load(open(f"{d}/args.json", encoding="utf-8"))
with open(f"{d}/records.jsonl", encoding="utf-8") as f:
    first = f.readline().strip()
if not first:
    # A run that died before its first row leaves an EMPTY records.jsonl
    # (observed 2026-08-31: gemma-4's heterogeneous head widths failed every
    # batch). Nothing was collected under any prompt, so there is nothing to
    # compare -- resuming is a fresh start.
    print(f"[smoke-guard] {d}/records.jsonl is empty (no rows collected); "
          "nothing to compare, starting fresh")
    sys.exit(0)
rec = json.loads(first)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True,
                                    cache_dir=common.model_cache_dir(model_id))
now, why = None, ""
try:
    now = common.make_chat_prompt(tok, rec["problem"], args["prompt_mode"],
                                  dataset=args.get("dataset", ""))
except ValueError as e:
    why = str(e)
if now == rec["prompt"]:
    print(f"[smoke-guard] stored prompts match the current builder; resuming {d}")
    sys.exit(0)
print(f"[smoke-guard] {d}\n  was collected under a DIFFERENT prompt than the "
      "current builder produces:")
if now is None:
    print("  " + why)
else:
    sys.stdout.write("".join(difflib.unified_diff(
        rec["prompt"].splitlines(True), now.splitlines(True),
        "stored", "current", n=0)))
print("\n  Resuming would report the OLD prompt's generations. Either\n"
      f"    mv {d} {d}_stale\n"
      "  or name a fresh tree:\n"
      "    SMOKE_TAG=<tag> ./run_compass.sh smoke        # -> "
      f"{d}_<tag>")
sys.exit(0 if dry else 1)
PYEOF
}

# THE FITTING POOL MUST CARRY SINGLE-PASS LABELS WHEN IT IS FITTED ON
# (2026-08-27, paramount). Every selection artifact -- the balanced draw, the
# per-head probes, the outcome target, the com directions and the head
# ranking -- reads the pool's `correct` column as-is. Under
# FIT_LABELS=singlepass that column must be the frozen grader's, untouched by
# the LLM answer formatter: a formatter relabel that is live on the pool
# (`correct_singlepass` present and differing from `correct`) rotated com off
# the reasoning axis once already (llama, 2026-08-24). `format` reverts such a
# relabel, but nothing stopped a fitting stage from running before it did --
# so every fitting stage now refuses, rather than fit on the wrong labels.
# FIT_LABELS=formatter is the documented opposite policy (qwen) and is not
# checked here. CPU, seconds; skipped when nothing is collected yet.
check_fit_labels() {
  [ "$FIT_LABELS" = singlepass ] || return 0
  [ -f "$TRAIN_DIR/ground_truth.csv" ] || return 0
  $PY - "$TRAIN_DIR/ground_truth.csv" <<'PYEOF'
import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if "correct_singlepass" not in (reader.fieldnames or []):
        sys.exit(0)                      # never relabeled: labels are single-pass
    changed = sum(1 for r in reader
                  if str(r.get("correct_singlepass", "")).strip() not in ("",)
                  and str(r["correct"]).strip() != str(r["correct_singlepass"]).strip())
if changed:
    raise SystemExit(
        f"\n[fit-labels] {path}\n  carries a LIVE LLM-formatter relabel: {changed} row(s) have "
        "correct != correct_singlepass, but FIT_LABELS=singlepass.\n  Fitting (target / "
        "elicit) on formatter labels would select heads and a com\n  "
        "direction for the wrong quantity. Put the pool back first:\n"
        "    ./run_compass.sh format        # reverts correct <- correct_singlepass")
print(f"[fit-labels] {path}: formatter relabel present but fully reverted; single-pass labels in force")
PYEOF
}

# --------------------------------------------------------------- STAGES

# Calibrate MAX_NEW_TOKENS before paying for the full collection. Generates a
# few dozen rows at a DELIBERATELY GENEROUS cap, then prints what each smaller
# cap would have cost in correct answers. Smoking at the cap you intend to use
# tells you nothing -- you cannot see the answers it would have cut off.
stage_smoke() {
  banner "smoke: $SMOKE_NUM rows of $SMOKE_SPLIT at max_new_tokens=$SMOKE_MAX_NEW_TOKENS -> $SMOKE_DIR"
  check_smoke_prompt
  run $PY -m src.utils.collect.collection --mode collect \
    --model_name "$MODEL_ID" --dataset "$DATASET" \
    --prompt_mode "$PROMPT_MODE" \
    --subject "$SUBJECT" --split "$SMOKE_SPLIT" --num_samples "$SMOKE_NUM" \
    --out_dir "$SMOKE_DIR" --max_new_tokens "$SMOKE_MAX_NEW_TOKENS" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" --seed "$SEED" "$@"
  run $PY -m src.utils.diagnostics.check_collection "$SMOKE_DIR"
  # THE LOGIT-LENS CHECK, on every dataset and BEFORE the formatter: one
  # prompt-only forward per smoke row at the last prompt token gives
  # p(first token | correct) and p(first token | incorrect) under the stored
  # single-pass labels; their difference is delta_p, the outcome target that
  # `target` builds the head ranking from, and its
  # total variation is judged against the same MIN_TV floor check_target
  # enforces later. Runs first so the verdict is on the labels the llama
  # fitting policy (FIT_LABELS=singlepass) uses, untouched by any relabel.
  banner "smoke: first-token outcome contrast (logit lens, delta_p vs MIN_TV=$MIN_TV)"
  run $PY -m src.utils.diagnostics.first_token_delta "$SMOKE_DIR" \
    --batch_size "$BATCH_SIZE" --dtype "$DTYPE" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" --min_tv "$MIN_TV"
  # Datasets with an LLM formatting prompt (src.core.grading
  # FORMATTING_PROMPTS) also get the second-stage formatter by default: it
  # restates each generation as "Answer: \[ \boxed{} \]" with a second model
  # call, re-grades through the same frozen grader, and prints the formatted
  # accuracy plus the correct/incorrect length histogram. Stored labels are
  # untouched; verdicts land in $SMOKE_DIR/llm_formatted_answers.jsonl.
  if format_supported "$DATASET"; then
    banner "smoke: LLM answer formatter + formatted histogram"
    run $PY -m src.core.grading --collection "$SMOKE_DIR" \
      --batch_size "$BATCH_SIZE" --dtype "$DTYPE"
  fi
  echo
  echo "Pick the smallest cap that keeps ~100% of correct answers, then:"
  echo "  MAX_NEW_TOKENS=<cap> ./run_compass.sh collect --nohup"
  echo "Also read the accuracy line: if the qa prompt already elicits reasoning on"
  echo "this dataset there is little left to elicit, and the target will be small."
}

stage_collect() {
  banner "collect: $SPLITS x $PROMPT_MODE -> $ROOT"
  run $PY -m src.utils.collect.collection --mode sweep \
    --model_name "$MODEL_ID" --dataset "$DATASET" --sweep_prompt_modes "$PROMPT_MODE" \
    --sweep_splits $SPLITS \
    --subject "$SUBJECT" --num_samples "$NUM_SAMPLES" \
    --out_dir "$ROOT" --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" --seed "$SEED" "$@"
}

# The CoT REFERENCE CEILING (2026-08-25): the vendor's chain-of-thought eval
# prompt (common.COT_PROMPT_BUILDERS) on the full test split, so every
# steering delta can be read as "how much of the CoT ceiling was recovered".
# Collect-only: nothing is fitted on test_cot. Needs PROMPT_MODE=cot so the
# tree resolves to $ROOT/test_cot (the mode is part of every path); refuses
# to run under any other mode rather than overwrite a baseline tree. The
# generation cap defaults to 2048 (HARP's own eval setting) unless
# MAX_NEW_TOKENS was set explicitly. On datasets with a formatting prompt the
# LLM answer formatter is run in REPORT mode afterwards (labels untouched),
# so the ceiling is also available under the EVAL_GRADING=formatter
# instrument the headline numbers use.
stage_cot() {
  [ "$PROMPT_MODE" = cot ] || {
    echo "the cot stage needs PROMPT_MODE=cot (got $PROMPT_MODE): it collects \$ROOT/test_cot" >&2
    exit 1; }
  SPLITS=test
  [ -n "$MNT_WAS_SET" ] || MAX_NEW_TOKENS=2048
  banner "cot reference: $DATASET/test x cot (max_new_tokens=$MAX_NEW_TOKENS) -> $ROOT/test_cot"
  stage collect "$@"
  run $PY -m src.utils.diagnostics.check_collection "$ROOT/test_cot"
  if format_supported "$DATASET"; then
    banner "cot reference: LLM answer formatter (report only)"
    run $PY -m src.core.grading --collection "$ROOT/test_cot" \
      --batch_size "$BATCH_SIZE" --dtype "$DTYPE"
  fi
}

# Does this dataset have an LLM formatting prompt? Mirrors
# src.core.grading FORMATTING_PROMPTS (+ aliases).
format_supported() { case "$1" in math|math500|harp|gsm8k|gsm_plus|svamp) return 0 ;; *) return 1 ;; esac; }

# Relabel the fitting pool with the LLM answer formatter. Runs between
# collect and plan so every downstream chooser -- the outcome target, com --
# reads formatter-graded correctness. Resumable: rows already formatted cost
# nothing, and an idempotent re-apply is a no-op.
stage_format() {
  format_supported "$DATASET" || {
    echo "skip format ($DATASET has no formatting prompt; labels are single-pass)"; return 0; }
  case "$FIT_LABELS" in
    formatter)
      banner "format: LLM formatter relabels $TRAIN_DIR (fit labels = formatter verdicts)"
      # FORCE=1 forwards --force: needed when the collection already carries
      # a balanced* column from an earlier fit, since the apply guard
      # otherwise refuses to change labels under a live draw. FORCE=1 also
      # makes `balance` redraw afterwards, which is exactly the pairing.
      run $PY -m src.core.grading --collection "$TRAIN_DIR" \
        --batch_size "$BATCH_SIZE" --dtype "$DTYPE" ${FORCE:+--force} ;;
    singlepass)
      # The fitting pool must carry the frozen single-pass labels. A
      # collection relabeled under the formatter policy earlier is put back
      # (correct <- correct_singlepass); one never relabeled is a no-op.
      # CPU only. If labels actually change, target/elicit re-run (their
      # stamps hash the csv).
      banner "format: fit labels = single-pass; reverting any formatter relabel in $TRAIN_DIR"
      run $PY -m src.core.grading --collection "$TRAIN_DIR" --label_policy singlepass ;;
    *) echo "unknown FIT_LABELS=$FIT_LABELS (singlepass | formatter)" >&2; exit 1 ;;
  esac
}

# The rows band and alpha are tuned on, as a (subject, idx) CSV at
# $SCAN_ROWS_CSV: at most SCAN_ROWS untruncated rows of $SCAN_DIR, stratified
# by (subject, correct) and seeded. With a SCAN_SPLIT that is the separate
# scan collection (a split collected in full, e.g. qwen3's 4,500-row
# calibration, is sampled down); without one it is a hold-out of the fitting
# pool, which `elicit` then excludes from the com fit. Written once; a file
# already there is kept (the scan resumes against it).
stage_scanrows() {
  [ -n "${DRY_RUN:-}" ] && { echo "+ scanrows: $SCAN_ROWS of $SCAN_DIR -> $SCAN_ROWS_CSV"; return 0; }
  [ -f "$SCAN_ROWS_CSV" ] && { echo "scan rows already carved: $SCAN_ROWS_CSV"; return 0; }
  [ -f "$SCAN_DIR/ground_truth.csv" ] || {
    echo "no $SCAN_DIR/ground_truth.csv -- collect the scan split first:" >&2
    echo "  SPLITS=$SCAN_SPLIT NUM_SAMPLES=$SCAN_ROWS ./run_compass.sh collect" >&2; return 1; }
  mkdir -p "$(dirname "$SCAN_ROWS_CSV")"
  # Rows of the fitting pool are never scan rows: a no-op when the scan split
  # is disjoint from the pool (MATH calibration vs a train pool), the whole
  # point when the pool is a subset of the scan split (GSM8K calib vs train).
  $PY - "$SCAN_DIR" "$SCAN_ROWS_CSV" "$SCAN_ROWS" "$SEED" "$TRAIN_DIR" <<'PYEOF'
import os, sys
from src.core import io
d, out, n, seed, train = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
fit = set()
if os.path.isfile(os.path.join(train, "ground_truth.csv")) and os.path.abspath(train) != os.path.abspath(d):
    fit = {(r["subject"], r["idx"]) for r in io.load_ground_truth(train)}
pool = [r for r in io.load_ground_truth(d)
        if r["truncated"] == "0" and (r["subject"], r["idx"]) not in fit]
rows = io.stratified_sample(pool, n, seed + 1)
io.write_csv(out, [{"subject": r["subject"], "idx": r["idx"]} for r in rows],
             ["subject", "idx"])
print(f"scan rows: {len(rows)} of {len(pool)} untruncated rows of {d} "
      f"(excluding {len(fit)} fitting-pool rows) -> {out}")
PYEOF
}

stage_target() {
  check_fit_labels
  banner "target: first token of correct minus incorrect rows, both under $PROMPT_MODE"
  run mkdir -p "$TARGET_DIR"
  run $PY -m src.steering.elicitation_target \
    --exp1_collection "$TRAIN_DIR" --out "$TARGET_NPZ" \
    --model_name "$MODEL_ID" --dtype "$DTYPE" \
    --direct_mode "$PROMPT_MODE" \
    --agg "$TARGET_AGG" \
    --n_prompts "$TARGET_N_PROMPTS" --batch_size "$BATCH_SIZE" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" --seed "$SEED" "$@"
}

# The whole chain up to (but not including) the GPU scan, for any model on any
# dataset: measure the target, prove it is worth steering toward, rank the
# heads, derive the bands from this model's own profile, and print the exact
# scan commands with dose-matched alphas. Nothing here needs to be told a layer
# index or an alpha.
stage_plan() {
  stage_target
  check_target
  stage_elicit
  banner "plan: bands from the elbow, alphas matched on dose"
  run $PY -m src.steering.operating_point plan \
    --scores_npz "$SCORES_NPZ" --heads_file "$HEADS_FILE" \
    --collection "$TRAIN_DIR" --elbow_frac "$ELBOW_FRAC" \
    --anchor_alphas $ANCHOR_ALPHAS --n_alpha "$N_ALPHA" \
    --emit "$PLAN_FILE"
}

# plan -> every scan of the grid, sequentially -> report -> the best band's
# whole dose ladder on the FULL test split -> final report. One process, one
# GPU, nothing concurrent. The oversight structure is unchanged: this stage
# only automates the transcription a human would do between stages (run the
# printed scan commands; read the report; type BEST_BAND/BEST_ALPHA), and every
# intermediate artifact lands exactly where the manual stages put it, so the
# run can be inspected -- or stopped and continued by hand -- at any point.
# What it does NOT do is choose a single test config: it promotes the best
# band's full ladder (N_ALPHA runs, <=5), so the pick between alphas is still
# made by a human reading the final report.

# plan -> every band in the plan grid scanned sequentially on the dev rows ->
# report. The shared front half of `automate-all` and `automate-transfer`.
run_plan_grid() {
  stage_plan
  # The scan split (MATH: `calibration`), SCAN_ROWS of it, collected once.
  if [ -n "$SCAN_SPLIT" ] && [ -z "${DRY_RUN:-}" ] && [ ! -f "$SCAN_DIR/ground_truth.csv" ]; then
    banner "scan split: $SCAN_SPLIT x $PROMPT_MODE, $SCAN_ROWS rows -> $SCAN_DIR"
    SPLITS=$SCAN_SPLIT NUM_SAMPLES=$SCAN_ROWS stage_collect
  fi
  [ -n "${DRY_RUN:-}" ] || [ -f "$PLAN_FILE" ] || {
    echo "no $PLAN_FILE — plan did not emit the grid" >&2; return 1; }
  # shellcheck disable=SC1090
  # The dry-run placeholder must survive band_var's indirect expansion, so it
  # has to be a legal shell-identifier fragment (no <>).
  [ -n "${DRY_RUN:-}" ] && PLAN_BANDS="FROM-PLAN" || source "$PLAN_FILE"
  local band alphas var
  for band in $PLAN_BANDS; do
    if [ -n "${DRY_RUN:-}" ]; then alphas="<from-plan>"; else
      var="PLAN_ALPHAS_${band//-/_}"
      alphas=${!var:?$PLAN_FILE names band $band but holds no $var}
    fi
    banner "scan band $band  alphas {$alphas}  (dev rows, sequential)"
    BANDS=$band; ALPHAS=$alphas
    stage_scan
  done
  stage_report
}

# The operating point: the scan config with the highest dev-row accuracy,
# ties to the smaller alpha. Echoes "<band> <alpha>". One chooser --
# src.steering.operating_point pick -- read from the same gen files the
# report tabulates, so what this picks is what a reader of the report sees.
# Only configs covering the dev slice's ACTUAL row count compete (a small
# fitting split can cap it below $SCAN_ROWS).
pick_best_config() {
  $PY -m src.steering.operating_point pick \
    --scan_root "$SCAN_ROOT" --rows_csv "$SCAN_ROWS_CSV"
}

stage_automate_all() {
  run_plan_grid
  [ -n "${DRY_RUN:-}" ] && return 0
  local best_band var
  best_band=$(pick_best_config | cut -d' ' -f1)
  var="PLAN_ALPHAS_${best_band//-/_}"
  local best_alphas=${!var:?report picked band $best_band but $PLAN_FILE holds no $var}
  banner "automate-all: best band $best_band -> FULL test at each of {$best_alphas}"
  BEST_BAND=$best_band BEST_ALPHA=$best_alphas stage_best
  stage_report
}

# automate-all, but the chosen config is evaluated on TRANSFER_DATASET instead
# of on this dataset's own test split.
#
# WHY IT EXISTS. `best` steers $TEST_DIR -- this dataset's `test`. That is the
# right target only when `test` is disjoint from everything the selection was
# fitted on. It is NOT when the eval set is a SUBSET of test: fitting on MATH
# `calibration` and then reporting `best` on MATH `test` would report the
# headline number on 5000 rows of which 500 are the intended eval set and 4500
# are the fitting pool itself. This stage sends the same chosen config to
# `transfer` instead, which collects an unsteered baseline on the transfer set,
# PRINTS it, and steers only those rows.
#
#   TRANSFER_DATASET=math500 TRAIN_SPLIT=calibration SPLITS=calibration \
#     ./run_compass.sh automate-transfer
#
# Unlike automate-all, this promotes ONE (band, alpha) rather than the winning
# band's whole ladder: the transfer set is evaluated in full, so each extra
# alpha is another full pass over it. Set TRANSFER_ALPHA to override the
# scan's pick with a list and get the ladder back.
stage_automate_transfer() {
  : "${TRANSFER_DATASET:?set TRANSFER_DATASET (the dataset to steer, e.g. math500)}"
  run_plan_grid
  [ -n "${DRY_RUN:-}" ] && { TRANSFER_BAND=FROM-SCAN TRANSFER_ALPHA=0 stage_transfer; return 0; }
  local best band alpha
  best=$(pick_best_config)
  band=${best% *}; alpha=${best#* }
  # An explicit TRANSFER_BAND / TRANSFER_ALPHA still wins: the scan's pick is
  # the default, not a lock-in.
  band=${TRANSFER_BAND:-$band}; alpha=${TRANSFER_ALPHA:-$alpha}
  banner "automate-transfer: scan picked band $band alpha $alpha -> $TRANSFER_DATASET"
  TRANSFER_BAND=$band TRANSFER_ALPHA=$alpha stage_transfer
}

# The target is the first-token contrast between correct and incorrect rows.
# Below MIN_TV the two classes open alike, every head score below is noise,
# and a scan would burn GPU discovering that.
# Checked here rather than only inside the target stage, because that stage
# skips when it is already current and would then verify nothing.
check_target() {
  [ -n "${DRY_RUN:-}" ] && return 0
  $PY - "$TARGET_NPZ" "$MIN_TV" <<'PYEOF'
import sys
import numpy as np
path, min_tv = sys.argv[1], float(sys.argv[2])
z = np.load(path, allow_pickle=True)
tv = float(np.abs(z["delta_p"]).sum() / 2)
print(f"target total variation = {tv:.3f}  (floor {min_tv:g})   "
      f"correct vs incorrect first token under {str(z['direct_mode'])}")
if tv < min_tv:
    raise SystemExit(
        f"\nTOTAL VARIATION {tv:.3f} IS BELOW THE FLOOR {min_tv:g}.\n"
        "Correct and incorrect rows open on the same token under this prompt, "
        "so there is no outcome target to steer toward and the head ranking "
        "would be noise. The smoke's first-token contrast "
        "(src.utils.diagnostics.first_token_delta) shows this before collect; change "
        "the prompt or the dataset, or lower MIN_TV deliberately if you know "
        "why it is small.")
PYEOF
}

stage_elicit() {
  banner "elicit: top-$K per band {$BANDS}  direction=com"
  check_fit_labels
  [ -n "${DRY_RUN:-}" ] || [ -f "$TARGET_NPZ" ] || {
    echo "no $TARGET_NPZ — run target first" >&2; return 1; }
  run mkdir -p "$ELICIT_DIR"
  # com (mean(correct) - mean(incorrect) per head) is computed here, over
  # every labelled row of the pool. Without a SCAN_SPLIT the scan rows are a
  # hold-out of this very pool, so they are carved first and excluded from
  # the fit; with one they live in another split and nothing is excluded.
  local exclude=()
  if [ -z "$SCAN_SPLIT" ]; then
    stage_scanrows || return 1
    exclude=(--exclude_rows_file "$SCAN_ROWS_CSV")
  fi
  run $PY -m src.steering.head_ranker \
    --exp1_train_collection "$TRAIN_DIR" --exp4_train_collection "$TRAIN_DIR" \
    "${exclude[@]}" \
    --target_npz "$TARGET_NPZ" --model_name "$MODEL_ID" \
    --K "$K" --bands $BANDS \
    --elbow_frac "$ELBOW_FRAC" \
    --heads_out "$HEADS_FILE" --out_npz "$SCORES_NPZ" \
    --directions_out "$DIRECTIONS_NPZ" "$@"
}

# One steering sweep. Every band writes to its own directory, so a band can be
# added or re-run later without disturbing the others, and gen_*.jsonl resumes
# per row after a kill.
#
# WHICH ROWS. `scan` names the dev rows outright (--row_ids_file), so every band
# and every alpha is compared on the IDENTICAL problems and none of them is in
# the test split. `best` takes the whole test split. Tuning on test rows and then
# reporting on a superset of them, which is what a --subset of the test split
# does, makes the headline number optimistic by an unmeasurable amount.
sweep() {
  local band=$1 heads=$2 alphas=$3 rows_csv=$4 out=$5 mnt=$6 collection=$7 kind=${8:-eval}
  run mkdir -p "$out"
  banner "steer band $band  alpha={$alphas}  rows=${rows_csv:-full split} -> $out"
  run $PY -m src.steering.steering_driver --mode sweep \
    --model_name "$MODEL_ID" --dtype "$DTYPE" \
    --exp1_train_collection "$TRAIN_DIR" --exp1_test_collection "$collection" \
    --probe_report "$ELICIT_DIR" --out_dir "$out" \
    --heads $heads --heads_tag "$band" --K "$K" \
    --alpha $alphas --direction com \
    ${rows_csv:+--row_ids_file "$rows_csv"} \
    --subset 0 --seed "$SEED" --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$mnt" --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
    --steer_steps all
  # ONE grading instrument per purpose. steering_driver grades the steered rows
  # inline with the frozen single-pass grader. Whether a sweep is then
  # re-graded through the LLM formatter (and APPLIED: gen_*.jsonl `correct`
  # becomes the formatter verdict with correct_singlepass kept, summary_*.json
  # patched with *_singlepass originals, sweep_summary.csv rebuilt) depends on
  # what the sweep is FOR:
  #   kind=scan  selection on the fitting pool -> follows FIT_LABELS, so the
  #              steered rows are graded by the same instrument as the
  #              baseline_correct they are compared against;
  #   kind=eval  best / transfer -> follows EVAL_GRADING, whose baseline the
  #              transfer stage relabels with the same instrument first.
  # The formatter reads the dataset from the collection's args.json and skips
  # itself, announced, where no formatting prompt exists.
  local regrade=0
  case "$kind" in
    scan) [ "$FIT_LABELS" = formatter ] && regrade=1 ;;
    eval) [ "$EVAL_GRADING" = formatter ] && regrade=1 ;;
    *) echo "sweep: unknown kind '$kind' (scan | eval)" >&2; return 1 ;;
  esac
  # Only THIS call's configs (2026-09-08). The grading CLI takes every
  # gen_*.jsonl under --gen_dir by default, so a best/transfer re-run into a
  # directory already holding a ladder re-walked every rung: each a resumed
  # no-op (verdicts on disk) or a few newly-gradable rows, but a full report
  # block per rung and a confusing log. --gen_glob names the one file this
  # alpha wrote; printf %g spells alpha exactly as the driver's tag does
  # (8.0 -> 8, 16.88 -> 16.88).
  if [ "$regrade" = 1 ]; then
    local a gen
    for a in $alphas; do
      gen="gen_H${band}_alpha$(printf %g "$a")_com_cos.jsonl"
      banner "LLM-formatted regrade of $out/$gen (applied to gen file + summary)"
      run $PY -m src.core.grading --collection "$collection" \
        --gen_dir "$out" --gen_glob "$gen" --batch_size "$BATCH_SIZE" --dtype "$DTYPE"
    done
  fi
}

# The accuracy sweep_summary.csv holds for ONE (alpha, n) of a scan directory
# -- the same file and the same row filter pick_best_config uses, so what the
# early stop sees is what the report will show. Prints nothing and fails when
# the row is absent.
scan_accuracy() {
  local out=$1 alpha=$2 n_rows=$3
  $PY - "$out/sweep_summary.csv" "$alpha" "$n_rows" <<'PYEOF'
import csv, sys
path, alpha, n_rows = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
for row in csv.DictReader(open(path)):
    if row.get("random_heads") not in ("", "0", "False", None):
        continue
    # a pre-2026-08-29 constant rung at this alpha is a different intervention
    # (rows written before the column existed are constant): cosine only.
    if (row.get("schedule") or "constant") != "cosine":
        continue
    if int(float(row["n"])) != n_rows or abs(float(row["alpha"]) - alpha) > 1e-6:
        continue
    print(row["accuracy"]); break
else:
    sys.exit(1)
PYEOF
}

# One band's ladder, ascending, stopping at the first dip below the running
# peak (see SCAN_EARLY_STOP). Each rung is its own sweep() call, so per-row
# resume, the FIT_LABELS regrade and the cumulative sweep_summary.csv all
# behave exactly as they do for a whole-ladder call.
scan_band_early_stop() {
  local band=$1 heads=$2 alphas=$3 out=$4
  local n_scan=0
  [ -n "${DRY_RUN:-}" ] || n_scan=$(($(wc -l < "$SCAN_ROWS_CSV") - 1))
  local sorted; sorted=$(printf '%s\n' $alphas | sort -g | tr '\n' ' ')
  local best=-1 best_alpha= a acc ran=()
  echo "early stop on: band $band alphas {$sorted} run ascending; stop on the first dip below the peak (tol $SCAN_DIP_TOL)"
  for a in $sorted; do
    sweep "$band" "$heads" "$a" "$SCAN_ROWS_CSV" "$out" "$STEER_MAX_NEW_TOKENS" "$SCAN_DIR" scan
    ran+=("$a")
    [ -n "${DRY_RUN:-}" ] && continue
    acc=$(scan_accuracy "$out" "$a" "$n_scan") || {
      echo "band $band alpha $a: no n=$n_scan row in $out/sweep_summary.csv -- the sweep did not finish" >&2
      return 1; }
    if awk -v x="$acc" -v b="$best" -v t="$SCAN_DIP_TOL" 'BEGIN { exit !(x < b - t) }'; then
      local rest; rest=$(printf '%s\n' $sorted | awk -v a="$a" '$1 > a' | tr '\n' ' ')
      echo "band $band alpha $a: accuracy $acc < peak $best (alpha $best_alpha) -- stopping this band; not run: {$rest}"
      { echo "# written by run_compass.sh stage_scan (SCAN_EARLY_STOP=1, tol $SCAN_DIP_TOL)"
        echo "peak_alpha=$best_alpha"; echo "peak_accuracy=$best"
        echo "dip_alpha=$a"; echo "dip_accuracy=$acc"
        echo "ran=\"${ran[*]}\""; echo "not_run=\"$rest\""; } > "$out/early_stop.txt"
      return 0
    fi
    if awk -v x="$acc" -v b="$best" 'BEGIN { exit !(x > b) }'; then best=$acc; best_alpha=$a; fi
    echo "band $band alpha $a: accuracy $acc   (peak so far $best at alpha $best_alpha)"
  done
  [ -n "${DRY_RUN:-}" ] || echo "band $band: ladder exhausted without a dip (peak $best at alpha $best_alpha)"
}

stage_scan() {
  if [ -f "$HEADS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$HEADS_FILE"
    # The head list names layers, and a layer index means nothing without the
    # model it indexes. Refuse a heads.sh written for a different one.
    if [ -n "${HEADS_MODEL:-}" ] && [ "$HEADS_MODEL" != "$MODEL_ID" ]; then
      echo "$HEADS_FILE was written for $HEADS_MODEL, not $MODEL_ID." >&2
      echo "Re-run elicit for this model." >&2
      exit 1
    fi
  elif [ -z "${DRY_RUN:-}" ]; then
    echo "no $HEADS_FILE — run elicit first" >&2; return 1
  fi
  stage_scanrows || return 1
  # BANDS=auto was resolved by elicit against the score profile; the resolved
  # list is recorded in heads.sh, so the scan steers exactly the bands that were
  # ranked rather than re-deriving them and risking a different answer.
  local bands=$BANDS
  if [ "$bands" = auto ]; then
    bands=${HEADS_BANDS:-}
    [ -n "$bands" ] || { [ -n "${DRY_RUN:-}" ] && bands="<from-elicit>" || {
      echo "BANDS=auto but $HEADS_FILE records no HEADS_BANDS." >&2
      echo "Run: ./run_compass.sh elicit   (or ./run_compass.sh plan)" >&2
      exit 1; }; }
    echo "bands=auto -> $bands   (resolved by elicit)"
  fi
  for band in $bands; do
    local var; var=$(band_var "E$band")
    local heads=${!var:-}
    if [ -z "$heads" ]; then
      # NOT a skip. A band with no head list means heads.sh was written for a
      # DIFFERENT set of bands, so the sweep about to run is not the sweep that
      # was asked for. Skipping it exited 0 with an empty log that read as a
      # successful run; this stops instead and says what to do.
      [ -n "${DRY_RUN:-}" ] || {
        echo "$var is not in $HEADS_FILE." >&2
        echo "That file holds:" >&2
        sed -n 's/^\(HEADS_[A-Za-z0-9_]*\)=.*/  \1/p' "$HEADS_FILE" >&2
        echo "BANDS has changed since elicit ran. Re-run it, then scan:" >&2
        echo "  BANDS=\"$BANDS\" ./run_compass.sh elicit" >&2
        exit 1; }
      heads="<chosen-by-elicit>"
    fi
    if [ "$SCAN_EARLY_STOP" = 1 ]; then
      scan_band_early_stop "E$band" "$heads" "$ALPHAS" "$SCAN_ROOT/E$band"
    else
      sweep "E$band" "$heads" "$ALPHAS" "$SCAN_ROWS_CSV" "$SCAN_ROOT/E$band" \
            "$STEER_MAX_NEW_TOKENS" "$SCAN_DIR" scan
    fi
  done
}

stage_best() {
  # The operating point: the scan's pick unless both are set by hand (an
  # explicit BEST_BAND/BEST_ALPHA still wins; the pick is the default).
  if [ -z "$BEST_BAND" ] || [ -z "$BEST_ALPHA" ]; then
    if [ -n "${DRY_RUN:-}" ]; then BEST_BAND=FROM-SCAN; BEST_ALPHA=0; else
      read -r BEST_BAND BEST_ALPHA < <(pick_best_config) && [ -n "$BEST_ALPHA" ] || {
        echo "no operating point yet: run the scan first, or set BEST_BAND/BEST_ALPHA" >&2
        return 1; }
      echo "operating point from the scan: band $BEST_BAND alpha $BEST_ALPHA   (highest dev accuracy, ties to the smaller alpha)"
    fi
  fi
  [ -f "$HEADS_FILE" ] || { echo "no $HEADS_FILE — run elicit first" >&2; return 1; }
  # shellcheck disable=SC1090
  source "$HEADS_FILE"
  local var; var=$(band_var "E$BEST_BAND")
  [ -n "${!var:-}" ] || [ -n "${DRY_RUN:-}" ] || {
    echo "band $BEST_BAND was never scored (BANDS={$BANDS})" >&2; return 1; }
  # The unsteered baseline on the FULL test split, collected here (resumes
  # per row; a finished collection costs one pass over the manifest), then
  # relabeled under EVAL_GRADING so the baseline_correct stamped into every
  # steered row and the printed deltas are on the same instrument as the
  # steered rows, then its accuracy printed BEFORE any steering runs.
  banner "baseline: $DATASET/test x $PROMPT_MODE -> $TEST_DIR"
  run $PY -m src.utils.collect.collection --mode collect \
    --model_name "$MODEL_ID" --dataset "$DATASET" \
    --prompt_mode "$PROMPT_MODE" \
    --subject "$SUBJECT" --split test --num_samples 0 \
    --out_dir "$TEST_DIR" --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" --seed "$SEED"
  if [ "$EVAL_GRADING" = formatter ] && format_supported "$DATASET"; then
    banner "baseline: LLM formatter relabels $TEST_DIR (EVAL_GRADING=formatter)"
    run $PY -m src.core.grading --collection "$TEST_DIR" \
      --batch_size "$BATCH_SIZE" --dtype "$DTYPE"
  fi
  banner "baseline accuracy ($DATASET/test, unsteered)"
  run $PY -m src.utils.diagnostics.check_collection "$TEST_DIR"
  # The full TEST split, and the first time this configuration has seen it.
  sweep "E$BEST_BAND" "${!var:-<from-heads.sh>}" "$BEST_ALPHA" "" "$BEST_ROOT/E$BEST_BAND" \
        "$STEER_MAX_NEW_TOKENS" "$TEST_DIR" eval
}

# Steer a WHOLE OTHER dataset with THIS run's selection -- nothing re-fitted.
# The heads come from this run's heads.sh and the injected com vectors from its
# directions.npz, exactly as `best` uses them; the only new GPU work is (a) a
# baseline collection of the transfer dataset (made once, resumed per row) and
# (b) the steered generation. Use it when the transfer set is a harder test
# split of the SAME task (gsm_plus is perturbed gsm8k test), so the question is
# whether the source selection carries, not what a fresh selection would be --
# which is why band and alpha are required inputs, not searched here.
#
#   TRANSFER_DATASET=gsm_plus TRANSFER_BAND=30-31 TRANSFER_ALPHA=16 \
#     ./run_compass.sh transfer
#
# Run it under the SOURCE run's env (DATASET=gsm8k etc.), because every source
# path -- heads.sh, directions.npz, train collection -- resolves from it. The
# baseline collection lands in the transfer dataset's own tree
# (zs_<tag><tslug>_<subject>/test_<mode>), the steered rows in this run's
# transfer<tslug>_* tree. TRANSFER_ALPHA may list several alphas.
stage_transfer() {
  : "${TRANSFER_DATASET:?set TRANSFER_DATASET (the dataset to steer, e.g. gsm_plus)}"
  [ "$TRANSFER_DATASET" != "$DATASET" ] || {
    echo "TRANSFER_DATASET=$TRANSFER_DATASET is this run's own dataset; use scan/best" >&2
    return 1; }
  # The operating point: the scan's pick unless both are set by hand, exactly
  # as `best` resolves it (TRANSFER_ALPHA may list several alphas).
  if [ -z "${TRANSFER_BAND:-}" ] || [ -z "${TRANSFER_ALPHA:-}" ]; then
    if [ -n "${DRY_RUN:-}" ]; then TRANSFER_BAND=FROM-SCAN; TRANSFER_ALPHA=0; else
      read -r TRANSFER_BAND TRANSFER_ALPHA < <(pick_best_config) && [ -n "$TRANSFER_ALPHA" ] || {
        echo "no operating point yet: run the scan first, or set TRANSFER_BAND/TRANSFER_ALPHA" >&2
        return 1; }
      echo "operating point from the scan: band $TRANSFER_BAND alpha $TRANSFER_ALPHA   (highest dev accuracy, ties to the smaller alpha)"
    fi
  fi
  local tslug; tslug=$(data_slug "$TRANSFER_DATASET") || return 1
  # Transfer defaults to a 2048 cap for BOTH the baseline collection and the
  # steered generation: the transfer sets are the report's headline numbers,
  # and clipping answers there costs accuracy invisibly. An explicit
  # MAX_NEW_TOKENS / STEER_MAX_NEW_TOKENS still wins.
  local tmnt=$MAX_NEW_TOKENS tsmnt=$STEER_MAX_NEW_TOKENS
  [ -n "$MNT_WAS_SET" ] || tmnt=2048
  [ -n "$STEER_MNT_WAS_SET" ] || tsmnt=2048

  # The source selection must already be recorded (heads.sh + directions.npz,
  # written by `elicit` -- the logit-lens ranking). Not run implicitly: which
  # bands it ranks is a source-run decision, not a transfer-time side effect.
  [ -n "${DRY_RUN:-}" ] || [ -f "$HEADS_FILE" ] || {
    echo "no $HEADS_FILE — record the selection first:" >&2
    echo "  BANDS=\"$TRANSFER_BAND\" ./run_compass.sh plan" >&2; return 1; }
  # shellcheck disable=SC1090
  [ -f "$HEADS_FILE" ] && source "$HEADS_FILE"
  if [ -n "${HEADS_MODEL:-}" ] && [ "$HEADS_MODEL" != "$MODEL_ID" ]; then
    echo "$HEADS_FILE was written for $HEADS_MODEL, not $MODEL_ID." >&2; return 1
  fi
  local var; var=$(band_var "E$TRANSFER_BAND")
  local heads=${!var:-}
  [ -n "$heads" ] || {
    [ -n "${DRY_RUN:-}" ] || {
      echo "band $TRANSFER_BAND is not in $HEADS_FILE (bands: ${HEADS_BANDS:-?})." >&2
      echo "Re-run elicit with BANDS including it." >&2; return 1; }
    heads="<chosen-by-elicit>"; }

  # Baseline generation for every transfer row, same collector as `collect`,
  # in the transfer dataset's own tree. Resumes per row; a finished collection
  # costs one pass over the manifest. The eval-only loaders serve `test` from
  # their single split where needed and announce it.
  local troot=$EMBED_ROOT/zs_${MODEL_TAG}${tslug}_${SUBJECT}
  local tdir=$troot/test_${MODE_TAG}
  banner "transfer baseline: $TRANSFER_DATASET/test x $PROMPT_MODE -> $tdir"
  run $PY -m src.utils.collect.collection --mode collect \
    --model_name "$MODEL_ID" --dataset "$TRANSFER_DATASET" \
    --prompt_mode "$PROMPT_MODE" \
    --subject "$SUBJECT" --split test --num_samples 0 \
    --out_dir "$tdir" --max_new_tokens "$tmnt" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" --seed "$SEED"
  # Under EVAL_GRADING=formatter the transfer baseline is relabeled by the formatter
  # BEFORE steering, so the baseline the deltas are read against -- and the
  # baseline_correct field stamped into every steered row -- are
  # formatter-graded, matching how the selection itself was fitted.
  if [ "$EVAL_GRADING" = formatter ] && format_supported "$TRANSFER_DATASET"; then
    banner "transfer baseline: LLM formatter relabels $tdir (EVAL_GRADING=formatter)"
    run $PY -m src.core.grading --collection "$tdir" \
      --batch_size "$BATCH_SIZE" --dtype "$DTYPE"
  fi
  # The unsteered accuracy, printed BEFORE any steering runs: this is the
  # number every steered delta below is read against.
  banner "transfer baseline accuracy ($TRANSFER_DATASET, unsteered)"
  run $PY -m src.utils.diagnostics.check_collection "$tdir"

  local out=$ROOT/transfer${tslug}_${MODE_TAG}${SEL_SUFFIX}${FIT_SUFFIX}${K_SUFFIX}/E$TRANSFER_BAND
  sweep "E$TRANSFER_BAND" "$heads" "$TRANSFER_ALPHA" "" "$out" \
        "$tsmnt" "$tdir" eval
  # (the steered rows' formatter regrade now happens inside sweep(), applied)
  echo
  echo "inspect with: python3 -m src.utils.diagnostics.check_collection $out"
}

stage_report() {
  run $PY -m src.steering.operating_point report \
    --scan_root "$SCAN_ROOT" --best_root "$BEST_ROOT" --heads_file "$HEADS_FILE" \
    --rows_csv "$SCAN_ROWS_CSV"
}

# --------------------------------------------------------------- DISPATCH
stage() {
  local name=$1; shift
  case "$name" in
    smoke)   stage_smoke "$@" ;;
    collect) collect_complete \
               && { echo "skip collect (every split fully on disk)"; return 0; }
             stage_collect "$@"; stamp collect ;;
    balance|probes)
             echo "the $name stage was removed on 2026-08-28: the per-head probe fit and its" >&2
             echo "50/50 draw fed nothing downstream. com is computed inside elicit; the scan" >&2
             echo "rows come from scanrows (SCAN_SPLIT, or a hold-out of the pool)." >&2; return 1 ;;
    scanrows) stage_scanrows ;;
             # No done_if gate on these two. Their skip check lives in the
             # python, keyed on the IDENTITY of the inputs (resume_stamp): the
             # collection, the scan hold-out, the direction, and — for target —
             # a fingerprint of the prompt string itself. A bash test for
             # "the output file exists" cannot see any of that, and having both
             # would mean the weaker one decides.
    target)  stage_target "$@" ;;
    elicit)  stage_elicit "$@" ;;
             # everything up to the GPU scan, in one command
    plan)    stage_plan ;;
    scan)    stage_scan "$@" ;;      # per-config resume lives in steering_driver
    best)    stage_best "$@" ;;
             # this run's heads + com direction on ANOTHER dataset's full split
    transfer) stage_transfer "$@" ;;
    report)  stage_report ;;         # read-only, seconds: always run
             # the scan's (band, alpha) on ONE line, for scripts that build a
             # ladder or a control arm around the pick: `... pick | tail -1`
    pick)    pick_best_config ;;     # the operating point, one line: "<band> <alpha>"
             # `best` is NOT in `all`: it is the one run that touches the test
             # split, and which config it runs is a decision made by reading the
             # report, not by the pipeline.
             # put the fitting pool's labels under the FIT_LABELS policy
             # (singlepass: revert any formatter relabel; formatter: apply)
    format)  stage_format "$@" ;;
             # the vendor CoT prompt on the full test split: the reference
             # ceiling every steering delta is read against (PROMPT_MODE=cot)
    cot)     stage_cot "$@" ;;
    all)     stage collect; stage format; stage plan ;;
             # plan -> all scans sequentially -> report, and STOP. The
             # selection (heads.sh, plan.sh) and the dev-row scan are on disk;
             # nothing touches test or a transfer set. Read the report, then
             # `best` / `transfer` with the (band, alpha) it names.
    automate-scan) run_plan_grid
             [ -n "${DRY_RUN:-}" ] || {
               echo; echo "scan picked: band/alpha = $(pick_best_config)   (highest dev accuracy, ties to the smaller alpha)"
               echo "next: TRANSFER_DATASET=<set> TRANSFER_BAND=<band> TRANSFER_ALPHA=<alpha> ./run_compass.sh transfer"
               echo "  or: BEST_BAND=<band> BEST_ALPHA=<alpha> ./run_compass.sh best"; } ;;
             # plan -> all scans sequentially -> report -> best band's ladder
             # on full test -> report. Assumes collect/format are done
             # (or run `all` first). The alpha pick still belongs to a human.
    automate-all) stage_automate_all ;;
             # same, but the chosen config is evaluated on TRANSFER_DATASET --
             # for when this dataset's `test` is not disjoint from the pool the
             # selection was fitted on (MATH calibration -> MATH-500)
    automate-transfer) stage_automate_transfer ;;
    *) echo "unknown stage: $name" >&2
       echo "one of: smoke collect format scanrows target elicit plan scan report pick best transfer cot all automate-scan automate-all automate-transfer" >&2
       exit 1 ;;
  esac
}

STAGE=${1:-}
[ -n "$STAGE" ] || {
  echo "usage: ./run_compass.sh <stage> [--nohup]" >&2
  echo "  stage: smoke collect format scanrows plan scan report best transfer cot all" >&2
  echo "         automate-scan automate-all automate-transfer" >&2
  echo "         (plan = target + elicit + the band/alpha grid; also: target elicit)" >&2
  exit 1; }
shift

if [ "${1:-}" = --nohup ]; then
  shift
  mkdir -p logs
  log=logs/zs_${MODEL_TAG}${DATA_SLUG}_${MODE_TAG}${K_SUFFIX}_${STAGE}_$(date +%Y%m%d_%H%M%S).log
  nohup "$0" "$STAGE" "$@" > "$log" 2>&1 &
  echo "detached -> $log  (pid $!)"
  exit 0
fi

echo "model    $MODEL_ID  ($MODEL_TAG)"
echo "data     $DATASET / $SUBJECT / $PROMPT_MODE   selection: outcome (correct - incorrect first token)"
echo "root     $ROOT"
echo "K=$K  bands={$BANDS}  alphas={$ALPHAS}  scan rows=$SCAN_ROWS (${SCAN_SPLIT:-train dev slice})  early stop=$SCAN_EARLY_STOP (tol $SCAN_DIP_TOL)"
echo "direction com   (mean(correct) - mean(incorrect); ranked and injected)   schedule=cosine"
[ -n "${DRY_RUN:-}" ] && echo "(DRY_RUN: printing commands only)"
check_root
stage "$STAGE" "$@"
