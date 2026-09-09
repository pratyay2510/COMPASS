#!/bin/bash
# =============================================================================
# Ablation studies at the PROMOTED operating point (calib-split fit + cosine
# schedule, the finalized recipe). Each ablation varies exactly one thing;
# everything else -- band, alpha, target, com, rows, caps, grader -- is the
# headline cell's own. The (band, alpha) are read at run time from the
# promoted full-split tree (the best cosine rung of its sweep_summary.csv,
# the reported number), never typed in.
#
#   ./ablations.sh kheads llama_gsm8k     # K in {4 2 1}, band/alpha fixed
#   ./ablations.sh kheads qwen_math       #   "        ", eval on MATH-500
#   ./ablations.sh heads  llama_math      # head-choice controls, 3 reps each
#   ./ablations.sh heads  qwen_gsm8k
#   ./ablations.sh anticom llama_gsm8k    # the promoted cell run BACKWARDS
#   ./ablations.sh anticom qwen_math      #   (suppression; see below)
#   ./ablations.sh transfer llama_gsm8k gsm_plus   # promoted config, frozen,
#   ./ablations.sh transfer llama_math  harp       #   on another full split
#
# kheads -- number-of-heads ablation. Per K: `elicit` writes the top-K
#   selection into K's own _k<K> tree (target / com / scan rows are
#   K-independent and shared), then one full-split generation at the fixed
#   (band, alpha). The K=8 row is the headline run itself: read, never re-run.
#     KS="4 2" ./ablations.sh kheads llama_gsm8k
#
# heads -- head-CHOICE controls in the best band, REPS seeded repeats each:
#     randheads_r<n>  com direction @ K heads drawn at random FROM INSIDE THE
#                     SAME BAND -- does the attribution RANKING matter? The
#                     draw is NOT forced disjoint from the selected set (an
#                     overlap biases the control UP, i.e. against us); each
#                     draw prints its overlap.
#     randdir_r<n>    iid N(0,I) direction @ the K selected heads, same sigma
#                     rescale -- does the DIRECTION matter? Seeded per rep
#                     (an earlier job card reused one seed, so its reps were
#                     one draw three times; fixed here).
#   Steered rows are LLM-formatter re-graded and applied, like best/transfer.
#     REPS=3 ./ablations.sh heads qwen_gsm8k
#
# anticom -- the SUPPRESSION arm: the promoted (band, K=8 heads, dose, cosine
#   schedule, full split, grader) with the injection SIGN flipped, i.e. steered
#   along mean(incorrect)-mean(correct). One full-split pass into
#   <promoted>/anticom; the headline sweep is not written to.
#
# BAND=.. ALPHA=.. override the promoted-config lookup. One GPU, hours (one
# full-split pass per K / per rep). Every step resumes per row; rerun as-is
# after a kill. Never run two ablations of one cell at once, and not while
# anything else writes the same cell's trees (no lock).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CMD=${1:?usage: ./ablations.sh <kheads|heads|anticom|transfer> <cell> [transfer dataset]}
CELL=${2:?usage: ./ablations.sh <kheads|heads|anticom|transfer> <llama_gsm8k|llama_math|qwen_gsm8k|qwen_math> [transfer dataset]}
TDS=${3:-}
# every machine path (EMBED_ROOT, the venvs) comes from data/paths.py
eval "$(python3 data/paths.py --sh)"
: "${EMBED_ROOT:?set EMBED_ROOT in data/paths.py}"

# Stale exports from another session silently redirect every path.
unset SPLITS FIT_TAG SCAN_SPLIT NUM_SAMPLES PROMPT_MODE BANDS ALPHAS \
      BEST_BAND BEST_ALPHA TRANSFER_DATASET TRANSFER_BAND TRANSFER_ALPHA K K_SUFFIX

# The promoted cell's env, verbatim from its headline run. EVAL names the
# headline eval: best (own test split) | transfer (MATH-500's test split).
case "$CELL" in
  llama_gsm8k|llama_math)
    source "${VENV_LLAMA:?set VENV_LLAMA in data/paths.py}/bin/activate"
    export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MODEL_TAG=llama TRAIN_SPLIT=llamacalib ;;
  qwen_gsm8k|qwen_math)
    source "${VENV_QWEN3:?set VENV_QWEN3 in data/paths.py}/bin/activate"
    export MODEL_ID=Qwen/Qwen3-4B MODEL_TAG=qwen3_4b TRAIN_SPLIT=qwencalib FIT_LABELS=singlepass ;;
  *) echo "unknown cell '$CELL' (llama_gsm8k | llama_math | qwen_gsm8k | qwen_math)" >&2; exit 1 ;;
esac
case "$CELL" in
  *_gsm8k) export DATASET=gsm8k MAX_NEW_TOKENS=512  STEER_MAX_NEW_TOKENS=512;  EVAL=best ;;
  *_math)  export DATASET=math  MAX_NEW_TOKENS=2048 STEER_MAX_NEW_TOKENS=2048; EVAL=transfer ;;
esac

# The cell's trees. ELICIT holds the K=8 selection (heads.sh, directions.npz);
# TEST is the full split the headline number is reported on.
case "$DATASET" in gsm8k) SLUG=_gsm8k ;; math) SLUG= ;; esac
ROOT=$EMBED_ROOT/zs_${MODEL_TAG}${SLUG}_all
POOL=$ROOT/${TRAIN_SPLIT}_standard
ELICIT=$ROOT/outcome_standard_fit${TRAIN_SPLIT}
if [ "$EVAL" = best ]; then
  TEST=$ROOT/test_standard
  PROMOTED=$ROOT/best_standard_outcome_fit${TRAIN_SPLIT}
else
  TEST=$EMBED_ROOT/zs_${MODEL_TAG}_math500_all/test_standard
  PROMOTED=$ROOT/transfer_math500_standard_outcome_fit${TRAIN_SPLIT}
fi

# The headline operating point: the highest-accuracy cosine, non-random-heads
# rung of the promoted full-split tree. Echoes "<band> <alpha> <accuracy>".
promoted_config() {
  local csv
  csv=$(ls "$PROMOTED"/E*/sweep_summary.csv 2>/dev/null) || {
    echo "no sweep_summary.csv under $PROMOTED/E* -- the headline run is not on disk" >&2; return 1; }
  [ "$(echo "$csv" | wc -l)" = 1 ] || {
    echo "several bands under $PROMOTED -- set BAND/ALPHA explicitly:" >&2
    echo "$csv" >&2; return 1; }
  awk -F, -v band="$(basename "$(dirname "$csv")" | cut -c2-)" '
    NR > 1 && $8 == "cosine" && ($13 == "" || $13 == "0") && $17 > best { best = $17; alpha = $3 }
    END { if (alpha == "") exit 1; print band, alpha, best }' "$csv"
}

resolve_config() {
  local band alpha acc
  read -r band alpha acc < <(promoted_config)
  BAND=${BAND:-$band}; ALPHA=${ALPHA:-$alpha}
  echo "cell $CELL: promoted config band=$BAND alpha=$ALPHA (full-split accuracy $acc) -- held fixed"
}

# ------------------------------------------------------------------- kheads
kheads() {
  local KS=${KS:-"4 2 1"} k csv row
  resolve_config
  for k in $KS; do
    echo; echo "########## $CELL kheads K=$k  $(date) ##########"
    K=$k BANDS=$BAND ./run_compass.sh elicit
    case "$EVAL" in
      best)     K=$k BANDS=$BAND BEST_BAND=$BAND BEST_ALPHA=$ALPHA ./run_compass.sh best ;;
      transfer) K=$k BANDS=$BAND TRANSFER_DATASET=math500 \
                    TRANSFER_BAND=$BAND TRANSFER_ALPHA=$ALPHA ./run_compass.sh transfer ;;
    esac
  done
  echo; echo "=============== kheads $CELL: accuracy at band=$BAND alpha=$ALPHA ==============="
  for k in 8 $KS; do
    csv=$PROMOTED$([ "$k" = 8 ] || echo "_k$k")/E$BAND/sweep_summary.csv
    row=$(awk -F, -v a="$ALPHA" '
      NR > 1 && $8 == "cosine" && ($13 == "" || $13 == "0") && ($3+0) == (a+0) { print $17, "(n=" $14 ")" }' \
      "$csv" 2>/dev/null | tail -1)
    echo "  K=$k  ${row:-MISSING ($csv)}"
  done
}

# -------------------------------------------------------------------- heads
# One arm at the promoted operating point: $1 out dir, $2 direction, $3 seed,
# $4 alpha, $5.. heads.
run_arm() {
  local out=$1 dir=$2 seed=$3 alpha=$4; shift 4
  echo; echo "=============== $CELL $(basename "$out")  direction=$dir  alpha=$alpha  seed=$seed  heads: $* ==============="
  python3 -u -m src.steering.steering_driver --mode sweep \
    --model_name "$MODEL_ID" --dtype bf16 \
    --exp1_train_collection "$POOL" --exp1_test_collection "$TEST" \
    --probe_report "$ELICIT" --out_dir "$out" \
    --heads "$@" --heads_tag "E$BAND" --K 8 --alpha "$alpha" --direction "$dir" \
    --subset 0 --seed "$seed" --batch_size 32 \
    --max_new_tokens "$STEER_MAX_NEW_TOKENS" --max_prompt_tokens 2048 \
    --steer_steps all
  # same instrument as best/transfer (EVAL_GRADING=formatter): re-grade + apply
  python3 -u -m src.core.grading --collection "$TEST" \
    --gen_dir "$out" --batch_size 32 --dtype bf16
}

heads() {
  local REPS=${REPS:-3} r var heads draw
  resolve_config
  # shellcheck disable=SC1091
  source "$ELICIT/heads.sh"
  [ "$HEADS_MODEL" = "$MODEL_ID" ] || { echo "$ELICIT/heads.sh is for $HEADS_MODEL, not $MODEL_ID" >&2; exit 1; }
  var="HEADS_E${BAND//-/_}"; heads=${!var:?no $var in $ELICIT/heads.sh}
  [ -f "$TEST/ground_truth.csv" ] || { echo "no baseline at $TEST (run best/transfer first)" >&2; exit 1; }
  echo "selected heads (E$BAND): $heads"
  echo "controls -> $PROMOTED/controls  reps=$REPS"
  # arm 1: com direction @ heads drawn at random from inside the band
  for r in $(seq 1 "$REPS"); do
    draw=$(python3 - "$BAND" "$r" "$heads" <<'PY'
import sys, numpy as np
band, rep, selected = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
lo, hi = (int(x) for x in band.split("-"))
NH = 32                                     # heads per layer, both models
pool = [f"L{l}H{h}" for l in range(lo, hi + 1) for h in range(NH)]
rng = np.random.default_rng(20260828 + rep)
draw = list(rng.choice(pool, size=len(selected), replace=False))
print(" ".join(draw))
print(f"draw {rep}: {len(pool)} candidates in band {band}, overlap with selected = "
      f"{len(set(draw) & set(selected))}/{len(selected)}", file=sys.stderr)
PY
)
    run_arm "$PROMOTED/controls/randheads_r$r" com 42 "$ALPHA" $draw
  done
  # arm 2: random direction @ the selected heads, one direction PER REP
  for r in $(seq 1 "$REPS"); do
    run_arm "$PROMOTED/controls/randdir_r$r" random $((42 + r)) "$ALPHA" $heads
  done
  echo; echo "=============== heads $CELL: controls at band=$BAND alpha=$ALPHA ==============="
  echo "  real arm: $(awk -F, -v a="$ALPHA" 'NR>1 && $8=="cosine" && ($3+0)==(a+0) {print $17}' \
                     "$PROMOTED/E$BAND/sweep_summary.csv" | tail -1)"
  for r in $(seq 1 "$REPS"); do
    for arm in randheads randdir; do
      echo "  ${arm}_r$r: $(awk -F, 'NR>1 && $8=="cosine" {print $17}' \
        "$PROMOTED/controls/${arm}_r$r/sweep_summary.csv" 2>/dev/null | tail -1)"
    done
  done
}

# ------------------------------------------------------------------ anticom
# SUPPRESSION: the promoted cell run backwards. Same band, same K=8 heads,
# same com direction, same dose, same cosine schedule, same full split, same
# grader -- only the SIGN of the injection is flipped, from
# mean(correct)-mean(incorrect) to mean(incorrect)-mean(correct).
#
# --direction anticom is -com at the same (positive) alpha; the tag carries the
# direction (HE<band>_alpha<a>_anticom_cos), so it cannot collide with, or
# resume into, the promoted rung's files.
#
# Its own tree ($PROMOTED/anticom), so the headline sweep_summary.csv that
# promoted_config() reads is untouched.
anticom() {
  local var heads out
  resolve_config
  # shellcheck disable=SC1091
  source "$ELICIT/heads.sh"
  [ "$HEADS_MODEL" = "$MODEL_ID" ] || { echo "$ELICIT/heads.sh is for $HEADS_MODEL, not $MODEL_ID" >&2; exit 1; }
  var="HEADS_E${BAND//-/_}"; heads=${!var:?no $var in $ELICIT/heads.sh}
  [ -f "$TEST/ground_truth.csv" ] || { echo "no baseline at $TEST (run best/transfer first)" >&2; exit 1; }
  out=$PROMOTED/anticom
  echo "selected heads (E$BAND): $heads"
  echo "anti-com -> $out   (direction anticom at alpha $ALPHA, the promoted dose reversed)"
  run_arm "$out" anticom 42 "$ALPHA" $heads
  echo; echo "=============== anticom $CELL: band=$BAND dose=$ALPHA ==============="
  awk -F, -v a="$ALPHA" 'NR>1 && $8=="cosine" && ($3+0)==(a+0) {
    printf "  baseline      : %s (n=%s)\n  +com  (alpha %s): %s\n", $18, $14, a, $17 }' \
    "$PROMOTED/E$BAND/sweep_summary.csv" | tail -2
  awk -F, -v a="$ALPHA" 'NR>1 && $8=="cosine" && ($3+0)==(a+0) {
    printf "  anticom (alpha %s): %s   truncation %s\n", a, $17, $20 }' \
    "$out/sweep_summary.csv" 2>/dev/null | tail -1
}

# ----------------------------------------------------------------- transfer
# The promoted config, frozen, on ANOTHER dataset's full test split (the
# transfer cells: gsm8k fit -> gsm_plus, math fit -> harp). Not an
# ablation, but it needs the same promoted-config lookup. The cell's 512
# caps are dropped so run_compass's transfer default (2048, both caps)
# applies -- the cap every existing transfer row was generated under.
transfer_cmd() {
  [ -n "$TDS" ] || { echo "usage: ./ablations.sh transfer <cell> <dataset>  (e.g. gsm_plus | harp)" >&2; exit 1; }
  resolve_config
  unset MAX_NEW_TOKENS STEER_MAX_NEW_TOKENS
  TRANSFER_DATASET=$TDS TRANSFER_BAND=$BAND TRANSFER_ALPHA=$ALPHA ./run_compass.sh transfer
  local slug; case "$TDS" in gsm_plus) slug=_gsmplus ;; *) slug=_$TDS ;; esac
  echo; echo "=============== transfer $CELL -> $TDS at band=$BAND alpha=$ALPHA ==============="
  awk -F, -v a="$ALPHA" 'NR>1 && $8=="cosine" && ($3+0)==(a+0) {print "  accuracy:", $17, "(n=" $14 ")"}' \
    "$ROOT/transfer${slug}_standard_outcome_fit${TRAIN_SPLIT}/E$BAND/sweep_summary.csv" | tail -1
}

case "$CMD" in
  kheads)   kheads ;;
  heads)    heads ;;
  anticom)  anticom ;;
  transfer) transfer_cmd ;;
  *) echo "unknown command '$CMD' (kheads | heads | anticom | transfer)" >&2; exit 1 ;;
esac
