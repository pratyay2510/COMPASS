#!/usr/bin/env python3
"""THE grading protocol, start to finish, in one place.

A row's final label is produced in three steps:

  1. single-pass   the frozen per-family grader parses the raw generation
                   (extract -> normalise -> compare with the gold answer and
                   its aliases). Deterministic, CPU, pinned by
                   src.utils.diagnostics.test_grading.
  2. formatter     every row the single-pass grader marked INCORRECT is sent
                   to a second model call (the collection's own model) that
                   restates the generation as "Answer: \\boxed{...}"; the same
                   frozen grader then grades that restatement. Rows already
                   correct are never sent. Truncated rows are sent like any
                   other (the partial text can still hold the answer).
  3. OR            final = single-pass OR formatter verdict. The formatter can
                   only promote, never demote (it is itself a generation and
                   fails as one).

Running this module on a collection or on a directory of steered gen files
performs all three steps and RETURNS THE SCORE: it prints the final accuracy
over ALL rows of the split and writes the final label into `correct`, with
the single-pass label beside it as `correct_singlepass`. There is no
separate report/apply switch: what it prints is what it wrote.

    cd COMPASS
    python3 -m src.core.grading --collection <dir>                 # baseline rows
    python3 -m src.core.grading --collection <dir> --gen_dir <dir> # steered gen_*.jsonl

Denominator = every row of the split; a steered row the steerer skipped
(baseline-truncated, never generated) counts INCORRECT.

`--label_policy singlepass` is the one exception and exists for FITTING
POOLS only: it skips the formatter and writes the single-pass label into
`correct`, because com and head selection are fitted on single-pass labels
(FIT_LABELS=singlepass). Evaluation never uses it.

Layout of this file: (1) the llama grader, frozen; (2) the qwen grader and
the per-family registry; (3) the formatter; (4) the OR protocol + CLI.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

Key = Tuple[str, str]


# -----------------------------------------------------------------------------
# (1) The llama grader, FROZEN: answer extraction, normalisation, comparison
# -----------------------------------------------------------------------------


_TEXT_MACRO_RE = re.compile(r"\\(?:text|mbox|textbf|textit|mathrm|textrm|mathbf)\s*\{([^{}]*)\}")
_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?")
_SINGLE_VAR_LHS_RE = re.compile(r"^\s*[A-Za-z]\s*=\s*(.+?)\s*$")
# A number as a generation writes one, with optional thousands separators. The
# grouping alternative must be present or "1,000" is read as the two numbers
# 1 and 000 and the fallback extractor returns 000 -- which GSM8K, whose answers
# are routinely in the thousands, hits constantly. Only proper 3-digit groups
# match, so the answer SET "1,-2" is still two separate numbers.
_NUMBER_RE = r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:\\?%|percent)?"
_INEQUALITY_RE = re.compile(r"[<>]|\\neq|\\ne\b|\\leq|\\geq|\\le\b|\\ge\b")
# "Answer: 42" as its OWN line -- the final line of the qa_brief schema. Bold
# and italic markers are eaten because instruct models emit "**Answer:** 42"
# about as often as the plain form.
_ANSWER_LABEL_RE = re.compile(
    r"(?im)^[\s>*_#-]*(?:final\s+)?answer[\s*_]*:[\s*_]*(.+?)\s*$"
)
# Used to reject a label line that only introduces one of the marker phrases
# below, so "Answer: the final answer is 42" is still graded 42, not the
# sentence.
_ANSWER_MARKER_RE = re.compile(r"(?i)final answer is|answer is")


def strip_text_macros(s: str) -> str:
    """Unwrap \\text{...}/\\mbox{...} and friends, keeping their contents.

    Without this, normalize_answer's blanket removal of backslashes and braces
    turns gold "\\text{B}" into "textb", which can never match a model's "B".
    MATH stores every multiple-choice and word answer in this form.
    """
    for _ in range(3):
        nxt = _TEXT_MACRO_RE.sub(r"\1", s)
        if nxt == s:
            break
        s = nxt
    return s


def final_value(s: str) -> str:
    """Reduce a derivation chain to the value it ends on.

    Models asked for a bare answer often box the working instead: "x = 8",
    "4^3 = 64", "\\frac{6}{36} = \\frac{1}{6}". The asserted answer is the last
    right-hand side. Chains containing an inequality or comparison are left
    alone, since there the relation itself is the answer.
    """
    t = s.strip()
    if "=" not in t or _INEQUALITY_RE.search(t):
        return t
    m = _SINGLE_VAR_LHS_RE.match(t)
    if m:
        return m.group(1)
    return t.rsplit("=", 1)[-1].strip()


def strip_latex_wrappers(s: str) -> str:
    """Remove spacing/sizing LaTeX commands that never change answer meaning."""
    s = s.strip()
    for tok in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "~"):
        s = s.replace(tok, "")
    return s.strip()


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...} or \\fbox{...} in `text`.

    Uses brace matching, so nested braces inside the box are handled.
    Returns None when no boxed expression is found.
    """
    candidates: List[str] = []
    for pat in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(pat, start)
            if idx == -1:
                break
            brace = text.find("{", idx + len(pat))
            if brace == -1:
                start = idx + len(pat)
                continue
            depth = 0
            for j in range(brace, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[brace + 1 : j])
                        break
            start = idx + len(pat)
    return candidates[-1].strip() if candidates else None


def extract_final_answer(text: str) -> str:
    """Extract a final answer from a model generation or gold solution.

    Preference order: last boxed expression, an "Answer:" label line, number
    after a marker phrase ("final answer is", ...), last number-like token,
    last non-empty line.
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    # The label line of the qa_brief schema. Ranked above the marker phrases
    # below because those can only return a NUMBER (they regex the tail), so on
    # a LaTeX answer they would skip the label and grab a digit out of the
    # working line. Anchored to the start of a line so it cannot fire on prose
    # ("the answer: obviously ..."), and only accepted when the label is the
    # whole of what precedes the answer -- a label line that then spells out
    # "the final answer is X" is left to the markers.
    labelled = _ANSWER_LABEL_RE.findall(text)
    if labelled:
        tail = labelled[-1].strip().rstrip(".").strip()
        if tail and not _ANSWER_MARKER_RE.search(tail):
            return tail

    lower = text.lower()
    for marker in ("final answer is", "answer is", "therefore", "so,"):
        pos = lower.rfind(marker)
        if pos != -1:
            tail = text[pos + len(marker) :]
            nums = re.findall(_NUMBER_RE, tail)
            if nums:
                return nums[-1]

    nums = re.findall(_NUMBER_RE, text)
    if nums:
        return nums[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_frac(s: str) -> str:
    """Rewrite simple \\frac{a}{b} (and \\dfrac/\\tfrac) as a/b."""
    return re.sub(
        r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", s
    )


def normalize_answer(ans: Any) -> str:
    """Normalize an answer string for approximate exact matching.

    Handles boxed LaTeX, commas in numbers, whitespace, percent variants
    (100%, 100\\%), simple fractions, dollar signs, braces, leading '+',
    and trailing periods. Returns a lowercase canonical string.
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed

    s = strip_latex_wrappers(s)
    # Unwrap \text{...} before backslashes/braces are deleted below, otherwise
    # the macro name itself survives into the canonical form.
    s = strip_text_macros(s)
    s = _DEGREE_RE.sub("", s)  # 90^\circ == 90
    s = normalize_frac(s)

    # Normalize percent variants before removing backslashes.
    s = s.replace("\\%", "%")
    s = s.replace(" percent", "%").replace(" Percent", "%")

    s = s.replace("$", "").replace("\\", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(" ", "").replace(",", "")
    s = s.strip().strip(".").strip()

    if s.startswith("+"):
        s = s[1:]
    return s.lower()


def numeric_value(s: str) -> Optional[float]:
    """Parse a normalized answer as a float (supports a/b fractions), else None."""
    s = normalize_answer(s)
    if s.endswith("%"):
        # 100 and 100% are treated as equivalent; do not divide by 100.
        s = s[:-1]
    try:
        if "/" in s and re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", s):
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def _latex_to_sympy(s: str) -> str:
    """Best-effort LaTeX -> sympy-parsable source for simple MATH answers."""
    s = strip_text_macros(strip_latex_wrappers(s))
    s = _DEGREE_RE.sub("", s)
    s = s.replace("\\%", "").replace("%", "").replace("$", "")
    for _ in range(4):
        nxt = re.sub(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
        if nxt == s:
            break
        s = nxt
    s = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "(").replace("}", ")").replace("^", "**")
    return s.strip()


class _SymbolicTimeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ARG001
    raise _SymbolicTimeout()


def symbolic_equal(pred: str, gold: str, budget_s: float = 1.0) -> Optional[bool]:
    """Symbolic/high-precision-numeric equality, or None if not decidable.

    Catches answers that are correct but left unevaluated -- "-6 + 0 - 35" for
    -41, "8^6" for 262144, "6x^2 + 18x + 12x + 36" for "6x^2 + 30x + 36" --
    which no amount of string normalization can reach.

    Deliberately bounded and conservative:
      * a SIGALRM budget, because a few MATH answers (nested radicals, large
        powers) make sympy run for minutes;
      * `.equals` for expressions with free symbols, `evalf` otherwise, never a
        full `simplify`, for the same reason;
      * refuses any input containing a comma or a LaTeX environment, since
        "1,-2" is a two-element answer set and not the expression 1 - 2, and
        sympy cannot tell them apart.
    Returns None when the comparison could not be made, so callers can fall
    back rather than treat "undecided" as "unequal".
    """
    if any(ch in pred or ch in gold for ch in (",", ";")):
        return None
    if "\\begin" in pred or "\\begin" in gold:
        return None
    try:
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return None

    ps, gs = _latex_to_sympy(pred), _latex_to_sympy(gold)
    if not ps or not gs:
        return None

    tr = standard_transformations + (implicit_multiplication_application,)
    try:
        import signal
        prev = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, budget_s)
    except (ValueError, AttributeError):
        prev = None  # not on the main thread; run unbounded
    try:
        pe = parse_expr(ps, transformations=tr, evaluate=True)
        ge = parse_expr(gs, transformations=tr, evaluate=True)
        if pe.free_symbols or ge.free_symbols:
            return bool(pe.equals(ge))
        a, b = complex(pe.evalf(20)), complex(ge.evalf(20))
        return bool(abs(a - b) <= 1e-9 * max(1.0, abs(b)))
    except Exception:
        return None
    finally:
        if prev is not None:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)


def _answers_equal_literal(pred: str, gold: str, tol: float) -> bool:
    """String/numeric equality on already-normalized-ish inputs."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True

    p_num = numeric_value(p)
    g_num = numeric_value(g)
    if p_num is not None and g_num is not None:
        if abs(p_num - g_num) <= tol * max(1.0, abs(g_num)):
            return True

    if p.endswith("%") or g.endswith("%"):
        pp = p[:-1] if p.endswith("%") else p
        gg = g[:-1] if g.endswith("%") else g
        if pp == gg:
            return True
        pp_num = numeric_value(pp)
        gg_num = numeric_value(gg)
        if pp_num is not None and gg_num is not None:
            return abs(pp_num - gg_num) <= tol * max(1.0, abs(gg_num))
    return False


def answers_equal(pred: str, gold: str, tol: float = 1e-6,
                  symbolic: bool = True) -> bool:
    """Approximate answer equality.

    Tried in order, cheapest first: normalized string match, numeric comparison
    with relative tolerance, percent-stripped retry (all in
    _answers_equal_literal); the same again after reducing each side to the
    final value of a derivation chain; then a bounded symbolic comparison.

    Set symbolic=False to skip the sympy fallback (faster, and what the pure
    string matcher did before).
    """
    if _answers_equal_literal(pred, gold, tol):
        return True

    p_fin, g_fin = final_value(str(pred)), final_value(str(gold))
    if (p_fin, g_fin) != (str(pred), str(gold)):
        if _answers_equal_literal(p_fin, g_fin, tol):
            return True

    if symbolic and symbolic_equal(p_fin, g_fin) is True:
        return True
    return False


def answer_is_correct(pred: str, example: Dict[str, Any]) -> bool:
    """Whether `pred` matches an example's gold answer, aliases included.

    The one grading entry point collection uses, so a dataset that carries
    extra accepted surface forms per answer (HARP keeps the raw `$...$` form
    as an alias) is scored on all of them while MATH and GSM8K, which carry no
    `gold_aliases`, fall through to exactly the single-gold `answers_equal`
    call they always made.

    The aliases are matched with symbolic=False. They are alternative surface
    forms of the same value, never a different expression, so the sympy
    fallback can only cost time.
    """
    if answers_equal(pred, example.get("gold_answer", "")):
        return True
    return any(answers_equal(pred, alias, symbolic=False)
               for alias in (example.get("gold_aliases") or []) if str(alias).strip())


# -----------------------------------------------------------------------------
# (2) The qwen grader and the per-family registry
# -----------------------------------------------------------------------------
# The functions above are the LLAMA grader, frozen: every llama collection was
# labelled by them and their behaviour must not drift. Qwen gets its own
# extract/compare pair because its output surface differs in ways the llama
# grader mis-reads, each observed on zs_qwen3_4b_gsm8k_all/*_qabrief:
#
#   * markdown emphasis around the answer line: "**Answer: 22**". The label
#     regex eats LEADING emphasis but captured "22**" with the trailing stars
#     attached, which then failed the string compare. 75 of 94 flipped-to-wrong
#     rows in one steered config were this pattern, numerically correct.
#   * a trailing unit word: "Final answer: 135 minutes." against gold "135".
#     542 baseline rows graded 0-for-542 because of it (220 numerically right).
#   * Qwen3's <think>...</think> block, which can contain numbers that the
#     last-number fallback would grab.
#   * \boxed{} mid-derivation followed by a later "**Answer: N**" line: the
#     llama grader prefers the box unconditionally; the LAST asserted value is
#     the answer.
#
# Same registry contract as the prompt builders: resolve with grader_for(),
# which falls back to the llama pair for every family without an entry, so
# nothing about existing collections changes.

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)
# Edge decoration a qwen answer token arrives wrapped in: markdown emphasis,
# backticks, quotes. Stripped only at the EDGES of the candidate, never inside,
# so "2**10" as an answer value survives while "**22**" is unwrapped.
_EDGE_DECOR = "*_`~\"'"
# Scale/number words that change the value: "7 million" is not 7, "5 squared"
# is not 5. A tail containing one must NOT be stripped as a unit.
_UNIT_SCALE_WORDS = frozenset((
    "dozen", "dozens", "score", "hundred", "hundreds", "thousand", "thousands",
    "million", "millions", "billion", "billions", "trillion", "trillions",
    "tenth", "tenths", "hundredth", "hundredths", "thousandth", "thousandths",
    "half", "halves", "third", "thirds", "quarter", "quarters",
    "squared", "cubed", "percent",
))
# "<number><unit words>": a leading number followed by alphabetic unit text,
# attached ("25ml") or spaced ("25 ml") -- both observed in qwen/gsm8k label
# audits. Digits in the tail ("3 hours 30 minutes") refuse the match -- such an
# answer asserts more than its leading number. The charset admits the
# superscripts/degree of "cm³", "m²", "45°".
_UNIT_TAIL_RE = re.compile(
    r"^(-?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z][A-Za-z\s.\-/²³°]*)$"
)
# Currency symbols qwen writes on either side of the number ("€32", "25€",
# "8£", "2500 €"). "$" is already stripped by normalize_answer; these are not.
_CURRENCY_EDGE_RE = re.compile(r"^\s*[€£¥₹]\s*|\s*[€£¥₹]\s*$")
# "2:00 pm" asserts the value 2; only a :00 tail is an identity ("7:30" is not
# the number 7).
_TIME_ON_THE_HOUR_RE = re.compile(r"^(\d{1,2}):00(?:\s*[ap]\.?m\.?)?$", re.I)
# Connectives under which a multi-number sentence is NOT asserting its last
# number ("120 counters and 200 marbles", "between 3 and 5") -- the guard on
# the last-number fallback in answers_equal_qwen.
_SENTENCE_CONNECTIVES = frozenset(("and", "or", "between", "either", "versus", "vs"))


def _strip_answer_decoration(s: str) -> str:
    """Peel markdown emphasis, quotes and trailing periods off a candidate."""
    prev = None
    s = str(s).strip()
    while s != prev:
        prev = s
        s = s.strip().strip(_EDGE_DECOR).strip()
        if s.endswith("."):
            s = s[:-1].strip()
    return s


def _strip_unit_tail(s: str) -> str:
    """"135 minutes" / "25ml" / "€32" / "2:00 pm" -> the bare number.

    Refuses scale words ("7 million") because there the tail changes the value,
    and any tail containing digits, where the leading number is not the whole
    claim. Currency symbols come off both edges first; a :00 clock time is the
    hour it names ("7:30" is left alone).
    """
    s2 = s.strip()
    prev = None
    while s2 != prev:
        prev = s2
        s2 = _CURRENCY_EDGE_RE.sub("", s2).strip()
    t = _TIME_ON_THE_HOUR_RE.match(s2)
    if t:
        return t.group(1)
    m = _UNIT_TAIL_RE.match(s2)
    if not m:
        return s2 if s2 != s.strip() else s
    words = re.findall(r"[a-z]+", m.group(2).lower())
    if any(w in _UNIT_SCALE_WORDS for w in words):
        return s
    return m.group(1)


def extract_final_answer_qwen(text: str) -> str:
    """extract_final_answer for qwen-family output surface.

    Differences from the llama extractor, each tied to an observed failure:
    thinking blocks are stripped first; an answer-label line has its edge
    decoration peeled; and when BOTH a \\boxed{} and a label line exist, the
    one asserted LAST in the text wins rather than the box unconditionally.
    The marker/last-number/last-line fallbacks are unchanged.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    # A truncated generation can die inside an unclosed think block; nothing
    # after "<think>" is an asserted answer.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]

    boxed = extract_boxed(text)
    boxed_pos = max(text.rfind("\\boxed"), text.rfind("\\fbox")) \
        if boxed is not None else -1

    label_val, label_pos = None, -1
    for m in _ANSWER_LABEL_RE.finditer(text):
        tail = _strip_answer_decoration(m.group(1))
        if tail and not _ANSWER_MARKER_RE.search(tail):
            label_val, label_pos = tail, m.start()

    if boxed is not None and boxed_pos >= label_pos:
        return boxed.strip()
    if label_val:
        return label_val

    lower = text.lower()
    for marker in ("final answer is", "answer is", "therefore", "so,"):
        pos = lower.rfind(marker)
        if pos != -1:
            tail = text[pos + len(marker):]
            # Bold groups in the tail: the LAST one is the asserted answer when
            # it carries the value itself ("... is **$4840**.") or when the
            # tail has no number at all ("The final answer is **B**"). A bold
            # label mid-sentence ("the **total** is 42") loses to the number.
            bolds = re.findall(r"\*\*([^*\n]+?)\*\*", tail[:160])
            nums = re.findall(_NUMBER_RE, tail)
            if bolds:
                cand = _strip_answer_decoration(bolds[-1])
                if cand and (any(ch.isdigit() for ch in cand) or not nums):
                    return cand
            if nums:
                return nums[-1]

    nums = re.findall(_NUMBER_RE, text)
    if nums:
        return nums[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def answers_equal_qwen(pred: str, gold: str, tol: float = 1e-6,
                       symbolic: bool = True) -> bool:
    """answers_equal for qwen: the llama comparison, tried again after peeling
    qwen's decoration, unit-word tails and currency off both sides, then after
    reducing a sentence/derivation pred to the value it asserts.

    Strictly more permissive than the llama comparator on the FORMATTING axis
    only; a genuinely different value still fails every retry. The stripped
    retries run with symbolic=False because both sides are then bare numbers.
    """
    p, g = _strip_answer_decoration(pred), _strip_answer_decoration(gold)
    if answers_equal(p, g, tol, symbolic):
        return True
    pu, gu = _strip_unit_tail(p), _strip_unit_tail(g)
    if (pu, gu) != (p, g) and answers_equal(pu, gu, tol, symbolic=False):
        return True

    # A label tail that is a whole sentence or derivation ("11 - 2 - 3 - 4 =
    # 2 ounces", "1200 / (10 x 8) = 15. Each orc has to carry 15 pounds.",
    # "got 108 reward"): reduce to the final RHS, strip again, and lastly take
    # the pred's LAST number -- the same semantics the marker-phrase extraction
    # path has always had. Guarded three ways against inventing a match:
    # only for a NUMERIC gold (math-style datasets), never when the pred
    # carries a scale word ("7 million" is not 7), and never for a
    # multi-number sentence joined by a connective ("120 counters and 200
    # marbles" does not assert 200).
    if numeric_value(gu) is None:
        return False
    pf = _strip_unit_tail(_strip_answer_decoration(final_value(p)))
    if pf not in (p, pu) and answers_equal(pf, gu, tol, symbolic=False):
        return True
    if len(p.split()) > 1:
        words = set(re.findall(r"[a-z]+", p.lower()))
        nums = re.findall(_NUMBER_RE, pf if pf != p else p)
        if (nums and not (words & _UNIT_SCALE_WORDS)
                and not (len(nums) > 1 and words & _SENTENCE_CONNECTIVES)):
            return bool(answers_equal(nums[-1], gu, tol, symbolic=False))
    return False


def answer_is_correct_qwen(pred: str, example: Dict[str, Any]) -> bool:
    """answer_is_correct with the qwen comparator, aliases included."""
    if answers_equal_qwen(pred, example.get("gold_answer", "")):
        return True
    return any(answers_equal_qwen(pred, alias, symbolic=False)
               for alias in (example.get("gold_aliases") or []) if str(alias).strip())


class Grader(NamedTuple):
    """One family's grading triple. `extract` parses a generation into an
    asserted answer; `equal` compares two answer strings; `is_correct` is the
    alias-aware entry point collection and steering call."""
    extract: Any
    equal: Any
    is_correct: Any


_LLAMA_GRADER = Grader(extract_final_answer, answers_equal, answer_is_correct)
_QWEN_GRADER = Grader(extract_final_answer_qwen, answers_equal_qwen,
                      answer_is_correct_qwen)

# family -> Grader. Same contract as STANDARD_PROMPT_BUILDERS: a family with no entry
# gets the llama grader, so every existing collection keeps its exact labels.
GRADERS: Dict[str, Grader] = {
    "qwen": _QWEN_GRADER,
}


def grader_for(model_ref: Any) -> Grader:
    """The grading triple for a model id/tokenizer/model, by family."""
    from src.core.common import model_family  # lazy: grading is a leaf module
    return GRADERS.get(model_family(model_ref), _LLAMA_GRADER)


# =============================================================================
# (3) The formatter: second-stage answer restatement
# =============================================================================
# WHY. The frozen graders parse a generation with extract_final_answer, which
# prefers an explicit marker (\boxed{}, "Final answer:") and otherwise falls
# back to the last number. A correct answer that ends "... = \frac{109}{33}."
# with no marker is extracted as "33" and graded wrong. Fractional Reasoning
# (shengliu66/FractionalReason) side-steps this with a SECOND model call that
# rewrites the free-form reasoning into a fixed "Answer: \[ \boxed{...} \]"
# line before extraction. This is that stage. The formatting prompt is
# DATASET-SPECIFIC: "math" (also serving math500) is the FR MATH-500 prompt;
# "harp" states the competition-answer contract; "gsm8k" (also serving
# gsm_plus and svamp) states the single-number contract.

FORMATTER_SYSTEM = ("Your job is to extract the final short answer from the "
                    "more detailed answer.")

# One prompt per DATASET (the answer shape is the dataset's, not the model's).
FORMATTING_PROMPTS: Dict[str, str] = {
    "math": (
        'Generate the final answer for the query {query} based on the '
        'reasoning process {reasoning} in this format: '
        '"Answer: \\[ \\boxed{{your_answer_here}} \\]". The entire answer '
        'should be contained completely within the \\boxed{{}} command. '
        'Do not include any other text.'
    ),
    "harp": (
        'Generate the final answer for the query {query} based on the '
        'reasoning process {reasoning} in this format: '
        '"Answer: \\[ \\boxed{{your_answer_here}} \\]". The answer is a math '
        'competition short answer: a single number or LaTeX expression, '
        'stated exactly as derived in the reasoning. The entire answer '
        'should be contained completely within the \\boxed{{}} command. '
        'Do not include any other text.'
    ),
    # GSM8K / GSM-Plus (2026-08-25): word problems with a single numeric
    # answer. The graders read \boxed{} first and normalise thousands
    # separators, $ and %, so the contract only has to keep the model from
    # boxing an intermediate quantity or a unit-laden phrase. GSM-Plus golds
    # include fractions and decimals (its "numerical substitution" and
    # "problem understanding" perturbations), so "number" is stated broadly.
    "gsm8k": (
        'Generate the final answer for the query {query} based on the '
        'reasoning process {reasoning} in this format: '
        '"Answer: \\[ \\boxed{{your_answer_here}} \\]". The answer is a '
        'single number (an integer, decimal or fraction) with no units, '
        'currency symbols or words, taken exactly as the final result of '
        'the reasoning, not an intermediate value. The entire answer should '
        'be contained completely within the \\boxed{{}} command. Do not '
        'include any other text.'
    ),
}
FORMATTING_ALIASES = {"math500": "math", "gsm_plus": "gsm8k", "svamp": "gsm8k"}

# Per-(family, dataset-key) overrides of the prompt above. A family lands here
# only with a measurement behind it.
#
# gemma / math (2026-09-01). MEASURED on gemma-4-12B-it, math500 test (500
# rows): the shared prompt above produced NO \boxed{} on 294/500 rows (58.8%),
# median output 105 words where one line was asked for. It was not restating
# the reasoning -- it was re-deriving the problem from scratch and hitting
# --max_new_tokens before ever reaching the box, so the verdict fell back to
# the last number of a half-finished derivation. A two-part instruction
# ("generate X ... in this format Y") reads as a SCHEMA to fill in and
# LICENSES the working it was meant to suppress; the fix is to say less and
# make the task a copy rather than a generation: reasoning BEFORE the
# instruction, no "based on the reasoning process" clause, and no query (the
# answer is already in the reasoning; showing the problem invites a re-solve).
FORMATTING_PROMPT_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("gemma", "math"): (
        'Here is a worked solution:\n\n{reasoning}\n\n'
        'Copy its final answer, exactly as the solution states it, into this '
        'line and output nothing else:\n'
        'Answer: \\[ \\boxed{{your_answer_here}} \\]\n'
        'Do not solve the problem. Do not explain. Do not recompute anything. '
        'If the solution states no final answer, box its last expression.'
    ),
}


def formatting_prompt_for(dataset: str, model_name: Optional[str] = None) -> Optional[str]:
    """The formatting template for a dataset (None if the dataset has none)."""
    from src.core import common
    key = FORMATTING_ALIASES.get(dataset, dataset)
    prompt = FORMATTING_PROMPTS.get(key)
    if prompt is None:
        return None
    if model_name is not None:
        override = FORMATTING_PROMPT_OVERRIDES.get((common.model_family(model_name), key))
        if override is not None:
            print(f"formatting prompt: per-family override ({common.model_family(model_name)}, "
                  f"{key}) -- the shared {key} prompt is NOT used for this model")
            return override
    return prompt


def build_prompt(tokenizer: Any, template: str, problem: str, reasoning: str) -> str:
    from src.core import common
    messages = [
        {"role": "system", "content": FORMATTER_SYSTEM},
        {"role": "user", "content": template.format(query=problem, reasoning=reasoning)},
    ]
    extra = common.chat_template_kwargs(tokenizer)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra)
    except Exception:  # families whose template rejects the system role
        messages = [{"role": "user",
                     "content": FORMATTER_SYSTEM + "\n\n" + messages[1]["content"]}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra)


class Formatter:
    """Lazy model wrapper: loads once, formats batches, resumes via out files."""

    def __init__(self, model_name: str, dtype: str, template: str,
                 batch_size: int, max_new_tokens: int):
        self.model_name = model_name
        self.dtype = dtype
        self.template = template
        self.batch_size = max(1, batch_size)
        self.max_new_tokens = max_new_tokens
        self._model = self._tokenizer = self._stops = None

    def _ensure_loaded(self):
        if self._model is None:
            from src.core import common
            # NOTE the order: (tokenizer, model) -- swapping them once cost a
            # smoke run.
            self._tokenizer, self._model = common.load_model_and_tokenizer(
                self.model_name, self.dtype)
            self._tokenizer.padding_side = "left"
            self._stops = common.stop_token_ids(self._tokenizer, self._model)

    def ensure_formatted(self, items: List[Dict[str, Any]], out_path: str,
                         force: bool, desc: str) -> Dict[Key, str]:
        """items: dicts with subject/idx/problem/reasoning. Returns
        key -> formatted_text, generating only what out_path lacks."""
        done: Dict[Key, str] = {}
        if os.path.exists(out_path) and not force:
            for line in open(out_path, encoding="utf-8"):
                row = json.loads(line)
                done[(row["subject"], str(row["idx"]))] = row["formatted_text"]
        todo = [it for it in items if (it["subject"], str(it["idx"])) not in done]
        if not todo:
            return done
        self._ensure_loaded()
        tok, model = self._tokenizer, self._model
        mode = "w" if force else "a"
        with open(out_path, mode, encoding="utf-8") as out_f:
            for i in tqdm(range(0, len(todo), self.batch_size), desc=desc, unit="batch"):
                batch = todo[i:i + self.batch_size]
                prompts = [build_prompt(tok, self.template, it["problem"], it["reasoning"])
                           for it in batch]
                enc = tok(prompts, return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc, max_new_tokens=self.max_new_tokens,
                        do_sample=False, eos_token_id=self._stops,
                        pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
                for it, text in zip(batch, texts):
                    out_f.write(json.dumps({
                        "subject": it["subject"], "idx": it["idx"],
                        "formatted_text": text.strip()}) + "\n")
                    done[(it["subject"], str(it["idx"]))] = text.strip()
                out_f.flush()
        return done


# =============================================================================
# (4) The protocol: single-pass -> formatter -> OR, and the CLI that runs it
# =============================================================================

def singlepass_label(row: Dict[str, Any]) -> int:
    """The single-pass label a row currently carries: `correct_singlepass`
    when a past grading wrote it, else `correct` (a collection straight from
    the collector, where `correct` IS the single-pass label)."""
    v = row.get("correct_singlepass", row.get("correct", 0))
    return int(str(v).strip() or 0)


def truncated_row(row: Dict[str, Any]) -> bool:
    """Whether the generation hit the token cap (stored as 0/1, bool or
    string depending on the writer). A truncated row is graded like any
    other; this only feeds the reported count."""
    return str(row.get("truncated", "0")).strip() in ("1", "True", "true")


def skipped_row(row: Dict[str, Any]) -> bool:
    """A steered row the steerer never generated (its baseline was
    truncated): no text to grade, counts INCORRECT."""
    return bool(int(row.get("skipped", 0) or 0)) or row.get("generated_text") is None \
        or row.get("steered") is False


def singlepass(grader: Grader, text: str, gold_answer: str, gold_aliases: List[str]) -> bool:
    """Step 1: the frozen grader on the raw generation."""
    return bool(grader.is_correct(grader.extract(text), {
        "gold_answer": gold_answer, "gold_aliases": gold_aliases}))


def verdict(grader: Grader, formatted_text: str, gold_answer: str,
            gold_aliases: List[str]) -> bool:
    """Step 2: the same frozen grader on the formatter's restatement."""
    return bool(grader.is_correct(grader.extract(formatted_text), {
        "gold_answer": gold_answer, "gold_aliases": gold_aliases}))


def final_label(sp: bool, formatter_ok: Optional[bool]) -> bool:
    """Step 3: the OR. A row with no formatter verdict (never sent, or the
    dataset has no formatting prompt) keeps its single-pass label."""
    return bool(sp or formatter_ok)


_HIST_BINS = [(0, 1, "<= 1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, 10, "6-10"),
              (11, 20, "11-20"), (21, 40, "21-40"), (41, 80, "41-80"),
              (81, 160, "81-160"), (161, 320, "161-320"), (321, 10**9, "> 320")]


def histogram(rows: List[Tuple[int, bool]], title: str) -> None:
    counts = []
    for lo, hi, lab in _HIST_BINS:
        sel = [ok for w, ok in rows if lo <= w <= hi]
        counts.append((lab, sum(sel), len(sel) - sum(sel), len(sel)))
    widest = max(c[3] for c in counts) or 1
    n_c = sum(c[1] for c in counts)
    print(f"\n{title} ({len(rows)} rows: {n_c} correct, {len(rows)-n_c} incorrect)")
    for lab, c, i, tot in counts:
        bar = "#" * round(50 * c / widest) + "-" * round(50 * i / widest) if tot else "."
        print(f"  {lab:>8}  {bar}  {tot}  ({c} correct, {i} incorrect)")
    print("            # = correct   - = incorrect   (scaled to the widest bin)")


def grade_rows(rows: List[Dict[str, Any]], meta: Dict[Key, Dict[str, Any]],
               grader: Grader, fmtr: Optional[Formatter], verdict_path: str,
               force: bool, desc: str, use_formatter: bool) -> Dict[Key, Dict[str, int]]:
    """The protocol over a list of rows. Returns key -> {"sp", "fmt", "final"}
    (fmt = -1 when the row was not formatted). Skipped rows are absent."""
    sp: Dict[Key, bool] = {}
    for r in rows:
        k = (r["subject"], str(r["idx"]))
        if skipped_row(r) or k not in meta:
            continue
        m = meta[k]
        sp[k] = singlepass(grader, r["generated_text"], str(m["gold_answer"]),
                           m.get("gold_aliases") or [])
    fmt_text: Dict[Key, str] = {}
    if use_formatter and fmtr is not None:
        items = [{"subject": r["subject"], "idx": r["idx"],
                  "problem": meta[(r["subject"], str(r["idx"]))]["problem"],
                  "reasoning": r["generated_text"]}
                 for r in rows
                 if (r["subject"], str(r["idx"])) in sp and not sp[(r["subject"], str(r["idx"]))]]
        print(f"{desc}: formatting {len(items)} single-pass-incorrect row(s) of {len(sp)}")
        fmt_text = fmtr.ensure_formatted(items, verdict_path, force, desc)
    out: Dict[Key, Dict[str, int]] = {}
    for k, s in sp.items():
        m = meta[k]
        fv = None
        if k in fmt_text and not s:
            fv = verdict(grader, fmt_text[k], str(m["gold_answer"]), m.get("gold_aliases") or [])
        out[k] = {"sp": int(s), "fmt": -1 if fv is None else int(fv),
                  "final": int(final_label(s, fv))}
    return out


def print_score(title: str, grades: Dict[Key, Dict[str, int]], n_all: int,
                n_trunc: int, n_skipped: int) -> Dict[str, Any]:
    n_sp = sum(g["sp"] for g in grades.values())
    n_final = sum(g["final"] for g in grades.values())
    n_fmt = sum(1 for g in grades.values() if g["fmt"] >= 0)
    promoted = sum(1 for g in grades.values() if g["final"] and not g["sp"])
    print(f"\n{title}")
    print(f"  rows                      {n_all}  ({n_trunc} truncated, graded on partial text"
          + (f"; {n_skipped} skipped, counted wrong" if n_skipped else "") + ")")
    print(f"  single-pass accuracy      {n_sp}/{n_all} = {100.0 * n_sp / n_all:.2f}")
    print(f"  formatter                 {n_fmt} row(s) sent, {promoted} promoted wrong->right")
    print(f"  FINAL accuracy (OR)       {n_final}/{n_all} = {100.0 * n_final / n_all:.2f}")
    return {"n": n_all, "n_truncated": n_trunc, "n_skipped": n_skipped,
            "singlepass_correct": n_sp, "singlepass_accuracy": 100.0 * n_sp / n_all,
            "formatted_rows": n_fmt, "promoted": promoted,
            "final_correct": n_final, "final_accuracy": 100.0 * n_final / n_all}


# ---------------------------------------------------------------- collection

def _write_collection_labels(coll: str, records: List[Dict[str, Any]],
                             grades: Dict[Key, Dict[str, int]], policy: str) -> None:
    """Write `correct` (= final under policy=final, = single-pass under
    policy=singlepass) and `correct_singlepass` into records.jsonl and
    ground_truth.csv. Atomic (tmp + rename)."""
    label = (lambda g: g["final"]) if policy == "final" else (lambda g: g["sp"])
    gt_path = os.path.join(coll, "ground_truth.csv")
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gt_rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "correct_singlepass" not in fields:
        fields.insert(fields.index("correct") + 1, "correct_singlepass")
    changed = 0
    for r in gt_rows:
        g = grades.get((r["subject"], str(r["idx"])))
        if g is None:
            r.setdefault("correct_singlepass", r["correct"])
            continue
        new = str(label(g))
        changed += (new != str(r["correct"]).strip())
        r["correct"], r["correct_singlepass"] = new, str(g["sp"])
    tmp = gt_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(gt_rows)
    os.replace(tmp, gt_path)

    rec_path = os.path.join(coll, "records.jsonl")
    tmp = rec_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            g = grades.get((r["subject"], str(r["idx"])))
            if g is not None:
                r["correct"], r["correct_singlepass"] = int(label(g)), int(g["sp"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, rec_path)
    print(f"  labels written: `correct` = {policy} label ({changed} changed), "
          f"`correct_singlepass` beside it -> {coll}")


def grade_collection(args: argparse.Namespace, grader: Grader,
                     fmtr: Optional[Formatter]) -> Dict[str, Any]:
    coll = args.collection.rstrip("/")
    records = [json.loads(l) for l in open(os.path.join(coll, "records.jsonl"), encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    meta = {(r["subject"], str(r["idx"])): r for r in records}
    verdict_path = os.path.join(coll, "llm_formatted_answers.jsonl")
    grades = grade_rows(records, meta, grader, fmtr, verdict_path, args.force,
                        "Formatting", use_formatter=(args.label_policy == "final"))
    score = print_score(f"collection {coll}", grades, len(records),
                        sum(1 for r in records if truncated_row(r)), 0)
    hist = [(len(r["generated_text"].split()), bool(grades[(r["subject"], str(r["idx"]))]["final"]))
            for r in records if (r["subject"], str(r["idx"])) in grades]
    histogram(hist, "answer-length histogram, FINAL labels")
    if not args.limit:
        _write_collection_labels(coll, records, grades, args.label_policy)
        score["label_policy"] = args.label_policy
        with open(os.path.join(coll, "grade.json"), "w", encoding="utf-8") as f:
            json.dump(score, f, indent=2)
    return score


# ------------------------------------------------------------------ gen dir

def _write_gen_labels(gen_path: str, rows: List[Dict[str, Any]],
                      grades: Dict[Key, Dict[str, int]], n_all: int) -> None:
    """Write the final label into gen_<tag>.jsonl (`correct`, with
    `correct_singlepass` beside it) and patch every label-derived field of
    summary_<tag>.json so pick / report / transfer read the final grade."""
    from src.steering import steering_driver
    tmp = gen_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            g = grades.get((r["subject"], str(r["idx"])))
            if g is not None:
                r["correct"], r["correct_singlepass"] = int(g["final"]), int(g["sp"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, gen_path)

    out_dir = os.path.dirname(gen_path)
    tag = os.path.basename(gen_path)[len("gen_"):-len(".jsonl")]
    summary_path = os.path.join(out_dir, f"summary_{tag}.json")
    if not os.path.exists(summary_path):
        print(f"  (no {os.path.basename(summary_path)} to patch)")
        return
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    recs = [r for r in rows if (r["subject"], str(r["idx"])) in grades]
    corr = np.array([int(r["correct"]) for r in recs])
    base = np.array([int(r.get("baseline_correct", 0)) for r in recs])
    subjects = np.array([r["subject"] for r in recs])
    w2r, r2w = steering_driver.flip_counts(base, corr)
    n_all = max(int(summary.get("n") or 0), n_all)
    patched = {
        "n": n_all,
        "accuracy": float(corr.sum()) / n_all,
        "baseline_accuracy": float(base.sum()) / n_all,
        "delta": float(corr.sum() - base.sum()) / n_all,
        "flipped_to_correct": w2r,
        "flipped_to_wrong": r2w,
        "per_subject": steering_driver.per_subject_table(subjects, base, corr),
    }
    for k, v in patched.items():
        summary.setdefault(f"{k}_singlepass", summary.get(k))
        summary[k] = v
    summary["grading"] = "llm_formatter"
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)
    print(f"  labels written: {os.path.basename(gen_path)} + {os.path.basename(summary_path)} "
          f"carry the FINAL grade (single-pass preserved as *_singlepass)")


def grade_gen_dir(args: argparse.Namespace, grader: Grader,
                  fmtr: Optional[Formatter]) -> None:
    coll = args.collection.rstrip("/")
    meta = {(r["subject"], str(r["idx"])): r
            for r in map(json.loads, open(os.path.join(coll, "records.jsonl"), encoding="utf-8"))}
    n_split = len(meta)
    gen_files = sorted(f for f in glob.glob(os.path.join(args.gen_dir, args.gen_glob))
                       if not f.endswith("_llmfmt.jsonl"))
    if not gen_files:
        raise SystemExit(f"no {args.gen_glob} under {args.gen_dir}")
    wrote = False
    for path in gen_files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if args.limit:
            rows = rows[: args.limit]
        name = os.path.basename(path)
        grades = grade_rows(rows, meta, grader, fmtr, path[:-len(".jsonl")] + "_llmfmt.jsonl",
                            args.force, f"Formatting {name}",
                            use_formatter=(args.label_policy == "final"))
        n_skipped = sum(1 for r in rows if skipped_row(r))
        # Denominator = every addressed row. A best_/transfer_ tree is a
        # whole-split run by construction, so the split size is its floor.
        tree = os.path.basename(os.path.dirname(os.path.dirname(path)))
        n_all = len(rows) if args.limit else max(
            len(rows), n_split if tree.startswith(("best_", "transfer_")) else 0)
        base = sum(int(r.get("baseline_correct", 0)) for r in rows if not skipped_row(r))
        score = print_score(f"=== {name}", grades, n_all,
                            sum(1 for r in rows if truncated_row(r)), n_skipped + (n_all - len(rows)))
        print(f"  baseline (collection labels)  {base}/{n_all} = {100.0 * base / n_all:.2f}"
              f"   delta {(score['final_correct'] - base) * 100.0 / n_all:+.2f}")
        if not args.limit and grades:
            _write_gen_labels(path, rows, grades, n_all)
            wrote = True
    if wrote:
        from src.steering import steering_driver
        steering_driver.rebuild_sweep_summary(args.gen_dir)


# --------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True,
                   help="collection dir with records.jsonl + args.json (with --gen_dir: "
                        "the baseline collection the gen files are joined on)")
    p.add_argument("--gen_dir", default="",
                   help="grade every gen_*.jsonl in this dir instead of the collection's rows")
    p.add_argument("--gen_glob", default="gen_*.jsonl",
                   help="which gen files under --gen_dir to grade (default all)")
    p.add_argument("--label_policy", choices=["final", "singlepass"], default="final",
                   help="what lands in `correct`: the FINAL (single-pass OR formatter) label, "
                        "the default and the only evaluation label; or the single-pass label "
                        "alone, which skips the formatter and exists for FITTING POOLS "
                        "(FIT_LABELS=singlepass) only")
    p.add_argument("--model_name", default="",
                   help="formatter model (default: the collection's own model)")
    p.add_argument("--dataset", default="",
                   help="formatting-prompt key (default: the collection's dataset)")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0,
                   help="grade only the first N rows and write NOTHING (a smoke)")
    p.add_argument("--force", action="store_true",
                   help="re-run the formatter instead of resuming from its verdict file")
    args = p.parse_args()

    coll = args.collection.rstrip("/")
    if not os.path.exists(os.path.join(coll, "records.jsonl")):
        raise FileNotFoundError(f"no records.jsonl under {coll}")
    with open(os.path.join(coll, "args.json"), encoding="utf-8") as fh:
        coll_args = json.load(fh)
    model_name = args.model_name or coll_args["model_name"]
    dataset = args.dataset or coll_args.get("dataset", "")
    grader = grader_for(model_name)
    fmtr = None
    if args.label_policy == "final":
        template = formatting_prompt_for(dataset, model_name)
        if template is None:
            print(f"no formatting prompt for dataset {dataset!r}: the final label is the "
                  f"single-pass label (have: {sorted(FORMATTING_PROMPTS)})")
        else:
            fmtr = Formatter(model_name, args.dtype, template, args.batch_size,
                             args.max_new_tokens)
    print(f"model      {model_name}   dataset {dataset}   label policy {args.label_policy}")
    if args.gen_dir:
        grade_gen_dir(args, grader, fmtr)
    else:
        grade_collection(args, grader, fmtr)


if __name__ == "__main__":
    main()
