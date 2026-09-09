#!/usr/bin/env python3
"""Regression cases for grading.answers_equal (the single-pass grader).

Every collection's `correct` column is produced by a single call in
collection.py (SampleRecord.correct), so these cases pin the exact behaviour that
labels depend on. Run after any edit to the matcher:

    python3 test_grading.py

The SHOULD_MATCH block is grouped by the three bugs fixed on 2026-07-29; the
SHOULD_NOT_MATCH block guards against the fix over-reaching, which would be far
worse than the original under-matching (it would silently invent correct labels).
"""

from __future__ import annotations

import sys

from src.core.grading import answers_equal, extract_final_answer

SHOULD_MATCH = [
    # --- bug 1: \text{...} was never unwrapped, so gold "\text{B}" normalized
    # to "textb" and could not match a model's "B". Also degree marks.
    (r"B", r"\text{B}"),
    (r"(C)", r"\text{(C)}"),
    (r"even", r"\text{even}"),
    (r"Friday", r"\text{Friday}"),
    (r"9 a.m.", r"9\text{ a.m.}"),
    (r"90", r"90^\circ"),
    (r"108", r"108^{\circ}"),
    # --- bug 2: no symbolic comparison, so correct-but-unevaluated answers failed.
    ("-6 + 0 - 35", "-41"),
    ("8^6", "262144"),
    ("-6 + 8", "2"),
    ("6x^2 + 18x + 12x + 36", "6x^2 + 30x + 36"),
    (".35625", "0.35625"),
    # --- bug 3: the model boxes the working; only the final RHS is the answer.
    (r"x = 8", "8"),
    (r"11 * 33 = 363", "363"),
    (r"\frac{6}{36} = \frac{1}{6}", r"\frac{1}{6}"),
    (r"9 * 8 * 7 * 6 = 3024", "3024"),
    (r"(5+\sqrt{3})(5-\sqrt{3}) = 25 - 3 = 22", "22"),
    # --- behaviour that predates the fix and must survive it.
    ("100", "100"),
    (r"\frac{1}{2}", "0.5"),
    ("50%", "50"),
    (r"\frac{5\sqrt{3}}{3}", r"\frac{5\sqrt{3}}{3}"),
    ("-2x + 20", "-2x + 20"),
    ("1,000", "1000"),
]

SHOULD_NOT_MATCH = [
    # plain wrong answers
    ("16", "121"),
    ("-35", "-8"),
    (r"\frac{3}{5}", r"\frac{2}{3}"),
    ("2 + 5i", "3 + 5i"),
    ("4.99", "4.95"),
    ("2200", "2220"),
    (r"4\sqrt{3}", r"4\sqrt{2}"),
    # answer SETS: "1,-2" must not be read as the expression 1 - 2, which would
    # make it equal "0,-1". This is why symbolic_equal refuses commas.
    ("0, -1", "1,-2"),
    ("1, -1", "-2,-1,1,2"),
    ("(-5, 0)", "(0,0)"),
    # \text{} unwrapping must not collapse distinct words
    (r"\text{east}", r"\text{South}"),
    (r"\text{A}", r"\text{B}"),
    # chain-stripping must not discard a real inequality/interval answer
    ("x = 5", "7"),
    (r"\begin{pmatrix} -11/5 \\ 38/5 \end{pmatrix}",
     r"\begin{pmatrix} 1 \\ 3 \end{pmatrix}"),
    ("x > 3", "x > 5"),
]

# The other half of the label: what pred_answer the collector pulls out of a
# generation before grading it. Only the no-\boxed{} fallback is interesting --
# a boxed answer is taken verbatim -- and that fallback is what GSM8K leans on,
# since its answers are frequently comma-grouped thousands.
SHOULD_EXTRACT = [
    (r"So the answer is \boxed{18}.", "18"),
    ("The final answer is 18", "18"),
    ("She makes $1,250 total.", "1,250"),
    ("Therefore the answer is 1,000", "1,000"),
    ("x = 1,000,000", "1,000,000"),
    # a comma-separated answer set stays two numbers, not one grouped one
    ("The answer is 1,-2", "-2"),
    # the qa_brief schema: the Answer: label wins over any number in the work
    ("Work: 48/2 = 24, 48+24 = 72\nAnswer: 72", "72"),
    ("Work: 3 of the 8 are red\n**Answer:** \\frac{3}{8}", "\\frac{3}{8}"),
    ("Work: 2 + 2\nAnswer: 4.", "4"),
    # a label line that only introduces a marker phrase defers to the marker
    ("Answer: the final answer is 42", "42"),
    # ... and a box still outranks the label
    ("Work: \\boxed{9}\nAnswer: 3", "9"),
    # "answer:" mid-sentence is NOT a label line: the old fallback still rules
    ("so the answer: 7 apples", "7"),
]

# --- the QWEN grader (grading.grader_for -> extract_final_answer_qwen /
# answers_equal_qwen). Each group pins a failure observed on
# zs_qwen3_4b_gsm8k_all/*_qabrief, 2026-08-18: markdown emphasis around the
# answer, trailing unit words, think blocks, and \boxed{} mid-derivation
# followed by a later Answer line.
QWEN_SHOULD_MATCH = [
    # trailing unit words (542 baseline rows graded 0-for-542 over these)
    ("135 minutes", "135"),
    ("40 minutes.", "40"),
    ("250 liters", "250"),
    ("7 pounds", "7"),
    ("15 miles per hour", "15"),
    ("72 square feet", "72"),
    ("18 dollars", "18"),
    ("$18", "18"),
    # markdown emphasis survived extraction in older gen files' stored preds
    ("22**", "22"),
    ("**495**", "495"),
    ("*5*", "5"),
    # both at once, plus the llama grader's own behaviors passing through
    ("**40 minutes**", "40"),
    ("50%", "50"),
    ("1,000", "1000"),
    (r"x = 8", "8"),
    # --- patterns from the 2026-08-18 label audit (all were missed corrects) ---
    # currency symbols beyond $, either side, attached or spaced
    ("€32", "32"),
    ("25€", "25"),
    ("8£", "8"),
    ("2500 €", "2500"),
    ("€450", "450"),
    # attached units, including superscripts
    ("25ml", "25"),
    ("6kg", "6"),
    ("980 cm³", "980"),
    ("2350 square feet", "2350"),
    # clock time on the hour is the hour it names
    ("2:00 pm", "2"),
    # derivation/sentence label tails asserting their final value
    ("11 - 2 - 3 - 4 = 2 ounces", "2"),
    ("1200 / (10 x 8) = 15. Each orc has to carry 15 pounds of swords.", "15"),
    ("€1500 - €1423 = €77", "77"),
    ("50 + 6 + 1 = 57 inches", "57"),
    ("got 108 reward", "108"),
]

QWEN_SHOULD_NOT_MATCH = [
    # genuinely wrong terse answers from the same rows
    ("135 minutes", "1350"),
    ("5 days", "7"),
    ("160 seconds", "223"),
    # scale words change the value: NOT unit tails
    ("7 million", "7"),
    ("5 dozen", "5"),
    ("3 tenths", "3"),
    ("5 squared", "5"),
    # a tail with digits asserts more than its leading number
    ("3 hours 30 minutes", "3"),
    # emphasis stripping must not equate different values
    ("**22**", "23"),
    # --- guards on the audit-driven relaxations ---
    # a clock time NOT on the hour is not its hour
    ("7:30", "7"),
    # multi-part sentences joined by connectives assert no single number
    ("120 counters and 200 marbles", "320"),
    ("120 counters and 200 marbles", "200"),
    ("between 3 and 5", "5"),
    # unit CONVERSION is a value change, not formatting (gold is minutes)
    ("6.5 hours", "390"),
    # scale words survive the sentence fallback too
    ("that makes 7 million", "7"),
    # currency stripping must not equate different values
    ("€32", "23"),
]

QWEN_SHOULD_EXTRACT = [
    # the steered-generation ending that exploded flipped-to-wrong
    ("$$\n90 - 68 = 22\n$$\n\n**Final answer: 22**", "22"),
    ("Adding these together: 495.\n\n**Answer: 495**.", "495"),
    ("**Answer:** 42", "42"),
    ("Final answer: 40 minutes.", "40 minutes"),
    # think blocks are not answers, even when they contain numbers
    ("<think>maybe 7? no.</think>\nAnswer: 9", "9"),
    ("<think>unclosed and truncated 123", ""),
    # the LAST asserted value wins: label after a mid-derivation box...
    ("First $\\boxed{68}$ apples.\nThen 90-68=22.\n\nAnswer: 22", "22"),
    # ... and a box after a label still wins, as in the llama grader
    ("Answer: 3\nWait: $\\boxed{9}$", "9"),
    # edge decoration is peeled from the label capture, not from values
    ("So 2**3 = 8.\n\n**Answer: 8**", "8"),
    ("Answer: `12`", "12"),
    # no label, no box: the llama fallbacks are unchanged
    ("The final answer is 18", "18"),
    ("She makes $1,250 total.", "1,250"),
    # a bolded token after a marker phrase is the answer, numeric or not
    ("Option values: 3, 7. The final answer is **B**", "B"),
    ("The values are 3 and 7. The final answer is **18**", "18"),
    # ... but the LAST bold group is the assertion, and a bold LABEL
    # mid-sentence loses to the number (regression row gsm8k idx 5452)
    ("So, Esperanza's **gross monthly salary** is **$4840**.", "$4840"),
    ("So, the **total** is 42", "42"),
]


def main() -> int:
    from src.core.grading import answers_equal_qwen, extract_final_answer_qwen

    failures = []
    for pred, gold in SHOULD_MATCH:
        if not answers_equal(pred, gold):
            failures.append(f"  MISSED MATCH      {pred!r} vs {gold!r}")
    for pred, gold in SHOULD_NOT_MATCH:
        if answers_equal(pred, gold):
            failures.append(f"  SPURIOUS MATCH    {pred!r} vs {gold!r}")
    for text, want in SHOULD_EXTRACT:
        got = extract_final_answer(text)
        if got.strip() != want:
            failures.append(f"  BAD EXTRACTION    {text!r} -> {got!r}, want {want!r}")

    for pred, gold in QWEN_SHOULD_MATCH:
        if not answers_equal_qwen(pred, gold):
            failures.append(f"  QWEN MISSED MATCH   {pred!r} vs {gold!r}")
    for pred, gold in QWEN_SHOULD_NOT_MATCH:
        if answers_equal_qwen(pred, gold):
            failures.append(f"  QWEN SPURIOUS MATCH {pred!r} vs {gold!r}")
    for text, want in QWEN_SHOULD_EXTRACT:
        got = extract_final_answer_qwen(text)
        if got.strip() != want:
            failures.append(f"  QWEN BAD EXTRACTION {text!r} -> {got!r}, want {want!r}")
    # the llama grader must be UNTOUCHED by the qwen one: its own suites above
    # already ran, but pin the exact divergence cases too
    for pred, gold in (("135 minutes", "135"), ("22**", "22")):
        if answers_equal(pred, gold):
            failures.append(f"  LLAMA GRADER DRIFT  {pred!r} vs {gold!r} now matches")

    n = (len(SHOULD_MATCH) + len(SHOULD_NOT_MATCH) + len(SHOULD_EXTRACT)
         + len(QWEN_SHOULD_MATCH) + len(QWEN_SHOULD_NOT_MATCH)
         + len(QWEN_SHOULD_EXTRACT) + 2)
    if failures:
        print(f"FAILED {len(failures)}/{n}")
        print("\n".join(failures))
        return 1
    print(f"OK: {len(SHOULD_MATCH)} should-match, "
          f"{len(SHOULD_NOT_MATCH)} should-not-match, "
          f"{len(SHOULD_EXTRACT)} extraction, "
          f"{len(QWEN_SHOULD_MATCH)}+{len(QWEN_SHOULD_NOT_MATCH)}+"
          f"{len(QWEN_SHOULD_EXTRACT)} qwen, {n}/{n} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
