"""Answer extraction and grading for GSM8k and MATH datasets."""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in text, handling nested braces."""
    for match in reversed(list(re.finditer(r"\\boxed\{", text))):
        start = match.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1].strip()
    return None


def extract_gsm8k_answer(solution: str) -> Optional[str]:
    """Extract the numeric answer from a GSM8k solution string (after ####)."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", solution)
    if m:
        return m.group(1).replace(",", "")
    return None


def _try_numeric(a: str, b: str) -> Optional[bool]:
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return None


def _try_sympy(a: str, b: str) -> Optional[bool]:
    try:
        from sympy import simplify, sympify
        return simplify(sympify(a) - sympify(b)) == 0
    except Exception:
        return None


def grade_answer(predicted: Optional[str], ground_truth: str, dataset: str) -> bool:
    """Return True if predicted matches ground_truth."""
    if predicted is None or not ground_truth:
        return False

    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    if predicted == ground_truth:
        return True

    numeric = _try_numeric(predicted, ground_truth)
    if numeric is not None:
        return numeric

    if dataset == "math":
        sympy_result = _try_sympy(predicted, ground_truth)
        if sympy_result is not None:
            return sympy_result

    return False
