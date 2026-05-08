from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", "", (answer or "").strip().lower())


def extract_answer(text: str) -> str:
    matches = BOXED_RE.findall(text or "")
    return normalize_answer(matches[-1] if matches else text)


def extract_question(text: str) -> str:
    matches = QUESTION_RE.findall(text or "")
    return (matches[-1] if matches else text or "").strip()


def majority_vote(responses: Iterable[str]) -> tuple[str, float, dict[str, int]]:
    answers = [extract_answer(x) for x in responses]
    answers = [x for x in answers if x]
    if not answers:
        return "", 0.0, {}
    counts = Counter(answers)
    answer, count = counts.most_common(1)[0]
    return answer, count / len(answers), dict(counts)


def uncertainty_interval(consensus_ratio: float, num_intervals: int) -> int:
    r = min(1.0, max(0.0, float(consensus_ratio)))
    return min(int(num_intervals * r), num_intervals - 1)

