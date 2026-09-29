"""
CER / WER Metrics cho Handwritten Text Recognition.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# =========================================================
# Levenshtein
# =========================================================
def levenshtein_counts(
    reference: Sequence,
    hypothesis: Sequence
) -> Tuple[int, int, int]:
    """
    Tính số phép:
    - S: Substitution
    - D: Deletion
    - I: Insertion
    Trả về:
        (S, D, I)
    """
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [["same"] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "same"
                continue
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1
            substitution = dp[i - 1][j - 1] + 1
            best = min(deletion, insertion, substitution)
            dp[i][j] = best
            if best == substitution:
                op[i][j] = "S"
            elif best == deletion:
                op[i][j] = "D"
            else:
                op[i][j] = "I"
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        current = op[i][j]
        if current == "same":
            i -= 1
            j -= 1
        elif current == "S":
            S += 1
            i -= 1
            j -= 1
        elif current == "D":
            D += 1
            i -= 1
        elif current == "I":
            I += 1
            j -= 1
    return S, D, I

# =========================================================
# Distance
# =========================================================
def levenshtein_distance(
    reference: Sequence,
    hypothesis: Sequence
) -> int:
    S, D, I = levenshtein_counts(reference, hypothesis)
    return S + D + I

# =========================================================
# Safe division
# =========================================================
def _safe_rate(
    errors: int,
    reference_length: int
) -> float:
    if reference_length == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / reference_length

# =========================================================
# CER
# =========================================================
def cer(
    reference: str,
    hypothesis: str,
    as_percent: bool = True
) -> float:
    """
    Character Error Rate.
    CER = (S + D + I) / N * 100
    """
    S, D, I = levenshtein_counts(
        list(reference),
        list(hypothesis)
    )
    value = _safe_rate(
        S + D + I,
        len(reference)
    )
    return value * 100.0 if as_percent else value

# =========================================================
# WER
# =========================================================
def wer(
    reference: str,
    hypothesis: str,
    as_percent: bool = True
) -> float:
    """
    Word Error Rate.
    WER = (S + D + I) / N * 100
    """
    S, D, I = levenshtein_counts(
        reference.split(),
        hypothesis.split()
    )
    value = _safe_rate(
        S + D + I,
        len(reference.split())
    )
    return value * 100.0 if as_percent else value

# =========================================================
# Batch CER / WER
# =========================================================
def batch_cer_wer(
    references: Iterable[str],
    hypotheses: Iterable[str],
    as_percent: bool = True
) -> Tuple[float, float]:
    """
    Tính CER và WER trên toàn bộ batch/dataset.
    """
    total_char_errors = 0
    total_ref_chars = 0
    total_word_errors = 0
    total_ref_words = 0
    for ref, hyp in zip(references, hypotheses):
        S, D, I = levenshtein_counts(
            list(ref),
            list(hyp)
        )
        total_char_errors += S + D + I
        total_ref_chars += len(ref)
        S, D, I = levenshtein_counts(
            ref.split(),
            hyp.split()
        )
        total_word_errors += S + D + I
        total_ref_words += len(ref.split())
    char_rate = _safe_rate(
        total_char_errors,
        total_ref_chars
    )
    word_rate = _safe_rate(
        total_word_errors,
        total_ref_words
    )
    if as_percent:
        char_rate *= 100.0
        word_rate *= 100.0
    return char_rate, word_rate

# =========================================================
# MetricAccumulator
# =========================================================
@dataclass
class MetricAccumulator:
    char_errors: int = 0
    char_ref_length: int = 0
    word_errors: int = 0
    word_ref_length: int = 0
    def update(
        self,
        references: Iterable[str],
        hypotheses: Iterable[str]
    ) -> None:
        for ref, hyp in zip(references, hypotheses):
            S, D, I = levenshtein_counts(
                list(ref),
                list(hyp)
            )
            self.char_errors += S + D + I
            self.char_ref_length += len(ref)
            S, D, I = levenshtein_counts(
                ref.split(),
                hyp.split()
            )
            self.word_errors += S + D + I
            self.word_ref_length += len(ref.split())
    def compute(
        self,
        as_percent: bool = True
    ) -> Tuple[float, float]:
        char_rate = _safe_rate(
            self.char_errors,
            self.char_ref_length
        )
        word_rate = _safe_rate(
            self.word_errors,
            self.word_ref_length
        )
        if as_percent:
            char_rate *= 100.0
            word_rate *= 100.0
        return char_rate, word_rate
    def reset(self) -> None:
        self.char_errors = 0
        self.char_ref_length = 0
        self.word_errors = 0
        self.word_ref_length = 0

__all__ = [
    "levenshtein_counts",
    "levenshtein_distance",
    "cer",
    "wer",
    "batch_cer_wer",
    "MetricAccumulator",
]