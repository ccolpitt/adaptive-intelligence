"""Statistical helpers for gate decisions and hypothesis criteria.

Every promotion / regression decision in the loop is a proportion comparison
made from finite game samples, so every decision has a false-positive rate
(promoting a challenger that is not actually better) and a false-negative
rate (rejecting one that is). These helpers make both explicit:

- False positives are controlled by one-sided confidence levels (z):
  z = 1.645 -> ~5% FP rate per decision; z = 2.326 -> ~1%.
- False negatives are controlled by sample size: with n games, effects
  smaller than about z * sqrt(0.5/n) are UNDETECTABLE. ``detectable_effect``
  reports this so every experiment can state up front what "no effect" does
  and does not mean. An undetected effect is recorded as inconclusive, not
  refuted.

Draws are counted as half-wins throughout (score = (W + D/2) / N), and the
binomial machinery is applied to that score fraction — a standard,
slightly conservative approximation given how rare Connect 4 draws are.
"""

from __future__ import annotations

import math

# One-sided z values.
Z_95 = 1.645
Z_99 = 2.326


def wilson_interval(score: float, n: int, z: float = Z_95) -> tuple:
    """Wilson score interval for a proportion estimated as ``score`` from
    ``n`` trials. Robust for small n and extreme proportions, unlike the
    normal approximation."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (score + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(score * (1.0 - score) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_lower(score: float, n: int, z: float = Z_95) -> float:
    return wilson_interval(score, n, z)[0]


def significantly_better(
    score_a: float, n_a: int, score_b: float, n_b: int, z: float = Z_95
) -> bool:
    """One-sided two-proportion test: is A's true score plausibly above B's?
    Returns True only when the observed difference exceeds z standard errors
    — i.e. H0 (A <= B) is rejected at the level implied by z."""
    return (score_a - score_b) > z * _se_diff(score_a, n_a, score_b, n_b)


def significantly_worse(
    score_a: float, n_a: int, score_b: float, n_b: int, z: float = Z_95
) -> bool:
    """One-sided: is A's true score plausibly below B's? Used as the
    regression gate — we block promotion only on EVIDENCE of regression,
    not on noise."""
    return (score_b - score_a) > z * _se_diff(score_a, n_a, score_b, n_b)


def detectable_effect(n: int, z: float = Z_95) -> float:
    """Smallest score difference detectable at level z with n games per arm
    (worst case p=0.5). Effects below this read as 'inconclusive' — state
    this in every hypothesis so null results are not mistaken for refutation."""
    if n <= 0:
        return 1.0
    return z * math.sqrt(0.5 / n)


def _se_diff(p_a: float, n_a: int, p_b: float, n_b: int) -> float:
    if n_a <= 0 or n_b <= 0:
        return float("inf")
    var_a = max(p_a * (1.0 - p_a), 1e-9) / n_a
    var_b = max(p_b * (1.0 - p_b), 1e-9) / n_b
    return math.sqrt(var_a + var_b)
