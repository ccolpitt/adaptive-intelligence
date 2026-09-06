"""Statistical helpers: sanity of Wilson bounds and one-sided comparisons —
the machinery that keeps gate decisions honest about noise."""

import pytest

from harness.stats import (
    Z_95,
    detectable_effect,
    significantly_better,
    significantly_worse,
    wilson_interval,
    wilson_lower,
)


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(0.6, 100)
    assert lo < 0.6 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_narrows_with_n():
    lo_small, hi_small = wilson_interval(0.6, 20)
    lo_big, hi_big = wilson_interval(0.6, 500)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_small_sample_is_humble():
    # 7/10 wins is NOT significant evidence of >50% true strength at z=1.645.
    assert wilson_lower(0.7, 10) < 0.5
    # 80/100 clearly is.
    assert wilson_lower(0.8, 100) > 0.5
    # The same 70% win rate becomes significant once the sample is large.
    assert wilson_lower(0.7, 200) > 0.5


def test_wilson_edge_cases():
    assert wilson_interval(0.0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(1.0, 50)
    assert lo < 1.0  # perfect record still gets an honest lower bound
    lo, hi = wilson_interval(0.0, 50)
    assert hi > 0.0


def test_significance_needs_both_effect_and_sample():
    # Big effect, big sample: significant.
    assert significantly_better(0.70, 200, 0.50, 200)
    # Same effect, tiny sample: not significant (false-positive control).
    assert not significantly_better(0.70, 10, 0.50, 10)
    # Tiny effect, big sample: not significant.
    assert not significantly_better(0.52, 200, 0.50, 200)


def test_worse_is_mirror_of_better():
    assert significantly_worse(0.50, 200, 0.70, 200)
    assert not significantly_worse(0.68, 200, 0.70, 200)  # noise, not regression


def test_detectable_effect_shrinks_with_n():
    assert detectable_effect(50) > detectable_effect(500)
    # With ~168 games (ladder aggregate) at z=1.645, effects under ~9pp of
    # score are invisible — hypotheses must say so.
    assert 0.05 < detectable_effect(168, Z_95) < 0.12
