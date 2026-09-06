"""Selection gate: table-driven tests over all four quadrants of the double
gate — promotion iff (beats champion) AND (no benchmark regression)."""

import pytest

from harness.promotion_gate import double_gate


CASES = [
    # (wr, challenger_bench, champion_bench, expect_promote, label)
    (0.60, 0.80, 0.75, True, "beats champion, improves benchmark"),
    (0.60, 0.75, 0.75, True, "beats champion, holds benchmark exactly"),
    (0.60, 0.74, 0.75, True, "beats champion, within regression tolerance"),
    (0.60, 0.60, 0.75, False, "beats champion, REGRESSES benchmark (v32 trap)"),
    (0.50, 0.90, 0.75, False, "improves benchmark, loses to champion"),
    (0.40, 0.50, 0.75, False, "loses both gates"),
    (0.55, 0.75, 0.75, True, "exactly at win-rate threshold"),
    (0.549, 0.75, 0.75, False, "just below win-rate threshold"),
]


@pytest.mark.parametrize("wr,cb,champ_b,expect,label", CASES)
def test_double_gate_quadrants(wr, cb, champ_b, expect, label):
    d = double_gate(wr, cb, champ_b, win_rate_threshold=0.55, benchmark_regression_tolerance=0.02)
    assert d.promote is expect, label
    assert d.promote == (d.relative_pass and d.absolute_pass)
    assert d.reason  # every decision explains itself


def test_v32_trap_is_named_in_reason():
    d = double_gate(0.70, 0.40, 0.80)
    assert not d.promote
    assert d.relative_pass and not d.absolute_pass
    assert "REGRESSED" in d.reason


# -- statistical gate -------------------------------------------------------

from harness.promotion_gate import statistical_double_gate  # noqa: E402

BENCH_N = 168  # ladder aggregate sample size


def stat_gate(h2h, n, bench_c, bench_champ):
    return statistical_double_gate(h2h, n, bench_c, BENCH_N, bench_champ, BENCH_N)


def test_stat_gate_promotes_clear_winner():
    d = stat_gate(0.65, 200, 0.30, 0.25)
    assert d.promote


def test_stat_gate_lucky_small_sample_does_not_promote():
    # 60% of 20 games: point estimate says "better", statistics say "unproven".
    d = stat_gate(0.60, 20, 0.30, 0.30)
    assert not d.promote
    assert "not proven yet" in d.reason


def test_stat_gate_same_winrate_promotes_with_enough_games():
    # The same 60% IS evidence at 200 games — false negatives are a sample
    # size problem, not a threshold problem.
    d = stat_gate(0.60, 200, 0.30, 0.30)
    assert d.promote


def test_stat_gate_benchmark_noise_does_not_block():
    # Slightly lower benchmark score, well within noise at n=168: promote.
    d = stat_gate(0.65, 200, 0.28, 0.30)
    assert d.promote


def test_stat_gate_significant_regression_blocks():
    # Beats the champion head-to-head but collapses on the ladder: the v32
    # trap. 0.10 vs 0.30 at n=168 is way beyond noise.
    d = stat_gate(0.70, 200, 0.10, 0.30)
    assert not d.promote
    assert d.relative_pass and not d.absolute_pass
    assert "v32" in d.reason
