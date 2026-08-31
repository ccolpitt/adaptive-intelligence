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
