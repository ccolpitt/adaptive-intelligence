"""The "select" step — double-gated champion/challenger selection.

Promote iff (a) the challenger beats the incumbent champion (relative gate)
AND (b) it does not regress on the fixed benchmark (absolute gate). The
relative gate alone is the connect4-rl v32 failure mode: 32 promotions with
unknown absolute strength.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    relative_pass: bool
    absolute_pass: bool
    reason: str


def double_gate(
    challenger_win_rate: float,
    challenger_benchmark: float,
    champion_benchmark: float,
    win_rate_threshold: float = 0.55,
    benchmark_regression_tolerance: float = 0.02,
) -> GateDecision:
    """``benchmark_regression_tolerance`` absorbs benchmark sampling noise;
    it is NOT a licence to regress (tolerance << any real improvement)."""
    relative = challenger_win_rate >= win_rate_threshold
    absolute = challenger_benchmark >= champion_benchmark - benchmark_regression_tolerance
    if relative and absolute:
        reason = (
            f"promoted: beat champion ({challenger_win_rate:.2f} >= "
            f"{win_rate_threshold}) and held benchmark "
            f"({challenger_benchmark:.3f} vs {champion_benchmark:.3f})"
        )
    elif relative and not absolute:
        reason = (
            f"rejected: beat champion ({challenger_win_rate:.2f}) but REGRESSED on "
            f"benchmark ({challenger_benchmark:.3f} < {champion_benchmark:.3f} - "
            f"{benchmark_regression_tolerance}) — the v32 trap, caught"
        )
    elif absolute and not relative:
        reason = f"rejected: lost to champion ({challenger_win_rate:.2f} < {win_rate_threshold})"
    else:
        reason = (
            f"rejected: lost to champion ({challenger_win_rate:.2f}) and regressed "
            f"on benchmark ({challenger_benchmark:.3f} vs {champion_benchmark:.3f})"
        )
    return GateDecision(
        promote=relative and absolute,
        relative_pass=relative,
        absolute_pass=absolute,
        reason=reason,
    )
