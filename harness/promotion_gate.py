"""The "select" step — double-gated champion/challenger selection.

Promote iff (a) the challenger beats the incumbent champion (relative gate)
AND (b) it does not regress on the fixed benchmark (absolute gate). The
relative gate alone is the connect4-rl v32 failure mode: 32 promotions with
unknown absolute strength.

Two implementations:

- ``double_gate`` — the original fixed-threshold version (Phase 0).
- ``statistical_double_gate`` — the noise-aware version the loop uses.
  Relative gate: promote only if the Wilson lower confidence bound of the
  head-to-head score exceeds 0.5 (one-sided; z controls the false-promotion
  rate). Absolute gate: block only if the challenger's benchmark score is
  SIGNIFICANTLY worse than the champion's (one-sided two-proportion test) —
  benchmark noise alone never blocks, and beating the champion while
  measurably regressing on the benchmark never promotes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import Z_95, significantly_worse, wilson_lower


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


def statistical_double_gate(
    challenger_h2h_score: float,
    h2h_games: int,
    challenger_benchmark: float,
    challenger_benchmark_games: int,
    champion_benchmark: float,
    champion_benchmark_games: int,
    z_promote: float = Z_95,
    z_regress: float = Z_95,
) -> GateDecision:
    """Noise-aware double gate.

    False positives (promoting a non-improvement): controlled by z_promote —
    the head-to-head Wilson lower bound must clear 0.5, so a lucky streak in
    a small sample cannot promote.
    False negatives (rejecting a real improvement): a true edge smaller than
    ~z*sqrt(0.5/n) is undetectable at the given game count; such challengers
    are rejected THIS iteration but keep training — rejection here is "not
    proven yet", never "proven bad".
    """
    lb = wilson_lower(challenger_h2h_score, h2h_games, z_promote)
    relative = lb > 0.5
    regressed = significantly_worse(
        challenger_benchmark,
        challenger_benchmark_games,
        champion_benchmark,
        champion_benchmark_games,
        z_regress,
    )
    absolute = not regressed

    h2h_txt = f"h2h {challenger_h2h_score:.2f} over {h2h_games} games (95% LB {lb:.3f})"
    bench_txt = f"bench {challenger_benchmark:.3f} vs champion {champion_benchmark:.3f}"
    if relative and absolute:
        reason = f"promoted: {h2h_txt} > 0.5, no significant benchmark regression ({bench_txt})"
    elif relative and not absolute:
        reason = (
            f"rejected: {h2h_txt} > 0.5 BUT SIGNIFICANT benchmark regression "
            f"({bench_txt}) — the v32 trap, caught statistically"
        )
    elif not relative and absolute:
        reason = (
            f"rejected (not proven yet): {h2h_txt} LB <= 0.5; improvement, if real, "
            f"is below the detectable effect at n={h2h_games}"
        )
    else:
        reason = f"rejected: {h2h_txt} LB <= 0.5 AND significant benchmark regression ({bench_txt})"
    return GateDecision(
        promote=relative and absolute,
        relative_pass=relative,
        absolute_pass=absolute,
        reason=reason,
    )
