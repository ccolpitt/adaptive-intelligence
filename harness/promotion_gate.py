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

from .stats import Z_95, significantly_better, significantly_worse, wilson_lower, wilson_upper


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    relative_pass: bool
    absolute_pass: bool
    reason: str
    # Signals for the hypothesis generator (ADR-007): a non-promotion at
    # PARITY (h2h CI straddles 0.5 and benchmark not below the champion's)
    # means the lever that produced it is probably sound — scale it rather
    # than abandon it.
    parity: bool = False
    bench_gain_significant: bool = False


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


def statistical_either_gate(
    challenger_h2h_score: float,
    h2h_games: int,
    challenger_benchmark: float,
    challenger_benchmark_games: int,
    champion_benchmark: float,
    champion_benchmark_games: int,
    z_promote: float = Z_95,
    z_regress: float = Z_95,
) -> GateDecision:
    """Gate v2 (ADR-005, human-directed): promote on EITHER strength proof,
    provided the other axis shows no significant regression.

    Path A (relative): significantly beats the champion head-to-head
    (Wilson LB > 0.5) AND no significant benchmark regression.
    Path B (absolute): significantly better on the fixed benchmark
    (one-sided two-proportion test) AND no significant head-to-head
    regression (Wilson UPPER bound must not fall below 0.5 — true parity
    passes, a proven losing matchup does not).

    Rationale: the double gate let the relative match VETO absolute
    progress (exp-012 iter 11: challenger benchmark 0.116 > champion 0.083,
    blocked on h2h 0.41 — the reverse of the v32 failure). Principle 1 says
    the absolute yardstick is the ground truth; head-to-head demotes from
    co-equal gate to regression guard on Path B. Note iter-11 itself would
    STILL fail this gate: its h2h upper bound (~0.47) is significantly
    below parity, so it regressed, not merely differed.
    """
    lb = wilson_lower(challenger_h2h_score, h2h_games, z_promote)
    ub = wilson_upper(challenger_h2h_score, h2h_games, z_promote)
    h2h_superior = lb > 0.5
    h2h_regressed = ub < 0.5
    bench_superior = significantly_better(
        challenger_benchmark, challenger_benchmark_games,
        champion_benchmark, champion_benchmark_games, z_promote,
    )
    bench_regressed = significantly_worse(
        challenger_benchmark, challenger_benchmark_games,
        champion_benchmark, champion_benchmark_games, z_regress,
    )
    path_a = h2h_superior and not bench_regressed
    path_b = bench_superior and not h2h_regressed

    h2h_txt = (f"h2h {challenger_h2h_score:.2f} over {h2h_games} games "
               f"(95% CI {lb:.3f}-{ub:.3f})")
    bench_txt = (f"bench {challenger_benchmark:.3f} vs champion "
                 f"{champion_benchmark:.3f}")
    if path_a and path_b:
        reason = f"promoted (both paths): {h2h_txt} beats champion AND {bench_txt} superior"
    elif path_a:
        reason = f"promoted (relative path): {h2h_txt} > 0.5, no significant benchmark regression ({bench_txt})"
    elif path_b:
        reason = (f"promoted (absolute path): {bench_txt} significantly superior, "
                  f"h2h at parity or better ({h2h_txt})")
    elif h2h_superior and bench_regressed:
        reason = (f"rejected: {h2h_txt} beats champion BUT SIGNIFICANT benchmark "
                  f"regression ({bench_txt}) — the v32 trap, caught")
    elif bench_superior and h2h_regressed:
        reason = (f"rejected: {bench_txt} superior BUT significant h2h regression "
                  f"({h2h_txt}, UB < 0.5) — stronger on the yardstick, proven "
                  f"weaker against the incumbent; needs both")
    else:
        reason = (f"rejected (not proven yet): neither {h2h_txt} LB > 0.5 nor "
                  f"{bench_txt} significant at n={challenger_benchmark_games}")
    return GateDecision(
        promote=path_a or path_b,
        relative_pass=h2h_superior,
        absolute_pass=not bench_regressed,
        reason=reason,
    )


def statistical_strict_gate(
    challenger_h2h_score: float,
    h2h_games: int,
    challenger_benchmark: float,
    challenger_benchmark_games: int,
    champion_benchmark: float,
    champion_benchmark_games: int,
    z_promote: float = Z_95,
    z_regress: float = Z_95,
) -> GateDecision:
    """Gate v3 (ADR-007, human-directed): promote ONLY if the challenger
    beats the champion head-to-head (Wilson LB > 0.5) AND improves on the
    absolute benchmark (score strictly above the champion's; whether that
    gain is statistically significant is reported, not required — at 168
    ladder games significance would need ~+0.09 and would freeze promotion).

    Everything else is rejected, but rejections carry a PARITY signal: h2h
    confidence interval containing 0.5 with benchmark >= champion's. Parity
    is not failure — it says the lever is probably sound and under-dosed
    (e.g. "more episodes got us to parity; double them"). The hypothesis
    generator reads this flag; the gate does not act on it.
    """
    lb = wilson_lower(challenger_h2h_score, h2h_games, z_promote)
    ub = wilson_upper(challenger_h2h_score, h2h_games, z_promote)
    h2h_superior = lb > 0.5
    h2h_parity = lb <= 0.5 <= ub
    bench_improved = challenger_benchmark > champion_benchmark
    bench_significant = significantly_better(
        challenger_benchmark, challenger_benchmark_games,
        champion_benchmark, champion_benchmark_games, z_promote,
    )
    parity = h2h_parity and challenger_benchmark >= champion_benchmark
    promote = h2h_superior and bench_improved

    h2h_txt = (f"h2h {challenger_h2h_score:.2f} over {h2h_games} games "
               f"(95% CI {lb:.3f}-{ub:.3f})")
    bench_txt = (f"bench {challenger_benchmark:.3f} vs champion {champion_benchmark:.3f}"
                 + (" [significant]" if bench_significant else " [within noise]"))
    if promote:
        reason = f"promoted: beats champion ({h2h_txt}) AND improves benchmark ({bench_txt})"
    elif h2h_superior:
        reason = (f"rejected: beats champion ({h2h_txt}) but does NOT improve the "
                  f"benchmark ({bench_txt}) — strict gate requires both")
    elif parity:
        reason = (f"rejected at PARITY: {h2h_txt} straddles 0.5 and {bench_txt} — "
                  f"lever likely sound and under-dosed; hypothesis generator: scale it")
    elif bench_improved:
        reason = (f"rejected: improves benchmark ({bench_txt}) but loses head-to-head "
                  f"({h2h_txt})")
    else:
        reason = f"rejected: neither beats champion ({h2h_txt}) nor improves benchmark ({bench_txt})"
    return GateDecision(
        promote=promote,
        relative_pass=h2h_superior,
        absolute_pass=bench_improved,
        reason=reason,
        parity=parity,
        bench_gain_significant=bench_significant,
    )


GATE_RULES = {
    "double": statistical_double_gate,   # ADR-era original: h2h win, bench as regression guard
    "either": statistical_either_gate,   # ADR-005: either axis wins, other must not regress
    "strict": statistical_strict_gate,   # ADR-007: must win BOTH axes
}
