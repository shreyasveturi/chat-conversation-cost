"""Hand-checkable tests for the Study A cost arithmetic.

The whole of Study A rests on these formulas being right, so they are verified
against numbers worked out by hand rather than against the code's own output.

Run:  python study_a/test_arithmetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_costs import per_turn_frame  # noqa: E402

# Zero overhead keeps the hand-arithmetic readable; overhead is tested separately.
NO_OVERHEAD = {"tokens_per_message": 0, "tokens_reply_priming": 0, "system_prompt_tokens": 0}


def check(label: str, got, want) -> None:
    ok = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got}, want {want}")
    if not ok:
        raise AssertionError(label)


def test_three_equal_turns() -> None:
    """Three turns, every message exactly 10 tokens.

    Worked by hand:
      u_t = a_t = 10 for all t
      I_1 = 10                 (nothing before it)
      I_2 = (10+10) + 10 = 30
      I_3 = (10+10+10+10) + 10 = 50
      C_3 = 10 + 30 + 50 = 90
      sum u_t = 30  ->  history_share at turn 3 = (90-30)/90 = 2/3
      novel_t = u_t + a_{t-1}: 10, 20, 20 -> sum 50
                            ->  repeat_share at turn 3 = (90-50)/90 = 4/9
      relative cost of turn 3 = 50/10 = 5
    """
    print("test_three_equal_turns")
    conv = [{"conversation_id": "x", "n_turns": 3, "turns": [("", "")] * 3}]
    df = per_turn_frame(conv, [[(10, 10)] * 3], NO_OVERHEAD)

    check("I_1", int(df.input_tokens[0]), 10)
    check("I_2", int(df.input_tokens[1]), 30)
    check("I_3", int(df.input_tokens[2]), 50)
    check("C_3", int(df.cum_input_tokens[2]), 90)
    check("history_share@3", float(df.history_share_cum[2]), 2 / 3)
    check("repeat_share@3", float(df.repeat_share_cum[2]), 4 / 9)
    check("relative_cost@3", float(df.relative_cost_vs_turn1[2]), 5.0)


def test_first_turn_is_all_new() -> None:
    """At turn 1 there is no history, so no share of the spend can be resent."""
    print("test_first_turn_is_all_new")
    conv = [{"conversation_id": "x", "n_turns": 1, "turns": [("", "")]}]
    df = per_turn_frame(conv, [[(42, 99)]], NO_OVERHEAD)
    check("history_share@1", float(df.history_share_cum[0]), 0.0)
    check("repeat_share@1", float(df.repeat_share_cum[0]), 0.0)
    check("relative_cost@1", float(df.relative_cost_vs_turn1[0]), 1.0)


def test_quadratic_growth() -> None:
    """With equal-length messages, C_T should be exactly quadratic in T.

    u = a = 1 token, no overhead:  I_t = 2(t-1) + 1,  so C_T = T^2.
    """
    print("test_quadratic_growth")
    n = 20
    conv = [{"conversation_id": "x", "n_turns": n, "turns": [("", "")] * n}]
    df = per_turn_frame(conv, [[(1, 1)] * n], NO_OVERHEAD)
    for t in (1, 5, 10, 20):
        check(f"C_{t} == {t}^2", int(df.cum_input_tokens[t - 1]), t * t)


def test_overhead_is_applied() -> None:
    """Per-message and priming overhead must show up in the billed input."""
    print("test_overhead_is_applied")
    cfg = {"tokens_per_message": 4, "tokens_reply_priming": 3, "system_prompt_tokens": 11}
    conv = [{"conversation_id": "x", "n_turns": 2, "turns": [("", "")] * 2}]
    df = per_turn_frame(conv, [[(10, 10)] * 2], cfg)
    # I_1 = system 11 + (10+4) + priming 3 = 28
    check("I_1 with overhead", int(df.input_tokens[0]), 28)
    # I_2 = 11 + (14 + 14) + 14 + 3 = 56
    check("I_2 with overhead", int(df.input_tokens[1]), 56)


def test_repeat_share_never_exceeds_history_share() -> None:
    """repeat_share is the stricter metric and must never be the larger one."""
    print("test_repeat_share_never_exceeds_history_share")
    import random
    rng = random.Random(0)
    pairs = [(rng.randint(1, 500), rng.randint(1, 3000)) for _ in range(40)]
    conv = [{"conversation_id": "x", "n_turns": 40, "turns": [("", "")] * 40}]
    df = per_turn_frame(conv, [pairs], NO_OVERHEAD)
    worst = float((df.repeat_share_cum - df.history_share_cum).max())
    check("max(repeat - history) <= 0", worst <= 1e-12, True)


if __name__ == "__main__":
    for fn in (
        test_three_equal_turns,
        test_first_turn_is_all_new,
        test_quadratic_growth,
        test_overhead_is_applied,
        test_repeat_share_never_exceeds_history_share,
    ):
        fn()
    print("\nAll arithmetic tests passed.")
