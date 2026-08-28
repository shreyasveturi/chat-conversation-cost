"""Study A1: what a long conversation actually costs, in input tokens.

This is arithmetic, not an experiment. Every turn of a chat resends the entire
prior thread as input, so cumulative input tokens grow roughly with the square of
turn count. This script measures that on real conversations.

Definitions (all per conversation, turns indexed from 1):

    u_t   tokens in the user message at turn t   (+ per-message format overhead)
    a_t   tokens in the assistant reply at turn t (+ per-message format overhead)
    H_t   resent history at turn t = system + sum_{i<t} (u_i + a_i)
    I_t   input tokens billed at turn t = H_t + u_t + reply-priming
    C_T   cumulative input tokens through turn T = sum_{t<=T} I_t

Two "wasted share" metrics, because the honest answer depends on what you count
as new, and we would rather show both than quietly pick the flattering one:

    history_share   (C_T - sum u_t) / C_T
        Share of everything you have ever sent that was NOT your new question.
        This is the headline. It answers "how much of my quota went on
        re-transmitting the thread rather than on asking things?"

    repeat_share    (C_T - sum novel_t) / C_T,  novel_t = u_t + a_{t-1}
        Stricter. Counts an assistant reply as new the first time it is sent
        back as input, and only calls a token repeated once the model has
        genuinely seen it as input before. Always lower than history_share.

Usage:
    python study_a/compute_costs.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import yaml


def load_conversations(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def tokenize_all(conversations: list[dict], encoding_name: str) -> list[list[tuple[int, int]]]:
    """Return, per conversation, a list of (user_tokens, assistant_tokens)."""
    enc = tiktoken.get_encoding(encoding_name)

    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for conv in conversations:
        start = len(flat)
        for user, assistant in conv["turns"]:
            flat.append(user)
            flat.append(assistant)
        spans.append((start, len(flat)))

    print(f"[tokenize] {len(flat):,} messages with {encoding_name} ...", flush=True)
    lengths = [len(ids) for ids in enc.encode_ordinary_batch(flat, num_threads=8)]

    out = []
    for start, end in spans:
        chunk = lengths[start:end]
        out.append(list(zip(chunk[0::2], chunk[1::2])))
    return out


def per_turn_frame(conversations: list[dict], token_pairs, tok_cfg: dict) -> pd.DataFrame:
    per_msg = tok_cfg["tokens_per_message"]
    priming = tok_cfg["tokens_reply_priming"]
    system = tok_cfg["system_prompt_tokens"]

    rows = []
    for conv, pairs in zip(conversations, token_pairs):
        history = system
        cum_input = 0
        cum_user = 0
        cum_novel = 0
        first_input = None
        prev_assistant = 0

        for idx, (u_raw, a_raw) in enumerate(pairs, start=1):
            u = u_raw + per_msg
            a = a_raw + per_msg

            input_tokens = history + u + priming
            novel = u + (prev_assistant if idx > 1 else system)

            cum_input += input_tokens
            cum_user += u
            cum_novel += novel
            if first_input is None:
                first_input = input_tokens

            rows.append({
                "conversation_id": conv["conversation_id"],
                "n_turns": conv["n_turns"],
                "turn": idx,
                "user_tokens": u,
                "assistant_tokens": a,
                "history_tokens": history,
                "input_tokens": input_tokens,
                "cum_input_tokens": cum_input,
                "cum_user_tokens": cum_user,
                "history_share_cum": (cum_input - cum_user) / cum_input,
                "repeat_share_cum": (cum_input - cum_novel) / cum_input,
                "history_share_turn": history / input_tokens,
                "relative_cost_vs_turn1": input_tokens / first_input,
            })

            history += u + a
            prev_assistant = a

    return pd.DataFrame(rows)


METRICS = [
    "input_tokens",
    "cum_input_tokens",
    "history_share_cum",
    "history_share_turn",
    "repeat_share_cum",
    "relative_cost_vs_turn1",
]


def aggregate(df: pd.DataFrame, max_turn: int, min_n: int) -> pd.DataFrame:
    """Median with interquartile band at each turn index, plus the surviving n.

    Note the survivorship caveat: at turn index k only conversations with at
    least k turns contribute, and long conversations are not a random subset of
    all conversations. We report n at every index and truncate once it thins out.
    See also the balanced-panel variant below.
    """
    grouped = df[df.turn <= max_turn].groupby("turn")
    out = pd.DataFrame({"n": grouped.size()})
    for metric in METRICS:
        out[f"{metric}_p25"] = grouped[metric].quantile(0.25)
        out[f"{metric}_p50"] = grouped[metric].median()
        out[f"{metric}_p75"] = grouped[metric].quantile(0.75)
    out = out[out.n >= min_n].reset_index()
    return out


def headline(df: pd.DataFrame, agg: pd.DataFrame, cfg: dict) -> dict:
    def at(turn: int, col: str):
        row = agg[agg.turn == turn]
        return None if row.empty else float(row.iloc[0][col])

    last_turn = int(agg.turn.max())
    # The last reported turn is always included, so downstream consumers (the
    # README renderer, the charts) can always look up the end of the curve
    # whether or not it happens to land on a round number.
    marks = sorted({1, 5, 10, 20, 30, last_turn})
    # Per-conversation end-state: the share at each conversation's own final turn.
    finals = df.sort_values("turn").groupby("conversation_id").tail(1)

    return {
        "n_conversations": int(df.conversation_id.nunique()),
        "n_turns_median": float(finals.n_turns.median()),
        "tokenizer": cfg["tokenizer"]["encoding"],
        "max_turn_reported": last_turn,
        "history_share_cum_at_turn": {
            str(t): at(t, "history_share_cum_p50") for t in marks
            if at(t, "history_share_cum_p50") is not None
        },
        "repeat_share_cum_at_turn": {
            str(t): at(t, "repeat_share_cum_p50") for t in marks
            if at(t, "repeat_share_cum_p50") is not None
        },
        "relative_cost_vs_turn1_at_turn": {
            str(t): at(t, "relative_cost_vs_turn1_p50") for t in marks
            if at(t, "relative_cost_vs_turn1_p50") is not None
        },
        "cum_input_tokens_at_turn": {
            str(t): at(t, "cum_input_tokens_p50") for t in marks
            if at(t, "cum_input_tokens_p50") is not None
        },
        "final_turn_history_share_median": float(finals.history_share_cum.median()),
        "final_turn_history_share_p25": float(finals.history_share_cum.quantile(0.25)),
        "final_turn_history_share_p75": float(finals.history_share_cum.quantile(0.75)),
    }


def quadratic_check(agg: pd.DataFrame) -> dict:
    """Fit log(cum_input) ~ alpha * log(turn) and report the exponent.

    The claim is that cumulative input grows about quadratically. If alpha comes
    out near 2, the claim holds on this sample; if it does not, we say so.
    """
    sub = agg[agg.turn >= 2]
    x = np.log(sub.turn.to_numpy(dtype=float))
    y = np.log(sub.cum_input_tokens_p50.to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2))
    return {"growth_exponent": float(slope), "log_log_r2": r2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--conversations", default="data/out/study_a_conversations.jsonl")
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(Path(args.conversations))
    print(f"[load] {len(conversations):,} conversations", flush=True)

    # Every downstream statistic groups by conversation_id. If the sampler ever
    # fails to populate it (e.g. a renamed upstream column), the ids collapse to
    # a single group and every "median across conversations" silently becomes a
    # median across one. Fail loudly instead.
    ids = [c["conversation_id"] for c in conversations]
    if len(set(ids)) != len(ids):
        raise SystemExit(
            f"conversation_id is not unique ({len(set(ids)):,} distinct for "
            f"{len(ids):,} conversations). Check the sampler's id column."
        )

    pairs = tokenize_all(conversations, cfg["tokenizer"]["encoding"])
    df = per_turn_frame(conversations, pairs, cfg["tokenizer"])

    sa = cfg["study_a"]
    agg = aggregate(df, sa["report_to_turn"], sa["min_conversations_per_turn"])

    # Balanced-panel robustness check: only conversations long enough to appear
    # at every reported turn index, so the curve cannot be an artefact of the
    # sample composition changing as turns advance.
    panel_len = int(agg.turn.max())
    balanced = df[df.n_turns >= panel_len]
    agg_balanced = aggregate(balanced, panel_len, 1)

    df.to_csv(tables / "study_a_per_turn.csv", index=False)
    agg.to_csv(tables / "study_a_by_turn_index.csv", index=False)
    agg_balanced.to_csv(tables / "study_a_by_turn_index_balanced.csv", index=False)

    head = headline(df, agg, cfg)
    head["growth"] = quadratic_check(agg)
    # At the final turn index the balanced panel is by construction identical to
    # the unbalanced one, so comparing there tests nothing. The informative
    # comparison is at a mid turn, where the unbalanced curve still contains
    # short conversations that the balanced panel excludes.
    def share_at(frame, turn):
        row = frame[frame.turn == turn]
        return None if row.empty else float(row.iloc[0]["history_share_cum_p50"])

    compare_turn = min(10, panel_len)
    head["balanced_panel"] = {
        "min_turns": panel_len,
        "n_conversations": int(balanced.conversation_id.nunique()),
        "compare_turn": compare_turn,
        "history_share_balanced_at_compare_turn": share_at(agg_balanced, compare_turn),
        "history_share_unbalanced_at_compare_turn": share_at(agg, compare_turn),
        "history_share_cum_at_final_turn": float(
            agg_balanced.iloc[-1]["history_share_cum_p50"]
        ),
    }
    (tables / "study_a_headline.json").write_text(json.dumps(head, indent=2))

    print("\n=== Study A headline ===")
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
