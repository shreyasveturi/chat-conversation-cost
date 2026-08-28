"""Study A2: is the resent history load-bearing?

Study A1 shows the history is re-transmitted. It does not show the answer needed
it. A2 asks whether the same turn can be answered from less context.

The pilot has two halves, and only one of them needs a model:

  --accounting  (no API, runs now)
      How much input would truncation or compression actually save at turn N?
      Pure arithmetic over the sampled conversations. This bounds the prize: if
      truncation saved little, the quality question would not be worth asking.

  --generate / --judge  (needs an API key and budget)
      Regenerate the answer at turn N three ways — full history, last-k turns
      only, and a summary of turns 1..N-k plus the last k verbatim — then judge
      them pairwise against the full-history answer using the Study B protocol:
      three-point ordinal scale, randomised presentation order, every comparison
      run twice with the order flipped, judge from a different lab than the
      generator.

Report it as suggestive. 100 conversations cannot settle whether context is
necessary in general; it can only indicate whether the question deserves a
larger study.

Usage:
    python study_a/truncation_pilot.py --accounting
    python study_a/truncation_pilot.py --generate --model ... --base-url ...
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import tiktoken
import yaml

# Assumed length of a generated summary of the dropped turns. Compression cannot
# be costed without generating summaries, so the accounting pass treats this as
# a parameter and reports sensitivity rather than pretending to know it.
SUMMARY_TOKENS = (100, 200, 400)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def token_lengths(conversations: list[dict], encoding: str) -> list[list[tuple[int, int]]]:
    enc = tiktoken.get_encoding(encoding)
    flat, spans = [], []
    for conv in conversations:
        start = len(flat)
        for user, assistant in conv["turns"]:
            flat.append(user)
            flat.append(assistant)
        spans.append((start, len(flat)))
    lengths = [len(x) for x in enc.encode_ordinary_batch(flat, num_threads=8)]
    return [list(zip(lengths[a:b][0::2], lengths[a:b][1::2])) for a, b in spans]


def accounting(conversations: list[dict], cfg: dict, turn_n: int, ks: list[int]) -> dict:
    """Input tokens at turn N under full / truncated / compressed history."""
    tok = cfg["tokenizer"]
    per_msg, priming = tok["tokens_per_message"], tok["tokens_reply_priming"]
    pairs = token_lengths(conversations, tok["encoding"])

    rows = []
    for conv, pr in zip(conversations, pairs):
        if len(pr) < turn_n:
            continue
        msgs = [(u + per_msg, a + per_msg) for u, a in pr]
        # Input at turn N with the whole thread in front of it.
        history_full = sum(u + a for u, a in msgs[: turn_n - 1])
        current_user = msgs[turn_n - 1][0]
        full = history_full + current_user + priming

        row = {"conversation_id": conv["conversation_id"], "full": full}
        for k in ks:
            kept = msgs[max(0, turn_n - 1 - k): turn_n - 1]
            history_k = sum(u + a for u, a in kept)
            row[f"trunc_k{k}"] = history_k + current_user + priming
            for s in SUMMARY_TOKENS:
                row[f"comp_k{k}_s{s}"] = history_k + s + current_user + priming
        rows.append(row)

    def saving(col: str) -> dict:
        vals = [1 - r[col] / r["full"] for r in rows if r["full"] > 0]
        return {
            "median_saving": statistics.median(vals),
            "p25": statistics.quantiles(vals, n=4)[0],
            "p75": statistics.quantiles(vals, n=4)[2],
        }

    out = {
        "turn_n": turn_n,
        "n_conversations": len(rows),
        "median_full_input_tokens": statistics.median(r["full"] for r in rows),
        "truncation": {f"k={k}": saving(f"trunc_k{k}") for k in ks},
        "compression": {
            f"k={k}, summary={s}t": saving(f"comp_k{k}_s{s}")
            for k in ks for s in SUMMARY_TOKENS
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--conversations", default="data/out/study_a_conversations.jsonl")
    ap.add_argument("--out", default="results/tables/study_a2_accounting.json")
    ap.add_argument("--turn", type=int, default=10, help="turn N to answer")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6],
                    help="how many recent turns to keep verbatim")
    ap.add_argument("--accounting", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.generate or args.judge:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")):
            raise SystemExit(
                "The generate/judge half of A2 needs an API key and a budget.\n"
                "Set OPENROUTER_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY).\n\n"
                "Note the pilot design before spending: 100 conversations x 3\n"
                "history conditions x 2 flipped judge orderings = 600 judge calls\n"
                "plus 300 generations. See METHODOLOGY.md for the protocol.\n\n"
                "`--accounting` runs now and needs nothing."
            )
        raise SystemExit("Generation/judging not yet implemented - see README status.")

    if not args.accounting:
        raise SystemExit("Pass --accounting (free) or --generate/--judge (needs a key).")

    conversations = load(Path(args.conversations))
    result = accounting(conversations, cfg, args.turn, args.k)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"Study A2 accounting - answering turn {result['turn_n']}, "
          f"n={result['n_conversations']:,}")
    print(f"median full-history input: {result['median_full_input_tokens']:,.0f} tokens\n")
    print("Truncation - keep only the last k turns:")
    for key, v in result["truncation"].items():
        print(f"  {key:<8} saves {v['median_saving']:6.1%} of input "
              f"(IQR {v['p25']:.1%}-{v['p75']:.1%})")
    print("\nCompression - summary of dropped turns + last k verbatim:")
    for key, v in result["compression"].items():
        print(f"  {key:<22} saves {v['median_saving']:6.1%}")
    print(f"\n[a2] wrote {args.out}")
    print("\nThis bounds the prize only. Whether the dropped context was needed "
          "is the generate/judge half, which has not been run.")


if __name__ == "__main__":
    main()
