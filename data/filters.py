"""Shared filtering logic for Study A and Study B.

Both studies draw from the same population, so the filters live in one place.
Every filter returns a reason string when it rejects, and the caller is expected
to tally those reasons into a filter log. A filter you cannot audit is a filter
you should not trust.
"""

from __future__ import annotations

import re
import unicodedata

# --- coding heuristic -------------------------------------------------------
# Deliberately conservative on the keyword side (>=2 distinct hits required)
# because single words like "function" or "class" appear constantly in ordinary
# English. The fenced-code-block rule is the high-precision half of the filter.

CODING_KEYWORDS = frozenset(
    """
    python javascript typescript java kotlin swift golang rust cpp c++ csharp
    php ruby perl scala haskell matlab sql html css jquery react angular vue
    svelte nodejs npm yarn webpack django flask fastapi spring laravel rails
    tensorflow pytorch numpy pandas sklearn opencv unity unreal godot
    def function() class import export const let var async await promise
    traceback stacktrace exception nullpointer segfault compile compiler
    debug debugger breakpoint refactor repo repository commit git github
    api endpoint json xml yaml regex algorithm bigo runtime array dataframe
    integer boolean nullable dereference pointer malloc struct typedef enum
    docker kubernetes nginx apache linux bash shell terminal sudo chmod
    stdout stderr localhost postgres mysql mongodb redis sqlite orm
    frontend backend fullstack devops css3 html5 dom xpath selenium
    syntaxerror typeerror valueerror indexerror keyerror importerror
    """.split()
)

CODE_BLOCK_RE = re.compile(r"```")
# Common code-ish shapes that a keyword list misses: assignment with semicolon,
# arrow functions, decorators, shell prompts.
CODE_SHAPE_RE = re.compile(
    r"(^\s*(def|class|function|import|from|package|public\s+static)\s)"
    r"|(=>\s*\{)"
    r"|(^\s*[$#]\s+\w+\s)"
    r"|(\w+\.\w+\([^)]*\)\s*;)",
    re.MULTILINE,
)

WORD_RE = re.compile(r"[a-z0-9+#]+")


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def coding_score(text: str) -> tuple[int, bool, bool]:
    """Return (distinct keyword hits, has fenced code block, has code shape)."""
    norm = _normalise(text)
    words = set(WORD_RE.findall(norm))
    hits = len(words & CODING_KEYWORDS)
    return hits, bool(CODE_BLOCK_RE.search(text)), bool(CODE_SHAPE_RE.search(text))


def is_coding(text: str, min_keyword_hits: int, drop_on_code_block: bool) -> bool:
    hits, has_block, has_shape = coding_score(text)
    if drop_on_code_block and (has_block or has_shape):
        return True
    return hits >= min_keyword_hits


# --- conversation shape -----------------------------------------------------


def extract_turns(conversation: list[dict]) -> list[tuple[str, str]]:
    """Collapse a WildChat conversation into [(user_text, assistant_text), ...].

    WildChat alternates user/assistant, but not every conversation is clean:
    trailing user messages with no reply, or consecutive same-role messages,
    both occur. We pair greedily and drop an unmatched trailing user message,
    because a turn with no assistant reply has no reply to have paid for.
    """
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            # Consecutive user messages: concatenate, they were sent as one
            # context block before the next reply.
            pending_user = content if pending_user is None else f"{pending_user}\n{content}"
        elif role == "assistant":
            if pending_user is None:
                continue  # assistant with no preceding user turn; skip
            turns.append((pending_user, content))
            pending_user = None
    return turns


# --- top-level record filter ------------------------------------------------


def check_record(record: dict, cfg: dict) -> tuple[bool, str]:
    """Apply the shared filters to one dataset record.

    Returns (keep, reason). `reason` is "ok" when kept, otherwise the name of
    the first filter that rejected it.
    """
    f = cfg["filters"]

    if record.get("language") != f["language"]:
        return False, "not_english"

    if f["drop_toxic"] and record.get("toxic"):
        return False, "toxic_flagged"

    conversation = record.get("conversation") or []
    turns = extract_turns(conversation)
    if not turns:
        return False, "no_paired_turns"

    first_user = turns[0][0].strip()
    if len(first_user) < f["min_user_chars"]:
        return False, "first_turn_too_short"
    if len(first_user) > f["max_user_chars"]:
        return False, "first_turn_too_long"

    cf = f["coding"]
    if is_coding(first_user, cf["min_keyword_hits"], cf["drop_on_code_block"]):
        return False, "coding"

    return True, "ok"
