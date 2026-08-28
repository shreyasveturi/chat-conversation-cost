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
# Version 2. v1 was measured at 78% precision / 88% recall on a 200-prompt blind
# sample (see METHODOLOGY.md §1.3). Both of its failure modes were structural
# rather than a matter of vocabulary coverage:
#
#   Recall: a fixed 2-keyword threshold cannot fire on short prompts. "Write
#   mini injector in C", "How to read first byte from a bytearray" and
#   "add some space between bottomNavigationBar and bottom of screen" are
#   unmistakably coding and hit zero or one keyword. Evidence must be judged
#   relative to prompt length.
#
#   Precision: the same fixed threshold over-fires on long prose that merely
#   mentions technical words — an interview role-play, a physics paragraph, a
#   worldbuilding prompt. A 900-word essay containing "api" and "linux" is not
#   a coding request.
#
# So the threshold now scales with length, and two high-precision structural
# signals (code punctuation shapes, camelCase/snake_case identifiers) are added
# so that short code questions are caught without loosening the prose rule.

CODING_KEYWORDS = frozenset(
    """
    python javascript typescript java kotlin swift golang rust cpp c++ csharp
    php ruby perl scala haskell matlab sql html css jquery react angular vue
    svelte nodejs npm yarn webpack django flask fastapi spring laravel rails
    tensorflow pytorch numpy pandas sklearn opencv unity unreal godot flutter
    def function class import export const let var async await promise
    traceback stacktrace exception nullpointer segfault compile compiler
    debug debugger breakpoint refactor repo repository commit git github
    api endpoint json xml yaml regex algorithm bigo runtime array dataframe
    dataframes integer boolean nullable dereference pointer malloc struct
    typedef enum bytearray websocket websockets tcp udp http https ssh
    docker kubernetes nginx apache linux bash shell terminal sudo chmod
    stdout stderr localhost postgres mysql mongodb redis sqlite orm
    frontend backend fullstack devops css3 html5 dom xpath selenium
    syntaxerror typeerror valueerror indexerror keyerror importerror
    hibernate jdbc dart scaffold widget npm pip conda venv
    """.split()
)

CODE_BLOCK_RE = re.compile(r"```")

# Structural code punctuation. Each alternative is meant to be rare in ordinary
# English prose, because these fire on a single match.
CODE_SHAPE_RE = re.compile(
    r"(^\s*(def|class|function|import|from|package|public\s+static|#include|using\s+namespace)\s)"
    r"|(=>\s*\{)"
    r"|(^\s*[$#]\s+\w+\s)"
    # method/attribute call: pd.concat(...), obj.method(...) — v1 required a
    # trailing semicolon, which no Python or JS snippet has.
    r"|(\b\w+\.\w+\s*\([^)]*\))"
    # assignment from a call: x = foo(...)
    r"|(\b\w+\s*=\s*\w+(\.\w+)*\s*\()"
    # C/C++/Java declarations and pointers
    r"|(\b(int|char|float|double|void|bool|string)\s*\*?\s*\w+\s*[;=\[])"
    r"|(::\w+)|(->\w+)|(\w+<\w+>)",
    re.MULTILINE,
)

# camelCase / snake_case / SCREAMING_CASE identifiers. Two or more distinct ones
# is a strong code signal that needs no vocabulary at all — this is what catches
# "bottomNavigationBar", "combined_df", "OnTryBuyItem".
IDENTIFIER_RE = re.compile(r"\b(?:[a-z]+(?:[A-Z][a-z0-9]+)+|[a-z0-9]+_[a-z0-9_]+)\b")

WORD_RE = re.compile(r"[a-z0-9+#]+")

# Long agent/system scaffolds (AutoGPT dumps and similar). These are developer
# traffic rather than consumer chat, but they are not coding *requests*, so
# conflating them with the coding filter made its precision look worse than it
# is. They get their own drop reason.
AGENT_SCAFFOLD_RE = re.compile(
    r"(short term memory)|(GOALS:\s*\n)|(COMMANDS:\s*\n)"
    r"|(You are .{0,40}(GPT|AI|autonomous agent))"
    r"|(\bconstraints:\s*\n?\s*1\.)",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def coding_score(text: str) -> tuple[int, bool, bool]:
    """Return (distinct keyword hits, has fenced code block, has code shape).

    Kept for backwards compatibility and for error analysis.
    """
    norm = _normalise(text)
    words = set(WORD_RE.findall(norm))
    hits = len(words & CODING_KEYWORDS)
    return hits, bool(CODE_BLOCK_RE.search(text)), bool(CODE_SHAPE_RE.search(text))


def is_agent_scaffold(text: str) -> bool:
    """Long autonomous-agent prompt dumps, which are not consumer chat."""
    return len(text) > 600 and bool(AGENT_SCAFFOLD_RE.search(text))


def is_coding(text: str, min_keyword_hits: int = 2,
              drop_on_code_block: bool = True) -> bool:
    """Length-aware coding classifier.

    `min_keyword_hits` is the threshold for a mid-length prompt; short prompts
    need less evidence and long prose needs more. Passing 1 (as the sensitivity
    analysis does) shifts the whole ladder down by one.
    """
    norm = _normalise(text)
    tokens = WORD_RE.findall(norm)
    n_words = len(tokens)
    hits = len(set(tokens) & CODING_KEYWORDS)

    if drop_on_code_block and CODE_BLOCK_RE.search(text):
        return True
    if drop_on_code_block and CODE_SHAPE_RE.search(text):
        return True
    if len(set(IDENTIFIER_RE.findall(text))) >= 2:
        return True

    # Evidence required scales with length: a 15-word question mentioning one
    # programming term is almost certainly a coding question, while a 600-word
    # essay mentioning two is almost certainly not.
    if n_words <= 40:
        threshold = max(1, min_keyword_hits - 1)
    elif n_words <= 200:
        threshold = min_keyword_hits
    else:
        threshold = min_keyword_hits + 1
    return hits >= threshold


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

    # Reported separately from "coding": these are developer traffic but not
    # coding requests, and lumping them together misstated the coding filter's
    # precision.
    if is_agent_scaffold(first_user):
        return False, "agent_scaffold"

    cf = f["coding"]
    if is_coding(first_user, cf["min_keyword_hits"], cf["drop_on_code_block"]):
        return False, "coding"

    return True, "ok"
