# Methodology

This document is the full protocol. It exists so that someone who distrusts the
headline number can find out exactly how it was produced and where it might be
wrong. Limitations are stated in the same voice as the results, not buried.

Phase 1 contains two studies:

- **Study A** — what long conversations cost. Pure arithmetic over real chat
  logs. No model calls, no judge.
- **Study B** — model over-service, prompt by prompt. Requires generation and
  LLM-as-judge grading. *Not yet run; see the status note in the README.*

---

## 1. Data

### 1.1 Source

`allenai/WildChat-1M` — one million real user conversations with ChatGPT-family
assistants, collected by AI2. It is the closest public approximation to a
consumer chat query distribution.

The dataset is public and **not gated**: it needs no Hugging Face token and no
terms acceptance. It was gated under AI2's ImpACT licence until 26 June 2024,
when AI2 relicensed it to [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)
and applied that change retroactively. Older instructions saying otherwise —
including an earlier version of this file — are out of date.

Under ODC-BY, redistribution of the database and of derived subsets is
permitted with attribution, which is why the validation prompt samples in
`data/out/` are committed. AI2 de-identified the corpus with Microsoft Presidio
plus hand-written rules before release.

We do not write our own prompts. Self-authored prompt sets unconsciously skew
easy, and a number derived from them means nothing.

`lmsys/lmsys-chat-1m` is a reasonable second source. It is not used here:
WildChat alone is sufficient for Study A, and adding a second corpus with a
different collection methodology would mix two populations without adding much.

### 1.2 Shard sampling

The corpus is stored as parquet shards. We do not read all of them. We read
every *n*-th shard (`dataset.shard_stride` in `config.yaml`, currently 3), which
spreads the sample across the whole file rather than concentrating it in
whichever period happens to be stored first — WildChat is broadly chronological,
so reading only the head would bias toward the earliest collection window.

WildChat-1M ships as 14 shards of roughly 60,000 conversations each. About 1.3%
of records survive the filters at ≥10 turns, so a shard yields ~770 usable
conversations and stride 3 (shards 0, 3, 6, 9, 12) gives a pool of ~3,850 —
comfortably above the 3,000 target.

The exact shard list is recorded in the filter log, so the sample is
reproducible.

If the repo listing is unavailable — no token, or offline — the sampler falls
back to whatever shards are already in the local Hugging Face cache rather than
failing outright. This is a convenience, not a sampling strategy: the stride is
not applied, and the shard set may be partial. The filter log records
`shard_list_from_local_cache: true` whenever this happened, and the README
surfaces it as a caveat. The results currently published were **not** produced
this way: they read shards 0, 3, 6, 9 and 12 at the configured stride, as the
filter log records.

### 1.3 Filters

Applied identically in both studies, so the two are drawn from one population.
Implemented in `data/filters.py`; every rejection is tallied by reason into
`data/out/study_a_filter_log.json`.

| Filter | Rule | Why |
|---|---|---|
| Language | WildChat `language == "English"` | Tokenizer behaviour and the judging protocol are both English-calibrated. |
| Toxicity | WildChat `toxic` flag set → drop | Removes adversarial, jailbreak and NSFW traffic, which is not the consumer usage we are measuring. |
| Length floor | first user turn < 8 chars | Drops empty turns, "hi", and connection tests. |
| Length ceiling | first user turn > 20,000 chars | A pasted 50-page document is a real use case but it dominates token statistics and is not representative. |
| Agent scaffold | long AutoGPT-style prompt dumps → drop | Developer traffic, not consumer chat. Reported separately from coding so neither filter's accuracy is misstated. |
| Coding | see below | Public chat logs skew technical; this is a consumer-chat study. |
| Near-duplicate | SHA-1 of whitespace-collapsed, lowercased first 300 chars | Templated openers are re-sent by many users. |
| Turn count (Study A only) | fewer than 10 turns → drop | Study A is about long threads. |

**The coding filter** runs in two stages. Stage 1, used in both studies, is a
heuristic over the first user turn (`is_coding` in `data/filters.py`). It drops
a prompt on any of:

- a fenced code block;
- a code-shaped construct — a `def`/`class`/`import`/`#include` opener, an arrow
  function, a shell prompt, a method call such as `pd.concat(...)`, an
  assignment from a call, a C-style declaration, `::`, `->` or `<T>`;
- two or more distinct camelCase / snake_case identifiers, which is a strong
  code signal needing no vocabulary at all;
- enough hits against a ~140-word programming vocabulary, where "enough" scales
  with prompt length: **1 hit** for prompts of ≤40 words, **2** up to 200 words,
  **3** beyond that.

The length-scaled threshold is the key design choice, and it exists because the
fixed-threshold first version failed in two measurable ways. See the validation
below.

Stage 2 is a model classifier pass over the survivors. **It applies to Study B
only**, where we are already paying for inference. Study A therefore relies on
the heuristic alone.

#### Validation of the coding filter

Eyeballing only the survivors would measure nothing: it can reveal false
negatives but says nothing about ordinary prompts wrongly discarded. So
`data/validate_coding_filter.py` samples **both sides** of the decision — 100
prompts the filter flagged and 100 it let through — shuffles them, and hides the
verdict in a separate key file. The population sampled is the one the filter
acts on inside the pipeline: records already past the language, toxicity, length
and agent-scaffold filters, with at least 10 turns.

**v1**, a fixed two-keyword threshold, scored **77.8% precision / 87.5% recall**
on 197 labelled prompts, implying about 440 coding conversations still in the
pool. Its two failure modes were structural rather than gaps in vocabulary:

1. **A fixed threshold cannot fire on short prompts.** "Write mini injector in
   C", "How to read first byte from a bytearray" and "add some space between
   bottomNavigationBar and bottom of screen" are unmistakably coding and hit
   zero or one keyword.
2. **The same threshold over-fires on long prose** that merely mentions
   technical words — an interview role-play, a physics paragraph, a fantasy
   worldbuilding prompt. A 900-word essay containing "api" and "linux" is not a
   coding request.

**v2** addresses both: the keyword threshold now scales with prompt length
(short prompts need less evidence, long prose needs more), the code-shape regex
was fixed to match calls without a trailing semicolon — v1's pattern could not
match any Python or JavaScript snippet — and camelCase/snake_case identifier
density was added as a vocabulary-free structural signal. Long autonomous-agent
prompt dumps (AutoGPT-style "CONSTRAINTS: 1. ~4000 word limit …") now get their
own `agent_scaffold` drop reason; they are developer traffic rather than
consumer chat, but they are not coding *requests*, and conflating them made the
coding filter's precision look worse than it was.

**v2 was tuned on the v1 validation set and then evaluated on a fresh held-out
sample** (different seed, 181 prompts not present in the first set). This
matters: on the tuning set v2 appeared to reach 83.7% precision, but on held-out
data it scores **74.1% precision / 91.3% recall** — a ~10-point optimism gap
that is exactly what in-sample evaluation of a tuned classifier produces. The
held-out figures are the ones reported.

| | v1 | v2 |
|---|---:|---:|
| precision | 77.8% | 74.1% |
| recall | 87.5% | 91.3% |
| miss rate among survivors | 11.2% | **6.4%** |
| est. coding conversations left in pool | ~440 | **~228** |

v2 therefore trades precision for recall and roughly halves contamination. That
is the right trade here — the goal is a clean consumer-chat sample, there is
ample data, and a wrongly-dropped borderline prompt costs less than a coding
prompt surviving into Study B's task mix. The cost is real and worth stating:
v2 flags 610 conversations where v1 flagged 313, and at 74% precision roughly
160 of those are ordinary prompts discarded for looking technical.

> **Direct v1-vs-v2 comparison on one sample is not sound**, and the repo does
> not claim it. Each validation set is stratified by the filter being tested, so
> running v1 against v2's sample structurally penalises v1's recall (it scores
> an implausible 31.9% there). The comparable quantity is the population-level
> miss rate in the table above, which is estimated the same way for both.

> **Caveat on the labels.** These labels were produced by the assistant, not by
> a human. The brief calls for the repo owner to hand-label a subset, and that
> check is still outstanding. Both blind CSVs and keys are committed
> (`data/out/coding_validation*`) so anyone can relabel and re-score with
> `python data/validate_coding_filter.py --score --prefix coding_validation_test`.
> Labelling used the strict definition "asks for code, debugging, or a
> programming explanation"; sysadmin, networking-concept and data-science-theory
> prompts were labelled *not* coding, which is a defensible but arguable line.

**How much does that line matter?** Enough to report. Re-labelling under a
broader rubric — where technical/developer-domain traffic counts as coding even
with no code requested — moves 8 of the 179 held-out prompts (Linux server
setup, `tcp_quickack`/`tcp_mtu_probing` sysctl questions, RedHat + Active
Directory, an AWS AMI lab):

| | strict rubric | broad rubric |
|---|---:|---:|
| precision | 74.1% | **81.2%** |
| recall | 91.3% | 90.8% |
| est. coding left in pool | ~228 | ~266 |

**Precision is rubric-dependent by about 7 points; recall is not.** That
asymmetry is worth knowing: the claim "this filter catches ~91% of coding
prompts" is robust to where you draw the line, while "~74% of what it drops is
really coding" is not — much of the disputed remainder is sysadmin and
networking, which a reasonable person could file either way. The strict figures
are the ones published, because they are the less flattering pair.

#### Does the contamination matter?

For Study A, no. `study_a/sensitivity_coding.py` re-runs the entire cost
computation on an aggressively re-filtered subset (one keyword hit is enough to
drop), removing a further 108 conversations. The headline metrics barely move:

| metric | baseline (n=3,000) | strict (n=2,892) | delta |
|---|---:|---:|---:|
| history share @ turn 10 | 96.9% | 96.9% | +0.06pp |
| history share @ turn 30 | 98.8% | 98.8% | +0.00pp |
| growth exponent | 1.951 | 1.958 | +0.006 |

This is the expected result rather than a lucky one: Study A measures the
*shape* of a conversation — how many turns, how long each message is — and that
arithmetic does not care whether the topic is code. Residual coding content is
not load-bearing here.

**It will matter for Study B**, where the task mix directly determines the
over-service estimate and coding is exactly the category where a cheap model is
most likely to fall short. The stage-2 classifier pass is therefore not optional
for Study B.

## 2. Study A — the cost of a long conversation

### 2.1 The claim

Chat assistants are stateless. Every turn resends the entire prior thread as
input. Cumulative input tokens therefore grow roughly with the square of turn
count: each successive answer costs more than the last for the same amount of
new information.

This is arithmetic, not an experiment. It requires no model calls and no judge,
which is what makes it the most defensible result in the project.

### 2.2 Definitions

For one conversation, turns indexed from 1, where turn *t* is a user message
followed by an assistant reply:

```
u_t   tokens in the user message at turn t      (+ per-message format overhead)
a_t   tokens in the assistant reply at turn t   (+ per-message format overhead)
H_t   resent history at turn t = system + Σ_{i<t} (u_i + a_i)
I_t   input tokens billed at turn t = H_t + u_t + reply-priming
C_T   cumulative input tokens through turn T = Σ_{t≤T} I_t
```

Two "wasted share" metrics are reported. The honest answer depends on what you
count as new, so both are published rather than quietly picking the flattering
one.

**`history_share` (headline)**

```
history_share(T) = (C_T − Σ_{t≤T} u_t) / C_T
```

The share of everything ever sent that was *not* your new question. This is the
metric that answers what a capped user actually wants to know: how much of my
quota went on re-transmitting the thread rather than on asking things?

**`repeat_share` (stricter, reported alongside)**

```
novel_t = u_t + a_{t−1}          (at t = 1, novel_1 = u_1 + system)
repeat_share(T) = (C_T − Σ_{t≤T} novel_t) / C_T
```

This counts an assistant reply as *new* the first time it is sent back as input,
and only calls a token repeated once the model has genuinely seen it as input
before. It is always the lower of the two. Where the README quotes one number it
quotes `history_share`, and the charts plot both.

**Relative cost** is `I_t / I_1` — what one turn costs as a multiple of the
first turn in the same conversation.

### 2.3 Turn pairing

WildChat conversations are not always clean. Consecutive same-role messages
occur, and a trailing user message with no reply occurs. We pair greedily:
consecutive user messages are concatenated (they were sent as one context block
before the next reply), an assistant message with no preceding user turn is
skipped, and an unmatched trailing user message is dropped — a turn with no
reply has no reply to have paid for.

### 2.4 Verification

The arithmetic is the whole of Study A, so it is tested against hand-worked
numbers rather than against its own output. `study_a/test_arithmetic.py` checks,
among others, that with unit-length messages and zero overhead the cumulative
input is exactly `T²`; that turn 1 has no resent history; that format overhead
lands where it should; and that `repeat_share` never exceeds `history_share`.

Run it with `python study_a/test_arithmetic.py`.

### 2.5 Aggregation and the survivorship caveat

We report the median with an interquartile band across conversations at each
turn index, never a single illustrative conversation.

At turn index *k*, only conversations with at least *k* turns contribute — and
long conversations are **not a random subset** of all conversations. A curve
that rises with turn index could in principle reflect the sample composition
changing rather than any within-conversation effect.

Two things guard against reading too much into that:

1. `n` is reported at every turn index, and the curve is truncated once fewer
   than `min_conversations_per_turn` (currently 100) conversations remain.
2. A **balanced panel** is computed alongside: the same curves restricted to
   conversations long enough to appear at *every* reported turn index, so the
   composition is fixed. It is written to
   `results/tables/study_a_by_turn_index_balanced.csv`.

   The comparison is made at a **mid turn**, not at the final one. At the final
   turn index the balanced panel is by construction the same set of
   conversations as the unbalanced one, so comparing there tests nothing — an
   earlier version of this document made exactly that mistake.

   **Result on the current sample:** restricting to the 118 conversations that
   run to turn 30, the resent-history share at turn 10 is 96.02%, against 96.98%
   for the full 3,000-conversation sample — a gap of about one percentage point.
   Composition therefore accounts for very little of the rise, which is a
   within-conversation effect rather than a change in which conversations are
   being averaged. Note this gap was near zero on an earlier, smaller sample; it
   is small but not identically zero, and longer conversations do skew very
   slightly toward a lower resent share at any given turn.

Note also that the ≥10-turn requirement means Study A describes *long*
conversations specifically. It is not a claim about the average chat, most of
which are one or two turns.

### 2.6 Mandatory caveats

**Prompt caching.** Providers cache repeated prefixes, and a cached prefix is
billed at a fraction of the normal input rate or not at all. Whether and how
caching applies inside a consumer subscription is not observable from outside.
**Every token count in Study A is therefore an upper bound on real cost.** The
*shape* of the curve — quadratic growth, rising resent share — is unaffected by
caching, because caching changes the price of a resent token, not the fact that
it is resent. The magnitude is affected, and could be substantially so.

**Tokenizer approximation.** One public tokenizer is used throughout:
`tiktoken`'s `o200k_base`, the GPT-4o/5-family encoding. Other model families
tokenize differently, typically within roughly ±20% on English prose. Absolute
counts are therefore indicative rather than exact, and cross-model token
comparisons should not be read as precise. Anthropic exposes a token-counting
endpoint that would give exact Claude-family numbers; it was not used here
because this session had no Anthropic API credentials, and using the harness's
own credentials for research calls would be a misuse of them.

**Chat format overhead.** Providers wrap each message in role and delimiter
tokens. We model this explicitly as `tokens_per_message = 4` and
`tokens_reply_priming = 3`, following the widely-cited OpenAI cookbook
approximation. It is small relative to message content. A consequence worth
naming: at turn 1 `history_share` is not exactly zero but a fraction of a
percent, because the reply-priming tokens are not part of your question either.

**System prompts.** Consumer products prepend a system prompt whose length is
not public. It is set to 0 tokens, which makes our numbers *conservative* — a
real system prompt is resent on every turn too, and would raise the resent
share.

**Output tokens are excluded.** Study A measures input only. Output tokens are
generated once and not resent, so they do not participate in the quadratic
effect, but they do consume quota. The resent-history share is a share of
*input* spend, not of total spend.

---

### 2.7 Study A2 — is the history load-bearing? (partially run)

Study A1 shows the history is re-transmitted. It does not show the answer needed
it. A2 asks whether the same turn can be answered from less context, and splits
into two halves.

**The accounting half (run).** `study_a/truncation_pilot.py --accounting` needs
no model and bounds the prize — how much input would be saved if you *could*
drop the old context. Answering turn 10 with only the last k turns:

| kept context | median input saved (turn 10) | (turn 20) |
|---|---:|---:|
| last 2 turns only | 75.6% | 88.2% |
| last 4 turns only | 53.1% | 77.5% |
| last 6 turns only | 30.9% | 67.3% |
| summary + last 4 turns (200-token summary) | 43.5% | 71.7% |

The summary length is a parameter, not a measurement — compression cannot be
costed without generating summaries — so it is swept (100/200/400 tokens) rather
than assumed.

**The quality half (NOT run).** Regenerating the turn from full, truncated and
compressed history and judging the three against each other needs API budget
that this project has not spent. Without it, the accounting above says only what
*could* be saved, never whether the answer would survive. Do not cite these
percentages as savings; they are an upper bound on an unproven manoeuvre.

> **Prior work has already done a version of this, and it matters.** Schelpe
> (2026), [*Byte-Exact Deduplication in Retrieval-Augmented Generation*](https://arxiv.org/abs/2605.09611),
> reports 80.34% context reduction in the multi-turn conversational regime
> (5,000 WildChat conversations) and validates it with a five-judge cross-vendor
> panel, concluding "zero measurable quality regression". That is stronger
> evidence on the necessity question than our planned 100-conversation pilot
> would produce, and it points the same way. Our A2, if run, would be a
> replication with a different removal method (truncation and summarisation
> rather than byte-exact dedup), not a first look.

### 2.8 The multiple is inflated by a small baseline

`relative_cost_vs_turn1` reaches 172× at turn 30, which sounds dramatic and is
partly an artefact of what it is divided by. The median WildChat opening message
is **29 tokens**. So 172× is 32 tokens → 6,610 tokens for a single message.

That is a genuine, steeply growing cost, and 6,610 tokens is not nothing when it
recurs every turn under a quota. But it is not a large absolute number, and the
multiple should never be quoted without the token counts beside it. The README
reports both on the same row for that reason.

---

## 3. Study B — model over-service

**Status: not yet run.** The protocol below is the pre-registered design,
written before any results exist so that it cannot be tuned to the outcome.

### 3.1 Sample

400 first-user-turns, stratified across the task taxonomy, seed recorded, drawn
with the filters in §1.3 plus the stage-2 classifier pass. First turn only —
this measures single-request routing, not multi-turn state.

### 3.2 Taxonomy

Exactly one label per prompt: factual lookup/recall · summarisation ·
drafting · explanation/teaching · arithmetic or simple logic · open
advice/recommendation · creative writing · translation/rewriting · multi-step
analysis or reasoning · other.

Model-assisted classification with a human-checked subsample; per-category
counts reported in the README.

### 3.3 Model ladder

At least five models spanning roughly two orders of magnitude in price, from at
least two labs. Model names and prices are **fetched, never remembered** — they
go stale. All prices live in `pricing.yaml` with the date and source URL of each,
so a rerun months later is a one-file edit.

### 3.4 Judging protocol

1. Generate a reference answer from the top-of-ladder model.
2. Compare each cheaper model's answer against the reference on a three-point
   ordinal scale — clearly worse / slightly worse / equivalent or better — not a
   1–10 score, which invites false precision.
3. The judge is from a **different lab** than the model being judged, to limit
   same-family self-preference.
4. Presentation order is randomised, and **every comparison is run twice with
   the order flipped**. Disagreement between the two orderings is a direct
   measure of judge noise and is reported, not hidden.
5. **Human validation:** the repo owner hand-grades a random 100-item subset
   blind. Agreement with the judge is reported as Cohen's κ with a confidence
   interval. *If agreement is poor, that is the headline finding and will be
   reported as such.*

### 3.5 Known judge biases

Stated up front because they cut against the result we are looking for:

- **Verbosity bias** — judges prefer longer answers, largely independent of
  quality. Cheaper models are often terser, so this biases *against* finding
  over-service, making our result conservative in that direction.
- **Position bias** — the answer shown first is favoured. Handled by the
  flipped-order double run.
- **Self-preference** — judges favour their own family's outputs. Handled by
  cross-lab judging.
- **Verifiable-answer unreliability** — LLM judges are weak where a ground truth
  exists. For the arithmetic/logic category, answers are graded against **the
  actual correct answer**, not against the reference model's answer.

See Zheng et al. (2023) for the original characterisation of position and
verbosity bias, and the self-preference literature cited in the README.

### 3.6 Analysis

The central output is deliberately **not a single percentage**. "Adequate" is a
judgement call, and any headline number is entirely determined by where the bar
is set. So the adequacy threshold is the x-axis: it is swept from lenient to
strict, and for each threshold we plot the cheapest model on the ladder that
still clears it, overall and per task category.

Bootstrap confidence intervals on every proportion. `58% ± 9pp` is a claim;
a bare `58%` is a decoration.

---

## 4. Pre-registered commitment

The finding gets published whichever way it comes out. If over-service turns out
to be 30% rather than 90%, that is still a real result, credibly obtained, and a
better public artifact than a project abandoned because the number was
unexciting.

The threshold, the sample and the judge will not be tuned to make the headline
larger.

## 5. What this project does not claim

- **Not** general quality parity between models. The claim is only ever "clears
  this specific bar on this specific suite".
- **Not** a measurement of user behaviour. Any retry-adjusted figure is an
  explicit modelling assumption with a sensitivity range, not an observation.
- **Not** a production context-compression system. Study A measures the cost of
  long threads; the A2 pilot asks whether that cost is avoidable. Neither ships
  a compressor.
- **Not** a claim to have invented routing. See the prior work section of the
  README.
