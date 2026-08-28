# The cost of a long conversation

**A measurement of how much of a chat assistant's usage goes on re-sending the
thread rather than answering the question — and, in a second study, how much
everyday usage is served by a model far stronger than the task requires.**

This is a research artifact, not a product. Everything here — the prompt set,
the intermediate tables, the filter logs, the code — is published so the numbers
can be checked rather than believed.

> **Status.** Study A (conversation length) is {{status_a}}.
> Study B (model tier) is **not yet run**; its protocol is pre-registered in
> [METHODOLOGY.md](METHODOLOGY.md) so it cannot be tuned to its own outcome.

---

## The finding

Chat assistants are stateless. Every turn resends the whole prior thread as
input, so the cost of a conversation grows with roughly the *square* of its
length — you pay for turn 1 again at turn 2, and again at turn 3, and so on.

Across **{{n_conversations}}** real English, non-coding WildChat conversations of
ten turns or more:

| | |
|---|---|
| Share of input spend that is resent history, by turn 10 | **{{hist_share_10}}** |
| Share of input spend that is resent history, by turn {{max_turn}} | **{{hist_share_max}}** |
| Cost of turn 10 vs. the first message in the same thread | **{{rel_cost_10}}** |
| Cost of turn {{max_turn}} vs. the first message | **{{rel_cost_max}}** |
| Input tokens for one message at turn 1 → turn {{max_turn}} (median) | **{{input_1}} → {{input_max}}** |
| Cumulative input tokens by turn {{max_turn}} (median) | **{{cum_tokens_max}}** |
| Fitted growth exponent of cumulative input vs. turn count | **{{growth_exponent}}** (R² = {{log_log_r2}}) |

The fitted exponent is the quadratic claim tested rather than asserted: a value
near 2 means cumulative cost really does grow with the square of turn count on
this sample.

### The charts

![Share of cumulative spend that is resent history](results/figures/study_a_fig2_history_share.png)

![Cumulative input tokens by turn](results/figures/study_a_fig1_cumulative_tokens.png)

![Cost of turn N relative to turn 1](results/figures/study_a_fig3_relative_cost.png)

---

## What you can actually do with this

Most people on a consumer plan never see a token price. They meet this cost as a
**usage cap** — the message that gets refused, the "you've reached your limit"
banner — which is why every consumer-facing number in this project is quota, not
money.

The lever is small, real, and almost nobody knows it:

> **Start a fresh conversation when the topic changes.**

Continuing one long thread out of convenience is the largest single lever a
capped user has. On this sample, asking a new question at turn {{max_turn}} of an
existing thread costs about **{{rel_cost_max}}** what the same question costs in
a fresh one — not because the question is harder, but because the thread is
dragged along with it.

That is the whole of the advice. It is not dressed up as more than it is — and
it is not ours alone. Anthropic's own guidance says the same thing:

> "Longer conversations that trigger automatic context management consume more
> of your usage limit." … "Try starting a new conversation if you're approaching
> your usage limit in a longer chat."
> — [Anthropic Help Center](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)

The honest counterweight, which that same guidance makes explicit: a fresh chat
costs you the context you have to re-establish. Starting over on a task that
genuinely depends on the thread means re-explaining it, and paying for that too.
The lever is real when **the topic changes** — it is not an argument for
short-thread purism.

---

## Caveats, stated up front

These are not fine print. They bound what the numbers above mean.

- **Prompt caching makes these an upper bound — and the discount is large.**
  On the Anthropic API, cache *reads* are billed at roughly **0.1×** the normal
  input rate, and a resent conversation prefix is precisely the stable-prefix
  case caching is built for. Anthropic also states that Claude.ai has
  [built-in caching](https://support.claude.com/en/articles/9797557-usage-limit-best-practices)
  for conversation context. So the *monetary* gap between turn 1 and turn 30 is
  far smaller than 172× wherever caching is active. Caching changes the price of
  a resent token, not the fact that it is resent: the **shape** of every curve
  here survives, the **magnitude** does not. Treat every number on this page as
  an upper bound on cost, and a measurement of redundancy rather than of spend.
- **"Resent" is not the same as "wasted", and this study does not measure
  necessity.** It shows the history is re-transmitted; it does not show the
  answer could have been produced without it. The pilot designed to test that —
  regenerating answers from truncated and compressed history — is specified in
  [METHODOLOGY.md](METHODOLOGY.md) and **has not been run**. Until it is, the
  supported claim is "long threads are expensive", not "long threads waste your
  quota".
- **One tokenizer, used consistently.** `tiktoken o200k_base`. Other model
  families tokenize differently; absolute counts are indicative, not exact.
- **This is about long threads specifically.** Conversations of fewer than ten
  turns are excluded by design. Most real chats are one or two turns and are not
  described by these curves.
- **Longer conversations are not a random subset.** At turn *k* only
  conversations with at least *k* turns contribute. `n` is reported at every
  turn index, curves are truncated once the sample thins, and a fixed-composition
  balanced panel tests whether the rise is an artifact of that. Restricting to
  the {{balanced_n}} conversations that run all the way to turn
  {{balanced_turn}}, the resent share at turn {{balanced_compare_turn}} is
  **{{balanced_compare}}**, against **{{unbalanced_compare}}** for the full
  sample — a gap of {{balanced_gap_pp}} percentage points. Composition therefore
  accounts for very little of the rise; the effect is within-conversation.
- **Input tokens only.** Output tokens consume quota too, but they are generated
  once and never resent, so they do not drive the quadratic effect.
- **The multiple is big partly because the baseline is tiny — read the absolute
  numbers.** A median WildChat opener is ~29 tokens, so "{{rel_cost_max}} the
  first message" means going from **{{input_1}} to {{input_max}} tokens** for one
  message. That is a real and fast-growing cost, but it is not a large absolute
  number, and anyone quoting the multiple without the token counts is
  overselling it. The multiple is the right shape; the tokens are the right
  magnitude.
- **The curve thins at high turn counts.** All {{n_conversations}} conversations
  contribute up to turn 10; by turn {{max_turn}} only {{n_at_max}} remain. The
  right-hand end of every chart is the noisiest part.
{{shard_note}}
- **The coding filter is imperfect, and we measured how imperfect.** Validated
  on blind samples drawn from *both* sides of the decision, then rebuilt and
  re-validated on a **held-out** set: **74% precision, 91% recall**, leaving
  roughly one in sixteen surviving conversations still coding-flavoured (down
  from one in nine before the rebuild). A sensitivity re-run that drops a
  further 3.6% of the sample moves the headline shares by **under 0.1pp** and
  the growth exponent by 0.006 — Study A measures conversation *shape*, not
  topic, so the residue is not load-bearing. It would be for Study B. Labels
  were model-generated; a human relabel is still outstanding. See
  [METHODOLOGY.md](METHODOLOGY.md) §1.3.

Full protocol and limitations: **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Prior work

This project did not invent LLM routing, and says so plainly. What it adds is
narrower and, as far as we can tell, unpublished: **how much everyday consumer
usage is over-served, measured over a realistic consumer query distribution
rather than a benchmark suite** — and the companion arithmetic on what
conversation length costs.

**Routing and cost reduction**

- **FrugalGPT** — Chen, Zaharia & Zou (2023), [arXiv:2305.05176](https://arxiv.org/abs/2305.05176).
  LLM cascades, prompt adaptation and approximation; reports large API cost
  reductions at comparable quality. The intellectual ancestor of this area.
- **RouteLLM** — Ong et al., LMSYS (2024), [arXiv:2406.18665](https://arxiv.org/abs/2406.18665),
  ICLR 2025; [blog](https://www.lmsys.org/blog/2024-07-01-routellm/).
  Routers trained on preference data to choose between a strong and a weak model.
  The closest prior work; it optimises the routing policy, where this project
  measures the size of the gap that makes routing worth doing.
- **Not Diamond** and **Martian** — commercial model routers, both predicting
  which model will do best on a query before running it. Evidence that the
  problem is considered real by people spending money on it.
- **RouterArena** — [arXiv:2510.00202](https://arxiv.org/abs/2510.00202), an open
  platform for comparing routers.

**LLM-as-judge reliability** — the load-bearing assumption in Study B

- **Zheng et al. (2023)**, *Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena*, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). The original
  characterisation of position, verbosity and self-enhancement bias.
- **Self-preference bias in LLM-as-a-judge** —
  [arXiv:2410.21819](https://arxiv.org/abs/2410.21819). Why the judge in Study B
  is drawn from a different lab than the model being judged.

**Inference price decline** — why any fixed routing policy has a shelf life

- **Epoch AI**, [LLM inference prices have fallen rapidly but unequally across
  tasks](https://epoch.ai/data-insights/llm-inference-price-trends) (12 March
  2025). Price to reach a fixed capability level falls somewhere between 9× and
  900× per year depending on the benchmark — roughly 40×/year for GPQA Diamond.
  This is precisely why `pricing.yaml` is dated and sourced rather than
  hardcoded, and why the policy in this repo is a lookup table meant to be
  re-derived, not a trained artifact.

---

## Reproducing

No API key, no dataset access request, no cost. WildChat-1M is public under
ODC-BY, so this runs from a clean checkout:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_study_a.sh
```

It downloads 1.19 GB of parquet shards (5 of WildChat's 14, at stride 3) on the
first run and caches them. A Hugging Face token is not required, though setting
`HF_TOKEN` raises the Hub's rate limits and speeds the download up.

`run_study_a.sh` verifies the arithmetic, samples and filters the corpus,
computes the per-turn cost tables, renders the charts, and regenerates this
README's numbers from the results. Every number above is injected from
`results/tables/study_a_headline.json` — none is typed by hand.

The sampling seed is `{{seed}}` and is recorded in the filter log along with the
exact shard list, so the sample is reproducible.

### What is committed

```
/data          sampling + filtering scripts, seed, filter logs (no raw dataset)
/study_a       tokenization, cumulative-cost computation, arithmetic tests
/study_b       generation, judging, scoring          (Phase 1, not yet run)
/analysis      chart generation, README rendering
/results       results tables (CSV) + charts (PNG/SVG)
/coach         consumer-facing explainer              (Phase 2, not started)
/proxy         developer routing proxy                (Phase 3, not started)
pricing.yaml   all model prices, dated and sourced    (populated in Study B)
```

The raw dataset is not committed — AI2's licence governs it, and it is a
`huggingface-cli download` away. The filter log, the derived per-turn token
tables and every chart *are* committed, which is what a reviewer actually needs.

## Filter log

Of **{{records_scanned}}** conversations scanned, **{{pool_size}}** survived
every filter and **{{n_conversations}}** were sampled. Full drop-reason
breakdown in `data/out/study_a_filter_log.json`:

{{drop_table}}

---

## Data attribution and licence

Contains information from [**WildChat-1M**](https://huggingface.co/datasets/allenai/WildChat-1M),
which is made available under the
[ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/).

> Zhao et al., *WildChat: 1M ChatGPT Interaction Logs in the Wild*,
> [arXiv:2405.01470](https://arxiv.org/abs/2405.01470). AI2, ODC-BY.

The full corpus is **not** redistributed here — it is a `hf_hub_download` away
and the pipeline fetches it. What *is* committed is the derived per-turn token
table, the filter logs, and the ~400 prompt excerpts that make up the coding
filter's validation sets, which ODC-BY permits with the attribution above. AI2
de-identified the corpus with Microsoft Presidio and hand-written rules before
release.

Code in this repository: MIT. See [LICENSE](LICENSE).
