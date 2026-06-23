---
name: deadeye-superforecaster
description: "The guided superforecasting loop for Deadeye distribution markets: read the market's outcome space, set reference-class base rates, gather and weight timestamped evidence, decompose into components, aggregate into a calibrated (mean, σ), commit a snapshot, and turn it into the highest-EV trade. Use when forecasting a continuous Deadeye market end-to-end. Backed by `deadeye forecast` + `deadeye trade`."
version: 1.0.0
license: MIT
platforms: [linux, macos, windows]
metadata:
  deadeye:
    tags: [forecasting, superforecasting, distribution-markets, bayesian, calibration, trading]
    category: forecasting
    related_skills: [bayes-forecast-scratchpad, evidence-ledger, deadeye-cli]
---

# Deadeye superforecaster loop

A Deadeye market holds a **probability distribution** (e.g. a Normal with mean μ
and sigma σ), not a yes/no contract. You profit by moving the market's curve
toward a **better-calibrated** distribution. Your whole job is to produce an
honest `(mean, σ)` and let `deadeye trade quote` size the EV-maximizing move.

The loop is auditable: every forecast is built from evidence and reference
classes recorded in a per-market **workspace** (`deadeye forecast …`), so you
can re-run it, defend it, and score it after resolution.

## Prerequisites

Before spending research time, run the readiness check:

```bash
deadeye doctor --market <MARKET>
```

You need a deployed account, gas STRK, claimed XP collateral, a reachable RPC,
and a readable active market before execution. Quoting is client-side and does
not require a math-runtime address; commands that explicitly ask for one will
tell you to deploy or inspect it with `deadeye admin deploy-math-runtime`.

## The loop

```
parse → base-rate → evidence → weight → decompose → aggregate → commit → size → trade → monitor → score
```

### 1. Parse the market (outcome space)

```bash
deadeye markets show  <MARKET>   # on-chain distribution (μ_mkt, σ_mkt), family, LP backing
deadeye markets info  <MARKET>   # the human question + resolution criteria
deadeye forecast new  <MARKET> --title "<question>" --lower <lo> --upper <hi> --resolution "<how it resolves>"
```

Write down exactly what resolves the market and the numeric range. If it isn't
scoreable, stop and clarify before forecasting.

### 2. Reference classes → base-rate prior

Anchor on the outside view first. Add 2–4 reference classes (numeric anchors for
a continuous market), each with an applicability weight and within-class sd:

```bash
deadeye forecast base-rate <MARKET> --name "CPI YoY, last 12 prints" --rate 3.2 --applicability 0.8 --uncertainty 0.3
deadeye forecast base-rate <MARKET> --name "Cleveland Fed nowcast"   --rate 3.0 --applicability 0.9 --uncertainty 0.2
```

The CLI blends them in **value space** into a prior `mean ± sd`. This is your
starting distribution before any inside-view evidence.

**Efficient-market prior + edge gate.** Before researching, classify the
market: backing/liquidity, number of independent traders, time to resolution,
and whether an external consensus or another market prices the same question.
For a liquid market resolving imminently, **default your prior to the de-vigged
market** and ask the gate question out loud: *"What do I know that the market
plausibly doesn't?"* If the honest answer is "nothing," set belief ≈ market and
size near zero — beating next-day consensus is rare, and the record should say
why you think this time is different (a recorded, differential piece of
evidence — not vibes).

### 3. Gather + record evidence

Use the **evidence-ledger** skill for the source taxonomy and scoring. Record
every item, timestamped and source-linked, with a stance and quality ratings:

```bash
deadeye forecast evidence <MARKET> --claim "Gasoline fell 3% in May" --stance down \
    --source "EIA weekly" --url https://... --reliability 0.9 --relevance 0.7
```

`--stance up|down|context|mixed`. Prefer primary/official sources; note when two
sources are not independent (you'll collapse them in step 4).

### 4. Weight evidence (and avoid double-counting)

Turn qualitative evidence into a likelihood ratio, and collapse correlated
clusters so you don't over-update. Use the **bayes-forecast-scratchpad** skill:

```bash
deadeye forecast bayes evidence-weight --input '{"reliability":0.9,"relevance":0.7,"independence":0.6,"recency":1,"bias_risk":0.1,"direction":"against","strength":"medium"}'
deadeye forecast bayes effective-count --input '{"n":4,"rho":0.7}'   # 4 correlated takes ≈ ? independent
```

### 5. Decompose the QUANTITY into structural components

Two different operations hide under "decompose" — do both, separately:

**(a) Build the number from parts (structural).** Split the target quantity
into additive components, each with its own central estimate and σ, and let the
component σ's *build* the total σ — never set the total σ by feel. Worked CPI
example (YoY contributions, percentage points):

| component        | contribution | σ    |
|------------------|-------------:|------|
| shelter          |         1.6  | 0.10 |
| energy           |        −0.3  | 0.15 |
| food             |         0.3  | 0.05 |
| core goods       |         0.2  | 0.10 |
| core services ex |         1.4  | 0.12 |

Total: μ = Σ contributions = 3.2; σ = √(Σ σᵢ² + 2·Σρᵢⱼσᵢσⱼ) — with mildly
correlated components this lands near 0.25, an *honest* σ derived from parts.

**(b) Blend independent estimators of the final number.** Separately, you may
hold several whole-number estimates (base-rate prior, structural build from
(a), inside-view model, the **de-vigged market itself** — never ignore it).
Blend those with `aggregate-normal` in step 6. **Warning:** estimators sharing
a source are correlated — consensus ⊂ market ⊂ prediction-market is one signal,
not three confirmations; collapse them or raise `rho` accordingly.

### 6. Aggregate into a calibrated (mean, σ)

```bash
# Snapshot the live market once (also feeds trade quote --from-state):
deadeye markets snapshot <MARKET> --output json > state.json

# The market curve is EVIDENCE — inject it mechanically, never hand-copy μ/σ:
deadeye forecast bayes aggregate-normal --from-state state.json --market-weight 0.8 --input '{
  "rho": 0.3,
  "components": [
    {"mu": 3.10, "sigma": 0.30, "weight": 1.0},   # base-rate prior
    {"mu": 2.95, "sigma": 0.20, "weight": 1.2}    # inside view (disinflation)
  ]
}'
```

`--from-state` appends the live market `(μ, σ)` as a component (zero extra
RPC) and echoes exactly what it blended against; skip it only with
`--no-market` and a reason. Scalar routines also take named flags directly —
`deadeye forecast bayes shrink-to-market --my-mu 4.32 --my-sigma 0.26
--market-mu 4.13 --market-sigma 0.55 --edge-strength 0.6` — no JSON quoting.

`rho` is the shared correlation across components (0 = independent). The output
`mean`, `sd`, and `variance` are your forecast. **Keep σ honest** — if you can't
name a credible path for the tail mass, don't fabricate spread, but don't
collapse it either.

**Shrink to the market.** When the edge gate (step 2) found only weak
differential information, shrink your posterior toward the market before
committing: `deadeye forecast bayes shrink-to-market --my-mu <μ> --my-sigma <σ>
--market-mu <μ_mkt> --market-sigma <σ_mkt> --edge-strength <0..1>` — edge 0
returns the market, edge 1 returns you; pick the factor from market quality and
how differential your recorded evidence really is.

**σ-floor from the surprise history.** Look up how much this quantity has
historically deviated from the day-before consensus (the *surprise* std, not
the level std) and **floor your σ at it**. Committing a σ below the historical
surprise std — or below the market σ — without a recorded justification is the
canonical overconfidence failure.

### 7. Commit the snapshot

Pass the same `--from-state state.json` so the committed snapshot records the
exact market (μ, σ) you blended against — that link is what makes later
scoring and audit honest.

```bash
deadeye forecast snapshot <MARKET> --mean 3.02 --sd 0.22 --method aggregate-normal \
    --from-state state.json \
    --rationale "Outside view ~3.1; energy disinflation pulls it down; market roughly agrees." \
    --reason-up "shelter stays sticky" --reason-down "energy keeps falling" \
    --change-my-mind "a hot core services print"
deadeye forecast show <MARKET>     # review the full workspace + the trade command
```

### 8. Decide the size (separate from the forecast)

The forecast says what you believe; the **size** says how much of your bankroll
rides on it. Never blur them — in one recorded session an agent tightened its
*stated* σ purely to make the optimizer stake more. That corrupts the forecast
record AND the calibration loop. Instead:

- Keep `(μ, σ)` exactly as committed in step 7 — σ is a function of *evidence
  only*.
- Choose the stake via a stated risk policy — the CLI enforces it as a hard
  cap on `trade quote` AND `trade execute` (the belief is never altered):
  `--bankroll 20000 --kelly 0.5` caps at half-Kelly; `--max-cvar 15` walks the
  stake down until the worst-5% loss fits (normal family); `--risk
  conservative|balanced|aggressive` maps to Kelly 0.25/0.5/1.0. The output's
  `sizing_basis` names the constraint that bound — record it.
- Record the decision (policy, fraction, resulting budget, `sizing_basis`)
  alongside the snapshot so the postmortem can judge the sizing separately
  from the forecast.

### 9. Turn the forecast into the highest-EV trade

Do **not** jump the market to your mean. `trade quote` runs the EV-max grid
search between the market's curve and yours under your budget:

```bash
deadeye trade quote   <MARKET> --mean 3.02 --variance 0.0484
deadeye trade execute <MARKET> --mean 3.02 --variance 0.0484 --max-collateral <XP_CAP>
```

(Or drive it from your belief directly: `trade quote <MARKET> --belief <mean> --budget <xp> --belief-sigma <sd>`.)

Both families are supported: normal markets take `--belief` in outcome space;
**lognormal markets take `--belief`/`--belief-sigma` in log space** (the same
(μ, σ) the chain stores). Single-trade movement is capped (σ ratio ≤ 4×,
|Δμ| ≤ 4σ_market — see "Per-trade movement limits" in the deadeye-cli skill):
when the quote reports `belief_utilization < 100%`, your full belief needs a
**ladder of trades** — execute, re-quote from the new state, repeat — sized
as a whole, since each rung locks its own lot and pays its own fee.

Mind RPC etiquette (see the deadeye-cli skill): snapshot state once with
`markets snapshot --output json`, explore candidates via `--from-state` with
zero further RPC, then one `--dry-run` and one `execute`. Never retry in a
loop — an empty/parse-error response means rate-limited; back off.

To run the ladder hands-off, `deadeye forecast loop <MARKET>` re-reads the
market each tick, re-loads your committed snapshot as the belief, and submits
at most one trade per tick — only when every EV / risk / budget gate passes.
It is **observe-only until `--execute`**, so you can watch the gates fire
first, and it self-throttles on `--interval` (don't wrap one-shot commands in a
shell loop). Re-committing `forecast snapshot` mid-loop takes effect live, so
this is also the clean way to keep a position tracking an evolving belief —
not a licence to skip step 10's discipline.

### 10. Monitor, re-run, score

Re-run when watched inputs move materially (re-record evidence, re-aggregate,
re-commit). **A market move against you is evidence to weigh, not a verdict to
adopt.** Ask what information the move plausibly carries — one counterparty
re-pricing on no news is weak (could be a single anchored trader); a large,
multi-party move on volume is strong. Record the move as an evidence item with
a stance, reliability, and weight, then re-aggregate — do NOT silently walk
your committed belief toward the new market level (use `markets trades` /
curve history from the indexer to see who moved it and how much, where
available). After the market resolves, run `deadeye forecast score <MARKET>` —
it pulls the settlement value and computes CRPS/z-score vs your committed
(μ, σ) — write a one-paragraph postmortem, and let `deadeye forecast
calibration` accumulate the record that tunes your future σ and sizing.

## Calibration discipline (do not skip)

- **Outside view before inside view.** Anchor on base rates; let evidence move
  you off them, not replace them.
- **Don't double-count.** Correlated sources share signal — collapse them with
  `effective-count` before pooling.
- **Stress-test the disconfirming case.** Before a confident, narrow σ, name the
  credible path where you're wrong and widen toward it.
- **The market is evidence.** Include the de-vigged market as a component;
  disagreeing with it is fine, ignoring it is not.
- **Never tighten σ to bet more.** Sizing is step 8's job. A σ you wouldn't
  defend as evidence-derived poisons both the trade and your calibration
  record.
- **XP is non-transferable and worthless off-platform** — only gas STRK has
  value. Always `trade quote` before `execute`; respect `--max-collateral`.
