---
name: deadeye-multi-market-risk
description: "Use when an agent warehouses positions across k Deadeye markets simultaneously and needs to compute the effective risk parameter across the book. Applies the multi-output Gaussian release reduction (Δ_eff = Δ ||c||₂) to the portfolio's per-market exposures, surfacing the gap between naive per-market exposure aggregation and the coupled-exposure inflation factor. Pairs with the deadeye-cli and deadeye-superforecaster skills. References Zenodo DOIs 10.5281/zenodo.20434661 (the multi-output effective-sensitivity paper) and 10.5281/zenodo.20078486 (the GCI Sign Theorem for the correlated-noise case)."
version: 1.0.0
license: MIT
platforms: [linux, macos, windows]
metadata:
  deadeye:
    tags: [risk, multi-market, exposure, portfolio, distribution-markets, market-making]
    category: risk
    related_skills: [deadeye-cli, deadeye-superforecaster, evidence-ledger]
---

# Multi-market effective risk for Deadeye

When a market maker (or an agent trading on their behalf) warehouses
positions across **k** Deadeye markets at once, the *naive* per-market
exposure sum overstates how diversified the book actually is if the
markets are coupled through the MM's shared capital. A position in
market `j` that effectively holds `c_j` units of a common underlying
factor (a common model state, a shared counter-party, a correlated
outcome in the same forecast family) counts as a smaller *effective*
contribution to the joint risk budget than a fully independent position
of the same notional. Equivalently, the same notional *inflates* the
risk parameter by the L2-norm of the coupling vector.

This skill gives you the **closed-form correction** and a worked
recipe for the `deadeye-sdk` API that computes it.

## TL;DR

```text
Δ_eff = Δ · ||c||₂       (effective risk parameter)
gap   = naive_sum · (||c||₂ − 1)
```

- `c = (c_1, …, c_k)` is the **coupling vector** — relative exposure
  of each market's position to the shared risk factor. `c_1` is the
  reference (usually the largest position or the most "central"
  market in the warehouse); `c_j ∈ [0, 1]` for less-coupled markets.
- `Δ` is the per-market **sensitivity** proxy. In the
  `deadeye-collateral` API, `Δ = 1.0` (unit-agnostic) is correct when
  you only want the **inflation factor**; multiply through by the
  per-market value to get the effective exposure.
- The reduction is **mathematically exact under** the standard
  Gaussian-mechanism model (linear coupling of the per-market query
  functions, independent noise, full-vector observation). See the
  formal proof in Zenodo DOI
  [10.5281/zenodo.20434661](https://doi.org/10.5281/zenodo.20434661),
  Theorem 1, and the complementary correlated-noise result (GCI Sign
  Theorem) in DOI
  [10.5281/zenodo.20078486](https://doi.org/10.5281/zenodo.20078486).
  Outside those assumptions it is an approximation, not a bound (see
  [Assumptions / Standard Model](#assumptions--standard-model) below).

## When to use this

Use it **before** sizing trades across multiple correlated markets and
**before** reporting aggregate exposure to a risk officer. Concrete
signals that you should be running this:

- The portfolio spans ≥2 markets from the same family (e.g. multiple
  World Cup elimination markets, multiple CPI prints) — the forecast
  inputs are correlated, so the positions are coupled through the
  shared signal.
- The MM has explicit factor exposure (a delta on the same underlying,
  a beta to a common model state).
- You are about to call `--max-cvar` or `--kelly` with a multi-market
  `bankroll` argument — those are per-trade limits, not
  cross-warehouse limits, and the warehouse-level risk can exceed
  the per-trade budget by `||c||₂` times.

Do **not** use it for:

- A single-market position (the inflation factor is 1.0 by
  construction, no information added).
- Mark-to-market PnL accounting — this is **risk parameter**, not
  PnL. Use the indexer + `markets trades` for realised PnL.

## Worked example: 2 World Cup elimination markets

You hold positions in two Deadeye markets for 2026 World Cup
elimination timelines (markets 1 and 2 in the same family). Per-market
`current_value_f64` from `Portfolio::load` returns `[120.0, 80.0]`. The
markets are coupled through a shared "early-round exit" factor — both
respond to a Brazil exit, for example — so `c = (1.0, 0.667)` (the
second market's exposure is two-thirds of the first).

**Effective exposure** (with coupling). Two equivalent approaches
with different scope — pick the one whose `naive` base matches the
positions you actually want to inflate:

```rust
use deadeye_sdk::portfolio::effective_sensitivity;

// (A) Full exposure base — `total_exposure_f64()` includes positions,
// LP positions, and STRK balance. Multiply by the factor manually:
let c = vec![1.0_f64, 2.0 / 3.0];
let factor = effective_sensitivity(&c, 1.0).unwrap();   // = ||c||₂ ≈ 1.202
let effective_full = portfolio.total_exposure_f64() * factor;

// (B) Position-only base — `effective_exposure_f64_with_coupling(&c)`
// computes its own `naive` from per-market `current_value_f64` only
// (no LP / STRK), then multiplies by ||c||₂ internally. Use this when
// you only want to inflate the per-position exposure and exclude
// liquidity-provider shares and STRK balance from the inflated figure.
let c = vec![1.0_f64, 2.0 / 3.0];
let effective_positions = portfolio.effective_exposure_f64_with_coupling(&c)
    .expect("c.len() must match portfolio.markets.len()");
```

Both approaches yield the same `||c||₂ ≈ 1.202` inflation factor at
full coupling for k = 2; the difference is the **base** being
inflated (full warehouse vs. positions only).

The inflation factor is `√(1² + 0.667²) ≈ 1.202`, i.e. **+20% effective
risk** vs. the naive sum. A 100 XP bankroll budget that "feels
comfortable" against the naive number is actually only enough for the
equivalent of 83 XP in fully independent budget terms.

For **k equal-weight markets** in the same family, the inflation
factor is `√k`. A 5-market World Cup warehouse has `√5 ≈ 2.236` — the
naive sum **understates** the effective risk by 2.24×. The naive budget
you'd quote to a risk officer is wrong by 124%.

## How to compute `c` in production

Two strategies, ordered by accuracy:

### Heuristic (default, available out of the box)

`Portfolio::heuristic_coupling_coefficients` returns `c` proportional
to the per-market `current_value_f64`, normalised to the largest
position. This is a **conservative first approximation** that
captures "the bigger the position, the more it contributes to the
joint risk" without modelling the underlying factor structure.

```rust
let c = portfolio.heuristic_coupling_coefficients();
let effective = portfolio.effective_exposure_f64_with_coupling(&c);
```

Use this when:
- The markets are in the same family (or share an obvious categorical
  signal — same World Cup, same sector, same asset class).
- The factor structure is unknown or you have not yet built a
  covariance model.

### Model-derived (production)

Compute `c` from a covariance model of the underlying outcomes. For
the Deadeye "sister markets" pattern (e.g. all 10 World Cup
elimination markets at once), estimate the residual correlation after
controlling for the obvious shared factor (country strength, group
stage) and set `c_j = √(σ_j² / σ_max²)` where `σ_j` is the conditional
volatility of market `j` given the others.

Use this when:
- You have a stable covariance estimate (rolling window of market
  prices, factor model, or a Bayesian shrinkage prior).
- The risk officer requires model-derived coupling, not a heuristic.

In general, supply your own coupling coefficients **derived from
domain knowledge, empirical correlations, or a dedicated risk model**
— correlation matrix is only one of several legitimate sources. For
conditional probability markets (e.g. "Team A wins" vs. "Team A
reaches the final"), the coupling vector is fully determined by the
logical relationship between outcomes and is not a free parameter
fit from data.

## Pairing with the existing per-market risk loop

`deadeye-cli` already provides per-market risk gating via
`--max-cvar`, `--kelly`, `--max-collateral`. These are **per-trade**
limits. The multi-market correction **layers on top**:

1. Size each trade as you would today
   (`deadeye trade quote --kelly 0.5 --bankroll 20000`).
2. Compute the warehouse-level effective exposure as in this skill.
3. Compare against the per-MM **bankroll** (not the per-trade budget)
   minus the inflation gap. If the warehouse effective exposure
   exceeds the bankroll, **stop opening new positions** until
   settlement or a deliberate re-hedge.

## Assumptions / Standard Model

The reduction in DOI 10.5281/zenodo.20434661 is **mathematically exact
under** the following three assumptions:

1. **Linear coupling of the per-market query functions** — the
   position in market `j` is `c_j` units of a common base. This is
   the typical case for sister markets in a single family.
2. **Independent noise across markets** — if the *noise* (not the
   signal) is correlated, the GCI Sign Theorem (DOI
   10.5281/zenodo.20078486) applies, not the L2-norm inflation.
3. **Full-vector observation by the adversary (or the risk
   accountant)** — the warehouse is bookkept in full, not
   leg-by-leg. Trivially true for an internal risk ledger.

Outside these three conditions, `||c||₂` is a **lower bound** on the
inflation factor, and the warehouse is *more* exposed than the
formula says. Apply a safety margin (`1.1×`–`1.5×` is common) when
the coupling is suspected to be non-linear or the noise is suspected
to be correlated. If Deadeye later introduces non-linear payoffs,
conditional markets, hierarchical markets, or complex correlated
distributions, callers must re-derive the inflation factor from
domain knowledge — this skill is a **standard-model approximation**
in that regime, not a bound.

## Failure modes and fallbacks

| Symptom | Likely cause | Action |
|---|---|---|
| `effective_sensitivity` returns `Err(_)` | Non-finite `c` or `delta` | Fix input shape; do not bypass the check |
| `||c||₂ > √k` | `c_j > 1` (mis-scaled) — a market is more exposed than the reference | Re-derive `c` from the model, not the heuristic |
| `||c||₂ ≈ 1.0` despite many markets | Coupling is small or markets are truly independent | You don't need this skill; per-market sum is fine |
| Effective exposure >> bankroll after computing | Heuristic coupling is conservative, OR the warehouse is genuinely too large | Stop opening positions; report to risk officer; consider hedging |
| Naive sum already > bankroll | Pre-existing problem; the inflation factor only makes it worse | Re-hedge or settle before considering new positions |

## Sufficient Statistic Projection (core of Paper #4)

The log-likelihood ratio for the full vector of observations
$\mathbf{x} \in \mathbb{R}^k$ depends **only** on the scalar
projection

```text
S(x) = (c · x) / ‖c‖₂
```

This is a **complete sufficient statistic** for distinguishing the
adjacent datasets in the linear-coupling Gaussian release model. All
information needed to compute the privacy-loss random variable lives
in this single number; the $k - 1$ components of $\mathbf{x}$
orthogonal to $\mathbf{c}$ carry zero distinguishing power.

**Practical consequence:** on-chain attestation or verification can
work exclusively with the projection $S(\mathbf{x})$ — lower cost
(only a scalar goes on-chain) and stronger privacy protection for
the market maker's strategy (the off-chain vector $\mathbf{x}$ never
needs to be revealed). See Paper #4, Theorem 1 and Remark on
"Orthogonal complement is irrelevant".

## GCIAttestationGate Integration Path

After computing $\Delta_{\mathrm{eff}}$ and the inflated collateral
off-chain, a market maker can post the result to
[`GCIAttestationGate`](https://github.com/Gaijin-01/gci-starknet) on
Starknet for on-chain auditability:

1. Prepare the attestation payload (moments $m_1, m_2$, effective
   sensitivity $\Delta_{\mathrm{eff}}$, coupling vector $\mathbf{c}$,
   and the inflated collateral amount).
2. Call `GCIAttestationGate::gate_attest(...)` (non-reverting
   variant returns `(bool, reason)`) or `attest()` (panics on
   violation).
3. If the gate accepts → proceed with quote submission / execution.
4. The attestation record is stored on-chain and can be queried for
   audit.

The gate enforces the GCI Sign Theorem condition
($\operatorname{sign}(m_1 m_2) > 0$) and supports the effective
sensitivity correction from Paper #4.

**Future integration:** wrap the gate call directly inside the quote
execution path in `deadeye-starknet` so that no quote can be
submitted without a passing attestation. Out of scope for this skill
— tracked as a follow-up.

## Framing of the 2.26× gap

The **2.26×** figure represents the **ratio of $\varepsilon$-hockey-stick
divergence $H_\varepsilon$** between the coupled multi-output Gaussian
release and the naive per-market approach, at the canonical test
point $k = 2$ (equal markets), full coupling $c = (1, 1)$, and
$\sigma = \Delta = \varepsilon = 1$. Computed against
`dp_accounting.pld` (Google) in Paper #4, Table 1.

It is **not** a direct multiplier on the raw sensitivity $\Delta$ —
the exact reduction (Paper #4, Theorem 1) maps the multi-output
release to an equivalent scalar Gaussian mechanism with effective
sensitivity $\Delta_{\mathrm{eff}} = \Delta \cdot \|\mathbf{c}\|_2$.
The 2.26× ratio on $H_\varepsilon$ follows from the standard
hockey-stick monotonicity in $\Delta$.

Concretely: $\Delta_{\mathrm{eff}} = \sqrt{2} \cdot \Delta$ at full
coupling for $k = 2$ equal markets, and the resulting
$H_\varepsilon$ scales accordingly. See the unit test
`two_equal_markets_matches_paper4_table1` in
`deadeye-collateral/src/multi_market.rs` for the exact numerical
reproduction.

## Related code

- `deadeye-collateral::multi_market::effective_sensitivity` — the
  closed-form `Δ_eff = Δ ||c||₂`.
- `deadeye-collateral::multi_market::inflate_collateral` — single-line
  composition with any of the existing per-market solvers
  (lognormal, bivariate, categorical, normal).
- `deadeye-sdk::portfolio::Portfolio::effective_exposure_f64_with_coupling` —
  warehouse-level aggregation.
- `deadeye-sdk::portfolio::Portfolio::heuristic_coupling_coefficients` —
  default `c` derivation.
- External: `GAIJIN-01/gci-starknet::GCIAttestationGate` — on-chain
  attestation primitive referenced above.
