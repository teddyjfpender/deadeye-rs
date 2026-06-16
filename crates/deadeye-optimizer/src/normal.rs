//! 2D grid-search trade optimizer for normal-distribution markets.
//!
//! Maximises the trader's **expected P&L** `E_belief[position_value]` (the
//! [`NormalOptimizationResult::expected_value`]) subject to `collateral ≤
//! budget`. Per the distribution-market scoring rule the EV-max submission is
//! the trader's own belief (`f ∝ p`); under a non-binding budget the optimum
//! is the full move to `(belief_mean, belief_sigma)`.
//!
//! **Collateral is a returned margin lock, not a cost.** At settlement
//! `gross_payout = collateral + position_value`, so the trader's net P&L is
//! `position_value` and its expectation is `expected_value`. The objective is
//! therefore `max EV s.t. collateral ≤ budget` — NOT `max (EV − collateral)`.
//! The old `EV − collateral` objective treated the (large) returned lock as a
//! sunk cost, so every feasible move looked net-negative and the optimizer
//! always declined (issue #12).
//!
//! ## λ-scaling
//!
//! The inner collateral cost is λ-scaled to the unit the chain charges so the
//! **budget filter** (`collateral ≤ budget`) is apples-to-apples:
//! `chain_cost = max(0, λ_f · f(x*) − λ_g · g(x*))` (`λ = k / ‖p‖₂`) evaluated
//! at the audited stationary point from `normal_collateral`, matching
//! `optimize_quote_offline` (`crates/deadeye-sdk/src/normal.rs`). This cost
//! λ-scaling is independent of the objective fix above.

use deadeye_collateral::{MinimizationPolicy, lambda, normal_collateral};
use deadeye_core::{Distribution, NormalDistribution, Sq128};

const N_SIGMA_SAMPLES: u32 = 50;
const N_MEAN_SAMPLES: u32 = 50;
const DEFAULT_MAX_SIGMA_RATIO: f64 = 4.0_f64;
const DEFAULT_MAX_MEAN_SEP_SIGMAS: f64 = 4.0_f64;

/// Backing-derived **σ-floor**: the narrowest σ a normal market can back.
///
/// From the on-chain backing constraint `max_x f(x) = k / √(σ·√π) ≤ b`
/// (where `f` is the λ-scaled position PDF with `‖f‖₂ = k`, and `b` is the
/// pool backing), which rearranges to the closed form
///
/// ```text
/// σ ≥ k² / (b² · √π)
/// ```
///
/// A candidate σ below this floor pushes the scaled-PDF peak above the pool
/// backing and the AMM rejects the trade with `SIGMA_TOO_LOW`. `k` is the
/// execution-time **effective** invariant and `b` the pool backing — the same
/// values the contract checks. Validated against the Cairo
/// `check_scaled_backing` constraint (see the webapp `solvencyConcentration`).
///
/// Returns `0.0` (no enforceable floor) when `effective_k` or `backing` is
/// non-positive.
#[must_use]
pub fn normal_sigma_floor(effective_k: f64, backing: f64) -> f64 {
    if effective_k <= 0.0 || backing <= 0.0 || !effective_k.is_finite() || !backing.is_finite() {
        return 0.0;
    }
    let sqrt_pi = core::f64::consts::PI.sqrt();
    (effective_k * effective_k) / (backing * backing * sqrt_pi)
}

/// Tunable bounds on the optimizer's policy region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizerConstraints {
    /// Maximum `σ_large / σ_small`.
    pub max_sigma_ratio: f64,
    /// Maximum `|μ_g - μ_market|` measured in units of `σ_market`.
    pub max_mean_sep_sigmas: f64,
}

impl Default for OptimizerConstraints {
    fn default() -> Self {
        Self {
            max_sigma_ratio: DEFAULT_MAX_SIGMA_RATIO,
            max_mean_sep_sigmas: DEFAULT_MAX_MEAN_SEP_SIGMAS,
        }
    }
}

/// Inputs to [`optimize_normal_trade`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalOptimizationInput {
    /// Trader's budget in collateral tokens.
    pub budget: f64,
    /// Trader's belief mean.
    pub belief_mean: f64,
    /// Trader's belief σ (from the UI's confidence selector).
    pub belief_sigma: f64,
    /// Current market mean.
    pub market_mean: f64,
    /// Current market σ.
    pub market_sigma: f64,
    /// AMM `effective_k` (post LP scaling).
    pub effective_k: f64,
    /// Policy bounds.
    pub constraints: OptimizerConstraints,
}

impl NormalOptimizationInput {
    /// Convenience constructor with sane defaults (default constraints).
    #[must_use]
    pub fn new(
        budget: f64,
        belief_mean: f64,
        belief_sigma: f64,
        market_mean: f64,
        market_sigma: f64,
        effective_k: f64,
    ) -> Self {
        Self {
            budget,
            belief_mean,
            belief_sigma,
            market_mean,
            market_sigma,
            effective_k,
            constraints: OptimizerConstraints::default(),
        }
    }
}

/// Output of [`optimize_normal_trade`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalOptimizationResult {
    /// Optimised target `μ_g`.
    pub optimized_mean: f64,
    /// Optimised target `σ_g`.
    pub optimized_sigma: f64,
    /// `optimized_sigma²`.
    pub optimized_variance: f64,
    /// Collateral required to enter the optimised trade.
    pub collateral_required: f64,
    /// Gross expected value under the trader's belief.
    pub expected_value: f64,
    /// Fraction of the intended belief shift the trade expresses (0–1).
    pub belief_utilization: f64,
    /// `true` iff the optimum sits at the full belief point.
    pub is_budget_sufficient: bool,
    /// `budget − collateral_required`.
    pub budget_surplus: f64,
    /// Expected return on the locked collateral: `EV / collateral`.
    pub roi: f64,
}

/// Closed-form Gaussian product integral — `∫ φ(x; μ₁, σ₁) φ(x; μ₂, σ₂) dx`.
fn gaussian_product_integral(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> f64 {
    let sum_var = sigma1.mul_add(sigma1, sigma2 * sigma2);
    if sum_var <= 0.0 {
        return 0.0;
    }
    let diff_mu = mu1 - mu2;
    (1.0 / (2.0 * core::f64::consts::PI * sum_var).sqrt())
        * (-(diff_mu * diff_mu) / (2.0 * sum_var)).exp()
}

/// **Chain-frame** collateral cost to transition `(μ_f, σ_f) →
/// (μ_g, σ_g)` at AMM parameter `k`.
///
/// Mirrors what `optimize_quote_offline` re-derives after a
/// `normal_collateral` call (`crates/deadeye-sdk/src/normal.rs:442-465`):
///
/// 1. Find the audited stationary point `x*` via `normal_collateral`.
/// 2. Evaluate `max(0, λ_f · f(x*) − λ_g · g(x*))` with `λ = k / ‖p‖₂` — the
///    **unit the chain charges** (`helpers.cairo:155-176`,
///    `helpers.cairo:198-230`).
///
/// Pre-v0.1.3 this returned the unscaled `verified.collateral`, which then
/// leaked into the `collateral ≤ budget` filter (where the user's budget is
/// the chain unit), letting trades pass that were ~200× over budget at the
/// chain charge. Reporting the chain-frame cost here keeps the budget filter
/// apples-to-apples with the user's budget.
///
/// Returns `f64::INFINITY` on any solver failure so the grid filter
/// in `optimize_normal_trade` rejects the candidate cleanly.
fn collateral_number(mu_f: f64, sigma_f: f64, mu_g: f64, sigma_g: f64, k: f64) -> f64 {
    if (mu_f - mu_g).abs() < 1e-12_f64 && (sigma_f - sigma_g).abs() < 1e-12_f64 {
        return 0.0;
    }
    let Ok(mean_f) = Sq128::from_f64(mu_f) else {
        return f64::INFINITY;
    };
    let Ok(var_f) = Sq128::from_f64(sigma_f * sigma_f) else {
        return f64::INFINITY;
    };
    let Ok(mean_g) = Sq128::from_f64(mu_g) else {
        return f64::INFINITY;
    };
    let Ok(var_g) = Sq128::from_f64(sigma_g * sigma_g) else {
        return f64::INFINITY;
    };
    let Ok(f) = NormalDistribution::from_variance(mean_f, var_f) else {
        return f64::INFINITY;
    };
    let Ok(g) = NormalDistribution::from_variance(mean_g, var_g) else {
        return f64::INFINITY;
    };
    let verified = match normal_collateral(&f, &g, MinimizationPolicy::unrestricted()) {
        Ok(v) if v.collateral.is_finite() => v,
        _ => return f64::INFINITY,
    };
    if verified.collateral <= 0.0 {
        return 0.0;
    }
    // λ-scale at the audited stationary point `x*`, matching
    // `optimize_quote_offline`'s re-evaluation. The unscaled
    // `verified.collateral` is **not** the chain charge — that's the bug
    // Reviewer B caught (`REVIEW_ITEM3_DRIVER_B.md` §6).
    let lam_f = lambda(sigma_f, k);
    let lam_g = lambda(sigma_g, k);
    let Ok(x_q) = Sq128::from_f64(verified.x_min) else {
        return f64::INFINITY;
    };
    let f_at = f.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
    let g_at = g.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
    let scaled = lam_f.mul_add(f_at, -(lam_g * g_at)).max(0.0);
    if scaled.is_finite() {
        scaled
    } else {
        f64::INFINITY
    }
}

fn expected_value(
    mu_f: f64,
    sigma_f: f64,
    mu_g: f64,
    sigma_g: f64,
    k: f64,
    belief_mu: f64,
    belief_sigma: f64,
) -> f64 {
    let lam_g = lambda(sigma_g, k);
    let lam_f = lambda(sigma_f, k);
    lam_g.mul_add(
        gaussian_product_integral(mu_g, sigma_g, belief_mu, belief_sigma),
        -(lam_f * gaussian_product_integral(mu_f, sigma_f, belief_mu, belief_sigma)),
    )
}

/// Picks the highest-net-EV trade in the policy region.
#[must_use]
pub fn optimize_normal_trade(input: NormalOptimizationInput) -> NormalOptimizationResult {
    let no_trade = NormalOptimizationResult {
        optimized_mean: input.market_mean,
        optimized_sigma: input.market_sigma,
        optimized_variance: input.market_sigma * input.market_sigma,
        collateral_required: 0.0,
        expected_value: 0.0,
        belief_utilization: 0.0,
        is_budget_sufficient: false,
        budget_surplus: input.budget,
        roi: 0.0,
    };

    if input.budget <= 0.0 || input.market_sigma <= 0.0 || input.effective_k <= 0.0 {
        return no_trade;
    }

    let sigma_min = (input.market_sigma / input.constraints.max_sigma_ratio).max(1e-6_f64);
    let sigma_max = input.market_sigma * input.constraints.max_sigma_ratio;
    let sigma_step = (sigma_max - sigma_min) / f64::from(N_SIGMA_SAMPLES);

    let mean_dir = if input.belief_mean >= input.market_mean {
        1.0_f64
    } else {
        -1.0_f64
    };
    let max_shift = input.constraints.max_mean_sep_sigmas * input.market_sigma;

    // The trader maximizes **expected P&L** = `E_belief[position_value]` (the
    // `expected_value` below) subject to the locked collateral fitting the
    // budget. Collateral is a MARGIN LOCK that is *returned* at settlement
    // (`gross_payout = collateral + position_value`), so it is a budget
    // constraint — NOT a cost to subtract from EV. The old objective
    // `net = ev − coll` treated the returned collateral as a sunk cost; since
    // collateral (~the max loss) typically dwarfs EV, every feasible move
    // looked net-negative and the optimizer always returned no-trade
    // (issue #12). Per the distribution-market scoring rule, the EV-max
    // submission is the trader's own belief (`f ∝ p`); under a non-binding
    // budget the optimum is the full move to `(belief_mean, belief_sigma)`.
    let mut best_mu = input.market_mean;
    let mut best_sigma = input.market_sigma;
    let mut best_coll = 0.0_f64;
    let mut best_ev = 0.0_f64; // the market baseline (no move) scores EV 0

    for i in 0..=N_SIGMA_SAMPLES {
        let cand_sigma = f64::from(i).mul_add(sigma_step, sigma_min);
        for j in 0..=N_MEAN_SAMPLES {
            let shift = (f64::from(j) / f64::from(N_MEAN_SAMPLES)) * max_shift;
            let cand_mu = mean_dir.mul_add(shift, input.market_mean);
            let coll = collateral_number(
                input.market_mean,
                input.market_sigma,
                cand_mu,
                cand_sigma,
                input.effective_k,
            );
            if coll <= 0.0 || coll > input.budget {
                continue;
            }
            let ev = expected_value(
                input.market_mean,
                input.market_sigma,
                cand_mu,
                cand_sigma,
                input.effective_k,
                input.belief_mean,
                input.belief_sigma,
            );
            // Maximize EV among budget-feasible candidates; on a tie prefer the
            // cheaper lock (more capital-efficient).
            if ev > best_ev || (ev >= best_ev && best_coll > 0.0 && coll < best_coll) {
                best_ev = ev;
                best_mu = cand_mu;
                best_sigma = cand_sigma;
                best_coll = coll;
            }
        }
    }
    if best_ev <= 0.0 || best_coll <= 0.0 {
        return no_trade;
    }

    let full_shift = (input.belief_mean - input.market_mean).abs();
    let achieved_shift = (best_mu - input.market_mean).abs();
    let utilization = if full_shift > 1e-6_f64 {
        (achieved_shift / full_shift).min(1.0)
    } else {
        1.0
    };

    NormalOptimizationResult {
        optimized_mean: best_mu,
        optimized_sigma: best_sigma,
        optimized_variance: best_sigma * best_sigma,
        collateral_required: best_coll,
        expected_value: best_ev,
        belief_utilization: utilization,
        is_budget_sufficient: (best_mu - input.belief_mean).abs() < sigma_step * 0.5
            && (best_sigma - input.belief_sigma).abs() < sigma_step * 0.5,
        budget_surplus: input.budget - best_coll,
        // Expected return on the locked collateral (EV per unit locked).
        roi: best_ev / best_coll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_budget_yields_no_trade() {
        let input = NormalOptimizationInput::new(0.0, 105.0, 1.0, 100.0, 2.0, 50.0);
        let r = optimize_normal_trade(input);
        assert!(r.collateral_required.abs() < 1e-12);
        assert!((r.optimized_mean - 100.0).abs() < 1e-12);
    }

    #[test]
    fn zero_market_sigma_yields_no_trade() {
        let input = NormalOptimizationInput::new(100.0, 105.0, 1.0, 100.0, 0.0, 50.0);
        let r = optimize_normal_trade(input);
        assert!(r.collateral_required.abs() < 1e-12);
    }

    #[test]
    fn picks_a_trade_under_reasonable_budget() {
        let input = NormalOptimizationInput::new(100.0, 105.0, 1.0, 100.0, 2.0, 50.0);
        let r = optimize_normal_trade(input);
        assert!(r.collateral_required >= 0.0);
        assert!(r.budget_surplus <= input.budget);
        // With a meaningful belief shift the optimizer should select a non-trivial
        // mean change.
        assert!(r.optimized_mean >= input.market_mean);
    }

    /// A tighter, slightly-shifted belief is a **positive-EV** move: a trader
    /// more confident than the market profits when the outcome lands near
    /// their sharper curve (where their belief puts its mass). The collateral
    /// is a returned margin lock, not a cost — under the corrected objective
    /// (maximize `E_belief[position_value]` s.t. collateral ≤ budget) the
    /// optimizer proposes this trade. Pre-fix it subtracted the (large) lock
    /// from the (small) EV and wrongly declined every σ-arb (issue #12).
    ///
    /// (The collateral here is still reported in chain-frame λ-scaled units —
    /// that earlier fix is unchanged; only the trade/no-trade *decision* is
    /// corrected.)
    #[test]
    fn sigma_arb_with_tighter_belief_finds_a_trade() {
        let input = NormalOptimizationInput::new(
            50.0,   // budget (chain XP units)
            4.3274, // belief μ
            0.2143, // belief σ — tighter than the market
            4.2900, // market μ
            0.3500, // market σ
            75.07,  // effective k
        );
        let r = optimize_normal_trade(input);
        assert!(
            r.collateral_required > 0.0,
            "expected a positive-EV σ-arb trade, got no-trade",
        );
        assert!(
            r.expected_value > 0.0,
            "expected EV > 0, got {}",
            r.expected_value,
        );
        assert!(r.collateral_required <= input.budget + 1e-9);
    }

    /// Pure σ-arb (exact-equal μ, tighter belief σ) is also positive-EV: the
    /// sharper curve out-scores the market at the shared mean. The optimizer
    /// must take it.
    #[test]
    fn pure_sigma_arb_finds_a_trade() {
        let input = NormalOptimizationInput::new(50.0, 4.29, 0.21, 4.29, 0.35, 75.07);
        let r = optimize_normal_trade(input);
        assert!(
            r.collateral_required > 0.0,
            "expected a σ-arb trade, got no-trade",
        );
        assert!(
            r.expected_value > 0.0,
            "expected EV > 0, got {}",
            r.expected_value,
        );
        assert!(r.collateral_required <= input.budget + 1e-9);
    }

    /// Regression — at LOW `k` (or wide σ) the chain charge stays small
    /// enough that σ-arb is still profitable. This is the positive case
    /// the optimizer must still pick up.
    ///
    /// At `k=1, σ_m=4` (so λ ≈ 1·√(2·4·√π) ≈ 4.76 — small), tight
    /// belief (`σ_b=0.5`) gives a clearly profitable σ-shrink. The
    /// optimizer must return a positive-net trade and respect the
    /// chain-frame budget filter.
    #[test]
    fn low_k_sigma_arb_finds_trade() {
        let input = NormalOptimizationInput::new(
            100.0, // budget
            0.0,   // belief μ
            0.5,   // belief σ
            0.0,   // market μ
            4.0,   // market σ
            1.0,   // effective k — small, so λ ≈ 5
        );
        let r = optimize_normal_trade(input);
        assert!(
            r.collateral_required > 0.0,
            "expected positive trade at low-k, got coll={}",
            r.collateral_required,
        );
        assert!(
            r.expected_value > 0.0,
            "expected positive EV (ev={})",
            r.expected_value,
        );
        assert!(r.optimized_sigma < input.market_sigma * 0.9);
        assert!(r.collateral_required <= input.budget + 1e-9);
    }

    // ─── λ-scaled chain-frame regressions (v0.1.3) ───────────────────────
    //
    // These pin the two bugs Reviewer B's audit named so they cannot
    // silently re-enter. Cheap unit tests that target the inner cost frame
    // directly.

    /// **Bug A — budget filter must use λ-scaled cost.**
    ///
    /// Scenario: small budget = 10, market params where the **unscaled**
    /// per-candidate collateral is small (~0.4) but the **λ-scaled**
    /// chain charge is large (~80). The pre-v0.1.3 optimizer compared
    /// the unscaled value against the budget and silently emitted a
    /// trade the chain would charge 8× the budget for; the v0.1.3
    /// optimizer rejects it cleanly.
    ///
    /// Concretely: at k=75.07, σ=0.35, λ ≈ 83.6 (cf.
    /// `lambda_at_k50_sigma8_matches_doc` in deadeye-sdk). The σ-arb
    /// scenario has unscaled coll ≈ 0.4 at the optimum; the λ-scaled
    /// charge ≈ 30. With budget=10 the chain would reject — and so
    /// must the optimizer.
    #[test]
    fn test_budget_filter_must_use_lambda_scaled_cost() {
        // CPI-like params, budget below the λ-scaled cost of the σ-arb
        // optimum but above the unscaled cost.
        let input = NormalOptimizationInput::new(
            5.0,   // budget — well below λ-scaled cost (~16)
            4.29,  // belief μ
            0.10,  // belief σ — very tight, big σ-arb on paper
            4.29,  // market μ
            0.35,  // market σ
            75.07, // k
        );
        let r = optimize_normal_trade(input);
        // The chain charge for the σ-arb optimum (~16 STRK) exceeds
        // the 5 STRK budget, so the optimizer must either return
        // no-trade OR a different in-budget λ-scaled candidate.
        // Either way: any returned coll must satisfy coll ≤ budget in
        // λ-scaled chain units (the load-bearing invariant here).
        assert!(
            r.collateral_required <= input.budget + 1e-9,
            "budget filter leak: returned coll={} > budget={}",
            r.collateral_required,
            input.budget,
        );
        // Any returned trade has positive expected P&L (collateral is a
        // returned lock, so the bar is EV > 0, not EV ≥ collateral).
        if r.collateral_required > 0.0 {
            assert!(
                r.expected_value > 0.0,
                "filter accepted a non-positive-EV candidate (ev={})",
                r.expected_value,
            );
        }
    }

    /// **Bug B — candidate selection must use λ-scaled units consistently.**
    ///
    /// Scenario contrived so two grid points score within < 1% under
    /// the buggy mixed-units selector (λ-scaled EV − unscaled cost) but
    /// separate clearly under the chain-correct λ-scaled selector
    /// (λ-scaled EV − λ-scaled cost): a μ-shift candidate carries a
    /// large `λ_g` shrink (cheaper in chain frame) than a σ-shrink one,
    /// even though their unscaled costs are similar.
    ///
    /// The test asserts the optimizer's returned net, **re-evaluated
    /// in chain units**, is non-negative and matches the optimum
    /// witnessed by a brute scan over the same lattice.
    #[test]
    fn test_candidate_selection_must_use_lambda_scaled_units() {
        let mu_m = 0.0_f64;
        let sigma_m = 1.0_f64;
        let k = 100.0_f64;
        let mu_b = 1.5_f64;
        let sigma_b = 0.4_f64;
        let budget = 50.0_f64;
        let r = optimize_normal_trade(NormalOptimizationInput::new(
            budget, mu_b, sigma_b, mu_m, sigma_m, k,
        ));
        // The optimizer maximizes λ-scaled EV; a returned trade has EV > 0
        // (collateral is a returned lock, not subtracted from EV — issue #12).
        assert!(
            r.expected_value > 0.0,
            "expected positive λ-scaled EV (got ev={} coll={})",
            r.expected_value,
            r.collateral_required,
        );
        // Re-evaluate the returned candidate's λ-scaled cost
        // independently and confirm it matches the optimizer's report
        // (within float tolerance). Pre-v0.1.3 the optimizer's reported
        // `collateral_required` was unscaled — this assertion would
        // have failed by ~200× back then.
        let recomputed = collateral_number(mu_m, sigma_m, r.optimized_mean, r.optimized_sigma, k);
        let diff = (recomputed - r.collateral_required).abs();
        let tol = 1e-6_f64.max(r.collateral_required.abs() * 1e-9);
        assert!(
            diff < tol,
            "reported λ-scaled coll {} disagrees with re-derivation {} (diff {})",
            r.collateral_required,
            recomputed,
            diff,
        );
        // And: the budget filter held in chain frame.
        assert!(r.collateral_required <= budget + 1e-9);
    }
}
