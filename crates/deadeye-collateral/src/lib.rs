//! Off-chain collateral computation for Deadeye AMM markets.
//!
//! The Deadeye AMM verifies on-chain but does not compute. The market maker
//! is responsible for finding the value of `x*` that minimises
//! `d(x) = g(x) - f(x)`, then supplying the resulting collateral together
//! with the trade and a square-root hint that allows the chain to
//! double-check `||g||₂`.
//!
//! This crate provides the off-chain side of that contract: numerical
//! primitives, a closed-form Newton-Raphson minimiser for the normal /
//! normal transition, and a typed [`MinimizationPolicy`] that mirrors the
//! "safe envelope" used by the on-chain verifier.

#![doc(html_no_source)]

use deadeye_core::{CoreError, Distribution, NormalDistribution, Sq128};
use thiserror::Error;

pub mod bivariate;
pub mod categorical;
pub mod lognormal;
pub mod multi_market;
pub use bivariate::{BivariateOptions, BivariateVerifiedMinimum, bivariate_collateral};
pub use categorical::{
    CategoricalVerifiedMinimum, categorical_collateral, categorical_l2_norm, categorical_lambda,
};
pub use lognormal::{LognormalOptions, LognormalVerifiedMinimum, lognormal_collateral};
pub use multi_market::{effective_sensitivity, inflate_collateral};

/// `√π`, used in the closed-form L2 norm of a Gaussian PDF.
pub const SQRT_PI: f64 = 1.772_453_850_905_516_f64;

/// Errors emitted by the collateral solver.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CollateralError {
    /// Wraps an arithmetic / domain error from [`deadeye_core`].
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Distribution failed the "safe envelope" pre-check (σ ratio, mean
    /// separation, mean magnitude).
    #[error("distribution pair rejected by minimisation policy: {reason}")]
    PolicyRejected {
        /// Symbolic reason, suitable for surfacing to operators.
        reason: PolicyRejection,
    },

    /// Newton-Raphson exceeded the iteration budget.
    #[error("Newton-Raphson did not converge after {iterations} iterations")]
    NewtonDidNotConverge {
        /// Iterations attempted.
        iterations: u32,
    },

    /// The verified minimum did not satisfy the on-chain shape contract.
    #[error("verification failed: {check}")]
    VerificationFailed {
        /// Which check failed.
        check: VerificationCheck,
    },
}

/// Symbolic reasons a distribution pair can be rejected by the policy gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyRejection {
    /// `max(σ_g, σ_f) / min(σ_g, σ_f)` exceeded `policy.max_sigma_ratio`.
    SigmaRatioTooLarge,
    /// `|μ_g - μ_f| / min(σ_g, σ_f)` exceeded `policy.max_mean_separation`.
    MeanSeparationTooLarge,
    /// `|μ_g|` or `|μ_f|` exceeded `policy.max_absolute_mean`.
    MeanMagnitudeTooLarge,
    /// One of the distributions is a point mass (variance 0).
    DegenerateDistribution,
}

impl core::fmt::Display for PolicyRejection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::SigmaRatioTooLarge => "sigma ratio out of envelope",
            Self::MeanSeparationTooLarge => "mean separation out of envelope",
            Self::MeanMagnitudeTooLarge => "absolute mean out of envelope",
            Self::DegenerateDistribution => "degenerate distribution",
        };
        f.write_str(s)
    }
}

/// Verification checks performed on a candidate minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerificationCheck {
    /// `d'(x*)` was not within tolerance of zero.
    NotStationary,
    /// `d''(x*) <= 0` — not a minimum.
    NotPositiveCurvature,
    /// `d(x*) > 0` — collateral is unnecessary, indicating solver landed on
    /// the wrong critical point.
    WrongSide,
}

impl core::fmt::Display for VerificationCheck {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::NotStationary => "stationary check failed (d'(x*) ≠ 0)",
            Self::NotPositiveCurvature => "curvature check failed (d''(x*) ≤ 0)",
            Self::WrongSide => "side check failed (d(x*) > 0)",
        };
        f.write_str(s)
    }
}

/// Operating envelope for the off-chain minimiser.
///
/// Mirrors the *numerical* bounds the on-chain math runtime expects. The
/// on-chain AMM itself does not gate trades by sigma-ratio or mean
/// separation — only its underlying math runtime imposes magnitude limits
/// (see `max_absolute_mean` in Cairo's `collateral_normal::minimize`).
/// Earlier revisions of this struct retained `max_sigma_ratio` and
/// `max_mean_separation` envelopes, but those guards rejected trades the
/// chain happily accepts (any μ-shift, σ-shrink, or σ-equal transition).
///
/// The fields are preserved for backwards compatibility with callers that
/// want a soft pre-check, but [`MinimizationPolicy::standard`] now uses
/// `f64::INFINITY` for both — i.e. the standard policy is fully
/// permissive. Use [`MinimizationPolicy::unrestricted`] if you also want
/// to disable the absolute-mean guard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimizationPolicy {
    /// Maximum ratio between the larger and smaller σ.
    pub max_sigma_ratio: f64,
    /// Maximum `|μ_g - μ_f|` measured in units of the narrower σ.
    pub max_mean_separation: f64,
    /// Maximum `|μ|` of either distribution.
    pub max_absolute_mean: f64,
    /// Tolerance used to consider Newton converged.
    pub tolerance: f64,
    /// Max Newton iterations before giving up.
    pub max_iterations: u32,
}

impl MinimizationPolicy {
    /// Default policy: matches the trade space the chain accepts.
    ///
    /// Drops the `max_sigma_ratio` and `max_mean_separation` envelope
    /// guards (set to `f64::INFINITY`) because the on-chain verifier
    /// itself never gated on them — only the math runtime's internal
    /// validity does (mean magnitude < 2^96, valid arithmetic). Keeps
    /// `max_absolute_mean` as a sanity guard to match the chain's
    /// implicit overflow bound.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_sigma_ratio: f64::INFINITY,
            max_mean_separation: f64::INFINITY,
            max_absolute_mean: 1_048_576.0,
            tolerance: 1e-12,
            max_iterations: 80,
        }
    }

    /// Fully unrestricted policy.
    ///
    /// All envelope guards (sigma ratio, mean separation, absolute mean)
    /// are disabled. Newton convergence and curvature/stationarity
    /// post-checks remain. Use this from chaos suites or scenarios that
    /// want zero off-chain gating — only the chain's final verifier
    /// decides validity.
    #[must_use]
    pub const fn unrestricted() -> Self {
        Self {
            max_sigma_ratio: f64::INFINITY,
            max_mean_separation: f64::INFINITY,
            max_absolute_mean: f64::INFINITY,
            tolerance: 1e-12,
            max_iterations: 80,
        }
    }

    /// Relaxed policy retained for back-compat; identical to
    /// [`MinimizationPolicy::standard`] now that the envelope guards are
    /// disabled by default.
    #[must_use]
    pub const fn relaxed() -> Self {
        Self::standard()
    }
}

impl Default for MinimizationPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

/// Verified minimum of `d(x) = g(x) - f(x)`.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedMinimum {
    /// Location of the minimum.
    pub x_min: f64,
    /// `d(x_min)`.
    pub d_min: f64,
    /// `max(0, -d_min)` — collateral required.
    pub collateral: f64,
    /// Iterations performed.
    pub iterations: u32,
}

/// `||p||₂` for a normal distribution with standard deviation σ.
///
/// `||p||₂ = 1 / √(2σ√π)`. Returns `0` when σ is non-positive.
///
/// Cairo source: `packages/market-normal/src/l2_norm.cairo:78-102`.
#[must_use]
pub fn l2_norm(sigma: f64) -> f64 {
    if sigma <= 0.0 || !sigma.is_finite() {
        return 0.0;
    }
    let denom_sq = 2.0 * sigma * SQRT_PI;
    1.0 / denom_sq.sqrt()
}

/// `λ = k / ||p||₂` for a normal distribution with σ and AMM parameter `k`.
///
/// Equivalently `λ = k · √(2σ√π)`. This is the per-distribution scaling
/// factor the on-chain verifier applies before checking the PDF-difference
/// curvature contract. Cairo source:
/// `packages/market-normal/src/invariant.cairo:65-75` and
/// `packages/onchain-normal-math/src/helpers.cairo:50-86`.
#[must_use]
pub fn lambda(sigma: f64, k: f64) -> f64 {
    let norm = l2_norm(sigma);
    if norm <= 0.0 { 0.0 } else { k / norm }
}

/// Computes the collateral required to transition `f → g`.
///
/// Minimises the **lambda-scaled** PDF difference
/// `d̃(x) = λ_g · g(x) − λ_f · f(x)` — exactly the same quantity the
/// on-chain verifier minimises (Cairo:
/// `packages/onchain-normal-math/src/helpers.cairo:190-230`).
///
/// Uses `f64` throughout; the on-chain verifier re-checks the result with
/// Q128.128 arithmetic, so this is an off-chain hint, not a binding answer.
///
/// ## Sign convention
///
/// The chain stores the collateral requirement as `C = max(0, -d_min)`
/// where `d = λ_g · g − λ_f · f`. We mirror that convention exactly: a
/// negative `d_min` indicates collateral is needed.
///
/// ## Lambda choice
///
/// The lambda factors used during minimisation cancel out of the
/// stationary equation `d'(x*) = 0`; any positive choice produces the
/// same `x*`. We use `k = 1`, which keeps all intermediate values close
/// to unit scale and avoids amplifying f64 round-off. The returned
/// `d_min` and `collateral` are reported in the **unscaled** `g − f`
/// frame (the on-chain verifier rescales by its own `k`).
pub fn normal_collateral(
    f: &NormalDistribution,
    g: &NormalDistribution,
    policy: MinimizationPolicy,
) -> Result<VerifiedMinimum, CollateralError> {
    if f.is_degenerate() || g.is_degenerate() {
        return Err(CollateralError::PolicyRejected {
            reason: PolicyRejection::DegenerateDistribution,
        });
    }
    // Fast path: identical distributions need no collateral. `d(x)=0`
    // everywhere, and the Newton iteration below would divide by `d''(x)=0`.
    if f.mean() == g.mean() && f.variance() == g.variance() {
        return Ok(VerifiedMinimum {
            x_min: f.mean().to_f64(),
            d_min: 0.0,
            collateral: 0.0,
            iterations: 0,
        });
    }

    let sigma_f = f.sigma().to_f64();
    let sigma_g = g.sigma().to_f64();
    let sigma_min = sigma_f.min(sigma_g);
    let sigma_max = sigma_f.max(sigma_g);
    let mu_f = f.mean().to_f64();
    let mu_g = g.mean().to_f64();

    // Soft envelope checks. By default `standard()` sets these to
    // `f64::INFINITY` so trades pass through; callers that want a tighter
    // gate can supply their own policy.
    if sigma_max / sigma_min > policy.max_sigma_ratio {
        return Err(CollateralError::PolicyRejected {
            reason: PolicyRejection::SigmaRatioTooLarge,
        });
    }
    if ((mu_g - mu_f).abs() / sigma_min) > policy.max_mean_separation {
        return Err(CollateralError::PolicyRejected {
            reason: PolicyRejection::MeanSeparationTooLarge,
        });
    }
    if mu_f.abs() > policy.max_absolute_mean || mu_g.abs() > policy.max_absolute_mean {
        return Err(CollateralError::PolicyRejected {
            reason: PolicyRejection::MeanMagnitudeTooLarge,
        });
    }

    // Lambda factors for the chain-side scaled minimisation. The choice
    // of `k` cancels out of `d'(x*) = 0`; using `k = 1` keeps everything
    // at unit scale for f64 precision.
    let lambda_f = lambda(sigma_f, 1.0_f64);
    let lambda_g = lambda(sigma_g, 1.0_f64);

    // Bracket the global minimum of the SCALED difference `d̃` with a
    // coarse grid scan, then refine with Newton. This is more robust than
    // the chain's bare `μ_f ± 2σ_f` initial guess for two scenarios that
    // the chain accepts but a naïve Newton struggles with:
    //
    //   * Equal-σ μ-shift: the minimum sits a fraction of σ inside μ_f on the side
    //     opposite to g; `μ_f ± 2σ_f` lands in a region where `d̃''<0` so Newton
    //     diverges further into the tail.
    //   * Same-μ σ-shrink: `d̃` has a local *maximum* at μ_f and twin symmetric
    //     minima in the tails (≈ √(2 ln(σ_f/σ_g))·σ_f σ_g / √(σ_f² − σ_g²) units
    //     out). A grid scan finds these reliably.
    //
    // The grid spans ±6 of the larger σ (matching the chain's implicit
    // numerical envelope for normals) and uses 96 samples — overkill for
    // a single-shot off-chain solve, but free at f64 rates.
    let x0 = find_grid_seed(f, g, lambda_f, lambda_g, mu_f, mu_g, sigma_max);

    let (x_min, iterations) = newton_minimise_scaled(f, g, lambda_f, lambda_g, x0, policy)?;

    let d_min = eval_d(f, g, x_min)?;
    // Final verification uses the LAMBDA-SCALED derivative + curvature,
    // matching the chain. Cairo: `scaled_verify_minimum_with_lambda`.
    let d_prime_scaled = eval_d_prime_scaled(f, g, lambda_f, lambda_g, x_min)?;
    let d_double_scaled = eval_d_double_prime_scaled(f, g, lambda_f, lambda_g, x_min)?;
    // `policy.tolerance` is the f64 Newton tolerance; allow a small
    // scaling cushion (proportional to lambda) for the post-check.
    let tol = policy.tolerance * lambda_f.max(lambda_g).max(1.0_f64);
    if d_prime_scaled.abs() > tol {
        return Err(CollateralError::VerificationFailed {
            check: VerificationCheck::NotStationary,
        });
    }
    if d_double_scaled <= 0.0 {
        return Err(CollateralError::VerificationFailed {
            check: VerificationCheck::NotPositiveCurvature,
        });
    }
    // The "wrong-side" guard uses the SCALED value: chain stores
    // collateral = max(0, -d̃_min); we accept any d_min that is negative
    // in the scaled frame.
    let d_min_scaled = eval_d_scaled(f, g, lambda_f, lambda_g, x_min)?;
    if d_min_scaled > tol {
        return Err(CollateralError::VerificationFailed {
            check: VerificationCheck::WrongSide,
        });
    }

    let collateral = if d_min < 0.0 { -d_min } else { 0.0 };
    Ok(VerifiedMinimum {
        x_min,
        d_min,
        collateral,
        iterations,
    })
}

/// Coarse grid search for the seed value of x*. Scans the scaled
/// difference `d̃ = λ_g·g − λ_f·f` on a window centred between the means
/// and returns the grid point with the smallest (most-negative) value.
///
/// This is the off-chain robustness layer: the chain itself relies on
/// `μ_f ± 2σ_f` as its starting point and falls back to caller-provided
/// `x*` hints when its built-in Newton fails. The off-chain solver is
/// responsible for *finding* that hint, hence the wider initial sweep.
fn find_grid_seed(
    f: &NormalDistribution,
    g: &NormalDistribution,
    lambda_f: f64,
    lambda_g: f64,
    mu_f: f64,
    mu_g: f64,
    sigma_max: f64,
) -> f64 {
    // Span ±6 σ_max around the midpoint of the means — enough headroom
    // for the off-axis tail minima of σ-shrink trades while still
    // staying inside the chain's numerical envelope (`|μ| < 2^96`).
    let centre = 0.5_f64 * (mu_f + mu_g);
    let half_width = 6.0_f64.mul_add(sigma_max, (mu_g - mu_f).abs());
    let lo = centre - half_width;
    let hi = centre + half_width;
    let samples: u32 = 96;
    let step = (hi - lo) / f64::from(samples);

    let mut best_x = centre;
    let mut best_d = f64::INFINITY;
    for i in 0..=samples {
        let x = step.mul_add(f64::from(i), lo);
        if let Ok(d) = eval_d_scaled(f, g, lambda_f, lambda_g, x)
            && d.is_finite()
            && d < best_d
        {
            best_d = d;
            best_x = x;
        }
    }
    best_x
}

/// Newton-Raphson minimisation of the lambda-scaled difference, mirroring
/// the chain's `newton_minimize` implementation in
/// `packages/newton/src/lib.cairo:173-233`.
///
/// Key safeguards (matching Cairo):
/// 1. Step clamp: `|step| ≤ 0.5 · max(1, |x|)` per iteration. Without this the
///    iteration can jump wildly when far from the minimum (this was the root
///    cause of the `NewtonDidNotConverge` failures for `N(42,64) →
///    N(43,64)`-style trades — `μ_f − 2σ_f = 26` has `d''<0` and the raw Newton
///    step shoots tens of σ deep into the tail).
/// 2. Convergence: `|d'(x)| < tolerance · scale`, NOT `|Δx| < tolerance`. The
///    scale factor `max(λ_f, λ_g, 1)` accounts for lambda-scaling.
/// 3. Minimum second-derivative gate at ≈ 2⁻⁶⁰.
fn newton_minimise_scaled(
    f: &NormalDistribution,
    g: &NormalDistribution,
    lambda_f: f64,
    lambda_g: f64,
    initial_guess: f64,
    policy: MinimizationPolicy,
) -> Result<(f64, u32), CollateralError> {
    let mut x = initial_guess;
    let scale = lambda_f.max(lambda_g).max(1.0_f64);
    let convergence_tol = policy.tolerance * scale;
    let min_d2 = 2.0_f64.powi(-60);
    let max_step_fraction = 0.5_f64;

    for iter in 0..policy.max_iterations {
        let d_prime = eval_d_prime_scaled(f, g, lambda_f, lambda_g, x)?;
        if d_prime.abs() < convergence_tol {
            return Ok((x, iter));
        }
        let d_double = eval_d_double_prime_scaled(f, g, lambda_f, lambda_g, x)?;
        if !d_prime.is_finite() || !d_double.is_finite() {
            return Err(CollateralError::NewtonDidNotConverge { iterations: iter });
        }
        if d_double.abs() < min_d2 {
            return Err(CollateralError::NewtonDidNotConverge { iterations: iter });
        }
        let raw_step = d_prime / d_double;
        let position_scale = x.abs().max(1.0_f64);
        let max_step = max_step_fraction * position_scale;
        let clamped_step = raw_step.clamp(-max_step, max_step);
        x -= clamped_step;
    }
    // One more derivative check at the final x — Cairo's algorithm
    // tolerates `|d'(x)| < tolerance` at the bound boundary too.
    let final_d_prime = eval_d_prime_scaled(f, g, lambda_f, lambda_g, x)?;
    if final_d_prime.abs() < convergence_tol {
        return Ok((x, policy.max_iterations));
    }
    Err(CollateralError::NewtonDidNotConverge {
        iterations: policy.max_iterations,
    })
}

fn eval_d(f: &NormalDistribution, g: &NormalDistribution, x: f64) -> Result<f64, CollateralError> {
    let x_q = Sq128::from_f64(x).map_err(CollateralError::Core)?;
    let g_pdf = g.pdf(x_q)?.to_f64();
    let f_pdf = f.pdf(x_q)?.to_f64();
    Ok(g_pdf - f_pdf)
}

fn eval_d_scaled(
    f: &NormalDistribution,
    g: &NormalDistribution,
    lambda_f: f64,
    lambda_g: f64,
    x: f64,
) -> Result<f64, CollateralError> {
    let x_q = Sq128::from_f64(x).map_err(CollateralError::Core)?;
    let g_pdf = g.pdf(x_q)?.to_f64();
    let f_pdf = f.pdf(x_q)?.to_f64();
    Ok(lambda_g.mul_add(g_pdf, -(lambda_f * f_pdf)))
}

fn eval_d_prime_scaled(
    f: &NormalDistribution,
    g: &NormalDistribution,
    lambda_f: f64,
    lambda_g: f64,
    x: f64,
) -> Result<f64, CollateralError> {
    let x_q = Sq128::from_f64(x).map_err(CollateralError::Core)?;
    let g_p = g.pdf_derivative(x_q)?.to_f64();
    let f_p = f.pdf_derivative(x_q)?.to_f64();
    Ok(lambda_g.mul_add(g_p, -(lambda_f * f_p)))
}

fn eval_d_double_prime_scaled(
    f: &NormalDistribution,
    g: &NormalDistribution,
    lambda_f: f64,
    lambda_g: f64,
    x: f64,
) -> Result<f64, CollateralError> {
    let x_q = Sq128::from_f64(x).map_err(CollateralError::Core)?;
    let g_pp = g.pdf_second_derivative(x_q)?.to_f64();
    let f_pp = f.pdf_second_derivative(x_q)?.to_f64();
    Ok(lambda_g.mul_add(g_pp, -(lambda_f * f_pp)))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use deadeye_core::{NormalDistribution, Sq128};

    use super::*;

    fn nd(mean: f64, variance: f64) -> NormalDistribution {
        NormalDistribution::from_variance(
            Sq128::from_f64(mean).unwrap(),
            Sq128::from_f64(variance).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn l2_norm_matches_closed_form() {
        // ||p||₂ = 1 / sqrt(2σ√π); for σ=1 this is 1/sqrt(2√π) ≈ 0.5311259.
        let v = l2_norm(1.0);
        assert!((v - 0.531_125_9).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn identical_distributions_require_zero_collateral() {
        let f = nd(100.0, 4.0);
        let g = nd(100.0, 4.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!(result.collateral.abs() < 1e-6);
    }

    #[test]
    fn shifted_mean_requires_positive_collateral() {
        let f = nd(100.0, 4.0);
        let g = nd(102.0, 4.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!(
            result.collateral > 0.0,
            "expected positive collateral, got {}",
            result.collateral
        );
    }

    #[test]
    fn standard_policy_accepts_large_sigma_ratio() {
        // The chain accepts σ ratio of 10× (σ²=1 → σ²=100). The off-chain
        // solver must not gate on this. This previously returned
        // `PolicyRejected{SigmaRatioTooLarge}`.
        let f = nd(0.0, 1.0);
        let g = nd(0.0, 100.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard());
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        let v = result.unwrap();
        assert!(v.collateral >= 0.0);
    }

    #[test]
    fn unrestricted_policy_disables_all_envelope_checks() {
        let p = MinimizationPolicy::unrestricted();
        assert!(p.max_sigma_ratio.is_infinite());
        assert!(p.max_mean_separation.is_infinite());
        assert!(p.max_absolute_mean.is_infinite());
    }

    #[test]
    fn degenerate_distribution_is_rejected() {
        let f = nd(0.0, 0.0);
        let g = nd(0.0, 1.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard());
        assert!(matches!(
            result,
            Err(CollateralError::PolicyRejected {
                reason: PolicyRejection::DegenerateDistribution
            })
        ));
    }

    /// Equal-σ pure-μ shift: `N(0, 1) → N(1, 1)`. Cairo precomputes
    /// `x* ≈ -0.5435` (`test_amm_contract.cairo:55-60`). The off-chain
    /// solver must converge to that value within 1e-9.
    #[test]
    fn equal_sigma_pure_mu_shift() {
        let f = nd(0.0, 1.0);
        let g = nd(1.0, 1.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        // Decoded Cairo limbs at line 56-60: -0.54362689559153698884… —
        // we accept ±1e-9 of that.
        // Cairo precomputes `x* ≈ -0.54362689559153698493` to f64
        // precision; our lambda-scaled Newton converges to that exact
        // double after 3 iterations, so we assert `|Δ| < 1e-15` (i.e.
        // within a handful of ulps of the precomputed Cairo value).
        let expected = -0.543_626_895_591_537_f64;
        assert!(
            (result.x_min - expected).abs() < 1e-15,
            "x* = {} (expected ≈ {expected})",
            result.x_min,
        );
        assert!(result.collateral > 0.0);
    }

    /// Pure σ-shrink, same μ. The chain accepts; the off-chain solver
    /// must find the symmetric tail minimum and produce positive
    /// collateral.
    #[test]
    fn shrinking_sigma() {
        let f = nd(0.0, 4.0);
        let g = nd(0.0, 1.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!(result.collateral > 0.0, "{result:?}");
        // x* is symmetric (±x_0); the solver picks one side.
        assert!(result.x_min.abs() > 0.5_f64);
    }

    /// "σ widens AND μ moves opposite to the σ change" — chain accepts,
    /// previous solver returned `NotPositiveCurvature`.
    #[test]
    fn widening_sigma_opposite_mu() {
        let f = nd(45.0, 100.0);
        let g = nd(38.0, 144.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!(result.collateral >= 0.0, "{result:?}");
    }

    /// Equal-σ pure-μ at non-trivial scale. Chain accepts; previous
    /// off-chain solver returned `NewtonDidNotConverge`.
    #[test]
    fn equal_sigma_pure_mu_shift_at_scale() {
        let f = nd(42.0, 64.0);
        let g = nd(43.0, 64.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!(result.collateral > 0.0, "{result:?}");
    }

    /// Identity round-trip: `N(42, 64) → N(42, 64)`. Must short-circuit
    /// with zero collateral.
    #[test]
    fn degenerate_round_trip() {
        let f = nd(42.0, 64.0);
        let g = nd(42.0, 64.0);
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!((result.x_min - 42.0).abs() < 1e-12);
        assert!(result.collateral.abs() < 1e-12);
        assert!(result.d_min.abs() < 1e-12);
        assert_eq!(result.iterations, 0);
    }

    /// Sigma-shrink with mean shift toward the narrower side: the
    /// previous solver's σ-weighted midpoint heuristic was reasonable
    /// for this, but the σ ratio (9/7 ≈ 1.29) and mean shift puts it
    /// near the chain-acceptable boundary.
    #[test]
    fn shrinking_sigma_with_mu_shift() {
        let f = nd(43.0, 81.0); // σ=9
        let g = nd(45.0, 49.0); // σ=7
        let result = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        assert!(result.collateral >= 0.0, "{result:?}");
    }

    #[test]
    #[ignore = "diagnostic probe — run with --ignored --nocapture"]
    #[expect(clippy::print_stderr, reason = "diagnostic")]
    fn probe_scaled_stationary_at_offline_xstar() {
        use deadeye_core::Distribution as _;
        let sf = 0.191_981_067_900_659_27_f64;
        let sg = 0.12_f64;
        let k = 200.0_f64;
        let f = NormalDistribution::from_sigma(
            Sq128::from_f64(4.205).unwrap(),
            Sq128::from_f64(sf).unwrap(),
        )
        .unwrap();
        let g = NormalDistribution::from_variance(
            Sq128::from_f64(4.174).unwrap(),
            Sq128::from_f64(0.0144).unwrap(),
        )
        .unwrap();
        let lf = Sq128::from_f64(lambda(sf, k)).unwrap();
        let lg = Sq128::from_f64(lambda(sg, k)).unwrap();
        let solved = normal_collateral(&f, &g, MinimizationPolicy::standard()).unwrap();
        eprintln!("solver x_min = {:.15}", solved.x_min);
        for (label, xf) in [("solver", solved.x_min), ("hard", 4.405_769_206_389_262)] {
            let x = Sq128::from_f64(xf).unwrap();
            let fd = f.pdf_derivative(x).unwrap();
            let gd = g.pdf_derivative(x).unwrap();
            let dprime = lg
                .checked_mul(gd)
                .unwrap()
                .checked_sub(lf.checked_mul(fd).unwrap())
                .unwrap();
            let fp = f.pdf(x).unwrap();
            let gp = g.pdf(x).unwrap();
            let dval = lg
                .checked_mul(gp)
                .unwrap()
                .checked_sub(lf.checked_mul(fp).unwrap())
                .unwrap();
            eprintln!(
                "[{label}] x*={xf:.12}  scaled d'={:.4e}  scaled d(=-collat)={:.4}  (tol=1e-3)",
                dprime.to_f64(),
                dval.to_f64(),
            );
        }
        eprintln!(
            "lambda_f={:.6} lambda_g={:.6}",
            lambda(sf, k),
            lambda(sg, k)
        );
    }
}
