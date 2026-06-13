//! Multi-market effective sensitivity.
//!
//! Computes k-output linearly-coupled market-maker exposure via the
//! effective sensitivity reduction. When a market maker warehouses
//! positions across k scalar markets simultaneously, the joint risk
//! budget inflates by the L2-norm of the coupling vector c.
//!
//! ## Assumptions / Standard Model
//!
//! The reduction `Δ_eff = Δ · ||c||₂` is **mathematically exact under**
//! the following conditions:
//!
//! 1. **Linear coupling** of per-market query functions (i.e. the
//!    joint release is a linear map of the per-market releases).
//! 2. **Independent Gaussian noise** across markets.
//! 3. **Full-vector observation** by the analyst (the analyst sees
//!    the entire k-dimensional release, not a per-market projection).
//!
//! See "Effective-Sensitivity Correction for Correlated Releases"
//! (Zenodo DOI 10.5281/zenodo.20434661, Theorem 1) for the proof. The
//! complementary correlated-noise case is the GCI Sign Theorem
//! (Paper #1, DOI 10.5281/zenodo.20078486).
//!
//! If Deadeye later introduces non-linear payoffs, conditional
//! markets, hierarchical markets, or complex correlated distributions,
//! this reduction becomes an **approximation** rather than an exact
//! bound. Callers operating outside the standard model must supply a
//! coupling vector derived from domain knowledge, empirical
//! correlations, or a dedicated risk model.

use deadeye_core::CoreError;

use crate::CollateralError;

/// Reject degenerate / non-finite inputs uniformly. `?` on the return
/// value lifts the [`CoreError`] into [`CollateralError::Core`] via the
/// existing `#[from]` impl — no new public API surface.
#[inline]
fn invalid_input(field: &'static str, msg: impl Into<String>) -> Result<f64, CollateralError> {
    Err(CoreError::invalid_input(field, msg).into())
}

/// Closed-form effective sensitivity for k-output linearly-coupled releases.
///
/// Given per-market (scalar) sensitivity `delta` and coupling vector
/// `c = (c_1, …, c_k)` with `c_1 = 1` (i.e. `c` is normalised to the
/// reference market), returns
///
/// ```text
/// Δ_eff = Δ · ||c||₂ = Δ · √(c_1² + c_2² + … + c_k²)
/// ```
///
/// This is the risk parameter that should be passed to a scalar
/// accountant in place of `Δ` when the MM warehouses the joint position
/// and the release is observable in full. The reduction is exact under
/// the [linear-coupling model](self#assumptions--standard-model) and
/// matches the 2.26× inflation of the privacy-loss random variable's
/// hockey-stick divergence `H_ε` at full coupling for k = 2 equal
/// markets (c = (1, 1), ||c||₂ = √2).
///
/// # Errors
///
/// Returns [`CollateralError::Core`] wrapping [`CoreError::InvalidInput`]
/// when `c` is empty, `delta` is negative or non-finite, `c` contains a
/// non-finite entry, or the final product overflows to a non-finite
/// value. We reject `delta = +∞` even though mathematically `∞ · ||c||₂`
/// could be defined, because the result is not a meaningful risk
/// surface for a downstream Gaussian accountant.
pub fn effective_sensitivity(c: &[f64], delta: f64) -> Result<f64, CollateralError> {
    if c.is_empty() {
        return invalid_input("c", "coupling vector is empty");
    }
    if !delta.is_finite() || delta < 0.0 {
        return invalid_input("delta", "must be finite and non-negative");
    }
    let norm_sq: f64 = c.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() {
        return invalid_input("c", "non-finite entry in coupling vector (NaN or Inf)");
    }
    let result = delta * norm_sq.sqrt();
    if !result.is_finite() {
        return invalid_input("result", "overflow: delta * ||c||_2 is non-finite");
    }
    Ok(result)
}

/// Multi-market collateral as a single-market collateral inflated by `||c||₂`.
///
/// Equivalent to `single_market_collateral * effective_sensitivity(c, 1.0)`
/// when the per-market solver has already returned a `collateral` figure
/// in the scalar frame. Use this to compose any of the existing per-market
/// solvers (lognormal, bivariate, categorical, normal) with multi-market
/// risk accounting without changing the solvers themselves.
///
/// # Errors
///
/// Returns [`CollateralError::Core`] wrapping [`CoreError::InvalidInput`]
/// when `c` is empty (degenerate `Δ_eff = 0` is meaningless for risk
/// accounting) or when the final product `single_market_collateral *
/// factor` is non-finite — even if both inputs are individually finite,
/// `1e308 * 2.0` overflows and we reject rather than silently propagate
/// `+inf` collateral.
#[must_use = "rejects empty `c` to avoid degenerate Δ_eff = 0 silent return"]
pub fn inflate_collateral(
    single_market_collateral: f64,
    c: &[f64],
) -> Result<f64, CollateralError> {
    if c.is_empty() {
        return invalid_input("c", "coupling vector is empty");
    }
    if !single_market_collateral.is_finite() {
        return invalid_input(
            "single_market_collateral",
            "must be finite (non-NaN, non-Inf)",
        );
    }
    let factor = effective_sensitivity(c, 1.0)?;
    let inflated = single_market_collateral * factor;
    if !inflated.is_finite() {
        return invalid_input(
            "result",
            "overflow: single_market_collateral * ||c||_2 is non-finite",
        );
    }
    Ok(inflated)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn zero_coupling_is_identity() {
        // c = (1, 0, …, 0) ||c||₂ = 1 → Δ_eff = Δ
        let c = [1.0, 0.0, 0.0, 0.0];
        assert!((effective_sensitivity(&c, 1.0).unwrap() - 1.0).abs() < TOL);
        assert!((effective_sensitivity(&c, 0.5).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn two_equal_markets_gives_sqrt2_inflation() {
        // k = 2, c = (1, 1) → ||c||₂ = √2
        let c = [1.0, 1.0];
        let result = effective_sensitivity(&c, 1.0).unwrap();
        assert!((result - core::f64::consts::SQRT_2).abs() < TOL);
    }

    #[test]
    fn two_equal_markets_matches_paper4_table1() {
        // Paper #4 Table 1: c=1, Δ=1, ||c||₂=√2, Δ_eff=√2.
        // Inflate a per-market collateral of 1.2694e-1 (naive, Δ=1) by
        // √2 and check we land near 1.7958e-1 (≈ 1.2694e-1 · 1.4142).
        let naive = 1.2694e-1_f64;
        let c = [1.0, 1.0];
        let inflated = inflate_collateral(naive, &c).unwrap();
        let expected = naive * core::f64::consts::SQRT_2;
        assert!(
            (inflated - expected).abs() < 1e-12,
            "inflated={inflated}, expected={expected}"
        );
    }

    #[test]
    fn monotonicity_in_c_norm() {
        // Δ_eff strictly increasing in ||c||₂ for fixed Δ.
        let delta = 1.0_f64;
        let r1 = effective_sensitivity(&[1.0, 0.3], delta).unwrap();
        let r2 = effective_sensitivity(&[1.0, 0.5], delta).unwrap();
        let r3 = effective_sensitivity(&[1.0, 0.7], delta).unwrap();
        let r4 = effective_sensitivity(&[1.0, 1.0], delta).unwrap();
        assert!(r1 < r2 && r2 < r3 && r3 < r4);
    }

    #[test]
    fn non_finite_c_is_rejected() {
        let c = [1.0, f64::NAN];
        effective_sensitivity(&c, 1.0).unwrap_err();
    }

    #[test]
    fn negative_delta_is_rejected() {
        let c = [1.0, 1.0];
        effective_sensitivity(&c, -1.0).unwrap_err();
    }

    #[test]
    fn k_market_uniform_coupling_grows_as_sqrt_k() {
        // c_j = 1/√k for all j → ||c||₂ = 1 (degenerate: no inflation).
        // Conversely, c_j = 1 for all j → ||c||₂ = √k.
        let k = 4;
        let c_uniform: Vec<f64> = vec![1.0; k];
        let result = effective_sensitivity(&c_uniform, 1.0).unwrap();
        assert!((result - (k as f64).sqrt()).abs() < TOL);
    }

    #[test]
    fn empty_coupling_vector_is_rejected() {
        // Degenerate case: a warehouse with no markets has no meaningful
        // effective sensitivity (||c||₂ = 0 would imply Δ_eff = 0, which
        // is mathematically true but operationally nonsense). Reject
        // rather than silently return zero.
        effective_sensitivity(&[], 1.0).unwrap_err();
        inflate_collateral(1.0, &[]).unwrap_err();
    }

    #[test]
    fn inflate_collateral_with_paper4_table1() {
        // Mirror `test_inflate_collateral` in standard form: base=1.0,
        // c=(1,1) → inflation by √2 ≈ 1.4142.
        let c = [1.0_f64, 1.0];
        let inflated = inflate_collateral(1.0, &c).unwrap();
        assert!(
            (inflated - core::f64::consts::SQRT_2).abs() < 1e-3,
            "inflated={inflated}, expected≈1.4142"
        );
    }

    // ----------------------------------------------------------------
    // Non-finite / overflow / stress tests (review feedback for PR #41)
    //
    // For large-value stress we only check that the result is finite
    // and positive; the exact sqrt behaviour at 1e150+ can vary across
    // platforms and optimisations, so we assert properties rather than
    // a specific value (see ChatGPT review point #2).
    // ----------------------------------------------------------------

    #[test]
    fn nan_in_c_is_rejected() {
        // NaN propagates through x*x and the sum, and the final
        // is_finite() guard catches it.
        effective_sensitivity(&[1.0, f64::NAN], 1.0).unwrap_err();
        effective_sensitivity(&[f64::NAN, 1.0], 1.0).unwrap_err();
    }

    #[test]
    fn infinity_in_c_is_rejected() {
        // 1e308 * 1e308 overflows; the guard catches it.
        effective_sensitivity(&[f64::INFINITY, 1.0], 1.0).unwrap_err();
        effective_sensitivity(&[1.0, f64::INFINITY], 1.0).unwrap_err();
    }

    #[test]
    fn nan_delta_is_rejected() {
        effective_sensitivity(&[1.0, 1.0], f64::NAN).unwrap_err();
    }

    #[test]
    fn positive_infinity_delta_is_rejected() {
        // Mathematically ∞ · ||c||₂ = ∞, but this is not a meaningful
        // risk parameter for a downstream accountant.
        effective_sensitivity(&[1.0, 1.0], f64::INFINITY).unwrap_err();
    }

    #[test]
    fn very_large_but_finite_c_does_not_overflow() {
        // 1e150² = 1e300, which is still finite; ||c||₂ ≈ √2 · 1e150.
        // We only assert properties, not exact values, to stay robust
        // across IEEE-754 implementations and sqrt edge cases.
        let c = [1e150_f64, 1e150];
        let factor = effective_sensitivity(&c, 1.0).unwrap();
        assert!(factor.is_finite(), "factor must be finite, got {factor}");
        assert!(factor > 0.0, "factor must be positive, got {factor}");
        // Sanity: the factor should be roughly √2 · 1e150.
        assert!((factor / 1e150 - core::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn c_at_overflow_boundary_is_rejected() {
        // 1e200² = 1e400, which exceeds f64::MAX (≈ 1.8e308) and
        // overflows to +Inf in the sum. The guard must reject.
        let c = [1e200_f64, 1e200];
        effective_sensitivity(&c, 1.0).unwrap_err();
    }

    #[test]
    fn inflate_collateral_nan_base_is_rejected() {
        // NaN propagates through multiplication; guard catches it.
        inflate_collateral(f64::NAN, &[1.0, 1.0]).unwrap_err();
    }

    #[test]
    fn inflate_collateral_infinite_base_is_rejected() {
        // Inf * finite_factor = Inf; guard catches it.
        inflate_collateral(f64::INFINITY, &[1.0, 1.0]).unwrap_err();
    }

    #[test]
    fn inflate_collateral_finite_inputs_yield_finite_output() {
        // 1e308 * √2 ≈ 1.414e308, which is below f64::MAX (≈ 1.798e308).
        // Both inputs are finite; the product is finite. This must succeed.
        let base = 1e308_f64;
        let c = [1.0, 1.0];
        let inflated = inflate_collateral(base, &c).unwrap();
        assert!(inflated.is_finite(), "expected finite, got {inflated}");
        assert!(inflated > 0.0);
        // Sanity: ratio should be √2.
        assert!((inflated / base - core::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn inflate_collateral_overflow_product_is_rejected() {
        // f64::MAX / √2 ≈ 1.272e308. Picking base = 1.5e308 with c=(1,1)
        // gives product ≈ 2.12e308, which exceeds f64::MAX (≈ 1.798e308)
        // and overflows to +Inf. The post-multiplication guard must catch
        // this. This is exactly the scenario the guard is designed for:
        // both inputs are finite, but the product is not.
        let base = 1.5e308_f64;
        let c = [1.0, 1.0];
        inflate_collateral(base, &c).unwrap_err();
    }

    #[test]
    fn inflate_collateral_negative_infinity_base_is_rejected() {
        // -Inf is non-finite; guard catches it before the product step.
        inflate_collateral(f64::NEG_INFINITY, &[1.0, 1.0]).unwrap_err();
    }
}
