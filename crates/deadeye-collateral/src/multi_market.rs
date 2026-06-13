//! Multi-market effective sensitivity.
//!
//! Computes k-output linearly-coupled market-maker exposure via the
//! effective sensitivity reduction. When a market maker warehouses
//! positions across k scalar markets simultaneously, the joint risk
//! budget inflates by the L2-norm of the coupling vector c.
//!
//! The reduction is exact under linear coupling of the per-market
//! query functions, independent Gaussian noise, and full-vector
//! observation. See "Effective-Sensitivity Correction for Correlated
//! Releases" (Zenodo DOI 10.5281/zenodo.20434661, Theorem 1) for the
//! proof. The complementary correlated-noise case
//! is the GCI Sign Theorem (Paper #1, DOI 10.5281/zenodo.20078486).

use crate::CollateralError;

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
/// and the release is observable in full. It is **exact** under the
/// linear-coupling model (see module-level docs) and matches the
/// 2.26× inflation of the privacy-loss random variable's
/// hockey-stick divergence `H_ε` at full coupling for k = 2 equal
/// markets (c = (1, 1), ||c||₂ = √2).
///
/// # Errors
///
/// Returns `CollateralError::VerificationFailed` if `c` is empty,
/// `delta < 0`, or the resulting norm is not finite (e.g. `c` contains
/// a non-finite entry).
pub fn effective_sensitivity(c: &[f64], delta: f64) -> Result<f64, CollateralError> {
    if c.is_empty() {
        return Err(CollateralError::VerificationFailed {
            check: crate::VerificationCheck::NotStationary,
        });
    }
    if !delta.is_finite() || delta < 0.0 {
        return Err(CollateralError::VerificationFailed {
            check: crate::VerificationCheck::NotStationary,
        });
    }
    let norm_sq: f64 = c.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() {
        return Err(CollateralError::VerificationFailed {
            check: crate::VerificationCheck::NotStationary,
        });
    }
    let result = delta * norm_sq.sqrt();
    if !result.is_finite() {
        return Err(CollateralError::VerificationFailed {
            check: crate::VerificationCheck::NotPositiveCurvature,
        });
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
/// Returns [`CollateralError::VerificationFailed`] when `c` is empty —
/// mathematically `Δ_eff = Δ · ||c||₂ = 0` is degenerate (a warehouse
/// with no markets is not a meaningful risk surface), so we reject it
/// rather than silently returning zero collateral.
#[must_use = "rejects empty `c` to avoid degenerate Δ_eff = 0 silent return"]
pub fn inflate_collateral(
    single_market_collateral: f64,
    c: &[f64],
) -> Result<f64, CollateralError> {
    if c.is_empty() {
        return Err(CollateralError::VerificationFailed {
            check: crate::VerificationCheck::NotStationary,
        });
    }
    let factor = effective_sensitivity(c, 1.0)?;
    Ok(single_market_collateral * factor)
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
        assert!(effective_sensitivity(&c, 1.0).is_err());
    }

    #[test]
    fn negative_delta_is_rejected() {
        let c = [1.0, 1.0];
        assert!(effective_sensitivity(&c, -1.0).is_err());
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
        assert!(effective_sensitivity(&[], 1.0).is_err());
        assert!(inflate_collateral(1.0, &[]).is_err());
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
}
