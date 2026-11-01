//! Multiple imputation by additive regression (`Hmisc::aregImpute`).
//!
//! A thin wrapper around [`anofox_regression::imputation::areg_impute`]:
//! each variable with missing values is imputed in turn from all the others
//! with flexible additive models (splines, dummies for categorical columns)
//! on bootstrap samples, and predictive mean matching picks observed donor
//! values. After the burn-in every further round yields one completed data
//! set, so the result is `n_impute` data sets rather than one. Fit the
//! downstream model to each and pool the estimates with Rubin's rules
//! (`anofox_regression::inference::pool::pool_rubin`).
//!
//! Unlike [`IterativeImputer`](crate::IterativeImputer) this is not a
//! transformer: it imputes the data it is given and keeps no model for new
//! rows. With the same seed the imputations replay R's
//! `set.seed(seed); aregImpute(...)`.

use anofox_ml_core::{Result, RustMlError};
use anofox_regression::imputation::{areg_impute, AregImputeOptions, DonorMatch, ImputeVariable};
use faer::Mat;
use ndarray::Array2;

/// How a column enters the imputation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AregVariable {
    /// Numeric, modelled with a restricted cubic spline.
    #[default]
    Numeric,
    /// Numeric, always linear.
    Linear,
    /// Categorical (the values are level codes).
    Categorical,
}

/// Donor selection (`match =`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AregMatch {
    /// Draw a donor with tricube weights on the distance between
    /// predictions, scaled by this fraction of the mean distance
    /// (`"weighted"`, `fweighted`; Hmisc's default 0.2).
    Weighted(f64),
    /// The donor with the closest prediction (`"closest"`).
    Closest,
}

/// Parameters of `aregImpute`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AregImputer {
    /// Number of completed data sets (`n.impute`).
    pub n_impute: usize,
    /// Spline knots (`nk`; 0 makes every variable linear).
    pub nk: usize,
    /// Model only the linear part of the variable being imputed (`tlinear`).
    pub tlinear: bool,
    /// Rounds before the first kept imputation; `None` is Hmisc's default
    /// (3, or 0 when only one column has missing values).
    pub burnin: Option<usize>,
    pub matching: AregMatch,
    /// Column kinds; `None` treats every column as numeric.
    pub kinds: Option<Vec<AregVariable>>,
    /// Seed of R's generator (`set.seed`).
    pub seed: i32,
}

impl Default for AregImputer {
    fn default() -> Self {
        Self {
            n_impute: 5,
            nk: 3,
            tlinear: true,
            burnin: None,
            matching: AregMatch::Weighted(0.2),
            kinds: None,
            seed: 1,
        }
    }
}

impl AregImputer {
    /// `n_impute` imputations with Hmisc's defaults otherwise.
    pub fn new(n_impute: usize) -> Self {
        Self {
            n_impute,
            ..Self::default()
        }
    }

    pub fn with_nk(mut self, nk: usize) -> Self {
        self.nk = nk;
        self
    }

    pub fn with_burnin(mut self, burnin: usize) -> Self {
        self.burnin = Some(burnin);
        self
    }

    pub fn with_matching(mut self, matching: AregMatch) -> Self {
        self.matching = matching;
        self
    }

    pub fn with_kinds(mut self, kinds: Vec<AregVariable>) -> Self {
        self.kinds = Some(kinds);
        self
    }

    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Impute `x` (`NaN` = missing). Needs at least two columns, each with at
    /// least two distinct observed values.
    pub fn impute(&self, x: &Array2<f64>) -> Result<MultipleImputation> {
        if x.is_empty() {
            return Err(RustMlError::EmptyInput("input array is empty".into()));
        }
        if self.n_impute == 0 {
            return Err(RustMlError::InvalidParameter(
                "n_impute must be at least 1".into(),
            ));
        }
        let p = x.ncols();
        let kinds: Vec<ImputeVariable> = match &self.kinds {
            Some(k) if k.len() != p => {
                return Err(RustMlError::ShapeMismatch(format!(
                    "{} column kinds for {p} columns",
                    k.len()
                )))
            }
            Some(k) => k
                .iter()
                .map(|v| match v {
                    AregVariable::Numeric => ImputeVariable::Numeric,
                    AregVariable::Linear => ImputeVariable::Linear,
                    AregVariable::Categorical => ImputeVariable::Categorical,
                })
                .collect(),
            None => vec![ImputeVariable::Numeric; p],
        };
        let options = AregImputeOptions {
            n_impute: self.n_impute,
            nk: self.nk,
            tlinear: self.tlinear,
            burnin: self.burnin,
            matching: match self.matching {
                AregMatch::Weighted(f) => DonorMatch::Weighted(f),
                AregMatch::Closest => DonorMatch::Closest,
            },
            seed: self.seed,
        };
        let m = Mat::<f64>::from_fn(x.nrows(), p, |i, j| x[[i, j]]);
        let res = areg_impute(&m, &kinds, &options)
            .map_err(|e| RustMlError::InvalidParameter(format!("aregImpute: {e}")))?;
        let datasets = (0..self.n_impute)
            .map(|k| {
                let c = res.complete(&m, k);
                Array2::from_shape_fn(x.dim(), |(i, j)| c[(i, j)])
            })
            .collect();
        Ok(MultipleImputation {
            datasets,
            missing: res.missing,
            r_squared: res.r_squared,
        })
    }
}

/// The completed data sets of [`AregImputer::impute`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultipleImputation {
    /// `n_impute` completed copies of the input.
    pub datasets: Vec<Array2<f64>>,
    /// Rows with missing values of each column.
    pub missing: Vec<Vec<usize>>,
    /// R² of the last imputation model of each column (NaN for complete
    /// columns).
    pub r_squared: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_regression::imputation::areg_impute;

    fn data() -> Array2<f64> {
        // 30 rows, x1 ≈ 2·x0 + noise, x2 categorical; a few holes.
        let mut x = Array2::from_shape_fn((30, 3), |(i, j)| {
            let t = i as f64;
            match j {
                0 => t * 0.5 + (t * 1.7).sin(),
                1 => t + (t * 0.9).cos() * 2.0,
                _ => (i % 3) as f64,
            }
        });
        for &(i, j) in &[(2, 0), (5, 1), (9, 0), (14, 1), (20, 2), (27, 1)] {
            x[[i, j]] = f64::NAN;
        }
        x
    }

    #[test]
    fn test_matches_areg_impute() {
        let x = data();
        let imp = AregImputer::new(3).with_kinds(vec![
            AregVariable::Numeric,
            AregVariable::Numeric,
            AregVariable::Categorical,
        ]);
        let r = imp.impute(&x).unwrap();
        assert_eq!(r.datasets.len(), 3);
        let m = Mat::<f64>::from_fn(30, 3, |i, j| x[[i, j]]);
        let opts = AregImputeOptions {
            n_impute: 3,
            ..AregImputeOptions::default()
        };
        let kinds = [
            ImputeVariable::Numeric,
            ImputeVariable::Numeric,
            ImputeVariable::Categorical,
        ];
        let direct = areg_impute(&m, &kinds, &opts).unwrap();
        for (k, ds) in r.datasets.iter().enumerate() {
            let c = direct.complete(&m, k);
            for i in 0..30 {
                for j in 0..3 {
                    assert_eq!(ds[[i, j]], c[(i, j)]);
                }
            }
            // Observed values kept, holes filled with observed donor values.
            for i in 0..30 {
                for j in 0..3 {
                    if x[[i, j]].is_nan() {
                        let v = ds[[i, j]];
                        assert!(x.column(j).iter().any(|&o| o == v));
                    } else {
                        assert_eq!(ds[[i, j]], x[[i, j]]);
                    }
                }
            }
        }
        assert_eq!(r.missing[0], vec![2, 9]);
    }

    #[test]
    fn test_errors() {
        let x = data();
        assert!(AregImputer::new(0).impute(&x).is_err());
        assert!(AregImputer::new(2)
            .with_kinds(vec![AregVariable::Numeric])
            .impute(&x)
            .is_err());
        // One column only.
        assert!(AregImputer::new(2)
            .impute(&x.column(0).to_owned().insert_axis(ndarray::Axis(1)))
            .is_err());
    }
}
