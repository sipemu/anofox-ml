//! Any registered estimator as a typed `Fit` / `Predict` model, so generic
//! wrappers (the conformal predictors) can use estimators chosen by name.

use anofox_ml_core::{Fit, Predict, PredictProba, Result, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::model::{DynModel, Out};

/// An estimator given by its registry `kind` and JSON `params`; `fit`
/// builds it with [`make`](crate::make) and fits it.
///
/// ```
/// use anofox_ml_core::{Fit, Predict};
/// use anofox_ml_registry::RegistryEstimator;
/// use ndarray::array;
///
/// let est = RegistryEstimator::new("DecisionTreeRegressor", serde_json::json!({ "max_depth": 2 }));
/// let x = array![[0.0], [1.0], [2.0], [3.0]];
/// let y = array![0.0, 0.0, 1.0, 1.0];
/// let fitted = est.fit(&x, &y).unwrap();
/// assert_eq!(fitted.predict(&x).unwrap(), y);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryEstimator {
    pub kind: String,
    pub params: serde_json::Value,
}

impl RegistryEstimator {
    pub fn new(kind: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            kind: kind.into(),
            params,
        }
    }
}

impl Fit<f64> for RegistryEstimator {
    type Fitted = RegistryFitted;

    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<RegistryFitted> {
        let mut model = crate::make(&self.kind, &self.params)?;
        model.fit(x, Some(y), None)?;
        Ok(RegistryFitted(model))
    }
}

/// A fitted registry model. Serialised as its kind plus its
/// [`DynModel::to_bytes`] output.
pub struct RegistryFitted(pub Box<dyn DynModel>);

impl RegistryFitted {
    pub fn model(&self) -> &dyn DynModel {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for RegistryFitted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RegistryFitted")
            .field(&self.0.kind())
            .finish()
    }
}

impl Predict<f64> for RegistryFitted {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        match self.0.predict(x)? {
            Out::A1(a) => Ok(a),
            Out::A2(_) => Err(RustMlError::InvalidParameter(format!(
                "{} predicts several outputs; a single one is needed here",
                self.0.kind()
            ))),
        }
    }
}

impl PredictProba<f64> for RegistryFitted {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.0.predict_proba(x)? {
            Out::A2(a) => Ok(a),
            Out::A1(a) => Ok(a.insert_axis(ndarray::Axis(1))),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Stored<K, B> {
    kind: K,
    bytes: B,
}

impl Serialize for RegistryFitted {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let bytes = self.0.to_bytes().map_err(serde::ser::Error::custom)?;
        Stored {
            kind: self.0.kind(),
            bytes,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for RegistryFitted {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = Stored::<String, Vec<u8>>::deserialize(d)?;
        crate::load(&s.kind, &s.bytes)
            .map(RegistryFitted)
            .map_err(serde::de::Error::custom)
    }
}
