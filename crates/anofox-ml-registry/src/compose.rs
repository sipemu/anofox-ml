//! sklearn's composition tools over registry estimators: `Pipeline`,
//! `FeatureUnion` and `ColumnTransformer`. Each step is an estimator spec
//! (kind and parameters), so a fitted composite is one registry model.

use anofox_ml_core::{Result, RustMlError};
use ndarray::{concatenate, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::estimator::RegistryFitted;
use crate::meta::EstimatorSpec;
use crate::model::Out;

fn err(msg: impl Into<String>) -> RustMlError {
    RustMlError::InvalidParameter(msg.into())
}

/// A transformer's output as a matrix (one column for 1-D outputs).
fn matrix(out: Out) -> Array2<f64> {
    match out {
        Out::A1(a) => a.insert_axis(Axis(1)),
        Out::A2(a) => a,
    }
}

fn fit_step(
    spec: &EstimatorSpec,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
) -> Result<RegistryFitted> {
    let mut model = crate::make(&spec.estimator, &spec.estimator_params)?;
    model.fit(x, y, w)?;
    Ok(RegistryFitted(model))
}

fn hstack(blocks: &[Array2<f64>], n_rows: usize) -> Result<Array2<f64>> {
    if blocks.is_empty() {
        return Ok(Array2::zeros((n_rows, 0)));
    }
    let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
    concatenate(Axis(1), &views).map_err(|e| err(e.to_string()))
}

/// sklearn's `Pipeline`: the steps but the last transform in order, the
/// last one is fitted on their output (and gets the sample weights).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct PipelineConfig {
    pub steps: Vec<EstimatorSpec>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct PipelineFit {
    steps: Vec<RegistryFitted>,
}

impl PipelineConfig {
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        y: Option<ArrayView1<f64>>,
        w: Option<ArrayView1<f64>>,
    ) -> Result<PipelineFit> {
        let Some((last, transformers)) = self.steps.split_last() else {
            return Err(err("Pipeline needs at least one step"));
        };
        let mut steps = Vec::with_capacity(self.steps.len());
        let mut current = x.to_owned();
        for spec in transformers {
            let model = fit_step(spec, current.view(), y, None)?;
            current = matrix(model.model().transform(current.view())?);
            steps.push(model);
        }
        steps.push(fit_step(last, current.view(), y, w)?);
        Ok(PipelineFit { steps })
    }
}

impl PipelineFit {
    fn last(&self) -> &RegistryFitted {
        self.steps.last().expect("a fitted pipeline has steps")
    }

    /// `x` through every step but the last.
    fn through(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut current = x.to_owned();
        for step in &self.steps[..self.steps.len() - 1] {
            current = matrix(step.model().transform(current.view())?);
        }
        Ok(current)
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Out> {
        self.last().model().predict(self.through(x)?.view())
    }

    pub fn predict_proba(&self, x: ArrayView2<f64>) -> Result<Out> {
        self.last().model().predict_proba(self.through(x)?.view())
    }

    pub fn decision_function(&self, x: ArrayView2<f64>) -> Result<Out> {
        self.last()
            .model()
            .decision_function(self.through(x)?.view())
    }

    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Out> {
        self.last().model().transform(self.through(x)?.view())
    }
}

/// sklearn's `FeatureUnion`: every transformer on all columns, their
/// outputs side by side (each times its weight).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct FeatureUnionConfig {
    pub transformer_list: Vec<EstimatorSpec>,
    pub transformer_weights: Option<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct FeatureUnionFit {
    transformers: Vec<RegistryFitted>,
    weights: Option<Vec<f64>>,
}

impl FeatureUnionConfig {
    pub fn fit(&self, x: ArrayView2<f64>, y: Option<ArrayView1<f64>>) -> Result<FeatureUnionFit> {
        if let Some(w) = &self.transformer_weights {
            if w.len() != self.transformer_list.len() {
                return Err(err("transformer_weights needs one weight per transformer"));
            }
        }
        let transformers = self
            .transformer_list
            .iter()
            .map(|s| fit_step(s, x, y, None))
            .collect::<Result<_>>()?;
        Ok(FeatureUnionFit {
            transformers,
            weights: self.transformer_weights.clone(),
        })
    }
}

impl FeatureUnionFit {
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let blocks = self
            .transformers
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut b = matrix(t.model().transform(x)?);
                if let Some(w) = &self.weights {
                    b.mapv_inplace(|v| v * w[i]);
                }
                Ok(b)
            })
            .collect::<Result<Vec<_>>>()?;
        hstack(&blocks, x.nrows())
    }
}

/// One `ColumnTransformer` step: a transformer (or `None` for
/// `"passthrough"`) and its columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ColumnStep {
    #[serde(default)]
    pub transformer: Option<EstimatorSpec>,
    pub columns: Vec<usize>,
}

/// sklearn's `ColumnTransformer`: every step on its columns, the outputs
/// side by side, then the columns no step used (`remainder =
/// "passthrough"`) or none of them (`"drop"`, the default).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ColumnTransformerConfig {
    pub transformers: Vec<ColumnStep>,
    pub remainder: String,
}

impl Default for ColumnTransformerConfig {
    fn default() -> Self {
        Self {
            transformers: Vec::new(),
            remainder: "drop".into(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ColumnTransformerFit {
    steps: Vec<(Option<RegistryFitted>, Vec<usize>)>,
    remainder: Vec<usize>,
    n_features: usize,
}

fn columns(x: ArrayView2<f64>, cols: &[usize]) -> Array2<f64> {
    x.select(Axis(1), cols)
}

impl ColumnTransformerConfig {
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        y: Option<ArrayView1<f64>>,
    ) -> Result<ColumnTransformerFit> {
        let p = x.ncols();
        if let Some(&c) = self
            .transformers
            .iter()
            .flat_map(|s| &s.columns)
            .find(|&&c| c >= p)
        {
            return Err(err(format!("column {c} is out of range for {p} features")));
        }
        let passthrough = match self.remainder.as_str() {
            "drop" => false,
            "passthrough" => true,
            other => {
                return Err(err(format!(
                    "remainder must be drop or passthrough, got {other}"
                )))
            }
        };
        let steps = self
            .transformers
            .iter()
            .map(|s| {
                let model = match &s.transformer {
                    Some(spec) => Some(fit_step(spec, columns(x, &s.columns).view(), y, None)?),
                    None => None,
                };
                Ok((model, s.columns.clone()))
            })
            .collect::<Result<_>>()?;
        let used: std::collections::HashSet<usize> = self
            .transformers
            .iter()
            .flat_map(|s| s.columns.iter().copied())
            .collect();
        let remainder = if passthrough {
            (0..p).filter(|c| !used.contains(c)).collect()
        } else {
            Vec::new()
        };
        Ok(ColumnTransformerFit {
            steps,
            remainder,
            n_features: p,
        })
    }
}

impl ColumnTransformerFit {
    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(RustMlError::ShapeMismatch(format!(
                "X has {} features, the ColumnTransformer was fitted on {}",
                x.ncols(),
                self.n_features
            )));
        }
        let mut blocks = Vec::with_capacity(self.steps.len() + 1);
        for (model, cols) in &self.steps {
            let part = columns(x, cols);
            blocks.push(match model {
                Some(m) => matrix(m.model().transform(part.view())?),
                None => part,
            });
        }
        if !self.remainder.is_empty() {
            blocks.push(columns(x, &self.remainder));
        }
        hstack(&blocks, x.nrows())
    }
}
