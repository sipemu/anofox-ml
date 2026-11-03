//! Pipeline / composite-transformer API (#37): refitting by reference,
//! supervised steps, `predict_proba`, use as a `Fit` estimator, and the
//! weighted CV helpers.

use anofox_ml_core::{
    cross_val_score_weighted_with_splits, k_fold, ColumnSelector, ColumnTransformer, FeatureUnion,
    Fit, FitUnsupervised, Pipeline, Predict, PredictProba, Result, Supervised, Transform,
};
use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Supervised transformer: keeps the single column most correlated with y.
#[derive(Clone)]
struct KeepBest;
struct FittedKeepBest(usize);

impl Fit<f64> for KeepBest {
    type Fitted = FittedKeepBest;
    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedKeepBest> {
        let yc = &y - y.mean().unwrap();
        let best = (0..x.ncols())
            .max_by(|&a, &b| {
                let s = |j: usize| {
                    let c = &x.column(j) - x.column(j).mean().unwrap();
                    c.dot(&yc).abs() / c.dot(&c).sqrt().max(1e-12)
                };
                s(a).total_cmp(&s(b))
            })
            .unwrap();
        Ok(FittedKeepBest(best))
    }
}

impl Transform<f64> for FittedKeepBest {
    fn transform_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        Ok(x.select(Axis(1), &[self.0]))
    }
}

/// Least-squares slope through the origin on the first column, with a
/// two-class "probability" for the predict_proba path.
struct Slope;
struct FittedSlope(f64);

impl Fit<f64> for Slope {
    type Fitted = FittedSlope;
    fn fit_view(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<FittedSlope> {
        let c = x.column(0);
        Ok(FittedSlope(c.dot(&y) / c.dot(&c)))
    }
}

impl Predict<f64> for FittedSlope {
    fn predict_view(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(x.column(0).mapv(|v| v * self.0))
    }
}

impl PredictProba<f64> for FittedSlope {
    fn predict_proba_view(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let p = x.column(0).mapv(|v| 1.0 / (1.0 + (-v * self.0).exp()));
        let mut out = Array2::zeros((x.nrows(), 2));
        out.column_mut(1).assign(&p);
        out.column_mut(0).assign(&p.mapv(|v| 1.0 - v));
        Ok(out)
    }
}

fn data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((12, 3), |(i, j)| match j {
        0 => (i % 5) as f64,
        1 => i as f64,
        _ => ((i * 7) % 4) as f64,
    });
    let y = x.column(1).mapv(|v| 3.0 * v);
    (x, y)
}

#[test]
fn pipeline_refits_by_reference_with_a_supervised_step() {
    let (x, y) = data();
    let pipe = Pipeline::new()
        .push_transformer("select", Supervised(KeepBest))
        .set_estimator("slope", Slope);
    // Refit twice from the same pipeline (it is borrowed, not consumed).
    let a = pipe.fit(&x, &y).unwrap();
    let b = pipe
        .fit(
            &x.slice(ndarray::s![..8, ..]).to_owned(),
            &y.slice(ndarray::s![..8]).to_owned(),
        )
        .unwrap();
    assert_eq!(a.get_transformer::<FittedKeepBest>("select").unwrap().0, 1);
    let pa = a.predict(&x).unwrap();
    assert!((pa - &y).iter().all(|d| d.abs() < 1e-12));
    assert_eq!(b.predict(&x).unwrap(), a.predict(&x).unwrap());
    // Without set_classifier there is no predict_proba.
    assert!(a.predict_proba(&x).is_err());
}

#[test]
fn set_classifier_forwards_predict_proba_and_pipeline_is_fit() {
    let (x, y) = data();
    let pipe = Pipeline::new()
        .push_transformer("select", Supervised(KeepBest))
        .set_classifier("slope", Slope);
    // Pipeline is usable through the generic Fit / Predict traits.
    fn fit_generic<E: Fit<f64>>(e: &E, x: &Array2<f64>, y: &Array1<f64>) -> E::Fitted {
        e.fit(x, y).unwrap()
    }
    let fitted = fit_generic(&pipe, &x, &y);
    let proba = PredictProba::predict_proba(&fitted, &x).unwrap();
    assert_eq!(proba.dim(), (12, 2));
    assert!(proba
        .rows()
        .into_iter()
        .all(|r| (r.sum() - 1.0).abs() < 1e-12));
    assert_eq!(Predict::predict(&fitted, &x).unwrap().len(), 12);
}

#[test]
fn supervised_step_needs_a_target() {
    let (x, _) = data();
    assert!(FitUnsupervised::fit(&Supervised(KeepBest), &x).is_err());
}

#[test]
fn composite_transformers_forward_y_to_supervised_branches() {
    let (x, y) = data();
    let ct = ColumnTransformer::new().push(
        "best_of_1_2",
        ColumnSelector::Indices(vec![1, 2]),
        Supervised(KeepBest),
    );
    let fu = FeatureUnion::new().push("best", Supervised(KeepBest));
    // Unsupervised fit fails for the supervised branch ...
    assert!(FitUnsupervised::fit(&ct, &x).is_err());
    // ... while a pipeline passes y through.
    let pipe = Pipeline::new()
        .push_transformer("ct", ct)
        .push_transformer("fu", fu)
        .set_estimator("slope", Slope);
    let f = pipe.fit(&x, &y).unwrap();
    assert_eq!(f.transform(&x).unwrap(), x.select(Axis(1), &[1]));
}

#[test]
fn weighted_cv_routes_fold_weights() {
    let (x, y) = data();
    let w = Array1::from_shape_fn(12, |i| 1.0 + (i * i) as f64);
    let splits = k_fold(12, 3).unwrap();
    // Weighted mean predictor; weighted MAE scorer.
    let scores = cross_val_score_weighted_with_splits(
        &x,
        &y,
        Some(&w),
        &splits,
        |_, yt, wt, xv| {
            let wt = wt.expect("train weights");
            assert_eq!(wt.len(), yt.len());
            Ok(Array1::from_elem(xv.nrows(), wt.dot(yt) / wt.sum()))
        },
        |yt, yp, wt| {
            let wt = wt.expect("test weights");
            Ok((yt - yp).mapv(f64::abs).dot(wt) / wt.sum())
        },
    )
    .unwrap();
    assert_eq!(scores.len(), 3);
    let none = cross_val_score_weighted_with_splits(
        &x,
        &y,
        None,
        &splits,
        |_, yt, wt, xv| {
            assert!(wt.is_none());
            Ok(Array1::from_elem(xv.nrows(), yt.mean().unwrap()))
        },
        |yt, yp, _| Ok((yt - yp).mapv(f64::abs).mean().unwrap()),
    )
    .unwrap();
    assert_ne!(scores, none);
    assert!(cross_val_score_weighted_with_splits(
        &x,
        &y,
        Some(&array![1.0]),
        &splits,
        |_, _, _, xv| Ok(Array1::zeros(xv.nrows())),
        |_, _, _| Ok(0.0),
    )
    .is_err());
}
