//! Estimators accept `ndarray` views: fitting and predicting on a strided,
//! non-contiguous view gives exactly the results of the owned copy.

use anofox_ml::prelude::*;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

/// Every other column of a wider matrix (non-contiguous rows) and every
/// other element of a longer target.
struct Data {
    x_wide: Array2<f64>,
    y_wide: Array1<f64>,
}

impl Data {
    fn new(classification: bool) -> Self {
        let n = 40;
        let x_wide = Array2::from_shape_fn((n, 8), |(i, j)| {
            ((i * 7 + j * 13) % 17) as f64 / 4.0 + (i as f64) * 0.05 * (j % 3) as f64
        });
        let y_wide = Array1::from_shape_fn(2 * n, |k| {
            let i = k / 2;
            let v = x_wide[[i, 0]] - 0.5 * x_wide[[i, 2]] + 0.25 * x_wide[[i, 4]];
            if classification {
                if v > 1.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                v
            }
        });
        Data { x_wide, y_wide }
    }
    fn x(&self) -> ArrayView2<'_, f64> {
        self.x_wide.slice(s![.., ..;2])
    }
    fn y(&self) -> ArrayView1<'_, f64> {
        self.y_wide.slice(s![..;2])
    }
}

macro_rules! check_supervised {
    ($name:ident, $est:expr, $classification:expr) => {
        #[test]
        fn $name() {
            let d = Data::new($classification);
            let (xv, yv) = (d.x(), d.y());
            assert!(!xv.is_standard_layout());
            let (xo, yo) = (xv.to_owned(), yv.to_owned());
            let est = $est;
            let from_view = est.fit(xv, yv).unwrap().predict(xv).unwrap();
            let from_owned = est.fit(&xo, &yo).unwrap().predict(&xo).unwrap();
            assert_eq!(from_view, from_owned);
        }
    };
}

macro_rules! check_transformer {
    ($name:ident, $est:expr) => {
        #[test]
        fn $name() {
            let d = Data::new(false);
            let xv = d.x();
            let xo = xv.to_owned();
            let est = $est;
            let from_view = FitUnsupervised::fit(&est, xv)
                .unwrap()
                .transform(xv)
                .unwrap();
            let from_owned = FitUnsupervised::fit(&est, &xo)
                .unwrap()
                .transform(&xo)
                .unwrap();
            assert_eq!(from_view, from_owned);
        }
    };
}

check_supervised!(decision_tree, DecisionTreeRegressor::new(), false);
check_supervised!(random_forest, RandomForestClassifier::new(10), true);
check_supervised!(gradient_boosting, GradientBoostingRegressor::new(), false);
check_supervised!(hist_gradient_boosting, HistGradientBoostingRegressor::new(), false);
check_supervised!(knn, KnnClassifier::new(3), true);
check_supervised!(gaussian_nb, GaussianNB::new(), true);
check_supervised!(lda, LinearDiscriminantAnalysis::new(), true);
check_supervised!(svc, Svc::new(), true);
check_supervised!(ridge, RidgeRegressor::new(), false);
check_supervised!(logistic, LogisticRegressor::new(), true);
check_supervised!(sgd, SgdRegressor::new(), false);
check_supervised!(mlp, MlpRegressor::default(), false);

check_transformer!(standard_scaler, StandardScaler::new());
check_transformer!(pca, Pca::new(2));

#[test]
fn kmeans() {
    let d = Data::new(false);
    let xv = d.x();
    let xo = xv.to_owned();
    let est = KMeans::new(3);
    let from_view = FitUnsupervised::fit(&est, xv).unwrap().predict(xv).unwrap();
    let from_owned = FitUnsupervised::fit(&est, &xo)
        .unwrap()
        .predict(&xo)
        .unwrap();
    assert_eq!(from_view, from_owned);
}

#[test]
fn pipeline() {
    let d = Data::new(true);
    let (xv, yv) = (d.x(), d.y());
    let (xo, yo) = (xv.to_owned(), yv.to_owned());
    let pipe = Pipeline::new()
        .push_transformer("scale", StandardScaler::new())
        .set_classifier("knn", KnnClassifier::new(3));
    let from_view = pipe.fit(xv, yv).unwrap().predict_proba(xv).unwrap();
    let from_owned = pipe.fit(&xo, &yo).unwrap().predict_proba(&xo).unwrap();
    assert_eq!(from_view, from_owned);
}

#[test]
fn weighted_fit() {
    let d = Data::new(false);
    let (xv, yv) = (d.x(), d.y());
    let w = Array1::from_shape_fn(xv.nrows(), |i| 1.0 + (i % 3) as f64);
    let (xo, yo) = (xv.to_owned(), yv.to_owned());
    let est = DecisionTreeRegressor::new();
    let from_view = est
        .fit_weighted_view(xv, yv, Some(w.view()))
        .unwrap()
        .predict(xv)
        .unwrap();
    let from_owned = est
        .fit_weighted(&xo, &yo, Some(&w))
        .unwrap()
        .predict(&xo)
        .unwrap();
    assert_eq!(from_view, from_owned);
}
