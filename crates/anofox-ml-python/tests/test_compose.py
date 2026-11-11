import pickle

import numpy as np
import pytest
import sklearn.compose as skc
import sklearn.decomposition as skd
import sklearn.linear_model as skl
import sklearn.pipeline as skp
import sklearn.preprocessing as skpp

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = rng.randn(120, 5) * [1, 10, 0.1, 3, 1] + [0, 5, 1, -2, 0]
y = X @ [1.0, 0.2, 5.0, -0.5, 0.0] + rng.randn(120) * 0.1
labels = np.where(y > np.median(y), "hi", "lo")


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def test_pipeline_regressor_and_classifier_match_sklearn():
    ours = am.Pipeline([("s", am.StandardScaler()), ("pca", am.PCA(n_components=3)),
                        ("r", am.Ridge(alpha=0.5))]).fit(X, y)
    ref = skp.Pipeline([("s", skpp.StandardScaler()), ("pca", skd.PCA(n_components=3)),
                        ("r", skl.Ridge(alpha=0.5))]).fit(X, y)
    close(ours.predict(X), ref.predict(X))
    assert ours._estimator_type == "regressor"
    close(ours.score(X, y), ref.score(X, y))
    clf = am.make_pipeline(am.StandardScaler(), am.LogisticRegression()).fit(X, labels)
    ref = skp.make_pipeline(skpp.StandardScaler(), skl.LogisticRegression()).fit(X, labels)
    assert list(clf.named_steps) == ["standardscaler", "logisticregression"]
    np.testing.assert_array_equal(clf.classes_, ref.classes_)
    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
    # Exactly the steps by hand; sklearn's lbfgs only to its tolerance.
    Z = am.StandardScaler().fit(X).transform(X)
    np.testing.assert_array_equal(clf.predict_proba(X),
                                  am.LogisticRegression().fit(Z, labels).predict_proba(Z))
    np.testing.assert_allclose(clf.predict_proba(X), ref.predict_proba(X), atol=1e-3)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict_proba(X), clf.predict_proba(X))


def test_transformer_pipeline_and_feature_union():
    ours = am.Pipeline([("s", am.StandardScaler()), ("p", "passthrough"),
                        ("pca", am.PCA(n_components=2))]).fit(X)
    ref = skp.Pipeline([("s", skpp.StandardScaler()), ("p", "passthrough"),
                        ("pca", skd.PCA(n_components=2))]).fit(X)
    close(np.abs(ours.transform(X)), np.abs(ref.transform(X)))
    union = am.FeatureUnion([("s", am.StandardScaler()), ("m", am.MinMaxScaler())],
                            transformer_weights={"m": 2.0}).fit(X)
    ref = skp.FeatureUnion([("s", skpp.StandardScaler()), ("m", skpp.MinMaxScaler())],
                           transformer_weights={"m": 2.0}).fit(X)
    close(union.transform(X), ref.transform(X))


def test_column_transformer_matches_sklearn():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X, columns=list("abcde"))
    for remainder in ("drop", "passthrough"):
        ours = am.ColumnTransformer([("s", am.StandardScaler(), ["a", "c"]),
                                     ("m", am.MinMaxScaler(), [1]),
                                     ("keep", "passthrough", [4])],
                                    remainder=remainder).fit(frame)
        ref = skc.ColumnTransformer([("s", skpp.StandardScaler(), ["a", "c"]),
                                     ("m", skpp.MinMaxScaler(), [1]),
                                     ("keep", "passthrough", [4])],
                                    remainder=remainder).fit(frame)
        close(ours.transform(frame), ref.transform(frame))
    # Inside a pipeline, with a boolean mask.
    mask = [True, False, True, False, False]
    pipe = am.Pipeline([("ct", am.ColumnTransformer([("s", am.StandardScaler(), mask)],
                                                     remainder="passthrough")),
                        ("r", am.Ridge())]).fit(X, y)
    ref = skp.Pipeline([("ct", skc.ColumnTransformer([("s", skpp.StandardScaler(), mask)],
                                                      remainder="passthrough")),
                        ("r", skl.Ridge())]).fit(X, y)
    close(pipe.predict(X), ref.predict(X))


def test_composition_errors():
    with pytest.raises(ValueError):
        am.Pipeline([("a", am.StandardScaler()), ("a", am.Ridge())]).fit(X, y)
    with pytest.raises(TypeError):
        am.Pipeline([("s", skpp.StandardScaler()), ("r", am.Ridge())]).fit(X, y)
    with pytest.raises(ValueError):
        am.ColumnTransformer([("s", am.StandardScaler(), [0])], remainder="x").fit(X)
    with pytest.raises(ValueError):
        am.FeatureUnion([("s", am.StandardScaler())]).fit(X, sample_weight=np.ones(120))
