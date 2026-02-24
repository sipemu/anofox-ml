"""Generate golden data for Gradient Boosting tests."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression


def generate():
    cases = []

    # Binary classification
    X_train, y_train = make_classification(
        n_samples=50, n_features=4, n_informative=3, n_redundant=0,
        n_classes=2, class_sep=2.5, random_state=42,
    )
    X_test = X_train[:10]

    clf = GradientBoostingClassifier(
        n_estimators=20, learning_rate=0.1, max_depth=3,
        min_samples_split=2, min_samples_leaf=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cases.append({
        "name": "gbt_classifier_binary",
        "algorithm": "GradientBoostingClassifier",
        "X_train": X_train.tolist(),
        "y_train": y_train.astype(float).tolist(),
        "X_test": X_test.tolist(),
        "y_pred": y_pred.astype(float).tolist(),
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    })

    # Multiclass classification
    X_multi, y_multi = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, class_sep=2.5, random_state=7,
    )
    X_test_multi = X_multi[:15]

    clf_multi = GradientBoostingClassifier(
        n_estimators=30, learning_rate=0.1, max_depth=3,
        random_state=42,
    )
    clf_multi.fit(X_multi, y_multi)
    y_pred_multi = clf_multi.predict(X_test_multi)

    cases.append({
        "name": "gbt_classifier_multiclass",
        "algorithm": "GradientBoostingClassifier",
        "X_train": X_multi.tolist(),
        "y_train": y_multi.astype(float).tolist(),
        "X_test": X_test_multi.tolist(),
        "y_pred": y_pred_multi.astype(float).tolist(),
        "n_estimators": 30,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    })

    # Regression
    X_reg, y_reg = make_regression(
        n_samples=50, n_features=4, n_informative=3,
        noise=5.0, random_state=42,
    )
    X_test_reg = X_reg[:10]

    reg = GradientBoostingRegressor(
        n_estimators=20, learning_rate=0.1, max_depth=3,
        min_samples_split=2, min_samples_leaf=1,
        random_state=42,
    )
    reg.fit(X_reg, y_reg)
    y_pred_reg = reg.predict(X_test_reg)

    cases.append({
        "name": "gbt_regressor",
        "algorithm": "GradientBoostingRegressor",
        "X_train": X_reg.tolist(),
        "y_train": y_reg.tolist(),
        "X_test": X_test_reg.tolist(),
        "y_pred": y_pred_reg.tolist(),
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    })

    return cases
