"""Generate golden data for Random Forest tests."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def generate():
    cases = []

    # Classifier - well-separated clusters
    X_train = np.array([
        [1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 2.0],
        [7.0, 8.0], [8.0, 7.0], [8.0, 9.0], [9.0, 8.0],
    ])
    y_train = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    X_test = np.array([[1.5, 1.5], [8.5, 8.5], [5.0, 5.0]])

    # Note: we can't exactly match sklearn's RF because of different bootstrap
    # sampling and random feature selection. We just verify predictions are reasonable.
    clf = RandomForestClassifier(
        n_estimators=10, max_depth=3, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cases.append({
        "name": "rf_classifier_basic",
        "algorithm": "RandomForestClassifier",
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "y_pred": y_pred.tolist(),
        "n_estimators": 10,
        "max_depth": 3,
    })

    # Regressor
    X_train_reg = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y_train_reg = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1])
    X_test_reg = np.array([[1.5], [4.5], [7.5]])

    reg = RandomForestRegressor(
        n_estimators=10, max_depth=3, random_state=42
    )
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)

    cases.append({
        "name": "rf_regressor_basic",
        "algorithm": "RandomForestRegressor",
        "X_train": X_train_reg.tolist(),
        "y_train": y_train_reg.tolist(),
        "X_test": X_test_reg.tolist(),
        "y_pred": y_pred_reg.tolist(),
        "n_estimators": 10,
        "max_depth": 3,
    })

    return cases
