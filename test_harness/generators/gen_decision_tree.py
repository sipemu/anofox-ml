"""Generate golden data for decision tree tests."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def generate():
    cases = []

    # Decision Tree Classifier - simple separable data
    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [7.0, 8.0],
            [8.0, 7.0],
            [8.0, 9.0],
            [9.0, 8.0],
        ]
    )
    y_train = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    X_test = np.array([[1.5, 1.5], [8.5, 8.5], [5.0, 5.0]])

    for max_depth in [None, 2, 1]:
        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        depth_str = "none" if max_depth is None else str(max_depth)
        cases.append(
            {
                "name": f"dt_classifier_gini_depth_{depth_str}",
                "algorithm": "DecisionTreeClassifier",
                "X_train": X_train.tolist(),
                "y_train": y_train.tolist(),
                "X_test": X_test.tolist(),
                "y_pred": y_pred.tolist(),
                "criterion": "gini",
                "max_depth": max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "feature_importances": clf.feature_importances_.tolist(),
            }
        )

    # Decision Tree Classifier with entropy
    clf_ent = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    clf_ent.fit(X_train, y_train)
    y_pred_ent = clf_ent.predict(X_test)

    cases.append(
        {
            "name": "dt_classifier_entropy",
            "algorithm": "DecisionTreeClassifier",
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_pred": y_pred_ent.tolist(),
            "criterion": "entropy",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "feature_importances": clf_ent.feature_importances_.tolist(),
        }
    )

    # Decision Tree Regressor
    X_train_reg = np.array(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    )
    y_train_reg = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1])
    X_test_reg = np.array([[1.5], [4.5], [7.5]])

    for max_depth in [None, 2, 1]:
        reg = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = reg.predict(X_test_reg)

        depth_str = "none" if max_depth is None else str(max_depth)
        cases.append(
            {
                "name": f"dt_regressor_depth_{depth_str}",
                "algorithm": "DecisionTreeRegressor",
                "X_train": X_train_reg.tolist(),
                "y_train": y_train_reg.tolist(),
                "X_test": X_test_reg.tolist(),
                "y_pred": y_pred_reg.tolist(),
                "max_depth": max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "feature_importances": reg.feature_importances_.tolist(),
            }
        )

    return cases
