"""Generate golden data for KNN tests."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def generate():
    cases = []

    # KNN Classifier - simple 2-cluster problem
    np.random.seed(42)
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 5.0],
            [5.0, 6.0],
        ]
    )
    y_train = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    X_test = np.array([[0.2, 0.3], [5.2, 5.3], [2.5, 2.5]])

    for k in [3, 5]:
        for weights in ["uniform", "distance"]:
            clf = KNeighborsClassifier(
                n_neighbors=k, weights=weights, metric="euclidean"
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            cases.append(
                {
                    "name": f"knn_classifier_k{k}_{weights}_euclidean",
                    "algorithm": "KnnClassifier",
                    "X_train": X_train.tolist(),
                    "y_train": y_train.tolist(),
                    "X_test": X_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "n_neighbors": k,
                    "weights": weights,
                    "metric": "euclidean",
                }
            )

    # KNN Classifier with manhattan distance
    clf_man = KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="manhattan")
    clf_man.fit(X_train, y_train)
    y_pred_man = clf_man.predict(X_test)

    cases.append(
        {
            "name": "knn_classifier_k3_uniform_manhattan",
            "algorithm": "KnnClassifier",
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_pred": y_pred_man.tolist(),
            "n_neighbors": 3,
            "weights": "uniform",
            "metric": "manhattan",
        }
    )

    # KNN Regressor
    X_train_reg = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y_train_reg = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    X_test_reg = np.array([[2.5], [5.5], [1.0]])

    for k in [3, 5]:
        for weights in ["uniform", "distance"]:
            reg = KNeighborsRegressor(
                n_neighbors=k, weights=weights, metric="euclidean"
            )
            reg.fit(X_train_reg, y_train_reg)
            y_pred_reg = reg.predict(X_test_reg)

            cases.append(
                {
                    "name": f"knn_regressor_k{k}_{weights}_euclidean",
                    "algorithm": "KnnRegressor",
                    "X_train": X_train_reg.tolist(),
                    "y_train": y_train_reg.tolist(),
                    "X_test": X_test_reg.tolist(),
                    "y_pred": y_pred_reg.tolist(),
                    "n_neighbors": k,
                    "weights": weights,
                    "metric": "euclidean",
                }
            )

    return cases
