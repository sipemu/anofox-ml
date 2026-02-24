"""Generate golden data for Gaussian Naive Bayes tests."""

import numpy as np
from sklearn.naive_bayes import GaussianNB


def generate():
    cases = []

    # Simple 2-class problem
    np.random.seed(42)
    X_train = np.array([
        [1.0, 2.0], [1.5, 1.8], [1.2, 2.1], [0.8, 1.9],
        [5.0, 6.0], [5.5, 5.8], [5.2, 6.1], [4.8, 5.9],
    ])
    y_train = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    X_test = np.array([[1.0, 1.5], [5.0, 5.5], [3.0, 3.5]])

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    cases.append({
        "name": "gaussian_nb_two_class",
        "algorithm": "GaussianNB",
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "y_pred": y_pred.tolist(),
        "class_prior": gnb.class_prior_.tolist(),
        "theta": gnb.theta_.tolist(),
        "var": gnb.var_.tolist(),
    })

    # 3-class problem
    X_train_3 = np.array([
        [0.0, 0.0], [0.5, 0.5], [0.2, 0.1],
        [5.0, 0.0], [5.5, 0.5], [5.2, 0.1],
        [2.5, 5.0], [2.0, 5.5], [3.0, 4.8],
    ])
    y_train_3 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    X_test_3 = np.array([[0.3, 0.2], [5.1, 0.3], [2.5, 5.2], [2.5, 2.5]])

    gnb3 = GaussianNB()
    gnb3.fit(X_train_3, y_train_3)
    y_pred_3 = gnb3.predict(X_test_3)

    cases.append({
        "name": "gaussian_nb_three_class",
        "algorithm": "GaussianNB",
        "X_train": X_train_3.tolist(),
        "y_train": y_train_3.tolist(),
        "X_test": X_test_3.tolist(),
        "y_pred": y_pred_3.tolist(),
        "class_prior": gnb3.class_prior_.tolist(),
        "theta": gnb3.theta_.tolist(),
        "var": gnb3.var_.tolist(),
    })

    return cases
