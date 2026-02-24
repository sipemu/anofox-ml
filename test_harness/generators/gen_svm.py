"""Generate golden data for SVM tests (LinearSVC and SVC with linear/rbf kernels)."""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification


def generate():
    cases = []

    # Well-separated binary classification data
    X_train, y_train = make_classification(
        n_samples=40, n_features=4, n_informative=3, n_redundant=0,
        n_classes=2, class_sep=3.0, random_state=42,
    )
    X_test = X_train[:10]

    # LinearSVC binary
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cases.append({
        "name": "linear_svc_binary",
        "algorithm": "LinearSvc",
        "X_train": X_train.tolist(),
        "y_train": y_train.astype(float).tolist(),
        "X_test": X_test.tolist(),
        "y_pred": y_pred.astype(float).tolist(),
        "C": 1.0,
        "max_iter": 5000,
    })

    # SVC with linear kernel (binary)
    svc_lin = SVC(kernel="linear", C=1.0, max_iter=5000, random_state=42)
    svc_lin.fit(X_train, y_train)
    y_pred_lin = svc_lin.predict(X_test)

    cases.append({
        "name": "svc_linear_binary",
        "algorithm": "Svc",
        "kernel": "linear",
        "X_train": X_train.tolist(),
        "y_train": y_train.astype(float).tolist(),
        "X_test": X_test.tolist(),
        "y_pred": y_pred_lin.astype(float).tolist(),
        "C": 1.0,
        "max_iter": 5000,
    })

    # SVC with RBF kernel (binary)
    svc_rbf = SVC(kernel="rbf", C=1.0, gamma=0.5, max_iter=5000, random_state=42)
    svc_rbf.fit(X_train, y_train)
    y_pred_rbf = svc_rbf.predict(X_test)

    cases.append({
        "name": "svc_rbf_binary",
        "algorithm": "Svc",
        "kernel": "rbf",
        "gamma": 0.5,
        "X_train": X_train.tolist(),
        "y_train": y_train.astype(float).tolist(),
        "X_test": X_test.tolist(),
        "y_pred": y_pred_rbf.astype(float).tolist(),
        "C": 1.0,
        "max_iter": 5000,
    })

    # Multiclass (3 classes)
    X_multi, y_multi = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, class_sep=3.0, random_state=7,
    )
    X_test_multi = X_multi[:15]

    # LinearSVC multiclass
    clf_multi = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    clf_multi.fit(X_multi, y_multi)
    y_pred_multi = clf_multi.predict(X_test_multi)

    cases.append({
        "name": "linear_svc_multiclass",
        "algorithm": "LinearSvc",
        "X_train": X_multi.tolist(),
        "y_train": y_multi.astype(float).tolist(),
        "X_test": X_test_multi.tolist(),
        "y_pred": y_pred_multi.astype(float).tolist(),
        "C": 1.0,
        "max_iter": 5000,
    })

    return cases
