"""Generate golden data for PCA tests."""

import numpy as np
from sklearn.decomposition import PCA


def generate():
    cases = []

    # 2D data with strong principal axis
    np.random.seed(42)
    X = np.random.randn(20, 2) @ np.array([[3.0, 1.0], [1.0, 0.5]]) + np.array([5.0, 10.0])

    for n_comp in [1, 2]:
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        X_inverse = pca.inverse_transform(X_transformed)

        cases.append({
            "name": f"pca_{n_comp}_components",
            "algorithm": "PCA",
            "X": X.tolist(),
            "n_components": n_comp,
            "components": pca.components_.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "mean": pca.mean_.tolist(),
            "X_transformed": X_transformed.tolist(),
            "X_inverse": X_inverse.tolist(),
        })

    # Higher-dimensional data
    np.random.seed(123)
    X_hd = np.random.randn(30, 5) @ np.diag([10, 5, 2, 0.5, 0.1]) + np.array([1, 2, 3, 4, 5])

    pca_hd = PCA(n_components=3)
    X_hd_transformed = pca_hd.fit_transform(X_hd)
    X_hd_inverse = pca_hd.inverse_transform(X_hd_transformed)

    cases.append({
        "name": "pca_5d_to_3d",
        "algorithm": "PCA",
        "X": X_hd.tolist(),
        "n_components": 3,
        "components": pca_hd.components_.tolist(),
        "explained_variance": pca_hd.explained_variance_.tolist(),
        "mean": pca_hd.mean_.tolist(),
        "X_transformed": X_hd_transformed.tolist(),
        "X_inverse": X_hd_inverse.tolist(),
    })

    return cases
