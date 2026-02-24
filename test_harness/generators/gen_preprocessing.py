"""Generate golden data for preprocessing tests."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def generate():
    cases = []

    # StandardScaler
    np.random.seed(42)
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)
    X_inverse = scaler.inverse_transform(X_transformed)

    cases.append(
        {
            "name": "standard_scaler_basic",
            "algorithm": "StandardScaler",
            "X": X.tolist(),
            "mean": scaler.mean_.tolist(),
            "std": np.sqrt(scaler.var_).tolist(),
            "X_transformed": X_transformed.tolist(),
            "X_inverse": X_inverse.tolist(),
        }
    )

    # StandardScaler with larger dataset
    np.random.seed(123)
    X2 = np.random.randn(20, 4) * np.array([1, 10, 100, 0.1]) + np.array(
        [5, 50, 500, 0.5]
    )
    scaler2 = StandardScaler()
    X2_transformed = scaler2.fit_transform(X2)
    X2_inverse = scaler2.inverse_transform(X2_transformed)

    cases.append(
        {
            "name": "standard_scaler_random",
            "algorithm": "StandardScaler",
            "X": X2.tolist(),
            "mean": scaler2.mean_.tolist(),
            "std": np.sqrt(scaler2.var_).tolist(),
            "X_transformed": X2_transformed.tolist(),
            "X_inverse": X2_inverse.tolist(),
        }
    )

    # MinMaxScaler default [0, 1]
    scaler3 = MinMaxScaler()
    X3_transformed = scaler3.fit_transform(X)
    X3_inverse = scaler3.inverse_transform(X3_transformed)

    cases.append(
        {
            "name": "minmax_scaler_default",
            "algorithm": "MinMaxScaler",
            "X": X.tolist(),
            "feature_min": 0.0,
            "feature_max": 1.0,
            "data_min": scaler3.data_min_.tolist(),
            "data_max": scaler3.data_max_.tolist(),
            "X_transformed": X3_transformed.tolist(),
            "X_inverse": X3_inverse.tolist(),
        }
    )

    # MinMaxScaler custom range [-1, 1]
    scaler4 = MinMaxScaler(feature_range=(-1, 1))
    X4_transformed = scaler4.fit_transform(X)
    X4_inverse = scaler4.inverse_transform(X4_transformed)

    cases.append(
        {
            "name": "minmax_scaler_custom_range",
            "algorithm": "MinMaxScaler",
            "X": X.tolist(),
            "feature_min": -1.0,
            "feature_max": 1.0,
            "data_min": scaler4.data_min_.tolist(),
            "data_max": scaler4.data_max_.tolist(),
            "X_transformed": X4_transformed.tolist(),
            "X_inverse": X4_inverse.tolist(),
        }
    )

    return cases
