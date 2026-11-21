import pickle

import numpy as np
import pytest
import sklearn.neighbors as skn

import anofox_ml_python as am

rng = np.random.RandomState(0)
X = rng.randn(40, 3)
Xq = rng.randn(7, 3)


def close(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("kw", [{}, {"metric": "manhattan"}, {"p": 3},
                                {"algorithm": "brute", "n_neighbors": 3}])
def test_kneighbors_matches_sklearn(kw):
    ours, ref = am.NearestNeighbors(**kw).fit(X), skn.NearestNeighbors(**kw).fit(X)
    for q in (Xq, None):
        do, io = ours.kneighbors(q)
        dr, ir = ref.kneighbors(q)
        close(do, dr)
        np.testing.assert_array_equal(io, ir)
        for mode in ("connectivity", "distance"):
            close(ours.kneighbors_graph(q, mode=mode).toarray(),
                  ref.kneighbors_graph(q, mode=mode).toarray())
    np.testing.assert_array_equal(ours.kneighbors(Xq, 2, return_distance=False),
                                  ref.kneighbors(Xq, 2, return_distance=False))
    close(pickle.loads(pickle.dumps(ours)).kneighbors(Xq)[0], ref.kneighbors(Xq)[0])


@pytest.mark.parametrize("q", [Xq, None])
@pytest.mark.parametrize("radius", [0.8, 1.5])
def test_radius_neighbors_matches_sklearn(q, radius):
    ours, ref = am.NearestNeighbors(radius=radius).fit(X), skn.NearestNeighbors(radius=radius).fit(X)
    do, io = ours.radius_neighbors(q, sort_results=True)
    dr, ir = ref.radius_neighbors(q, sort_results=True)
    assert len(io) == len(ir)
    for a, b, c, d in zip(do, dr, io, ir):
        close(a, b)
        np.testing.assert_array_equal(c, d)
    # Unsorted: the same sets (sklearn's order depends on its tree).
    for a, b in zip(ours.radius_neighbors(q, return_distance=False),
                    ref.radius_neighbors(q, return_distance=False)):
        assert sorted(a) == sorted(b)
    for mode in ("connectivity", "distance"):
        close(ours.radius_neighbors_graph(q, mode=mode).toarray(),
              ref.radius_neighbors_graph(q, mode=mode).toarray())
    close(ours.radius_neighbors_graph(q, radius=0.5).toarray(),
          ref.radius_neighbors_graph(q, radius=0.5).toarray())


@pytest.mark.parametrize("kw", [{}, {"mode": "connectivity", "radius": 1.2},
                                {"metric": "manhattan", "radius": 2.0}])
def test_radius_neighbors_transformer_matches_sklearn(kw):
    ours = am.RadiusNeighborsTransformer(**kw).fit(X)
    ref = skn.RadiusNeighborsTransformer(**kw).fit(X)
    close(ours.transform(Xq), ref.transform(Xq).toarray())
    close(ours.fit_transform(X), ref.fit_transform(X).toarray())


def test_errors():
    nn = am.NearestNeighbors(n_neighbors=3).fit(X[:3])
    with pytest.raises(ValueError):
        nn.kneighbors()  # needs n_neighbors < n_samples_fit
    with pytest.raises(ValueError, match="mode"):
        am.NearestNeighbors().fit(X).kneighbors_graph(mode="nope")
