import numpy as np
import pytest
from sklearn.feature_extraction.image import PatchExtractor as SkPatchExtractor

import anofox_ml_python as am

rng = np.random.RandomState(0)
grey = rng.rand(3, 12, 10)
rgb = rng.rand(2, 12, 11, 3)  # the default patch is a tenth: (1, 1)


@pytest.mark.parametrize("X", [grey, rgb], ids=["grey", "rgb"])
@pytest.mark.parametrize("kw", [{"patch_size": (3, 4)}, {},
                                {"patch_size": (2, 2), "max_patches": 7, "random_state": 0},
                                {"patch_size": (4, 3), "max_patches": 0.3, "random_state": 5},
                                {"patch_size": (2, 3), "max_patches": 10_000, "random_state": 1}])
def test_patch_extractor_matches_sklearn(X, kw):
    ours = am.PatchExtractor(**kw).transform(X)
    ref = SkPatchExtractor(**kw).fit(X).transform(X)
    assert ours.shape == ref.shape
    np.testing.assert_array_equal(ours, ref)


def test_errors():
    with pytest.raises(ValueError):
        am.PatchExtractor(patch_size=(20, 2)).transform(grey)
    with pytest.raises(ValueError):
        am.PatchExtractor().transform(grey[0])
