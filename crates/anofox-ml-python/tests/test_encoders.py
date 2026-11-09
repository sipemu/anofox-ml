"""LabelBinarizer, MultiLabelBinarizer and DictVectorizer against
scikit-learn 1.8."""

import pickle

import numpy as np
import pytest

import anofox_ml_python as am

sk = pytest.importorskip("sklearn")
from sklearn import feature_extraction, preprocessing  # noqa: E402


@pytest.mark.parametrize("y", [["b", "a", "c", "a"], [3, 1, 3], ["yes", "no", "no"], ["x", "x"]])
@pytest.mark.parametrize("labels", [{}, {"neg_label": -1, "pos_label": 2}])
def test_label_binarizer(y, labels):
    ours = am.LabelBinarizer(**labels).fit(y)
    ref = preprocessing.LabelBinarizer(**labels).fit(y)
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    new = list(y) + [y[0]]
    np.testing.assert_array_equal(ours.transform(new), ref.transform(new))
    Y = ref.transform(new)
    np.testing.assert_array_equal(ours.inverse_transform(Y), ref.inverse_transform(Y))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(ours)).transform(new), Y)


def test_multilabel_binarizer():
    y = [("sci-fi", "thriller"), ("comedy",), (), ("thriller", "comedy")]
    ours, ref = am.MultiLabelBinarizer(), preprocessing.MultiLabelBinarizer()
    np.testing.assert_array_equal(ours.fit_transform(y), ref.fit_transform(y))
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    Y = ref.transform(y)
    assert ours.inverse_transform(Y) == ref.inverse_transform(Y)
    ours = am.MultiLabelBinarizer(classes=["thriller", "comedy", "sci-fi", "drama"])
    ref = preprocessing.MultiLabelBinarizer(classes=["thriller", "comedy", "sci-fi", "drama"])
    np.testing.assert_array_equal(ours.fit_transform(y), ref.fit_transform(y))
    with pytest.raises(ValueError, match="duplicate"):
        am.MultiLabelBinarizer(classes=["a", "a"]).fit(y)


@pytest.mark.parametrize("sort", [True, False])
def test_dict_vectorizer(sort):
    D = [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1, "city": "Paris"},
         {"city": "London", "tags": ["a", "b"], "zero": 0}]
    ours = am.DictVectorizer(sort=sort)
    ref = feature_extraction.DictVectorizer(sort=sort, sparse=False)
    np.testing.assert_array_equal(ours.fit_transform(D), ref.fit_transform(D))
    assert list(ours.get_feature_names_out()) == list(ref.get_feature_names_out())
    assert ours.vocabulary_ == ref.vocabulary_
    new = [{"foo": 4, "unseen": 1, "city": "Rome"}]
    np.testing.assert_array_equal(ours.transform(new), ref.transform(new))
    X = ref.transform(D)
    assert ours.inverse_transform(X) == ref.inverse_transform(X)
