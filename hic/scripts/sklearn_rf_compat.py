"""Load Peakachu pretrained RandomForests on sklearn >= 1.3.

Published high-confidence pickles were dumped with sklearn < 1.3. Tree nodes
gained `missing_go_to_left` in 1.3, so a direct joblib.load raises ValueError.
Route sklearn.tree._tree.Tree through a Python proxy, add the missing field,
then rebuild real Trees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_NEW_DTYPE = np.dtype(
    {
        "names": [
            "left_child",
            "right_child",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "missing_go_to_left",
        ],
        "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8", "u1"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56],
        "itemsize": 64,
    }
)


def _convert_nodes(state: dict) -> dict:
    nodes = state.get("nodes")
    names = getattr(getattr(nodes, "dtype", None), "names", None)
    if not names or "missing_go_to_left" in names:
        return state
    new_nodes = np.empty(nodes.shape, dtype=_NEW_DTYPE)
    for name in names:
        if name in _NEW_DTYPE.names:
            new_nodes[name] = nodes[name]
    new_nodes["missing_go_to_left"] = 0
    out = dict(state)
    out["nodes"] = new_nodes
    return out


class _CompatTree:
    def __new__(cls, n_features, n_classes, n_outputs):
        obj = object.__new__(cls)
        obj._args = (n_features, n_classes, n_outputs)
        obj._state = None
        return obj

    def __setstate__(self, state):
        self._state = state

    def to_real(self):
        from sklearn.tree._tree import Tree

        tree = Tree(*self._args)
        tree.__setstate__(_convert_nodes(self._state))
        return tree


def _fill_missing_attrs(obj, proto):
    for k, v in proto.__dict__.items():
        if not hasattr(obj, k):
            setattr(obj, k, v)


def _normalize_tree_values(est) -> None:
    """Old sklearn stored class counts in tree_.value; 1.3+ treats them as proba."""
    tree = getattr(est, "tree_", None)
    if tree is None:
        return
    values = tree.value
    if values.size == 0 or float(np.nanmax(values)) <= 1.0 + 1e-6:
        return
    denom = values.sum(axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    try:
        values /= denom
    except ValueError:
        est.tree_.value = values / denom


def _modernize_forest(model):
    """Fill attributes sklearn>=1.3 expects on pickles from older RF APIs."""
    import types

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if getattr(model, "estimator", None) is None:
        base = getattr(model, "base_estimator", None)
        if base is not None:
            model.estimator = base
    if getattr(model, "estimator_", None) is None and getattr(model, "estimator", None) is not None:
        model.estimator_ = model.estimator
    if getattr(model, "n_features_in_", None) is None:
        n = getattr(model, "n_features_", None)
        if n is not None:
            model.n_features_in_ = n
    _fill_missing_attrs(model, RandomForestClassifier())
    tree_proto = DecisionTreeClassifier()
    for est in getattr(model, "estimators_", []) or []:
        _fill_missing_attrs(est, tree_proto)
        if getattr(est, "n_features_in_", None) is None and getattr(est, "n_features_", None) is not None:
            est.n_features_in_ = est.n_features_
        _normalize_tree_values(est)

    _orig_proba = model.predict_proba

    def _proba(self, X, orig=_orig_proba):
        p = np.asarray(orig(X), dtype=float)
        s = p.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return p / s

    model.predict_proba = types.MethodType(_proba, model)
    return model


def _replace_trees(obj, seen: set[int]):
    oid = id(obj)
    if oid in seen:
        return obj
    seen.add(oid)
    if isinstance(obj, _CompatTree):
        return obj.to_real()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_replace_trees(x, seen) for x in obj)
    if isinstance(obj, dict):
        return {k: _replace_trees(v, seen) for k, v in obj.items()}
    tree = getattr(obj, "tree_", None)
    if isinstance(tree, _CompatTree):
        obj.tree_ = tree.to_real()
    estimators = getattr(obj, "estimators_", None)
    if estimators is not None:
        obj.estimators_ = _replace_trees(estimators, seen)
    return obj


def _install_find_class_hook():
    from joblib.numpy_pickle import NumpyUnpickler

    if getattr(NumpyUnpickler.find_class, "_peakachu_compat", False):
        return
    orig = NumpyUnpickler.find_class

    def find_class(self, module, name):
        if name == "Tree" and "sklearn.tree" in module:
            return _CompatTree
        return orig(self, module, name)

    find_class._peakachu_compat = True  # type: ignore[attr-defined]
    NumpyUnpickler.find_class = find_class


_ORIG_JOBLIB_LOAD = None


def load_model(path: str | Path):
    import joblib

    global _ORIG_JOBLIB_LOAD
    _install_find_class_hook()
    loader = _ORIG_JOBLIB_LOAD or joblib.load
    model = loader(path)
    return _modernize_forest(_replace_trees(model, set()))


def patch() -> None:
    """Make joblib.load (used by peakachu score_*) accept old RF pickles."""
    import joblib

    global _ORIG_JOBLIB_LOAD
    if getattr(joblib.load, "_peakachu_compat", False):
        return
    _ORIG_JOBLIB_LOAD = joblib.load

    def _load(filename, *args, **kwargs):
        return load_model(filename)

    _load._peakachu_compat = True  # type: ignore[attr-defined]
    joblib.load = _load
