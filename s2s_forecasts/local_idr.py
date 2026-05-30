"""local_idr.py

A lightweight, non-pickle persistence wrapper for fitted IDR-like models.

This file is designed for the common workflow:

1. Train or obtain a fitted `idrobject`-style model.
2. Save only the model state to JSON (no joblib / pickle).
3. Load the state locally and call `.predict(...)`.

It expects the following fitted attributes on the source model:
- ecdf
- thresholds
- indices
- X
- y
- groups
- orders
- constraints

For the multivariate branch, it uses `neighbor_points` from a vendored
`isodisreg.partialorders` module if available.

If you have copied the upstream package locally, this file can live beside
it and import the helper functions directly.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    # If you vendored the package locally
    from isodisreg.partialorders import neighbor_points  # type: ignore
except Exception:  # pragma: no cover
    try:
        # If local_idr.py sits inside the vendored package directory
        from .partialorders import neighbor_points  # type: ignore
    except Exception:  # pragma: no cover
        neighbor_points = None


@dataclass
class predictions_idr:
    ecdf: np.ndarray
    points: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class idrpredict:
    predictions: list[predictions_idr]
    incomparables: Any


# ---------------------------------------------------------------------------
# JSON-safe save/load helpers
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Recursively convert common scientific Python objects to JSON-safe types."""
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        return {"__pd_dataframe__": True, "value": obj.to_dict(orient="split")}
    if isinstance(obj, pd.Series):
        return {"__pd_series__": True, "value": obj.tolist()}
    if isinstance(obj, pd.Index):
        return {"__pd_index__": True, "value": obj.tolist()}
    if isinstance(obj, pd.Timestamp):
        return {"__pd_timestamp__": True, "value": obj.isoformat()}
    if isinstance(obj, pd.Timedelta):
        return {"__pd_timedelta__": True, "value": obj.isoformat()}

    if isinstance(obj, np.ndarray):
        return {"__np_array__": True, "value": obj.tolist()}
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback for unusual objects like scipy sparse matrix, custom classes, etc.
    return {"__repr__": True, "value": repr(obj)}


def from_jsonable(obj: Any) -> Any:
    """Inverse of `to_jsonable` for the types we care about."""
    if isinstance(obj, dict):
        if obj.get("__pd_dataframe__"):
            return pd.DataFrame(**obj["value"])
        if obj.get("__pd_series__"):
            return pd.Series(obj["value"])
        if obj.get("__pd_index__"):
            return pd.Index(obj["value"])
        if obj.get("__pd_timestamp__"):
            return pd.Timestamp(obj["value"])
        if obj.get("__pd_timedelta__"):
            return pd.Timedelta(obj["value"])
        if obj.get("__np_array__"):
            return np.asarray(obj["value"], dtype=object)
        if obj.get("__repr__"):
            return obj["value"]
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj


def save_idr_model(model: Any, folder: str | Path) -> None:
    """Save a fitted model without pickle/joblib."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    state = {
        "ecdf": to_jsonable(model.ecdf),
        "thresholds": to_jsonable(model.thresholds),
        "indices": to_jsonable(model.indices),
        "X": to_jsonable(model.X),
        "y": to_jsonable(model.y),
        "groups": to_jsonable(model.groups),
        "orders": to_jsonable(model.orders),
        "constraints": to_jsonable(model.constraints),
        "meta": {
            "class_name": type(model).__name__,
            "format_version": 1,
        },
    }

    (folder / "model_state.json").write_text(json.dumps(state, indent=2))


def load_idr_state(folder: str | Path) -> dict[str, Any]:
    """Load a saved model state back into plain Python objects."""
    folder = Path(folder)
    raw = json.loads((folder / "model_state.json").read_text())

    state = {
        "ecdf": from_jsonable(raw["ecdf"]),
        "thresholds": from_jsonable(raw["thresholds"]),
        "indices": from_jsonable(raw["indices"]),
        "X": from_jsonable(raw["X"]),
        "y": from_jsonable(raw["y"]),
        "groups": from_jsonable(raw["groups"]),
        "orders": from_jsonable(raw["orders"]),
        "constraints": from_jsonable(raw["constraints"]),
        "meta": raw.get("meta", {}),
    }

    # Normalise X back into a DataFrame if it was stored as a split dict.
    if isinstance(state["X"], dict) and {"index", "columns", "data"}.issubset(state["X"].keys()):
        state["X"] = pd.DataFrame(**state["X"])
    elif not isinstance(state["X"], pd.DataFrame):
        raise TypeError("Loaded X is not a DataFrame; save_idr_model stored an unexpected structure")

    return state


# ---------------------------------------------------------------------------
# Small helpers used by predict
# ---------------------------------------------------------------------------

def ecdf_formal(thresholds: Iterable[float], y: Iterable[float]) -> np.ndarray:
    """Compute P(Y <= t) for each threshold t."""
    thresholds = np.asarray(list(thresholds), dtype=float)
    y = np.asarray(list(y), dtype=float)
    return np.array([(y <= t).mean() for t in thresholds], dtype=float)


def prepare_data(X: pd.DataFrame, groups: dict, orders: dict) -> pd.DataFrame:
    """Local copy of the upstream preprocessing step used before prediction."""
    X = X.copy()
    res = defaultdict(list)

    for key, val in sorted(groups.items()):
        res[val].append(key)

    for key, val in res.items():
        if len(val) > 1:
            if orders[str(int(key))] == "comp":
                continue
            tmp = -np.sort(-X[val], axis=1)
            if orders[str(int(key))] == "sd":
                X[val] = tmp
            else:
                X[val] = np.cumsum(tmp, axis=1)

    return X


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _wrap_prediction(ecdf: np.ndarray, points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> predictions_idr:
    return predictions_idr(
        ecdf=np.asarray(ecdf, dtype=float),
        points=np.asarray(points, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )


def _componentwise_neighbors(x: pd.DataFrame, X: pd.DataFrame):
    """Pure-Python fallback for neighbor_points.

    For each query row, returns:
    - smaller: rows in X that are <= x in every column and are maximal among them
    - greater: rows in X that are >= x in every column and are minimal among them
    """
    x_vals = x.to_numpy(dtype=float)
    X_vals = X.to_numpy(dtype=float)
    smaller_all: list[np.ndarray] = []
    greater_all: list[np.ndarray] = []

    for row in x_vals:
        le_mask = np.all(X_vals <= row, axis=1)
        ge_mask = np.all(X_vals >= row, axis=1)

        le_idx = np.where(le_mask)[0]
        ge_idx = np.where(ge_mask)[0]

        # keep only maximal smaller points
        smaller = []
        for i in le_idx:
            xi = X_vals[i]
            dominated_by_other = False
            for j in le_idx:
                if i == j:
                    continue
                xj = X_vals[j]
                if np.all(xi <= xj) and np.any(xi < xj):
                    dominated_by_other = True
                    break
            if not dominated_by_other:
                smaller.append(i)

        # keep only minimal greater points
        greater = []
        for i in ge_idx:
            xi = X_vals[i]
            dominates_other = False
            for j in ge_idx:
                if i == j:
                    continue
                xj = X_vals[j]
                if np.all(xj <= xi) and np.any(xj < xi):
                    dominates_other = True
                    break
            if not dominates_other:
                greater.append(i)

        smaller_all.append(np.asarray(smaller, dtype=int))
        greater_all.append(np.asarray(greater, dtype=int))

    return smaller_all, greater_all


# ---------------------------------------------------------------------------
# Local wrapper object
# ---------------------------------------------------------------------------

class LocalIDRModel:
    """A minimal local replacement for a fitted IDR model.

    The wrapper stores the fitted state and exposes `.predict(...)`.
    """

    def __init__(self, state: dict[str, Any]):
        self.ecdf = state["ecdf"]
        self.thresholds = state["thresholds"]
        self.indices = state["indices"]
        self.X = state["X"]
        self.y = state["y"]
        self.groups = state["groups"]
        self.orders = state["orders"]
        self.constraints = state["constraints"]
        self.meta = state.get("meta", {})

    @classmethod
    def load(cls, folder: str | Path) -> "LocalIDRModel":
        return cls(load_idr_state(folder))

    def save(self, folder: str | Path) -> None:
        save_idr_model(self, folder)

    def predict(self, data: pd.DataFrame | None = None, digits: int = 3) -> idrpredict:
        """Predict distributions for new data.

        This mirrors the upstream control flow closely enough for the
        `predict_idr_xarray(...)` and histogram workflow.
        """
        cdf = np.asarray(self.ecdf, dtype=float).copy()
        thresholds = np.asarray(self.thresholds, dtype=float).copy()
        order_indices: list[int] = []
        preds: list[predictions_idr] = []

        # Case 1: no new data
        if data is None:
            indices = np.asarray(self.indices)
            for i in range(indices.shape[0]):
                edf = np.round(cdf[i, :], digits)
                sel = np.hstack([edf[0] > 0, np.diff(edf) > 0])
                tmp = _wrap_prediction(
                    ecdf=edf[sel],
                    points=thresholds[sel],
                    lower=np.array([]),
                    upper=np.array([]),
                )
                for j in np.atleast_1d(indices[i]):
                    order_indices.append(int(j))
                    preds.append(tmp)

            preds_rearranged = [preds[k] for k in np.argsort(order_indices)]
            return idrpredict(predictions=preds_rearranged, incomparables=None)

        # Must be a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        X = self.X.copy()
        missing = [col for col in X.columns if col not in data.columns]
        if missing:
            raise ValueError(f"some variables of idr fit are missing in data: {missing}")

        data = prepare_data(data[X.columns], groups=self.groups, orders=self.orders)
        nVar = data.shape[1]

        # Case 2: one covariate
        if nVar == 1:
            xname = data.columns[0]
            X1 = np.asarray(X[X.columns[0]], dtype=float)
            x = np.asarray(data[xname], dtype=float)

            smaller = np.array([bisect.bisect_left(X1, a) for a in x])
            smaller = np.where(smaller == 0, 1, smaller) - 1

            wg = (
                np.interp(
                    x,
                    X1,
                    np.arange(1, X1.shape[0] + 1),
                    left=1,
                    right=X1.shape[0],
                )
                - np.arange(1, X1.shape[0] + 1)[smaller.astype(int)]
            )
            greater = smaller + (wg > 0).astype(int)
            ws = 1 - wg

            l = np.round(cdf[greater.astype(int), :], digits)
            u = np.round(cdf[smaller.astype(int), :], digits)

            def fun_preds(lrow, urow, ws_val, wg_val):
                ls = np.insert(lrow[:-1], 0, 0)
                us = np.insert(urow[:-1], 0, 0)
                ind = (ls < lrow) + (us < urow)

                l2 = lrow[ind]
                u2 = urow[ind]
                cdf2 = np.round(np.multiply(l2, wg_val) + np.multiply(u2, ws_val), digits)

                return _wrap_prediction(
                    ecdf=cdf2,
                    points=thresholds[ind],
                    lower=l2,
                    upper=u2,
                )

            preds = list(map(fun_preds, l, u, list(ws), list(wg)))
            return idrpredict(predictions=preds, incomparables=None)

        # Case 3: multivariate
        # Prefer the upstream helper when it works, but fall back to a
        # self-contained pure-Python implementation if the saved order structure
        # is not compatible with the vendored function.
        if neighbor_points is not None:
            try:
                nPoints = neighbor_points(data, X, order_X=self.constraints)
                smaller = nPoints[0]
                greater = nPoints[1]
            except Exception:
                smaller, greater = _componentwise_neighbors(data, X)
        else:
            smaller, greater = _componentwise_neighbors(data, X)
        incomparables = np.array([len(s) + len(g) for s, g in zip(smaller, greater)]) == 0

        # Fallback for incomparable points: use climatology / fitted ECDF
        if any(incomparables):
            y = self.y
            if isinstance(y, pd.DataFrame):
                y_flat = np.asarray(y.to_numpy()).ravel()
            elif isinstance(y, pd.Series):
                y_flat = np.asarray(y).ravel()
            else:
                y_flat = np.asarray(y, dtype=object).ravel()
                # flatten one more level for ragged nested lists
                if y_flat.dtype == object:
                    y_flat = np.concatenate([np.asarray(v).reshape(-1) for v in y_flat])

            edf = np.round(ecdf_formal(thresholds, y_flat), digits)
            sel = edf > 0
            edf = edf[sel]
            points = thresholds[sel]

            upr = np.where(edf == 1)[0]
            if upr.size and upr[0] < len(edf) - 1:
                points = np.delete(points, np.arange(upr[0], len(edf)))
                edf = np.delete(edf, np.arange(upr[0], len(edf)))

            tmp = _wrap_prediction(ecdf=edf, points=points, lower=edf, upper=edf)

            for i in np.where(incomparables)[0]:
                preds.append(tmp)
                order_indices.append(int(i))

        for i in np.where(~incomparables)[0]:
            if smaller[i].size > 0 and greater[i].size == 0:
                upper = np.round(np.amin(cdf[smaller[i].astype(int), :], axis=0), digits)
                sel = np.hstack([upper[0] != 0, np.diff(upper) != 0])
                upper = upper[sel]
                lower = np.zeros(len(upper))
                estimCDF = upper

            elif smaller[i].size == 0 and greater[i].size > 0:
                lower = np.round(np.amax(cdf[greater[i].astype(int), :], axis=0), digits)
                sel = np.hstack([lower[0] != 0, np.diff(lower) != 0])
                lower = lower[sel]
                upper = np.ones(len(lower))
                estimCDF = lower

            else:
                lower = np.round(np.amax(cdf[greater[i].astype(int), :], axis=0), digits)
                upper = np.round(np.amin(cdf[smaller[i].astype(int), :], axis=0), digits)
                sel = np.hstack([lower[0] != 0, np.diff(lower) != 0]) + np.hstack([
                    upper[0] != 0, np.diff(upper) != 0
                ])
                lower = lower[sel]
                upper = upper[sel]
                estimCDF = np.round(0.5 * (lower + upper), digits)

            tmp = _wrap_prediction(
                ecdf=estimCDF,
                points=thresholds[sel],
                lower=lower,
                upper=upper,
            )
            order_indices.append(int(i))
            preds.append(tmp)

        preds_rearranged = [preds[k] for k in np.argsort(order_indices)]
        return idrpredict(predictions=preds_rearranged, incomparables=np.where(incomparables))


# ---------------------------------------------------------------------------
# Convenience helper matching your earlier xarray flow
# ---------------------------------------------------------------------------

def predict_idr_xarray(models: dict[str, LocalIDRModel], fcst_new: Any):
    """Generate predictions for a forecast dataset by region.

    Expects fcst_new to behave like an xarray.Dataset with:
    - a region coordinate/dimension
    - variables tp_mean and tp_std
    """
    predictions = {}

    for region in fcst_new.region.values:
        if region not in models:
            raise KeyError(f"No saved IDR model found for region {region}")

        fcst_r = fcst_new.sel(region=region)
        X_new = pd.DataFrame(
            {
                "ens_mean": [fcst_r["tp_mean"].item()],
                "ens_std": [fcst_r["tp_std"].item()],
            }
        )

        preds = models[region].predict(X_new)
        predictions[region] = preds

    return predictions


__all__ = [
    "LocalIDRModel",
    "idrpredict",
    "predictions_idr",
    "save_idr_model",
    "load_idr_state",
    "predict_idr_xarray",
]
