"""A dependency-light (numpy-only) multi-head image+tabular fusion model.

This is a faithful, runnable implementation of the *training scheme* described
in ``docs/multimodal_fusion_guide.md`` — the part that makes fusion work under
modality asymmetry — not a production network:

- three heads: image-only, tabular-only, fusion;
- **loss masking** — synthetic (image-only) samples train ONLY the image head;
  real samples train all three (matches plan.md: "synthetic image는 image
  branch와 image auxiliary head 학습에만 사용");
- **modality dropout** — during training a real sample's tabular embedding is
  replaced by a learned ``null`` vector with probability ``dropout_p`` so the
  fusion head cannot collapse onto the (synthetic-rich) image branch;
- a learned null-tabular embedding so the fusion head has a defined behaviour
  when tabular is absent.

The encoders are single-hidden-layer MLPs so the whole thing trains on CPU with
numpy. Teammates can swap a torch CNN/transformer behind the same three-head /
loss-masking interface later; the evaluation (``fusion_eval``) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _bce_with_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    # numerically stable element-wise binary cross entropy
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


def _he(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0]
    return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)


@dataclass
class ClasswiseGatedResidualFusion:
    """Class-wise gated residual fusion for optional WL/CatBoost branches.

    ``combine_logits`` implements:

    fbm + has_wl_map * gate_wl * wl_logits
        + has_catboost_logits * gate_catboost * catboost_logits

    The gates are per class, so one modality can help specific defect classes
    without globally dominating the fusion output.
    """

    num_classes: int
    wl_gates: np.ndarray | None = None
    catboost_gates: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.wl_gates is None:
            self.wl_gates = np.ones(self.num_classes, dtype=float)
        if self.catboost_gates is None:
            self.catboost_gates = np.ones(self.num_classes, dtype=float)
        self.wl_gates = self._validate_gate(self.wl_gates, "wl_gates")
        self.catboost_gates = self._validate_gate(self.catboost_gates, "catboost_gates")

    def combine_logits(
        self,
        fbm_logits: np.ndarray,
        *,
        wl_logits: np.ndarray | None = None,
        has_wl_map: np.ndarray | None = None,
        catboost_logits: np.ndarray | None = None,
        has_catboost_logits: np.ndarray | None = None,
    ) -> np.ndarray:
        fbm = self._validate_logits(fbm_logits, "fbm_logits")
        combined = fbm.copy()
        if wl_logits is not None:
            wl = self._validate_logits(wl_logits, "wl_logits")
            wl_mask = self._row_mask(has_wl_map, fbm.shape[0])
            combined += wl_mask * self.wl_gates.reshape(1, -1) * wl
        if catboost_logits is not None:
            cat = self._validate_logits(catboost_logits, "catboost_logits")
            cat_mask = self._row_mask(has_catboost_logits, fbm.shape[0])
            combined += cat_mask * self.catboost_gates.reshape(1, -1) * cat
        return combined

    def combine_probabilities(self, fbm_logits: np.ndarray, **kwargs: object) -> np.ndarray:
        return _sigmoid(self.combine_logits(fbm_logits, **kwargs))

    def _validate_gate(self, gate: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(gate, dtype=float)
        if arr.shape != (self.num_classes,):
            raise ValueError(f"{name} must have shape ({self.num_classes},), got {arr.shape}")
        return arr

    def _validate_logits(self, logits: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(logits, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.num_classes:
            raise ValueError(f"{name} must have shape [N, {self.num_classes}], got {arr.shape}")
        return arr

    @staticmethod
    def _row_mask(mask: np.ndarray | None, n_rows: int) -> np.ndarray:
        if mask is None:
            return np.ones((n_rows, 1), dtype=float)
        arr = np.asarray(mask, dtype=float).reshape(-1)
        if arr.shape[0] != n_rows:
            raise ValueError(f"mask length must be {n_rows}, got {arr.shape[0]}")
        return arr.reshape(-1, 1)


@dataclass
class _Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m: dict = field(default_factory=dict)
    _v: dict = field(default_factory=dict)
    _t: int = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        for key, grad in grads.items():
            if key not in self._m:
                self._m[key] = np.zeros_like(grad)
                self._v[key] = np.zeros_like(grad)
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class WLResidualCatBoostFusionMLP:
    """FBM + WL residual map + direct CatBoost-logit fusion classifier.

    This class is intentionally numpy-only to match the rest of this example
    repo. It implements the requested residual fusion shape:

    ``fusion = fbm_logits + has_wl * gate_wl * wl_logits
              + has_cat * gate_cat * catboost_logits``

    CatBoost logits are treated as offline model outputs and receive no neural
    loss. The FBM and WL heads are trained with BCE, and the fusion loss is
    masked/weighted by row so synthetic WL maps can contribute at low weight.
    """

    hidden: int = 32
    lr: float = 3e-3
    epochs: int = 150
    l2: float = 1e-4
    seed: int = 0

    params: dict[str, np.ndarray] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    _img_mean: np.ndarray | None = None
    _img_std: np.ndarray | None = None
    _wl_mean: np.ndarray | None = None
    _wl_std: np.ndarray | None = None
    n_labels: int = 0

    def fit(
        self,
        images_flat: np.ndarray,
        wl_maps: np.ndarray | None,
        labels: np.ndarray,
        *,
        has_wl_map: np.ndarray,
        wl_loss_weight: np.ndarray | None = None,
        catboost_logits: np.ndarray | None = None,
        has_catboost_logits: np.ndarray | None = None,
        fusion_loss_weight: np.ndarray | None = None,
        fbm_loss_weight: np.ndarray | None = None,
    ) -> "WLResidualCatBoostFusionMLP":
        rng = np.random.default_rng(self.seed)
        y = self._as_2d(labels, "labels").astype(float)
        xi_raw = self._as_2d(images_flat, "images_flat")
        if xi_raw.shape[0] != y.shape[0]:
            raise ValueError("images_flat and labels must have the same row count")
        xw_raw = self._flatten_wl_maps(wl_maps, y.shape[0])
        cat = self._catboost_array(catboost_logits, y.shape)
        has_wl = self._row_mask(has_wl_map, y.shape[0], "has_wl_map")
        has_cat = self._row_mask(has_catboost_logits, y.shape[0], "has_catboost_logits")
        wl_weight = self._row_weights(wl_loss_weight, y.shape[0], default=has_wl) * has_wl
        fusion_weight = self._fusion_weights(fusion_loss_weight, has_wl, has_cat, wl_weight)
        fbm_weight = self._row_weights(fbm_loss_weight, y.shape[0], default=np.ones(y.shape[0]))

        xi = self._fit_standardize(xi_raw, rows=np.ones(y.shape[0], dtype=bool), kind="img")
        xw = self._fit_standardize(xw_raw, rows=has_wl.astype(bool), kind="wl")
        self.n_labels = y.shape[1]

        di, dw, hdim, lab = xi.shape[1], xw.shape[1], self.hidden, self.n_labels
        self.params = {
            "Wi1": _he((di, hdim), rng),
            "bi1": np.zeros(hdim),
            "Ww1": _he((dw, hdim), rng),
            "bw1": np.zeros(hdim),
            "Wih": _he((hdim, lab), rng),
            "bih": np.zeros(lab),
            "Wwh": _he((hdim, lab), rng),
            "bwh": np.zeros(lab),
            "gate_wl": np.ones(lab, dtype=float),
            "gate_cat": np.ones(lab, dtype=float),
        }
        self.history = {
            "loss_fbm": [],
            "loss_wl": [],
            "loss_fusion": [],
            "loss_total": [],
        }

        optim = _Adam(lr=self.lr)
        for _ in range(self.epochs):
            cache = self._forward(xi, xw, cat, has_wl, has_cat)
            losses = self._losses(cache, y, fbm_weight, wl_weight, fusion_weight)
            grads = self._backward(cache, xi, xw, cat, y, has_wl, has_cat, fbm_weight, wl_weight, fusion_weight)
            optim.step(self.params, grads)
            for key, value in losses.items():
                self.history[key].append(value)
        return self

    def predict_logits(
        self,
        images_flat: np.ndarray,
        wl_maps: np.ndarray | None,
        *,
        has_wl_map: np.ndarray,
        catboost_logits: np.ndarray | None = None,
        has_catboost_logits: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        xi_raw = self._as_2d(images_flat, "images_flat")
        n_rows = xi_raw.shape[0]
        xw_raw = self._flatten_wl_maps(wl_maps, n_rows)
        cat = self._catboost_array(catboost_logits, (n_rows, self.n_labels))
        has_wl = self._row_mask(has_wl_map, n_rows, "has_wl_map")
        has_cat = self._row_mask(has_catboost_logits, n_rows, "has_catboost_logits")
        xi = self._apply_standardize(xi_raw, self._img_mean, self._img_std)
        xw = self._apply_standardize(xw_raw, self._wl_mean, self._wl_std)
        cache = self._forward(xi, xw, cat, has_wl, has_cat)
        return {
            "fbm": cache["zi"],
            "wl": cache["zw"],
            "catboost": cat,
            "fusion": cache["zf"],
        }

    def predict_heads(
        self,
        images_flat: np.ndarray,
        wl_maps: np.ndarray | None,
        *,
        has_wl_map: np.ndarray,
        catboost_logits: np.ndarray | None = None,
        has_catboost_logits: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        logits = self.predict_logits(
            images_flat,
            wl_maps,
            has_wl_map=has_wl_map,
            catboost_logits=catboost_logits,
            has_catboost_logits=has_catboost_logits,
        )
        n_rows = logits["fbm"].shape[0]
        has_wl = self._row_mask(has_wl_map, n_rows, "has_wl_map").astype(bool)
        has_cat = self._row_mask(has_catboost_logits, n_rows, "has_catboost_logits").astype(bool)
        out = {
            "image": _sigmoid(logits["fbm"]),
            "wl": _sigmoid(logits["wl"]),
            "catboost": _sigmoid(logits["catboost"]),
            "fusion": _sigmoid(logits["fusion"]),
        }
        out["wl"][~has_wl] = np.nan
        out["catboost"][~has_cat] = np.nan
        out["fusion"][~(has_wl | has_cat)] = np.nan
        return out

    def _forward(
        self,
        xi: np.ndarray,
        xw: np.ndarray,
        cat: np.ndarray,
        has_wl: np.ndarray,
        has_cat: np.ndarray,
    ) -> dict[str, np.ndarray]:
        p = self.params
        pre_i = xi @ p["Wi1"] + p["bi1"]
        ai = np.maximum(0.0, pre_i)
        zi = ai @ p["Wih"] + p["bih"]
        pre_w = xw @ p["Ww1"] + p["bw1"]
        aw = np.maximum(0.0, pre_w)
        zw = aw @ p["Wwh"] + p["bwh"]
        zf = (
            zi
            + has_wl.reshape(-1, 1) * p["gate_wl"].reshape(1, -1) * zw
            + has_cat.reshape(-1, 1) * p["gate_cat"].reshape(1, -1) * cat
        )
        return {"pre_i": pre_i, "ai": ai, "zi": zi, "pre_w": pre_w, "aw": aw, "zw": zw, "zf": zf}

    def _losses(
        self,
        cache: dict[str, np.ndarray],
        y: np.ndarray,
        fbm_weight: np.ndarray,
        wl_weight: np.ndarray,
        fusion_weight: np.ndarray,
    ) -> dict[str, float]:
        loss_fbm = self._weighted_bce_loss(cache["zi"], y, fbm_weight)
        loss_wl = self._weighted_bce_loss(cache["zw"], y, wl_weight)
        loss_fusion = self._weighted_bce_loss(cache["zf"], y, fusion_weight)
        return {
            "loss_fbm": loss_fbm,
            "loss_wl": loss_wl,
            "loss_fusion": loss_fusion,
            "loss_total": loss_fbm + loss_wl + loss_fusion,
        }

    def _backward(
        self,
        cache: dict[str, np.ndarray],
        xi: np.ndarray,
        xw: np.ndarray,
        cat: np.ndarray,
        y: np.ndarray,
        has_wl: np.ndarray,
        has_cat: np.ndarray,
        fbm_weight: np.ndarray,
        wl_weight: np.ndarray,
        fusion_weight: np.ndarray,
    ) -> dict[str, np.ndarray]:
        p = self.params
        d_zi = self._weighted_head_grad(cache["zi"], y, fbm_weight)
        d_zw = self._weighted_head_grad(cache["zw"], y, wl_weight)
        d_zf = self._weighted_head_grad(cache["zf"], y, fusion_weight)

        d_gate_wl = (d_zf * has_wl.reshape(-1, 1) * cache["zw"]).sum(axis=0)
        d_gate_cat = (d_zf * has_cat.reshape(-1, 1) * cat).sum(axis=0)
        d_zi = d_zi + d_zf
        d_zw = d_zw + d_zf * has_wl.reshape(-1, 1) * p["gate_wl"].reshape(1, -1)

        d_wih = cache["ai"].T @ d_zi
        d_bih = d_zi.sum(axis=0)
        d_ai = d_zi @ p["Wih"].T
        d_pre_i = d_ai * (cache["pre_i"] > 0)
        d_wi1 = xi.T @ d_pre_i
        d_bi1 = d_pre_i.sum(axis=0)

        d_wwh = cache["aw"].T @ d_zw
        d_bwh = d_zw.sum(axis=0)
        d_aw = d_zw @ p["Wwh"].T
        d_pre_w = d_aw * (cache["pre_w"] > 0)
        d_ww1 = xw.T @ d_pre_w
        d_bw1 = d_pre_w.sum(axis=0)

        grads = {
            "Wi1": d_wi1,
            "bi1": d_bi1,
            "Ww1": d_ww1,
            "bw1": d_bw1,
            "Wih": d_wih,
            "bih": d_bih,
            "Wwh": d_wwh,
            "bwh": d_bwh,
            "gate_wl": d_gate_wl,
            "gate_cat": d_gate_cat,
        }
        for key in ("Wi1", "Ww1", "Wih", "Wwh"):
            grads[key] = grads[key] + self.l2 * p[key]
        return grads

    @staticmethod
    def _weighted_bce_loss(z: np.ndarray, y: np.ndarray, row_weight: np.ndarray) -> float:
        weights = np.asarray(row_weight, dtype=float).reshape(-1)
        denom = float(weights.sum())
        if denom <= 0:
            return 0.0
        per_row = _bce_with_logits(z, y).mean(axis=1)
        return float((per_row * weights).sum() / denom)

    @staticmethod
    def _weighted_head_grad(z: np.ndarray, y: np.ndarray, row_weight: np.ndarray) -> np.ndarray:
        weights = np.asarray(row_weight, dtype=float).reshape(-1)
        denom = float(weights.sum())
        if denom <= 0:
            return np.zeros_like(z)
        return (_sigmoid(z) - y) * weights.reshape(-1, 1) / (denom * z.shape[1])

    @staticmethod
    def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        return array

    @staticmethod
    def _flatten_wl_maps(wl_maps: np.ndarray | None, n_rows: int) -> np.ndarray:
        if wl_maps is None:
            return np.zeros((n_rows, 1), dtype=float)
        maps = np.asarray(wl_maps, dtype=float)
        if maps.shape[0] != n_rows:
            raise ValueError("wl_maps row count must match labels/images")
        return np.nan_to_num(maps.reshape(n_rows, -1), nan=0.0)

    @staticmethod
    def _catboost_array(catboost_logits: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
        if catboost_logits is None:
            return np.zeros(shape, dtype=float)
        logits = np.asarray(catboost_logits, dtype=float)
        if logits.shape != shape:
            raise ValueError(f"catboost_logits must have shape {shape}, got {logits.shape}")
        return np.nan_to_num(logits, nan=0.0)

    @staticmethod
    def _row_mask(mask: np.ndarray | None, n_rows: int, name: str) -> np.ndarray:
        if mask is None:
            return np.zeros(n_rows, dtype=float)
        arr = np.asarray(mask, dtype=float).reshape(-1)
        if arr.shape[0] != n_rows:
            raise ValueError(f"{name} length must be {n_rows}, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _row_weights(weights: np.ndarray | None, n_rows: int, *, default: np.ndarray) -> np.ndarray:
        if weights is None:
            return np.asarray(default, dtype=float).reshape(-1)
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.shape[0] != n_rows:
            raise ValueError(f"weight length must be {n_rows}, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _fusion_weights(
        fusion_loss_weight: np.ndarray | None,
        has_wl: np.ndarray,
        has_cat: np.ndarray,
        wl_weight: np.ndarray,
    ) -> np.ndarray:
        if fusion_loss_weight is not None:
            return np.asarray(fusion_loss_weight, dtype=float).reshape(-1)
        available = (has_wl > 0) | (has_cat > 0)
        weights = available.astype(float)
        synthetic_wl = (has_wl > 0) & (has_cat <= 0) & (wl_weight > 0) & (wl_weight < 1.0)
        weights[synthetic_wl] = wl_weight[synthetic_wl]
        return weights

    def _fit_standardize(self, x: np.ndarray, *, rows: np.ndarray, kind: str) -> np.ndarray:
        filled = np.nan_to_num(x, nan=0.0)
        selected = filled[rows]
        if selected.size == 0:
            mean = np.zeros(filled.shape[1], dtype=float)
            std = np.ones(filled.shape[1], dtype=float)
        else:
            mean = selected.mean(axis=0)
            std = selected.std(axis=0) + 1e-6
        if kind == "img":
            self._img_mean = mean
            self._img_std = std
        elif kind == "wl":
            self._wl_mean = mean
            self._wl_std = std
        else:
            raise ValueError(f"unknown standardization kind: {kind}")
        return self._apply_standardize(filled, mean, std)

    @staticmethod
    def _apply_standardize(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
        if mean is None or std is None:
            raise ValueError("model must be fit before prediction")
        return (np.nan_to_num(x, nan=0.0) - mean) / std


@dataclass
class FusionMLP:
    """Multi-head image+tabular fusion classifier (numpy)."""

    hidden: int = 32
    lr: float = 3e-3
    epochs: int = 150
    dropout_p: float = 0.3
    l2: float = 1e-4
    seed: int = 0

    params: dict[str, np.ndarray] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    _img_mean: np.ndarray | None = None
    _img_std: np.ndarray | None = None
    _tab_mean: np.ndarray | None = None
    _tab_std: np.ndarray | None = None
    n_labels: int = 0

    # -- public API --------------------------------------------------------
    def fit(
        self,
        images_flat: np.ndarray,
        tabular: np.ndarray,
        labels: np.ndarray,
        has_tabular: np.ndarray,
    ) -> "FusionMLP":
        rng = np.random.default_rng(self.seed)
        xi = self._fit_standardize_images(images_flat)
        xt = self._fit_standardize_tabular(tabular, has_tabular)
        y = labels.astype(float)
        has_tabular = has_tabular.astype(bool)
        self.n_labels = y.shape[1]

        di, dt, hdim, lab = xi.shape[1], xt.shape[1], self.hidden, self.n_labels
        self.params = {
            "Wi1": _he((di, hdim), rng), "bi1": np.zeros(hdim),
            "Wt1": _he((dt, hdim), rng), "bt1": np.zeros(hdim),
            "Wih": _he((hdim, lab), rng), "bih": np.zeros(lab),
            "Wth": _he((hdim, lab), rng), "bth": np.zeros(lab),
            "Wf": _he((2 * hdim, lab), rng), "bf": np.zeros(lab),
            "null_tab": np.zeros(hdim),
        }
        self.history = {"loss_image": [], "loss_tabular": [], "loss_fusion": [], "loss_total": []}

        optim = _Adam(lr=self.lr)
        for _ in range(self.epochs):
            dropped = has_tabular & (rng.random(has_tabular.shape[0]) < self.dropout_p)
            use_null = (~has_tabular) | dropped
            cache = self._forward(xi, xt, use_null)
            losses = self._losses(cache, y, has_tabular, dropped)
            grads = self._backward(cache, xi, xt, y, has_tabular, dropped, use_null)
            optim.step(self.params, grads)
            for key, value in losses.items():
                self.history[key].append(value)
        return self

    def predict_heads(
        self, images_flat: np.ndarray, tabular: np.ndarray, has_tabular: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return image / tabular / fusion probabilities.

        tabular & fusion probabilities are NaN where ``has_tabular`` is False,
        which is exactly the predictions-table contract ``fusion_eval`` expects.
        """
        xi = self._apply_standardize(images_flat, self._img_mean, self._img_std)
        tab_filled = np.where(np.isnan(tabular), 0.0, tabular)
        xt = self._apply_standardize(tab_filled, self._tab_mean, self._tab_std)
        has_tabular = has_tabular.astype(bool)

        ai = np.maximum(0.0, xi @ self.params["Wi1"] + self.params["bi1"])
        img_prob = _sigmoid(ai @ self.params["Wih"] + self.params["bih"])

        at_real = np.maximum(0.0, xt @ self.params["Wt1"] + self.params["bt1"])
        tab_prob = _sigmoid(at_real @ self.params["Wth"] + self.params["bth"])
        concat = np.concatenate([ai, at_real], axis=1)
        fusion_prob = _sigmoid(concat @ self.params["Wf"] + self.params["bf"])

        nan_rows = ~has_tabular
        tab_prob[nan_rows] = np.nan
        fusion_prob[nan_rows] = np.nan
        return {"image": img_prob, "tabular": tab_prob, "fusion": fusion_prob}

    # -- internals ---------------------------------------------------------
    def _forward(self, xi: np.ndarray, xt: np.ndarray, use_null: np.ndarray) -> dict:
        p = self.params
        pre_i = xi @ p["Wi1"] + p["bi1"]
        ai = np.maximum(0.0, pre_i)
        pre_t = xt @ p["Wt1"] + p["bt1"]
        at_real = np.maximum(0.0, pre_t)
        at = at_real.copy()
        at[use_null] = p["null_tab"]
        zi = ai @ p["Wih"] + p["bih"]
        zt = at_real @ p["Wth"] + p["bth"]
        concat = np.concatenate([ai, at], axis=1)
        zf = concat @ p["Wf"] + p["bf"]
        return {
            "pre_i": pre_i, "ai": ai, "pre_t": pre_t, "at_real": at_real,
            "at": at, "zi": zi, "zt": zt, "zf": zf, "concat": concat,
        }

    def _losses(self, cache: dict, y: np.ndarray, has_tabular: np.ndarray, dropped: np.ndarray) -> dict:
        mask_tab = has_tabular & ~dropped
        mask_fus = has_tabular
        loss_img = float(_bce_with_logits(cache["zi"], y).mean())
        loss_tab = float(_bce_with_logits(cache["zt"][mask_tab], y[mask_tab]).mean()) if mask_tab.any() else 0.0
        loss_fus = float(_bce_with_logits(cache["zf"][mask_fus], y[mask_fus]).mean()) if mask_fus.any() else 0.0
        return {
            "loss_image": loss_img,
            "loss_tabular": loss_tab,
            "loss_fusion": loss_fus,
            "loss_total": loss_img + loss_tab + loss_fus,
        }

    def _backward(
        self,
        cache: dict,
        xi: np.ndarray,
        xt: np.ndarray,
        y: np.ndarray,
        has_tabular: np.ndarray,
        dropped: np.ndarray,
        use_null: np.ndarray,
    ) -> dict:
        p = self.params
        hdim, lab = self.hidden, self.n_labels
        mask_img = np.ones(y.shape[0], dtype=bool)
        mask_tab = has_tabular & ~dropped
        mask_fus = has_tabular

        d_zi = self._head_grad(cache["zi"], y, mask_img, lab)
        d_zt = self._head_grad(cache["zt"], y, mask_tab, lab)
        d_zf = self._head_grad(cache["zf"], y, mask_fus, lab)

        # heads
        d_wih = cache["ai"].T @ d_zi
        d_bih = d_zi.sum(0)
        d_ai = d_zi @ p["Wih"].T

        d_wth = cache["at_real"].T @ d_zt
        d_bth = d_zt.sum(0)
        d_at_real = d_zt @ p["Wth"].T

        d_wf = cache["concat"].T @ d_zf
        d_bf = d_zf.sum(0)
        d_concat = d_zf @ p["Wf"].T
        d_ai += d_concat[:, :hdim]
        d_at = d_concat[:, hdim:]

        # route fusion's tabular gradient: null rows -> null_tab, else -> at_real
        d_null = d_at[use_null].sum(0)
        d_at_real_from_fusion = np.zeros_like(d_at_real)
        d_at_real_from_fusion[~use_null] = d_at[~use_null]
        d_at_real = d_at_real + d_at_real_from_fusion

        # encoders through relu
        d_pre_i = d_ai * (cache["pre_i"] > 0)
        d_wi1 = xi.T @ d_pre_i
        d_bi1 = d_pre_i.sum(0)
        d_pre_t = d_at_real * (cache["pre_t"] > 0)
        d_wt1 = xt.T @ d_pre_t
        d_bt1 = d_pre_t.sum(0)

        grads = {
            "Wi1": d_wi1, "bi1": d_bi1, "Wt1": d_wt1, "bt1": d_bt1,
            "Wih": d_wih, "bih": d_bih, "Wth": d_wth, "bth": d_bth,
            "Wf": d_wf, "bf": d_bf, "null_tab": d_null,
        }
        # L2 on weight matrices only
        for key in ("Wi1", "Wt1", "Wih", "Wth", "Wf"):
            grads[key] = grads[key] + self.l2 * p[key]
        return grads

    @staticmethod
    def _head_grad(z: np.ndarray, y: np.ndarray, mask: np.ndarray, lab: int) -> np.ndarray:
        d_z = np.zeros_like(z)
        n = int(mask.sum())
        if n > 0:
            d_z[mask] = (_sigmoid(z[mask]) - y[mask]) / (n * lab)
        return d_z

    # -- standardization ---------------------------------------------------
    def _fit_standardize_images(self, images_flat: np.ndarray) -> np.ndarray:
        self._img_mean = images_flat.mean(axis=0)
        self._img_std = images_flat.std(axis=0) + 1e-6
        return self._apply_standardize(images_flat, self._img_mean, self._img_std)

    def _fit_standardize_tabular(self, tabular: np.ndarray, has_tabular: np.ndarray) -> np.ndarray:
        real = tabular[has_tabular.astype(bool)]
        filled = np.where(np.isnan(tabular), 0.0, tabular)
        self._tab_mean = np.where(np.isnan(real).all(axis=0), 0.0, np.nanmean(real, axis=0))
        self._tab_std = np.nanstd(real, axis=0) + 1e-6
        return self._apply_standardize(filled, self._tab_mean, self._tab_std)

    @staticmethod
    def _apply_standardize(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
        return (x - mean) / std
