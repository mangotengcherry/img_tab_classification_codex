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
