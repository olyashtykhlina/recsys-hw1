"""iALS (Hu, Koren, Volinsky, 2008) для неявной обратной связи."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .utils import BaseRecommender, Stage


def _sparse_quadratic(Z, Y, M: sp.csr_matrix, chunk: int = 200_000):
    """Возвращает (M ∘ (Z Y^T)) @ Y, где M разреженная с весами (c_ui - 1); считается только по ненулям M."""
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    vals = np.empty(M.nnz, dtype=Z.dtype)
    for s in range(0, M.nnz, chunk):
        e = s + chunk
        vals[s:e] = np.einsum("ij,ij->i", Z[rows[s:e]], Y[M.indices[s:e]]) * M.data[s:e]
    return sp.csr_matrix((vals, M.indices, M.indptr), shape=M.shape) @ Y


def _cg_step(X, Y, C1: sp.csr_matrix, CP: sp.csr_matrix, reg: np.ndarray, n_steps: int):
    """Решает для всех строк сразу (Y^T C_u Y + reg_u I) x_u = Y^T C_u p_u методом сопряжённых градиентов.

    C1 — разреженная матрица (c_ui - 1) по наблюдаемым парам, CP — матрица c_ui * p_ui,
    reg — вектор коэффициентов регуляризации по строкам.
    Для ненаблюдаемых пар c = 1, p = 0, поэтому Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y.
    """
    YtY = Y.T @ Y

    def A(Z):
        return Z @ YtY + reg[:, None] * Z + _sparse_quadratic(Z, Y, C1)

    b = CP @ Y
    r = b - A(X)
    p = r.copy()
    rs = np.einsum("ij,ij->i", r, r)
    for _ in range(n_steps):
        Ap = A(p)
        denom = np.einsum("ij,ij->i", p, Ap)
        a = np.divide(rs, denom, out=np.zeros_like(rs), where=denom > 0)
        X += a[:, None] * p
        r -= a[:, None] * Ap
        rs_new = np.einsum("ij,ij->i", r, r)
        beta = np.divide(rs_new, rs, out=np.zeros_like(rs), where=rs > 0)
        p = r + beta[:, None] * p
        rs = rs_new
    return X


def _reg_weights(C: sp.csr_matrix, exponent: float) -> np.ndarray:
    """C_u = (sum_i c_ui)^nu * n / sum_v (sum_i c_vi)^nu — регуляризация, масштабированная активностью (лекция 4)."""
    mass = np.asarray(C.sum(axis=1)).ravel() ** exponent
    return mass * len(mass) / mass.sum()


class IALS(BaseRecommender):
    """min sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + reg (sum_u C_u ||x_u||^2 + sum_i C_i ||y_i||^2),

    базовый вариант: p_ui = 1 для наблюдаемых пар и 0 иначе, c_ui = 1 + alpha * [пара наблюдалась];
    C_u = (sum_i c_ui)^nu * m / sum_v (sum_i c_vi)^nu (nu = reg_exponent; при nu = 0 обычная L2).
    Шаг ALS для пользователя: x_u = (Y^T Y + reg C_u I + Y^T (C_u - I) Y)^{-1} Y^T C_u p_u,
    для объектов симметрично. Системы решаются методом сопряжённых градиентов с тёплым стартом
    (Takács et al., 2011).
    """

    name = "iALS"

    def __init__(self, factors: int = 64, reg: float = 1.0, alpha: float = 10.0,
                 reg_exponent: float = 0.0, iterations: int = 15, cg_steps: int = 3, seed: int = 42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.reg_exponent = reg_exponent
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.seed = seed

    # --- целевой сигнал: (p_ui, c_ui) для наблюдаемых пар в порядке stage.train
    def targets(self, stage: Stage):
        n = len(stage.train)
        return np.ones(n), np.full(n, 1.0 + self.alpha)

    def fit(self, stage: Stage, callback=None):
        tr = stage.train
        p, c = self.targets(stage)
        shape = (stage.n_users, stage.n_items)
        u, i = tr.u.to_numpy(), tr.i.to_numpy()
        C1 = sp.csr_matrix((c - 1.0, (u, i)), shape=shape)
        CP = sp.csr_matrix((c * p, (u, i)), shape=shape)
        C1T, CPT = C1.T.tocsr(), CP.T.tocsr()
        C1.sort_indices(); CP.sort_indices(); C1T.sort_indices(); CPT.sort_indices()
        C = sp.csr_matrix((c, (u, i)), shape=shape)
        reg_u = self.reg * _reg_weights(C, self.reg_exponent)
        reg_i = self.reg * _reg_weights(C.T.tocsr(), self.reg_exponent)

        rng = np.random.default_rng(self.seed)
        # пользователи решаются первыми, поэтому случайная инициализация нужна только объектам;
        # нулевой старт для X сохраняет масштаб решения даже при очень малых таргетах
        self.U = np.zeros((shape[0], self.factors))
        self.V = rng.normal(0, 0.01, (shape[1], self.factors))
        for it in range(self.iterations):
            self.U = _cg_step(self.U, self.V, C1, CP, reg_u, self.cg_steps)
            self.V = _cg_step(self.V, self.U, C1T, CPT, reg_i, self.cg_steps)
            if callback is not None:
                callback(self, it + 1)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T
