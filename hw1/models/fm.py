"""Factorization Machine (Rendle, 2010) с метапризнаками, обучение BPR-лоссом."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import BaseRecommender, Stage

DAY = 86400
AGE_EDGES = np.array([1, 2, 3, 5, 10, 20])        # возраст фильма, лет: <=0, 1, 2, 3-4, 5-9, 10-19, 20+
YEAR_EDGES = np.array([1960, 1980, 1990, 2000, 2005, 2010, 2013, 2015])   # эпохи выпуска
POP_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])   # оценок за 30 дней: 0, 1, 2-3, ..., 256+
POP_WINDOWS = (7, 30)                              # окна (дни) для числовых признаков log(1 + n_оценок)


class FactorizationMachine(BaseRecommender):
    """a(x) = w0 + sum_f w_f x_f + sum_{f<g} <v_f, v_g> x_f x_g
           = w0 + sum_f w_f x_f + 1/2 (||sum_f v_f x_f||^2 - sum_f ||v_f||^2 x_f^2).

    x(u, i, t) = [one-hot пользователя | one-hot фильма (use_item_id), жанры с весом 1/|G_i| и эпоха
    выпуска (use_meta) | контекст момента t (use_context): возраст фильма в момент t (корзины),
    число оценок фильма за 30 дней до t (лог-корзины), log(1 + n) за 7 и 30 дней (числа)].
    Признаки берутся только из прошлого относительно t: метаданные фильма известны заранее,
    популярность считается по оценкам строго до t. При рекомендации t = t_rec.
    Лосс BPR: -ln sigma(a(u, i+, t) - a(u, i-, t)), i- — случайный фильм каталога, не оценённый u.
    Adagrad, L2 для параметров, участвующих в батче; рекомендации строятся по EMA параметров.
    """

    name = "FM"

    def __init__(self, movies: pd.DataFrame, factors: int = 32, lr: float = 0.05, reg: float = 1e-4,
                 epochs: int = 20, batch_size: int = 4096, use_item_id: bool = True, use_meta: bool = True,
                 use_context: bool = True, ema: float | None = 0.99, seed: int = 42):
        if not (use_item_id or use_meta):
            raise ValueError("нужен хотя бы один признак фильма: id или метаданные")
        self.movies = movies.set_index("item_id")
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.ema = ema
        self.use_item_id = use_item_id
        self.use_meta = use_meta
        self.use_context = use_context
        self.seed = seed

    # ------------------------------------------------------------ признаки
    def _build_static(self, stage: Stage):
        """Статическая часть признаков фильма: id, жанры, эпоха выпуска (CSR n_items x n_feat)."""
        meta = self.movies.loc[stage.item_ids]
        n = stage.n_items
        rows, cols, vals, off = [], [], [], 0
        if self.use_item_id:
            rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(np.ones(n))
            off = n
        self.year = meta.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).to_numpy()
        if self.use_meta:
            genres = meta.genres.fillna("Unknown").str.split("|")
            vocab = sorted({g for gs in genres for g in gs})
            gidx = {g: k for k, g in enumerate(vocab)}
            for r, gs in enumerate(genres):
                for g in gs:
                    rows.append([r]); cols.append([off + gidx[g]]); vals.append([1.0 / len(gs)])
            off += len(vocab)
            yb = np.where(np.isnan(self.year), len(YEAR_EDGES) + 1,
                          np.searchsorted(YEAR_EDGES, np.nan_to_num(self.year), side="right"))
            rows.append(np.arange(n)); cols.append(off + yb); vals.append(np.ones(n))
            off += len(YEAR_EDGES) + 2
        self.ctx_offset = off
        n_ctx = (len(AGE_EDGES) + 2) + (len(POP_EDGES) + 1) + len(POP_WINDOWS)
        n_feat = off + (n_ctx if self.use_context else 0)
        X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n_feat))
        return X, n_feat

    def _fit_popularity_index(self, stage: Stage):
        tr = stage.train.sort_values(["i", "ts"])
        self._pop_key = tr.i.to_numpy().astype(np.int64) * (1 << 34) + tr.ts.to_numpy().astype(np.int64)

    def _count_before(self, items: np.ndarray, t: np.ndarray, days: int) -> np.ndarray:
        """Число оценок фильма в окне [t - days, t) — только прошлое относительно t."""
        key = items.astype(np.int64) * (1 << 34) + t.astype(np.int64)
        return (np.searchsorted(self._pop_key, key, side="left")
                - np.searchsorted(self._pop_key, key - days * DAY, side="left"))

    def _context(self, items: np.ndarray, t: np.ndarray):
        """Контекст момента t: возраст фильма (one-hot), популярность за 30 дней (one-hot)
        и log(1 + n_оценок) за 7 и 30 дней (числовые признаки, нормированы делением на log(1 + 256))."""
        cal_year = pd.to_datetime(t, unit="s").year.to_numpy()
        age = cal_year - self.year[items]
        ab = np.where(np.isnan(age), len(AGE_EDGES) + 1,
                      np.searchsorted(AGE_EDGES, np.nan_to_num(age), side="right"))
        counts = {d: self._count_before(items, t, d) for d in POP_WINDOWS}
        pb = np.searchsorted(POP_EDGES, counts[30], side="right")
        off_pop = self.ctx_offset + len(AGE_EDGES) + 2
        off_num = off_pop + len(POP_EDGES) + 1
        cols = [self.ctx_offset + ab, off_pop + pb] + [np.full(len(items), off_num + k)
                                                       for k in range(len(POP_WINDOWS))]
        vals = [np.ones(len(items)), np.ones(len(items))] + [np.log1p(counts[d]) / np.log1p(256)
                                                             for d in POP_WINDOWS]
        return np.stack(cols, 1), np.stack(vals, 1)

    def _design(self, items: np.ndarray, t: np.ndarray) -> sp.csr_matrix:
        X = self.X_static[items]
        if self.use_context:
            cols, vals = self._context(items, t)
            n, m = cols.shape
            ctx = sp.csr_matrix((vals.ravel(), cols.ravel(), np.arange(0, n * m + 1, m)), shape=X.shape)
            X = X + ctx
        return X

    # ------------------------------------------------------------ модель
    def _item_part(self, X: sp.csr_matrix):
        """Вклад признаков объекта: S = X V, b = X w + 1/2 (||S||^2 - X^2 ||v||^2)."""
        S = X @ self.V
        X2 = X.multiply(X)
        b = X @ self.w + 0.5 * (np.einsum("ij,ij->i", S, S) - X2 @ np.einsum("ij,ij->i", self.V, self.V))
        return S, b, X2

    def _bpr_grads(self, users, Xp, Xn):
        Su = self.U[users]
        Sp, bp, Xp2 = self._item_part(Xp)
        Sn, bn, Xn2 = self._item_part(Xn)
        diff = bp - bn + np.einsum("ij,ij->i", Su, Sp - Sn)
        g = 0.5 * (1.0 - np.tanh(0.5 * diff))           # sigma(-diff) без переполнения
        loss = np.logaddexp(0.0, -diff).sum()
        wp, wn = -g, g                                   # dL/da(x+), dL/da(x-)
        gU = (wp[:, None] * Sp + wn[:, None] * Sn)
        gw = Xp.T @ wp + Xn.T @ wn
        gV = (Xp.T @ (wp[:, None] * (Su + Sp)) - (Xp2.T @ wp)[:, None] * self.V
              + Xn.T @ (wn[:, None] * (Su + Sn)) - (Xn2.T @ wn)[:, None] * self.V)
        return loss, gU, gw, gV

    def fit(self, stage: Stage, callback=None):
        rng = np.random.default_rng(self.seed)
        self.stage_t_rec = stage.t_rec
        self.X_static, n_feat = self._build_static(stage)
        self._fit_popularity_index(stage)
        k = self.factors
        self.U = rng.normal(0, 0.01, (stage.n_users, k))
        self.V = rng.normal(0, 0.01, (n_feat, k))
        self.w = np.zeros(n_feat)
        acc_U, acc_V, acc_w = np.zeros_like(self.U), np.zeros_like(self.V), np.zeros_like(self.w)

        tr = stage.train
        u_all, i_all, t_all = tr.u.to_numpy(), tr.i.to_numpy(), tr.ts.to_numpy()
        seen = stage.X.tocsr()
        self.avg = [self.U.copy(), self.V.copy(), self.w.copy()] if self.ema else None
        self.history = []
        for epoch in range(self.epochs):
            order = rng.permutation(len(u_all))
            total = 0.0
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                u, ip, t = u_all[idx], i_all[idx], t_all[idx]
                # негатив: случайный фильм каталога, который пользователь не оценивал
                ineg = rng.integers(0, stage.n_items, len(idx))
                for _ in range(3):
                    bad = np.asarray(seen[u, ineg]).ravel() > 0
                    if not bad.any():
                        break
                    ineg[bad] = rng.integers(0, stage.n_items, bad.sum())
                Xp, Xn = self._design(ip, t), self._design(ineg, t)
                loss, gU, gw, gV = self._bpr_grads(u, Xp, Xn)
                total += loss
                # L2 для параметров, встретившихся в батче (как в SGD по объектам)
                cnt_v = np.asarray((Xp != 0).sum(0)).ravel() + np.asarray((Xn != 0).sum(0)).ravel()
                gV += self.reg * cnt_v[:, None] * self.V
                gw += self.reg * cnt_v * self.w
                uu, inv = np.unique(u, return_inverse=True)
                gUu = np.zeros((len(uu), k))
                np.add.at(gUu, inv, gU)
                gUu += self.reg * np.bincount(inv)[:, None] * self.U[uu]
                # Adagrad
                acc_U[uu] += gUu ** 2
                self.U[uu] -= self.lr * gUu / (np.sqrt(acc_U[uu]) + 1e-8)
                acc_V += gV ** 2
                self.V -= self.lr * gV / (np.sqrt(acc_V) + 1e-8)
                acc_w += gw ** 2
                self.w -= self.lr * gw / (np.sqrt(acc_w) + 1e-8)
                if self.avg is not None:                 # экспоненциальное среднее параметров
                    for a, p in zip(self.avg, (self.U, self.V, self.w)):
                        a *= self.ema
                        a += (1.0 - self.ema) * p
            self.history.append(total / len(order))
            if callback is not None and callback(self, epoch + 1) is False:
                break
        return self

    def score(self, users):
        """Скор кандидатов в момент рекомендации t_rec; слагаемые, зависящие только от пользователя, опущены."""
        U, V, w = self.avg if self.avg is not None else (self.U, self.V, self.w)
        items = np.arange(self.X_static.shape[0])
        X = self._design(items, np.full(len(items), self.stage_t_rec))
        S = X @ V
        b = X @ w + 0.5 * (np.einsum("ij,ij->i", S, S) - X.multiply(X) @ np.einsum("ij,ij->i", V, V))
        return U[users] @ S.T + b[None, :]
