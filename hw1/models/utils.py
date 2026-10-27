"""Общие функции: загрузка данных, временное разбиение, метрики, базовый класс рекомендера."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------- данные

REPO = "sidooms/MovieTweetings"
COMMIT = "44c525d0c766944910686c60697203cda39305d6"  # последнее обновление датасета, 24.11.2021
FILES = {
    "ratings.dat": "f8a46fe159d4bea12d5ea38922807ce1d8e7c2563785ebc84c479357d80feb13",
    "movies.dat": "0403d193fe09b41b2f1c5451e526d74c8c4d8629adb66646be51ba785018001f",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_movietweetings(data_dir: str = "data") -> None:
    """Скачивает ratings.dat и movies.dat из зафиксированного коммита GitHub и проверяет SHA-256."""
    os.makedirs(data_dir, exist_ok=True)
    for name, sha in FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/latest/{name}"
            print(f"скачиваю {url}")
            urllib.request.urlretrieve(url, path)
        got = _sha256(path)
        if got != sha:
            raise ValueError(f"{name}: SHA-256 {got} не совпадает с ожидаемым {sha}")
        print(f"{name}: OK ({os.path.getsize(path) / 2**20:.1f} MB, sha256 совпадает)")


def load_movietweetings(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"), sep="::", engine="python",
        names=["user_id", "item_id", "rating", "ts"],
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"), sep="::", engine="python",
        names=["item_id", "title", "genres"], encoding="utf-8",
    )
    return ratings, movies


# ---------------------------------------------------------------- временное разбиение


@dataclass
class Stage:
    """Один этап протокола: обучающая часть (всё до момента t_rec) и окно оценки [t_rec, t_end)."""

    name: str
    t_rec: int                    # момент формирования рекомендаций (unix time)
    train: pd.DataFrame           # u, i, rating, ts — индексы пользователей и объектов каталога
    X: sp.csr_matrix              # бинарная матрица train (n_users x n_items)
    eval_users: np.ndarray        # индексы пользователей, на которых считаем метрики
    relevant: sp.csr_matrix       # релевантные объекты окна (rating >= порога), строки = eval_users
    eval_ratings: pd.DataFrame    # все оценки окна по тёплым u/i (для RMSE)
    user_ids: np.ndarray          # индекс -> исходный user_id
    item_ids: np.ndarray          # индекс -> исходный item_id
    stats: dict = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def make_stage(df: pd.DataFrame, t_rec: int, t_end: int, name: str,
               min_item_train: int = 5, rel_threshold: int = 7) -> Stage:
    """Глобальное временное разбиение.

    train  = все взаимодействия с ts < t_rec;
    каталог = объекты, у которых в train >= min_item_train взаимодействий (статистика только по прошлому);
    тёплые пользователи = есть хотя бы одно взаимодействие с каталогом в train;
    окно оценки = [t_rec, t_end); релевантно = оценка >= rel_threshold на объект каталога.
    Холодные пользователи и объекты в оценку не попадают (их долю считаем в stats).
    """
    past = df[df.ts < t_rec]
    window = df[(df.ts >= t_rec) & (df.ts < t_end)]

    cnt = past.item_id.value_counts()
    item_ids = np.sort(cnt.index[cnt >= min_item_train].to_numpy())
    past = past[past.item_id.isin(item_ids)]
    user_ids = np.sort(past.user_id.unique())

    umap = pd.Series(np.arange(len(user_ids)), index=user_ids)
    imap = pd.Series(np.arange(len(item_ids)), index=item_ids)

    train = pd.DataFrame({
        "u": umap.loc[past.user_id].to_numpy(),
        "i": imap.loc[past.item_id].to_numpy(),
        "rating": past.rating.to_numpy(),
        "ts": past.ts.to_numpy(),
    }).sort_values(["u", "ts"], kind="stable").reset_index(drop=True)
    X = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (train.u.to_numpy(), train.i.to_numpy())),
        shape=(len(user_ids), len(item_ids)),
    )

    warm = window[window.user_id.isin(user_ids) & window.item_id.isin(item_ids)]
    eval_ratings = pd.DataFrame({
        "u": umap.loc[warm.user_id].to_numpy(),
        "i": imap.loc[warm.item_id].to_numpy(),
        "rating": warm.rating.to_numpy(),
        "ts": warm.ts.to_numpy(),
    })
    rel = eval_ratings[eval_ratings.rating >= rel_threshold]
    eval_users = np.sort(rel.u.unique())
    row = pd.Series(np.arange(len(eval_users)), index=eval_users)
    relevant = sp.csr_matrix(
        (np.ones(len(rel), dtype=np.float32), (row.loc[rel.u].to_numpy(), rel.i.to_numpy())),
        shape=(len(eval_users), len(item_ids)),
    )

    w_rel = window[window.rating >= rel_threshold]
    stats = {
        "train interactions": len(train),
        "train users": len(user_ids),
        "catalog items": len(item_ids),
        "window interactions": len(window),
        "window relevant (rating>=thr)": len(w_rel),
        "share relevant: cold user": float((~w_rel.user_id.isin(user_ids)).mean()),
        "share relevant: cold item": float((~w_rel.item_id.isin(item_ids)).mean()),
        "eval users": len(eval_users),
        "eval relevant pairs": int(relevant.nnz),
    }
    return Stage(name, t_rec, train, X, eval_users, relevant, eval_ratings, user_ids, item_ids, stats)


# ---------------------------------------------------------------- рекомендации и метрики


class BaseRecommender:
    """Модель реализует fit(stage) и score(users) -> плотная матрица (len(users) x n_items)."""

    name = "base"

    def fit(self, stage: Stage):
        raise NotImplementedError

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, users: np.ndarray, X_seen: sp.csr_matrix, k: int = 10,
                  batch: int = 2048) -> np.ndarray:
        """Top-k по убыванию скора; объекты, известные пользователю по train, исключаются."""
        out = np.empty((len(users), k), dtype=np.int64)
        for s in range(0, len(users), batch):
            ub = users[s:s + batch]
            scores = np.asarray(self.score(ub), dtype=np.float64)
            seen = X_seen[ub]
            r, c = seen.nonzero()
            scores[r, c] = -np.inf
            out[s:s + batch] = topk(scores, k)
        return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def per_user_metrics(recs: np.ndarray, relevant: sp.csr_matrix) -> dict:
    """NDCG@K_u = DCG@K_u / IDCG@K_u,  Recall@K_u = |T_u ∩ G_u| / |G_u| (G_u — релевантные объекты окна)."""
    k = recs.shape[1]
    rel = relevant.tocsr()
    hits = np.zeros(recs.shape, dtype=bool)
    for row in range(recs.shape[0]):
        g = rel.indices[rel.indptr[row]:rel.indptr[row + 1]]
        hits[row] = np.isin(recs[row], g)
    n_rel = np.diff(rel.indptr)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(axis=1)
    idcg = np.cumsum(disc)[np.minimum(n_rel, k) - 1]
    return {f"NDCG@{k}": dcg / idcg, f"Recall@{k}": hits.sum(axis=1) / n_rel}


def ranking_metrics(recs: np.ndarray, relevant: sp.csr_matrix, n_items: int) -> dict:
    """Средние по пользователям NDCG@K и Recall@K, а также Coverage@K = |I_P| / |I| (доля каталога в выдачах)."""
    k = recs.shape[1]
    out = {name: float(v.mean()) for name, v in per_user_metrics(recs, relevant).items()}
    out[f"Coverage@{k}"] = float(len(np.unique(recs)) / n_items)
    return out


def evaluate(model: BaseRecommender, stage: Stage, k: int = 10, return_recs: bool = False):
    recs = model.recommend(stage.eval_users, stage.X, k=k)
    metrics = ranking_metrics(recs, stage.relevant, stage.n_items)
    return (metrics, recs) if return_recs else metrics


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Доверительный интервал среднего по пользователям (перцентильный бутстрэп)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------- бейзлайн для ориентира


class Popular(BaseRecommender):
    """Непёрсонализированный ориентир: самые частые объекты train (за всё время или за последние days дней)."""

    def __init__(self, days: int | None = None):
        self.days = days
        self.name = "Popular" if days is None else f"Popular-{days}d"

    def fit(self, stage: Stage):
        tr = stage.train
        if self.days is not None:
            tr = tr[tr.ts >= stage.t_rec - self.days * 86400]
        cnt = np.bincount(tr.i.to_numpy(), minlength=stage.n_items).astype(np.float64)
        self.pop = cnt + 1e-6 * np.bincount(stage.train.i.to_numpy(), minlength=stage.n_items)
        return self

    def score(self, users):
        return np.broadcast_to(self.pop, (len(users), len(self.pop))).copy()
