"""iALS с самостоятельно разработанным целевым сигналом: «актуальное удовольствие»."""
from __future__ import annotations

import numpy as np

from .ials import IALS
from .utils import Stage

DAY = 86400


class IALSCustomTarget(IALS):
    """Тот же функционал и та же уверенность, что у базового iALS (c_ui = 1 + alpha), меняется только таргет:

        p_ui = s(r_ui) * d(t_rec - t_ui),
        s(r) = ((r - 1) / 9) ** gamma          — насколько понравился фильм (оценка 1..10 -> [0, 1]),
        d(dt) = 2 ** (-dt / half_life)          — насколько интерес актуален к моменту рекомендации.

    Для ненаблюдаемых пар по-прежнему p = 0, c = 1.
    gamma = 0 и half_life = None (бесконечность) возвращают базовый iALS: p_ui = 1.
    """

    name = "iALS-custom"

    def __init__(self, gamma: float = 1.0, half_life: float | None = 90.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.half_life = half_life

    def targets(self, stage: Stage):
        tr = stage.train
        r = tr.rating.to_numpy().astype(np.float64)
        satisfaction = ((r - 1.0) / 9.0) ** self.gamma
        age_days = (stage.t_rec - tr.ts.to_numpy()) / DAY
        if self.half_life is None:
            recency = np.ones_like(age_days)
        else:
            recency = 2.0 ** (-age_days / self.half_life)
        p = satisfaction * recency
        c = np.full(len(tr), 1.0 + self.alpha)
        return p, c
