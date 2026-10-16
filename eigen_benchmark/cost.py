"""§3.5 비용 — 라인아이템 계측, t(ε), E[T_success].

정직한 비용회계의 요점 두 가지:
1) **실패도 청구한다** — E[T_success] = 실행당 평균비용 / P(correct mode)
2) 사전학습·GPU 예열을 별도 라인으로 남긴다(숨기지 않는다)
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

DEFAULT_LEVELS = {
    "elam_1e-2": ("e_lam", "<=", 1e-2),
    "elam_1e-3": ("e_lam", "<=", 1e-3),
    "mac_0.95": ("mac", ">=", 0.95),
    "resid_1e-3": ("r_h", "<=", 1e-3),
}


@contextmanager
def timed(label: str):
    """with 블록의 벽시계 시간을 초 단위로 기록."""
    rec = {"label": label, "seconds": float("nan")}
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["seconds"] = time.perf_counter() - t0


def line_items(fns: dict) -> dict:
    """라벨→무인자 콜러블을 순차 실행하고 각 소요시간[s]을 반환."""
    out = {}
    for label, fn in fns.items():
        with timed(label) as rec:
            fn()
        out[label] = rec["seconds"]
    return out


def expected_time_to_success(mean_cost: float, p_correct: float) -> float:
    """E[T_success]. P(correct)=0이면 무한(=이 설정으로는 성공을 살 수 없다)."""
    if p_correct <= 0.0:
        return math.inf
    return float(mean_cost) / float(p_correct)


def time_to_accuracy(history, levels=None) -> dict:
    """수준별 최초 도달시각[s]. 미도달은 None.

    history: [(t_seconds, {지표명: 값})] 시간순. levels: {이름: (지표, "<="|">=", 임계)}"""
    levels = DEFAULT_LEVELS if levels is None else levels
    out = {name: None for name in levels}
    for t, m in history:
        for name, (key, op, thr) in levels.items():
            if out[name] is not None or key not in m:
                continue
            v = m[key]
            if v is None or not math.isfinite(float(v)):
                continue
            if (op == "<=" and v <= thr) or (op == ">=" and v >= thr):
                out[name] = float(t)
    return out
