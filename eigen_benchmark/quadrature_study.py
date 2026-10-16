"""§3.6 구적분리 — 기저차수를 고정하고 구적차수만 배증해 오차의 출처를 가른다.

구적오차가 기저오차로 위장하지 못하게 하는 것이 목적이다. 같은 기저를 두 구적차수로
풀어 값이 움직이지 않으면 남은 오차는 전부 기저(e_approx)의 것이다.
"""
from __future__ import annotations


def doubling_table(basis_factory, solver, n_q_list):
    """구적차수 목록에 대해 solver(basis, n_q)를 호출하고 변화량을 기록.

    반환: [{"n_q", "value", "abs_change"}] — abs_change는 직전 행 대비 절대변화(첫 행 None)."""
    rows = []
    prev = None
    for nq in n_q_list:
        val = float(solver(basis_factory(), nq))
        rows.append({"n_q": int(nq), "value": val,
                     "abs_change": None if prev is None else abs(val - prev)})
        prev = val
    return rows
