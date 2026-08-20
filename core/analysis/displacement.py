"""変位法による杭基礎の安定計算(道示Ⅳ(H24) 12.6)。

フーチングを剛体、杭を直杭・杭頭剛結と仮定し、フーチング底面中心の
変位 (u, v, θ) を未知数として釣合い式を解く。

    ⎡ nK1      0        nK2            ⎤ ⎧u⎫   ⎧H⎫
    ⎢ 0        nKv      Kv・Σxi        ⎥ ⎨v⎬ = ⎨V⎬
    ⎣ nK3      Kv・Σxi  nK4 + Kv・Σxi² ⎦ ⎩θ⎭   ⎩M⎭

各杭の杭頭反力:
    軸力     Ni = Kv・(v + xi・θ)   (押込み正)
    水平力   Hi = K1・u + K2・θ
    モーメント Mi = K3・u + K4・θ
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.models.pile import PileArrangement


@dataclass(frozen=True)
class PileReaction:
    index: int
    x: float  # フーチング中心からの橋軸方向距離 (m)
    axial: float  # 軸力 (kN、押込み正)
    shear: float  # 水平力 (kN)
    moment: float  # 杭頭モーメント (kN·m)


@dataclass(frozen=True)
class StabilityResult:
    u: float  # 水平変位 (m)
    v: float  # 鉛直変位 (m、沈下正)
    theta: float  # 回転角 (rad)
    reactions: list[PileReaction]

    @property
    def max_axial(self) -> float:
        return max(r.axial for r in self.reactions)

    @property
    def min_axial(self) -> float:
        return min(r.axial for r in self.reactions)

    @property
    def max_shear(self) -> float:
        return max(abs(r.shear) for r in self.reactions)

    @property
    def max_head_moment(self) -> float:
        return max(abs(r.moment) for r in self.reactions)


def pile_x_coordinates(arrangement: PileArrangement) -> list[float]:
    """全杭の橋軸方向座標(フーチング中心が原点)。"""
    xs = []
    for i in range(arrangement.nx):
        x = (i - (arrangement.nx - 1) / 2.0) * arrangement.spacing_x
        xs.extend([x] * arrangement.ny)
    return xs


def solve_stability(
    arrangement: PileArrangement,
    kv: float,
    k1: float,
    k2: float,
    k4: float,
    v_load: float,
    h_load: float,
    m_load: float,
) -> StabilityResult:
    """変位法によりフーチング変位と各杭の杭頭反力を求める。

    ``k2`` は結合項(K2 = K3)。荷重はフーチング底面中心に作用する値。
    """
    xs = np.array(pile_x_coordinates(arrangement), dtype=float)
    n = len(xs)
    sum_x = float(xs.sum())
    sum_x2 = float((xs**2).sum())

    a = np.array(
        [
            [n * k1, 0.0, n * k2],
            [0.0, n * kv, kv * sum_x],
            [n * k2, kv * sum_x, n * k4 + kv * sum_x2],
        ]
    )
    b = np.array([h_load, v_load, m_load], dtype=float)
    try:
        u, v, theta = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - 異常入力時のみ
        raise ValueError(f"剛性マトリクスが特異です: {exc}") from exc

    reactions = [
        PileReaction(
            index=i + 1,
            x=float(x),
            axial=kv * (v + x * theta),
            shear=k1 * u + k2 * theta,
            moment=k2 * u + k4 * theta,
        )
        for i, x in enumerate(xs)
    ]
    return StabilityResult(u=float(u), v=float(v), theta=float(theta), reactions=reactions)
