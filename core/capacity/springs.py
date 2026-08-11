"""杭のバネ定数(道示Ⅳ(H24) 9.5、12.6)。

軸方向バネ定数(12.6.1):
    Kv = a・Ap・Ep / L,  a = slope・(L/D) + intercept(工法別)

水平方向地盤反力係数(9.5.2):
    kH  = kH0・(BH/0.3)^(-3/4)
    kH0 = α・E0 / 0.3
    BH  = √(D/β),  β = ⁴√(kH・D / (4EI))
kH と β は相互に依存するため収束計算による。

杭頭バネ定数(12.6.2、半無限長の杭 βLe ≧ 3):
    K1 = 4EIβ³, K2 = K3 = −2EIβ², K4 = 2EIβ
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.loads import LoadCase
from core.models.pile import PileSpec
from core.models.soil import SoilProfile
from core.standards import (
    ALPHA_KH_NORMAL,
    ALPHA_KH_SEISMIC,
    E0_FROM_N,
    KV_A_COEF,
)


@dataclass(frozen=True)
class PileSection:
    """杭の断面諸元。"""

    area: float  # 断面積 Ap (m2)
    inertia: float  # 断面二次モーメント I (m4)
    young: float  # ヤング係数 E (kN/m2)

    @property
    def ei(self) -> float:
        return self.young * self.inertia


@dataclass(frozen=True)
class LateralSprings:
    """水平方向のバネ定数と関連量。"""

    e0: float  # 変形係数 (kN/m2)
    kh: float  # 水平方向地盤反力係数 (kN/m3)
    bh: float  # 換算載荷幅 (m)
    beta: float  # 特性値 (1/m)
    k1: float  # (kN/m)
    k2: float  # (kN/rad)
    k3: float  # (kN·m/m)
    k4: float  # (kN·m/rad)
    iterations: int
    beta_le: float  # βLe(半無限長の判定に用いる)

    @property
    def is_semi_infinite(self) -> bool:
        return self.beta_le >= 3.0


def axial_spring(pile: PileSpec, section: PileSection) -> float:
    """軸方向バネ定数 Kv (kN/m)(道示Ⅳ 12.6.1)。"""
    slope, intercept = KV_A_COEF[pile.method.value]
    a = slope * (pile.length / pile.diameter) + intercept
    if a <= 0:
        raise ValueError(
            f"係数 a が非正になりました (a={a:.3f})。L/D={pile.length / pile.diameter:.1f} "
            "が工法の適用範囲外の可能性があります"
        )
    return a * section.area * section.young / pile.length


def mean_e0(
    profile: SoilProfile, embedment: float, depth_range: float
) -> float:
    """杭頭直下 depth_range (m) 区間の変形係数 E0 の層厚加重平均 (kN/m2)。

    層に E0 が入力されていれば優先し、無ければ E0 = 2800N で推定する。
    """
    bottom = embedment + depth_range
    total, weighted = 0.0, 0.0
    for top, layer_bottom, layer in profile.layer_boundaries():
        seg_top = max(top, embedment)
        seg_bottom = min(layer_bottom, bottom)
        length = seg_bottom - seg_top
        if length <= 0:
            continue
        e0 = layer.e0 if layer.e0 is not None else E0_FROM_N * layer.n_value
        weighted += e0 * length
        total += length
    if total <= 0:
        raise ValueError("E0 の平均を取る区間が地盤モデル内にありません")
    return weighted / total


def lateral_springs(
    pile: PileSpec,
    section: PileSection,
    profile: SoilProfile,
    embedment: float,
    case: LoadCase,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LateralSprings:
    """kH・β と杭頭バネ定数 K1〜K4 を収束計算で求める。

    E0 の平均区間は 1/β 深さとし、β の更新に合わせて再評価する。
    """
    alpha = (
        ALPHA_KH_SEISMIC if case == LoadCase.LEVEL1_EQ else ALPHA_KH_NORMAL
    )
    d = pile.diameter
    ei = section.ei

    beta = 1.0  # 初期値 (1/m)
    kh = 0.0
    bh = 0.0
    e0 = 0.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        depth_range = min(1.0 / beta, profile.total_depth - embedment)
        e0 = mean_e0(profile, embedment, depth_range)
        kh0 = alpha * e0 / 0.3
        bh = math.sqrt(d / beta)
        kh = kh0 * (bh / 0.3) ** (-0.75)
        beta_new = (kh * d / (4.0 * ei)) ** 0.25
        if abs(beta_new - beta) < tol * max(1.0, beta):
            beta = beta_new
            break
        beta = beta_new
    else:
        raise RuntimeError(f"kH の収束計算が {max_iter} 回で収束しませんでした")

    # 収束後の値で kH・BH を再評価する
    depth_range = min(1.0 / beta, profile.total_depth - embedment)
    e0 = mean_e0(profile, embedment, depth_range)
    bh = math.sqrt(d / beta)
    kh = (alpha * e0 / 0.3) * (bh / 0.3) ** (-0.75)

    k1 = 4.0 * ei * beta**3
    k2 = -2.0 * ei * beta**2
    k4 = 2.0 * ei * beta
    return LateralSprings(
        e0=e0,
        kh=kh,
        bh=bh,
        beta=beta,
        k1=k1,
        k2=k2,
        k3=k2,
        k4=k4,
        iterations=iterations,
        beta_le=beta * pile.length,
    )
