"""杭のバネ定数(道示Ⅳ(H24) 9.6、12.6)。

軸方向バネ定数(12.6.1):
    Kv = a・Ap・Ep / L,  a = slope・(L/D) + intercept(工法別)

水平方向地盤反力係数(9.6):
    kH  = kH0・(BH/0.3)^(-3/4)
    kH0 = α・E0 / 0.3
    BH  = √(D/β),  β = ⁴√(kH・D / (4EI))
kH と β は相互に依存するため収束計算による。

杭頭バネ定数(12.6.2、半無限長の杭 βLe ≧ 3):
    K1 = 4EIβ³, K2 = K3 = −2EIβ², K4 = 2EIβ

.. note::
   地盤反力係数の条項番号は、第29回に提供された資料の記載に従い **9.6** と
   している(それ以前は 9.5.2 と書いていたが、これは照合していない記憶に
   よるものだった)。式そのものは変わらない。
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

from core.models.loads import LoadCase
from core.models.pile import PileSpec
from core.models.soil import SoilProfile

if TYPE_CHECKING:  # 循環インポートを避ける
    from core.soil.liquefaction import SoilReduction
from core.standards import (
    ALPHA_KH,
    E0_FROM_N,
    GROUP_PILE_KH_COEF,
    GROUP_PILE_SPACING_RATIO,
    KV_A_COEF,
    E0Method,
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
    alpha: float  # E0 → kH0 の換算係数
    kh: float  # 水平方向地盤反力係数 (kN/m3)
    bh: float  # 換算載荷幅 (m)
    beta: float  # 特性値 (1/m)
    k1: float  # (kN/m)
    k2: float  # (kN/rad)
    k3: float  # (kN·m/m)
    k4: float  # (kN·m/rad)
    iterations: int
    beta_le: float  # βLe(半無限長の判定に用いる)
    de: float = 1.0  # kH に乗じた液状化の低減係数(1.0 = 低減なし)
    group_factor: float = 1.0  # kH に乗じた群杭の補正係数 μ(1.0 = 補正なし)

    @property
    def is_semi_infinite(self) -> bool:
        return self.beta_le >= 3.0

    @property
    def is_group_corrected(self) -> bool:
        return self.group_factor < 1.0


def group_pile_factor(spacing: float, diameter: float) -> float:
    """群杭の水平方向地盤反力係数の補正係数 μ を返す。

        μ = 1 − 0.2 (2.5 − L/D)     (L < 2.5D)
        μ = 1                        (L ≧ 2.5D)

    ``spacing`` は杭中心間隔 L (m)、``diameter`` は杭径 D (m)。矩形配置で
    方向により間隔が異なる場合は、**小さいほうの間隔**を与える(μ が小さく
    なり安全側)。

    .. warning::
       本補正は**線形の地盤反力係数を用いる場合**のものである。基礎地盤の
       非線形性を考慮する場合(分布バネモデル)には適用しない。
       また群杭影響のもう一方の柱である「仮想ケーソン基礎とみなした押込み
       支持力の上限」は本ソフトでは未実装である。
    """
    if spacing <= 0 or diameter <= 0:
        raise ValueError("杭中心間隔・杭径は正の値である必要があります")
    ratio = spacing / diameter
    if ratio >= GROUP_PILE_SPACING_RATIO:
        return 1.0
    return 1.0 - GROUP_PILE_KH_COEF * (GROUP_PILE_SPACING_RATIO - ratio)


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


def _de_factor(
    reduction: "SoilReduction | None", embedment: float, depth_range: float
) -> float:
    """kH に乗じる低減係数。``reduction`` が無ければ 1.0。"""
    if reduction is None:
        return 1.0
    return reduction.mean_factor(embedment, embedment + depth_range)


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
    e0_method: E0Method = E0Method.N_VALUE,
    reduction: "SoilReduction | None" = None,
    group_factor: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LateralSprings:
    """kH・β と杭頭バネ定数 K1〜K4 を求める(道示Ⅳ 9.6)。

    手順は次のとおり(提供資料の「計算手順」に従う)。

    1. **常時(α = 1)** の条件で kH・β・BH が整合するまで反復計算する。
       E0 の平均区間は 1/β 深さとし、β の更新に合わせて再評価する。
    2. 得られた **BH を固定**し、当該荷重ケースの α(地震時は 2)を用いて
       kH を算定する。液状化の低減 DE はこの段階で乗じる。
    3. その kH から β を求め直し、K1〜K4 を算定する。

    .. important::
       **BH を求める kH は常時の値を用いる**。地震時の α で BH まで反復
       し直すと BH が小さくなり、kH が約 7%、K1 が約 6% 過大になる
       (地盤を硬く評価する = 非安全側)。第29回で修正した。

    ``e0_method`` は変形係数 E0 の推定方法。α の値がこれにより決まる
    (N値・平板載荷は常時1/地震時2、孔内水平載荷・室内試験は 4/8)。

    ``reduction`` を与えると、液状化に伴う土質定数の低減係数 DE を kH に
    乗じる(道示Ⅴ 8.2.4)。DE は E0 と同じ区間(常時の 1/β)で層厚加重
    平均する。**BH の決定には DE を効かせない**(BH は常時の条件で定める
    ため)。一方 β は低減後の kH から求め直すので、kH の低下は β の低下、
    ひいては地中部最大曲げモーメント位置の深部移動として現れる。

    ``group_factor`` は群杭の補正係数 μ(:func:`group_pile_factor`)。
    μ は常時にも効くので、BH を定める反復計算の内側で乗じる。
    """
    if not 0.0 < group_factor <= 1.0:
        raise ValueError("群杭の補正係数 μ は 0 < μ ≦ 1 である必要があります")
    alpha_normal, alpha_seismic = ALPHA_KH[e0_method]
    alpha = alpha_seismic if case == LoadCase.LEVEL1_EQ else alpha_normal
    d = pile.diameter
    ei = section.ei

    # --- 手順1: 常時の条件で BH を定める(DE は効かせない)---------------
    beta = 1.0  # 初期値 (1/m)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        depth_range = min(1.0 / beta, profile.total_depth - embedment)
        e0 = mean_e0(profile, embedment, depth_range)
        bh = math.sqrt(d / beta)
        kh_ref = (alpha_normal * e0 / 0.3) * (bh / 0.3) ** (-0.75) * group_factor
        beta_new = (kh_ref * d / (4.0 * ei)) ** 0.25
        if abs(beta_new - beta) < tol * max(1.0, beta):
            beta = beta_new
            break
        beta = beta_new
    else:
        raise RuntimeError(f"kH の収束計算が {max_iter} 回で収束しませんでした")

    # 収束後の値で BH・E0 を確定する。以降この BH は変えない
    depth_range = min(1.0 / beta, profile.total_depth - embedment)
    e0 = mean_e0(profile, embedment, depth_range)
    bh = math.sqrt(d / beta)

    # --- 手順2: 当該荷重ケースの α と DE で kH を求める -------------------
    de = _de_factor(reduction, embedment, depth_range)
    if de <= 0.0:
        raise ValueError(
            f"杭頭直下 {depth_range:.2f} m の区間が全て液状化と判定され"
            "(DE = 0)、水平地盤反力係数 kH が 0 になりました。"
            "杭頭バネ K1〜K4 は弾性床上の梁(Chang の式)を前提とするため、"
            "この状態では適用できません。分布バネモデル(BNWF)であれば"
            "節点ごとに扱えるため、レベル2の照査を用いてください"
        )
    kh = (alpha * e0 / 0.3) * (bh / 0.3) ** (-0.75) * de * group_factor

    # --- 手順3: kH から β と K1〜K4 を求める ------------------------------
    beta = (kh * d / (4.0 * ei)) ** 0.25
    k1 = 4.0 * ei * beta**3
    k2 = -2.0 * ei * beta**2
    k4 = 2.0 * ei * beta
    return LateralSprings(
        e0=e0,
        alpha=alpha,
        kh=kh,
        bh=bh,
        beta=beta,
        k1=k1,
        k2=k2,
        k3=k2,
        k4=k4,
        iterations=iterations,
        beta_le=beta * pile.length,
        de=de,
        group_factor=group_factor,
    )
