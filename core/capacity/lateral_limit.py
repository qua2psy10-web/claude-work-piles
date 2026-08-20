"""杭前面地盤の水平地盤反力度の上限値 pHU(道示Ⅳ(H24) 12.10)。

レベル2地震時には、地盤抵抗を弾塑性バネで表す。初期勾配が水平方向地盤反力
係数 kH、降伏後の頭打ちが本モジュールの pHU である。

    pHU = ηp・αp・pU

pHU は**反力度** (kN/m²) であり合力ではない。バネの力の上限にするには、
そのバネが受け持つ地盤抵抗面積(杭径 × 分担長)を掛ける。

受働土圧強度 pU
---------------
    pU = KEP・Σ(γ'ti・hi)+ 2c・√KEP

このうち **KEP(地震時クーロン受働土圧係数)は本ソフトでは算定しない**。
壁面摩擦角 δE、地表面勾配、設計水平震度を含む算定式であり、原典との照合が
できていないためである。利用者が層ごとに :attr:`SoilLayer.k_ep` として
与える。

.. important::
   ケーソン・フーチング前面(**基礎前面地盤**)の pHU は
   pHU = αp・pEP、αp = min(1 + 0.5 z/Be, 3.0) という別の規定である。
   本ソフトは基礎前面の水平抵抗を考慮しないため実装していない。
   同じ記号 pHU でも混用してはならない。
"""
from __future__ import annotations

import math

from core.models.soil import SoilLayer, SoilProfile, SoilType
from core.standards import (
    ALPHA_P_PILE,
    ALPHA_P_SOFT_CLAY,
    NON_FRONT_ROW_FACTOR_SAND,
    SOFT_CLAY_N_THRESHOLD,
)


def alpha_p(layer: SoilLayer) -> float:
    """単杭としての割増し係数 αp。

    砂質土・礫質土は 3.0、粘性土は 1.5。ただし N ≤ 2 の軟弱な粘性土は 1.0。
    """
    if (
        layer.soil_type == SoilType.CLAY
        and layer.n_value <= SOFT_CLAY_N_THRESHOLD
    ):
        return ALPHA_P_SOFT_CLAY
    return ALPHA_P_PILE[layer.soil_type.value]


def eta_p_alpha_p(
    layer: SoilLayer, diameter: float, spacing_perpendicular: float
) -> float:
    """群杭効果を含む係数 ηp・αp。

    砂質地盤では ηp・αp = min(s / D, αp)。s は**載荷直交方向**の杭中心間隔。
    粘性土地盤では ηp = 1.0 とし、αp をそのまま用いる。
    """
    a = alpha_p(layer)
    if layer.soil_type == SoilType.CLAY:
        return a
    if diameter <= 0:
        raise ValueError("杭径は正の値である必要があります")
    return min(spacing_perpendicular / diameter, a)


def passive_pressure(profile: SoilProfile, depth: float) -> float:
    """深度 ``depth`` における受働土圧強度 pU (kN/m²)。

        pU = KEP・σ'v + 2c・√KEP

    KEP は層ごとの :attr:`SoilLayer.k_ep`(利用者入力)を用いる。
    未入力の層ではエラーとする(既定値を置くと危険側にも安全側にも
    誤り得るため)。
    """
    layer = profile.layer_at(depth)
    if layer.k_ep is None:
        raise ValueError(
            f"層「{layer.name or '無名'}」の地震時受働土圧係数 KEP が未入力です。"
            "pHU の算定には層ごとの KEP が必要です"
            "(道示の算定式は本ソフトでは実装していません)"
        )
    _, sigma_v_eff = profile.stresses_at(depth)
    cohesion = layer.cohesion or 0.0
    return layer.k_ep * sigma_v_eff + 2.0 * cohesion * math.sqrt(layer.k_ep)


def p_hu(
    profile: SoilProfile,
    depth: float,
    diameter: float,
    spacing_perpendicular: float,
    front_row: bool = True,
) -> float:
    """深度 ``depth`` における水平地盤反力度の上限値 pHU (kN/m²)。

    Parameters
    ----------
    spacing_perpendicular:
        載荷直交方向の杭中心間隔 (m)。
    front_row:
        最前列の杭か。砂質地盤では最前列以外は 1/2 とする。
    """
    layer = profile.layer_at(depth)
    value = eta_p_alpha_p(layer, diameter, spacing_perpendicular) * passive_pressure(
        profile, depth
    )
    if not front_row and layer.soil_type != SoilType.CLAY:
        value *= NON_FRONT_ROW_FACTOR_SAND
    return value


def spring_force_limit(
    profile: SoilProfile,
    depth: float,
    diameter: float,
    spacing_perpendicular: float,
    tributary_length: float,
    front_row: bool = True,
) -> float:
    """深度 ``depth`` のバネが受け持てる水平力の上限 (kN)。

        R_max = pHU ×(杭径 D × 分担長 Δz)

    反力度 (kN/m²) を力 (kN) に換算する際の取り違えを防ぐため、換算を
    本関数に閉じ込めている。
    """
    if tributary_length <= 0:
        raise ValueError("分担長は正の値である必要があります")
    return (
        p_hu(profile, depth, diameter, spacing_perpendicular, front_row)
        * diameter
        * tributary_length
    )
