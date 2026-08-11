"""液状化の判定(道示Ⅴ(H24) 8.2)。

H24年道示では、液状化の判定はレベル2地震動(タイプI・タイプII)に対して行う。

判定フロー:
  1. 判定対象層のスクリーニング(8.2.2)
  2. FL = R / L の算定(8.2.3)
       R = cw・RL          (動的せん断強度比)
       L = rd・khg・σv/σ'v (地震時せん断応力比)
  3. FL ≦ 1.0 の層について土質定数の低減係数 DE を決定(8.2.4 表-8.2.1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.soil import SoilLayer, SoilProfile, SoilType
from core.standards import (
    DE_TABLE,
    KHG0_LIQUEFACTION,
    LIQUEFACTION_MAX_D10,
    LIQUEFACTION_MAX_D50,
    LIQUEFACTION_MAX_DEPTH,
    LIQUEFACTION_MAX_FC,
    LIQUEFACTION_MAX_GWL,
    LIQUEFACTION_MAX_IP,
    GroundMotionType,
    GroundType,
)


def n1_value(n: float, sigma_v_eff: float) -> float:
    """有効上載圧 100kN/m2 相当に換算したN値 N1(道示Ⅴ 8.2.3)

    N1 = 170・N / (σ'v + 70)
    """
    return 170.0 * n / (sigma_v_eff + 70.0)


def na_sand(n1: float, fc: float) -> float:
    """砂質土の粒度の影響を考慮した補正N値 Na(道示Ⅴ 8.2.3)"""
    if fc < 10.0:
        c1, c2 = 1.0, 0.0
    elif fc < 60.0:
        c1, c2 = (fc + 40.0) / 50.0, (fc - 10.0) / 18.0
    else:
        c1, c2 = fc / 20.0 - 1.0, (fc - 10.0) / 18.0
    return c1 * n1 + c2


def na_gravel(n1: float, d50: float) -> float:
    """礫質土の補正N値 Na(道示Ⅴ 8.2.3)

    Na = {1 − 0.36・log10(D50/2)}・N1
    """
    return (1.0 - 0.36 * math.log10(d50 / 2.0)) * n1


def rl_value(na: float) -> float:
    """繰返し三軸強度比 RL(道示Ⅴ 8.2.3)"""
    rl = 0.0882 * math.sqrt(na / 1.7)
    if na >= 14.0:
        rl += 1.6e-6 * (na - 14.0) ** 4.5
    return rl


def cw_value(rl: float, motion: GroundMotionType) -> float:
    """地震動特性による補正係数 cw(道示Ⅴ 8.2.3)"""
    if motion == GroundMotionType.LEVEL2_TYPE1:
        return 1.0
    # タイプII
    if rl <= 0.1:
        return 1.0
    if rl <= 0.4:
        return 3.3 * rl + 0.67
    return 2.0


def rd_value(depth: float) -> float:
    """地震時せん断応力比の深さ方向低減係数 rd = 1.0 − 0.015x"""
    return 1.0 - 0.015 * depth


def reduction_factor_de(fl: float, depth: float, r: float) -> float:
    """土質定数の低減係数 DE(道示Ⅴ(H24) 表-8.2.1)

    FL > 1.0(液状化しない)の場合は低減しない(DE = 1.0)。
    """
    if fl > 1.0:
        return 1.0
    fl_idx = 0 if fl <= 1.0 / 3.0 else (1 if fl <= 2.0 / 3.0 else 2)
    depth_idx = 0 if depth <= 10.0 else 1
    r_idx = 0 if r <= 0.3 else 1
    return DE_TABLE[(fl_idx, depth_idx, r_idx)]


def _screening_reason(
    layer: SoilLayer, depth: float, gwl: float
) -> str | None:
    """判定対象層のスクリーニング(道示Ⅴ 8.2.2)。対象なら None、対象外なら理由。"""
    if gwl > LIQUEFACTION_MAX_GWL:
        return f"地下水位が地表面から{LIQUEFACTION_MAX_GWL:.0f}mより深い"
    if depth > LIQUEFACTION_MAX_DEPTH:
        return f"深度{LIQUEFACTION_MAX_DEPTH:.0f}m超"
    if depth < gwl:
        return "地下水位以浅(非飽和)"
    if layer.soil_type == SoilType.CLAY:
        return "粘性土(砂質土・礫質土でない)"
    if not layer.is_alluvial:
        return "沖積層でない"
    if layer.fc is None:
        return "細粒分含有率FCが未入力"
    if layer.fc > LIQUEFACTION_MAX_FC and (
        layer.ip is None or layer.ip > LIQUEFACTION_MAX_IP
    ):
        return f"FC>{LIQUEFACTION_MAX_FC:.0f}%かつIP>{LIQUEFACTION_MAX_IP:.0f}"
    if layer.soil_type == SoilType.GRAVEL and layer.d50 is None:
        return "礫質土でD50が未入力"
    if layer.d50 is not None and layer.d50 > LIQUEFACTION_MAX_D50:
        return f"D50>{LIQUEFACTION_MAX_D50:.0f}mm"
    if layer.d10 is not None and layer.d10 > LIQUEFACTION_MAX_D10:
        return f"D10>{LIQUEFACTION_MAX_D10:.0f}mm"
    return None


@dataclass(frozen=True)
class SliceResult:
    """判定単位(スライス)ごとの結果。対象外の場合は数値フィールドが None。"""

    depth_top: float
    depth_bottom: float
    depth_mid: float
    layer_name: str
    soil_type: SoilType
    is_target: bool
    excluded_reason: str | None
    sigma_v: float | None = None
    sigma_v_eff: float | None = None
    n1: float | None = None
    na: float | None = None
    rl: float | None = None
    # タイプI
    l_type1: float | None = None
    r_type1: float | None = None
    fl_type1: float | None = None
    de_type1: float = 1.0
    # タイプII
    l_type2: float | None = None
    r_type2: float | None = None
    fl_type2: float | None = None
    de_type2: float = 1.0


@dataclass(frozen=True)
class LiquefactionAssessment:
    slices: list[SliceResult]
    liquefiable_type1: bool  # タイプIで FL≦1 の層があるか
    liquefiable_type2: bool


def evaluate_at(
    profile: SoilProfile,
    depth: float,
    ground_type: GroundType,
    cz_type1: float = 1.0,
    cz_type2: float = 1.0,
    depth_top: float | None = None,
    depth_bottom: float | None = None,
) -> SliceResult:
    """1深度についてFL・DEを算定する。"""
    layer = profile.layer_at(depth)
    top = depth if depth_top is None else depth_top
    bottom = depth if depth_bottom is None else depth_bottom
    reason = _screening_reason(layer, depth, profile.gwl)
    if reason is not None:
        return SliceResult(
            depth_top=top,
            depth_bottom=bottom,
            depth_mid=depth,
            layer_name=layer.name,
            soil_type=layer.soil_type,
            is_target=False,
            excluded_reason=reason,
        )

    sigma_v, sigma_v_eff = profile.stresses_at(depth)
    n1 = n1_value(layer.n_value, sigma_v_eff)
    if layer.soil_type == SoilType.GRAVEL:
        na = na_gravel(n1, layer.d50)  # type: ignore[arg-type]  # スクリーニング済み
    else:
        na = na_sand(n1, layer.fc)  # type: ignore[arg-type]  # スクリーニング済み
    rl = rl_value(na)
    rd = rd_value(depth)
    stress_ratio = sigma_v / sigma_v_eff

    per_motion: dict[GroundMotionType, tuple[float, float, float, float]] = {}
    for motion, cz in (
        (GroundMotionType.LEVEL2_TYPE1, cz_type1),
        (GroundMotionType.LEVEL2_TYPE2, cz_type2),
    ):
        khg = cz * KHG0_LIQUEFACTION[motion][ground_type]
        l_val = rd * khg * stress_ratio
        r_val = cw_value(rl, motion) * rl
        fl = r_val / l_val
        de = reduction_factor_de(fl, depth, r_val)
        per_motion[motion] = (l_val, r_val, fl, de)

    l1, r1, fl1, de1 = per_motion[GroundMotionType.LEVEL2_TYPE1]
    l2, r2, fl2, de2 = per_motion[GroundMotionType.LEVEL2_TYPE2]
    return SliceResult(
        depth_top=top,
        depth_bottom=bottom,
        depth_mid=depth,
        layer_name=layer.name,
        soil_type=layer.soil_type,
        is_target=True,
        excluded_reason=None,
        sigma_v=sigma_v,
        sigma_v_eff=sigma_v_eff,
        n1=n1,
        na=na,
        rl=rl,
        l_type1=l1,
        r_type1=r1,
        fl_type1=fl1,
        de_type1=de1,
        l_type2=l2,
        r_type2=r2,
        fl_type2=fl2,
        de_type2=de2,
    )


def assess_liquefaction(
    profile: SoilProfile,
    ground_type: GroundType,
    cz_type1: float = 1.0,
    cz_type2: float = 1.0,
    pitch: float = 1.0,
) -> LiquefactionAssessment:
    """地盤モデル全体の液状化判定。

    各層を pitch (m) 以下の等分スライスに分割し、スライス中央深度で評価する。
    """
    slices: list[SliceResult] = []
    for top, bottom, _layer in profile.layer_boundaries():
        n_div = max(1, math.ceil((bottom - top) / pitch - 1e-9))
        dz = (bottom - top) / n_div
        for i in range(n_div):
            s_top = top + i * dz
            s_bottom = s_top + dz
            mid = (s_top + s_bottom) / 2.0
            slices.append(
                evaluate_at(
                    profile,
                    mid,
                    ground_type,
                    cz_type1,
                    cz_type2,
                    depth_top=s_top,
                    depth_bottom=s_bottom,
                )
            )
    liq1 = any(s.fl_type1 is not None and s.fl_type1 <= 1.0 for s in slices)
    liq2 = any(s.fl_type2 is not None and s.fl_type2 <= 1.0 for s in slices)
    return LiquefactionAssessment(
        slices=slices, liquefiable_type1=liq1, liquefiable_type2=liq2
    )
