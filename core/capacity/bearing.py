"""杭の軸方向許容支持力(道示Ⅳ(H24) 12.4)。

極限支持力:
    Ru = qd・A + U・Σ Li・fi
許容支持力(押込み):
    Ra = (1/n)・(Ru − Ws) + Ws − W
許容引抜き力:
    Pa = (1/n)・Ruf + W        (Ruf は周面摩擦力のみ)

    n : 安全率(道示Ⅳ 表-12.4.1: 押込み / 表-12.4.3: 引抜き)
    W : 杭および杭内部の土の有効重量
    Ws: 杭で置換される部分の土の有効重量

.. warning::
   qd・f の推定式(:mod:`core.standards` の ``QD_SPECS`` / ``F_SPECS``)は
   道示Ⅳ 表-12.4.2(推定)の値を実装しているが、**実務適用前に原典との照合が必要**。
   詳細は ``docs/VERIFICATION.md`` を参照。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models.loads import LoadCase
from core.models.pile import ConstructionMethod, PileSpec, SupportType
from core.models.soil import SoilLayer, SoilProfile, SoilType
from core.standards import (
    F_MAX,
    F_SPECS,
    GAMMA_W,
    MIN_N_FOR_CLAY_FRICTION_FROM_N,
    QD_SPECS,
    SAFETY_FACTORS_PULL,
    SAFETY_FACTORS_PUSH,
    TIP_TREATMENT_QD_SOURCE,
    TipTreatment,
)


@dataclass(frozen=True)
class SkinFrictionSegment:
    """周面摩擦力の計算単位(1層分)。"""

    layer_name: str
    soil_type: SoilType
    length: float  # 杭が貫入する長さ (m)
    f: float  # 最大周面摩擦力度 (kN/m2)
    force: float  # U・Li・fi (kN)


@dataclass(frozen=True)
class BearingCapacity:
    """1本杭の軸方向支持力。"""

    qd: float  # 先端の極限支持力度 (kN/m2)
    tip_area: float  # 杭先端面積 Ap (m2)
    tip_resistance: float  # qd・A (kN)
    skin_segments: list[SkinFrictionSegment] = field(default_factory=list)
    skin_resistance: float = 0.0  # U・ΣLi・fi (kN)
    ru: float = 0.0  # 極限支持力 (kN)
    w_pile: float = 0.0  # 杭+内部土の有効重量 W (kN)
    w_soil: float = 0.0  # 置換土の有効重量 Ws (kN)
    n_tip: float = 0.0  # qd の算定に用いた先端付近の平均N値
    skin_bottom_depth: float = 0.0  # 周面摩擦を計上した下端深度 (m)
    tip_zone_excluded: bool = True  # 先端 1D 区間を除外したか
    support_type: SupportType = SupportType.END_BEARING  # 支持形式(安全率に影響)

    def safety_factor_push(self, case: LoadCase) -> float:
        """押込みの安全率 n(支持形式により異なる)。"""
        return SAFETY_FACTORS_PUSH[case.value][self.support_type.value]

    def safety_factor_pull(self, case: LoadCase) -> float:
        """引抜きの安全率 n。"""
        return SAFETY_FACTORS_PULL[case.value]

    def allowable_push(self, case: LoadCase) -> float:
        """許容押込み支持力 Ra (kN)"""
        n = self.safety_factor_push(case)
        return (self.ru - self.w_soil) / n + self.w_soil - self.w_pile

    def allowable_pull(self, case: LoadCase) -> float:
        """許容引抜き力 Pa (kN)。

            Pa =(1/n)・Ruf + W

        Ruf は周面摩擦力のみ(引抜きでは杭先端の地盤抵抗は期待できない)、
        W は杭の水中有効重量。この式形は提供解説資料の
        ``Rat =(1/n)× Rtu + Wp`` と一致することを確認済み。

        .. note::
           フーチング上の土の重量も引抜きに抵抗する側として評価できるが、
           本実装では算入していない(安全側)。また群杭のブロック破壊は
           照査していない。
        """
        n = self.safety_factor_pull(case)
        return self.skin_resistance / n + self.w_pile


def _method_key(method: ConstructionMethod) -> str:
    return method.value


def qd_method_key(
    method: ConstructionMethod, tip_treatment: TipTreatment | None = None
) -> str:
    """qd の算定に用いる工法キーを返す。

    中掘り杭工法は先端処理方式により算定法が変わる(道示Ⅳ):
      最終打撃方式        → 打込み杭の算定法
      セメントミルク噴出攪拌方式 → 中掘り杭の値
      コンクリート打設方式  → 場所打ち杭の値
    """
    if method == ConstructionMethod.INNER_DIGGING:
        treatment = tip_treatment or TipTreatment.CEMENT_MILK
        return TIP_TREATMENT_QD_SOURCE[treatment]
    return method.value


def tip_resistance_intensity(
    method: ConstructionMethod,
    layer: SoilLayer,
    n_tip: float,
    tip_treatment: TipTreatment | None = None,
) -> float:
    """杭先端の極限支持力度 qd (kN/m2)(道示Ⅳ)。

    ``n_tip`` は杭先端付近の平均N値。``tip_treatment`` は中掘り杭の
    先端処理方式(省略時はセメントミルク噴出攪拌方式)。
    """
    key_method = qd_method_key(method, tip_treatment)
    spec_by_soil = QD_SPECS[key_method]
    key = layer.soil_type.value
    if key not in spec_by_soil:
        # 支持層として想定していない土質は明示的にエラーとする
        raise ValueError(
            f"{method.value}では{layer.soil_type.value}を支持層にできません"
        )
    spec = spec_by_soil[key]

    if spec.kind == "N":
        qd = spec.coef * n_tip
    elif spec.kind == "qu":
        if layer.cohesion is None:
            raise ValueError(f"層「{layer.name}」の粘着力 c が未入力です")
        qd = spec.coef * (2.0 * layer.cohesion)  # qu = 2c
    elif spec.kind == "steps":
        for threshold, value in spec.steps:
            if n_tip >= threshold:
                qd = value
                break
        else:
            minimum = min(t for t, _ in spec.steps)
            raise ValueError(
                f"層「{layer.name}」は N={n_tip:.1f} で、{key_method}杭の"
                f"{key}支持層に必要な N ≧ {minimum:.0f} を満たしません"
            )
    else:  # pragma: no cover - 定義ミス時のみ
        raise ValueError(f"未知の qd 種別: {spec.kind}")

    return min(qd, spec.cap) if spec.cap is not None else qd


def skin_friction_intensity(
    method: ConstructionMethod, layer: SoilLayer
) -> float:
    """最大周面摩擦力度 f (kN/m2)(道示Ⅳ 表-12.4.2 と推定)。

    粘性土は「c または 10N」であり、両者を加算するものではない。
    c が入力されていればそれを、無ければ 10N を用いる。

    Raises
    ------
    ValueError
        N < 5 の軟弱粘性土層で粘着力 c が未入力の場合。この範囲では
        N 値による推定の信頼性が乏しく、道示は N 値式を用いないとしている。
    """
    spec = F_SPECS[_method_key(method)]
    key = layer.soil_type.value
    coef, kind = spec[key]
    if kind == "N":
        f = coef * layer.n_value
    else:  # "c"
        if layer.cohesion is None:
            if layer.n_value < MIN_N_FOR_CLAY_FRICTION_FROM_N:
                raise ValueError(
                    f"層「{layer.name}」は N={layer.n_value:.1f} の軟弱粘性土です。"
                    f"N < {MIN_N_FOR_CLAY_FRICTION_FROM_N:.0f} では N 値による"
                    "周面摩擦力度の推定を行わないため、土質試験による粘着力 c を"
                    "入力してください(道示Ⅳ の注記)"
                )
            # c が未入力の場合は c = 10N の目安を用いる
            f = coef * 10.0 * layer.n_value
        else:
            f = coef * layer.cohesion
    limit = F_MAX[_method_key(method)].get(key)
    return min(f, limit) if limit is not None else f


def average_n_near_tip(
    profile: SoilProfile, tip_depth: float, diameter: float
) -> float:
    """杭先端から上方 1D・下方 1D の範囲の平均N値(道示Ⅳ 12.4.1)。

    杭先端の極限支持力度 qd の算定に用いるN値は、先端位置の1点ではなく
    先端近傍の平均値を用いる。地盤モデルの範囲外は評価区間から除外する。
    """
    top = max(0.0, tip_depth - diameter)
    bottom = min(profile.total_depth, tip_depth + diameter)
    if bottom <= top:
        return profile.layer_at(tip_depth).n_value
    weighted = 0.0
    for layer_top, layer_bottom, layer in profile.layer_boundaries():
        seg_top = max(layer_top, top)
        seg_bottom = min(layer_bottom, bottom)
        if seg_bottom <= seg_top:
            continue
        weighted += layer.n_value * (seg_bottom - seg_top)
    return weighted / (bottom - top)


def compute_bearing_capacity(
    pile: PileSpec,
    profile: SoilProfile,
    embedment: float,
    n_tip: float | None = None,
    inner_soil: bool = False,
    exclude_tip_zone: bool = True,
) -> BearingCapacity:
    """1本杭の軸方向支持力を算定する。

    Parameters
    ----------
    embedment:
        地表面から杭頭(フーチング下面)までの深さ (m)。
    n_tip:
        杭先端付近の平均N値。省略時は先端から上下 1D の範囲の平均値
        (:func:`average_n_near_tip`)を用いる。
    inner_soil:
        中空杭で内部の土を重量に算入するか(閉端杭は True 相当)。
    exclude_tip_zone:
        杭先端から上方 1D の区間の周面摩擦力を計上しないか(道示Ⅳ 12.4.1)。
        載荷試験に基づく qd には先端近傍の周面摩擦の寄与が既に含まれるため、
        重複計上を避ける規定。既定で有効。
    """
    d = pile.diameter
    tip_depth = embedment + pile.length
    if tip_depth > profile.total_depth:
        raise ValueError(
            f"杭先端深度 {tip_depth:.1f} m が地盤モデル深さ "
            f"{profile.total_depth:.1f} m を超えています"
        )

    area = math.pi * d**2 / 4.0
    perimeter = math.pi * d
    tip_layer = profile.layer_at(tip_depth)
    n_value_tip = (
        average_n_near_tip(profile, tip_depth, d) if n_tip is None else n_tip
    )
    qd = tip_resistance_intensity(
        pile.method, tip_layer, n_value_tip, pile.tip_treatment
    )

    # 周面摩擦を計上する下端(先端から 1D 手前で打ち切る)
    skin_bottom = tip_depth - d if exclude_tip_zone else tip_depth
    skin_bottom = max(skin_bottom, embedment)

    segments: list[SkinFrictionSegment] = []
    skin = 0.0
    for top, bottom, layer in profile.layer_boundaries():
        seg_top = max(top, embedment)
        seg_bottom = min(bottom, skin_bottom)
        length = seg_bottom - seg_top
        if length <= 0:
            continue
        f = skin_friction_intensity(pile.method, layer)
        force = perimeter * length * f
        skin += force
        segments.append(
            SkinFrictionSegment(
                layer_name=layer.name,
                soil_type=layer.soil_type,
                length=length,
                f=f,
                force=force,
            )
        )

    ru = qd * area + skin
    w_pile = _pile_effective_weight(pile, profile, embedment, tip_depth, inner_soil)
    w_soil = _replaced_soil_weight(profile, embedment, tip_depth, area)
    return BearingCapacity(
        qd=qd,
        tip_area=area,
        tip_resistance=qd * area,
        skin_segments=segments,
        skin_resistance=skin,
        ru=ru,
        w_pile=w_pile,
        w_soil=w_soil,
        n_tip=n_value_tip,
        skin_bottom_depth=skin_bottom,
        tip_zone_excluded=exclude_tip_zone,
        support_type=pile.support_type,
    )


# 杭材の単位体積重量 (kN/m3)
GAMMA_CONCRETE = 24.5
GAMMA_STEEL = 77.0


def _pile_effective_weight(
    pile: PileSpec,
    profile: SoilProfile,
    embedment: float,
    tip_depth: float,
    inner_soil: bool,
) -> float:
    """杭および杭内部の土の有効重量 W (kN)。地下水位以深は浮力を控除する。"""
    area = math.pi * pile.diameter**2 / 4.0
    if pile.wall_thickness is not None:
        t = pile.wall_thickness / 1000.0
        steel_area = math.pi * (pile.diameter - t) * t
        inner_area = area - steel_area
    else:
        steel_area = 0.0
        inner_area = 0.0

    weight = 0.0
    # 杭体
    for depth_top, depth_bottom, gamma_reduction in _submergence_split(
        profile, embedment, tip_depth
    ):
        length = depth_bottom - depth_top
        if pile.wall_thickness is None:
            gamma = GAMMA_CONCRETE - gamma_reduction
            weight += area * length * gamma
        else:
            weight += steel_area * length * (GAMMA_STEEL - gamma_reduction)
            if inner_soil:
                layer = profile.layer_at((depth_top + depth_bottom) / 2.0)
                gamma_soil = (
                    layer.gamma_sat - gamma_reduction
                    if gamma_reduction > 0
                    else layer.gamma_t
                )
                weight += inner_area * length * gamma_soil
    return weight


def _replaced_soil_weight(
    profile: SoilProfile, embedment: float, tip_depth: float, area: float
) -> float:
    """杭で置換される部分の土の有効重量 Ws (kN)。"""
    weight = 0.0
    for top, bottom, layer in profile.layer_boundaries():
        seg_top = max(top, embedment)
        seg_bottom = min(bottom, tip_depth)
        if seg_bottom <= seg_top:
            continue
        above = max(0.0, min(seg_bottom, profile.gwl) - seg_top)
        below = (seg_bottom - seg_top) - above
        weight += area * (above * layer.gamma_t + below * (layer.gamma_sat - GAMMA_W))
    return weight


def _submergence_split(
    profile: SoilProfile, top: float, bottom: float
) -> list[tuple[float, float, float]]:
    """区間を地下水位で分割し、(上端, 下端, 浮力による単位重量控除) を返す。"""
    result = []
    gwl = profile.gwl
    if top < min(bottom, gwl):
        result.append((top, min(bottom, gwl), 0.0))
    if bottom > max(top, gwl):
        result.append((max(top, gwl), bottom, GAMMA_W))
    return result
