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

液状化に伴う低減(道示Ⅴ 8.2)
----------------------------
``reduction`` を与えると、液状化が生じると判定された層の**最大周面摩擦力度に
低減係数 DE を乗じる**(f′i = DE,i × fi)。押込み・引抜きの双方に適用する。
先端支持力度 qd は低減しない(支持層は液状化しない良質層であることが前提)。

**この低減は耐震設計上の扱いであり、常時の許容支持力照査には適用しない。**
詳細は :func:`compute_bearing_capacity` の docstring を参照。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.models.loads import LoadCase

if TYPE_CHECKING:  # 循環インポートを避ける
    from core.soil.liquefaction import SoilReduction
from core.models.pile import ConstructionMethod, PileSpec, PileType, SupportType
from core.models.soil import SoilLayer, SoilProfile, SoilType
from core.standards import (
    DEFAULT_WING_RATIO,
    F_MAX,
    F_SPECS,
    GAMMA_W,
    MIN_N_FOR_CLAY_FRICTION_FROM_N,
    QD_SPECS,
    QD_SPECS_ROTARY,
    QdSpec,
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
    force: float  # U・Li・fi (kN)。低減がある場合は低減後の値
    de: float = 1.0  # 液状化に伴う土質定数の低減係数(区間平均)

    @property
    def f_design(self) -> float:
        """低減後の周面摩擦力度 (kN/m2)。"""
        return self.f * self.de

    @property
    def is_reduced(self) -> bool:
        return self.de < 1.0


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
    tip_de: float = 1.0  # 先端付近(±1D)の DE。1.0 未満なら支持層が液状化する
    # 低減を適用しなかった場合の周面摩擦力 (kN)。低減量の把握に用いる。
    # :func:`compute_bearing_capacity` は常にこの値を設定する。
    skin_resistance_unreduced: float = 0.0

    @property
    def has_reduced_skin(self) -> bool:
        return any(s.is_reduced for s in self.skin_segments)

    @property
    def tip_zone_liquefies(self) -> bool:
        """杭先端付近が液状化すると判定されているか。

        本実装は先端支持力度 qd に DE を乗じていない(下記 :func:`compute_bearing_capacity`
        の注記を参照)ため、この場合は利用者に判断を促す必要がある。
        """
        return self.tip_de < 1.0

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


def tip_area(pile: PileSpec) -> float:
    """極限支持力式 Ru = qd・A + … に用いる杭先端面積 A (m2)(道示Ⅳ)。

    工法により基準とする径が異なる:

    * 鋼管ソイルセメント杭 — **ソイルセメント柱**の断面積(鋼管断面ではない)
    * 回転杭 — **先端羽根の投影面積 Aw**(羽根外径 = wing_ratio × 杭径)
    * その他 — 杭径による円の面積
    """
    if pile.pile_type == PileType.H_STEEL:
        raise NotImplementedError(
            "H鋼杭の先端面積・周長は円形断面では表せません"
            "(H形断面の寸法入力が必要)。未実装です"
        )
    if pile.pile_type == PileType.STEEL_PIPE_SOIL_CEMENT or (
        pile.method == ConstructionMethod.STEEL_PIPE_SOIL_CEMENT
    ):
        if pile.soil_cement_diameter is None:
            raise ValueError(
                "鋼管ソイルセメント杭の先端面積にはソイルセメント柱径 "
                "(soil_cement_diameter)の入力が必要です"
            )
        diameter = pile.soil_cement_diameter
    elif pile.method == ConstructionMethod.ROTARY:
        ratio = pile.wing_ratio or DEFAULT_WING_RATIO
        diameter = ratio * pile.diameter
    else:
        diameter = pile.diameter
    return math.pi * diameter**2 / 4.0


def qd_spec_for_soil(
    method: ConstructionMethod,
    soil_type_value: str,
    tip_treatment: TipTreatment | None = None,
    wing_ratio: float | None = None,
) -> QdSpec | None:
    """工法・土質に対応する qd の算定仕様を返す。無ければ None。"""
    if method == ConstructionMethod.ROTARY:
        ratio = wing_ratio or DEFAULT_WING_RATIO
        if ratio not in QD_SPECS_ROTARY:
            raise ValueError(
                f"回転杭の羽根外径比 {ratio} は未対応です。"
                f"対応値: {sorted(QD_SPECS_ROTARY)}"
            )
        return QD_SPECS_ROTARY[ratio].get(soil_type_value)
    return QD_SPECS[qd_method_key(method, tip_treatment)].get(soil_type_value)


def tip_resistance_intensity(
    method: ConstructionMethod,
    layer: SoilLayer,
    n_tip: float,
    tip_treatment: TipTreatment | None = None,
    wing_ratio: float | None = None,
) -> float:
    """杭先端の極限支持力度 qd (kN/m2)(道示Ⅳ)。

    ``n_tip`` は杭先端付近の平均N値。``tip_treatment`` は中掘り杭の
    先端処理方式(省略時はセメントミルク噴出攪拌方式)、``wing_ratio`` は
    回転杭の羽根外径比(省略時は 1.5)。
    """
    key_method = qd_method_key(method, tip_treatment)
    key = layer.soil_type.value
    spec = qd_spec_for_soil(method, key, tip_treatment, wing_ratio)
    if spec is None:
        # 支持層として想定していない土質は明示的にエラーとする
        raise ValueError(
            f"{method.value}では{layer.soil_type.value}を支持層にできません"
        )

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
    reduction: "SoilReduction | None" = None,
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
    reduction:
        液状化に伴う土質定数の低減係数 DE(道示Ⅴ 8.2)。与えると
        **最大周面摩擦力度に直接 DE を乗じる**:

            f′i = DE,i × fi

        押込み Ru = qd・A + U・Σ(Li・f′i)、引抜き Ruf = U・Σ(Li・f′i) の
        いずれにも適用する。「液状化層は一律に摩擦ゼロ」ではなく、FL に応じた
        DE による段階的な低減である(DE = 0 の層のみ寄与がゼロになる)。

        層内で f は一定なので、区間の DE を層厚加重平均したものを乗じた値は
        ∫f・DE dz と厳密に一致する(近似ではない)。

        .. important::
           **これは耐震設計上の扱いであり、常時の許容支持力照査には適用しない。**
           荷重ケースごとの使い分けは
           :meth:`core.analysis.stability.StabilityReport.bearing_for` を参照。

        .. note::
           **先端支持力度 qd には乗じない**。支持層は液状化しない良質層で
           あることが前提だからである。判定範囲(通常は地表面から 20 m)内で
           先端付近が液状化すると判定された場合は :attr:`BearingCapacity.tip_de`
           が 1.0 未満になるので、利用者に判断を促すこと。

        .. note::
           発注者の設計要領等で液状化層の周面摩擦をゼロ扱いとする指定がある
           場合は、その指定が優先される。本ソフトはその指定を表現する入力を
           持たないため、必要なら DE = 0 相当の扱いを別途検討すること。
    """
    d = pile.diameter
    tip_depth = embedment + pile.length
    if tip_depth > profile.total_depth:
        raise ValueError(
            f"杭先端深度 {tip_depth:.1f} m が地盤モデル深さ "
            f"{profile.total_depth:.1f} m を超えています"
        )

    # 先端面積は工法により基準径が異なる(鋼管ソイルセメント杭・回転杭)
    area = tip_area(pile)
    # 周面摩擦の対象となる杭周長。鋼管ソイルセメント杭は地盤と接するのが
    # ソイルセメント柱であるため、その径を用いる(原典未確認、物理的整合による)。
    shaft_diameter = (
        pile.soil_cement_diameter
        if pile.method == ConstructionMethod.STEEL_PIPE_SOIL_CEMENT
        and pile.soil_cement_diameter is not None
        else d
    )
    perimeter = math.pi * shaft_diameter
    tip_layer = profile.layer_at(tip_depth)
    n_value_tip = (
        average_n_near_tip(profile, tip_depth, d) if n_tip is None else n_tip
    )
    qd = tip_resistance_intensity(
        pile.method, tip_layer, n_value_tip, pile.tip_treatment, pile.wing_ratio
    )

    # 周面摩擦を計上する下端(先端から 1D 手前で打ち切る)
    skin_bottom = tip_depth - d if exclude_tip_zone else tip_depth
    skin_bottom = max(skin_bottom, embedment)

    segments: list[SkinFrictionSegment] = []
    skin = 0.0
    skin_unreduced = 0.0
    for top, bottom, layer in profile.layer_boundaries():
        seg_top = max(top, embedment)
        seg_bottom = min(bottom, skin_bottom)
        length = seg_bottom - seg_top
        if length <= 0:
            continue
        f = skin_friction_intensity(pile.method, layer)
        # 層内で f は一定なので、DE の層厚加重平均を乗じた値は ∫f・DE dz に等しい
        de = 1.0 if reduction is None else reduction.mean_factor(seg_top, seg_bottom)
        force = perimeter * length * f * de
        skin += force
        skin_unreduced += perimeter * length * f
        segments.append(
            SkinFrictionSegment(
                layer_name=layer.name,
                soil_type=layer.soil_type,
                length=length,
                f=f,
                force=force,
                de=de,
            )
        )

    # 先端支持力は低減しない。液状化する層が支持層になっていないかの確認用
    tip_de = (
        1.0
        if reduction is None
        else reduction.mean_factor(max(0.0, tip_depth - d), tip_depth + d)
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
        tip_de=tip_de,
        skin_resistance_unreduced=skin_unreduced,
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
