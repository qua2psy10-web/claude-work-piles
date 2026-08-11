"""杭種・工法の比較表(形式選定の支援)。

同一の地盤条件・杭長に対して、杭種 × 施工工法 × 杭径の組合せごとに
軸方向支持力を算定し、必要杭本数とともに一覧する。

支持力の算定に用いる qd・f は全工法について照合済み
(docs/VERIFICATION.md 第13〜14回)。本モジュールはそれらを組み替えて
比較するのみで、新たな基準値は導入しない。

.. note::
   杭種と施工工法の組合せ(:data:`APPLICABLE_METHODS`)は、実務で一般に
   用いられる範囲を整理したものであり、道示が規定するものではない。
   採用にあたっては工法の技術資料・適用範囲を確認すること。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.capacity.bearing import BearingCapacity, compute_bearing_capacity
from core.models.loads import LoadCase
from core.models.pile import (
    ConstructionMethod,
    PileSpec,
    PileType,
    SupportType,
)
from core.models.soil import SoilProfile
from core.standards import TipTreatment

# 杭種ごとに一般に適用される施工工法
APPLICABLE_METHODS: dict[PileType, tuple[ConstructionMethod, ...]] = {
    PileType.STEEL_PIPE: (
        ConstructionMethod.DRIVEN,
        ConstructionMethod.VIBRO,
        ConstructionMethod.INNER_DIGGING,
        ConstructionMethod.PREBORING,
        ConstructionMethod.ROTARY,
    ),
    PileType.STEEL_PIPE_SOIL_CEMENT: (ConstructionMethod.STEEL_PIPE_SOIL_CEMENT,),
    PileType.CAST_IN_PLACE: (ConstructionMethod.CAST_IN_PLACE,),
    PileType.PHC: (
        ConstructionMethod.DRIVEN,
        ConstructionMethod.VIBRO,
        ConstructionMethod.INNER_DIGGING,
        ConstructionMethod.PREBORING,
    ),
    PileType.SC: (
        ConstructionMethod.DRIVEN,
        ConstructionMethod.INNER_DIGGING,
        ConstructionMethod.PREBORING,
    ),
    PileType.RC: (
        ConstructionMethod.DRIVEN,
        ConstructionMethod.VIBRO,
        ConstructionMethod.INNER_DIGGING,
        ConstructionMethod.PREBORING,
    ),
    PileType.H_STEEL: (
        ConstructionMethod.DRIVEN,
        ConstructionMethod.VIBRO,
    ),
}

# 鋼管ソイルセメント杭のソイルセメント柱径の既定比(柱径 / 鋼管径)
DEFAULT_SOIL_CEMENT_RATIO = 1.4


@dataclass(frozen=True)
class ComparisonRow:
    """比較表の1行(杭種 × 工法 × 杭径)。"""

    pile_type: PileType
    method: ConstructionMethod
    diameter: float
    tip_treatment: TipTreatment | None = None
    bearing: BearingCapacity | None = None
    allowable_push: float | None = None  # 常時の許容押込み支持力 (kN)
    allowable_pull: float | None = None  # 常時の許容引抜き力 (kN)
    required_piles: int | None = None  # 鉛直荷重を支持するのに必要な本数
    error: str | None = None  # 算定できない場合の理由

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def ru(self) -> float | None:
        return self.bearing.ru if self.bearing is not None else None

    @property
    def tip_area(self) -> float | None:
        return self.bearing.tip_area if self.bearing is not None else None


def required_pile_count(vertical_load: float, allowable_push: float) -> int | None:
    """鉛直荷重を支持するのに必要な杭本数。許容値が非正なら None。"""
    if allowable_push <= 0:
        return None
    return math.ceil(vertical_load / allowable_push)


def compare(
    profile: SoilProfile,
    embedment: float,
    length: float,
    diameters: list[float],
    vertical_load: float,
    case: LoadCase = LoadCase.PERMANENT,
    pile_types: list[PileType] | None = None,
    support_type: SupportType = SupportType.END_BEARING,
    tip_treatment: TipTreatment = TipTreatment.CEMENT_MILK,
    wing_ratio: float = 1.5,
    soil_cement_ratio: float = DEFAULT_SOIL_CEMENT_RATIO,
) -> list[ComparisonRow]:
    """杭種 × 工法 × 杭径の組合せごとに支持力を算定して一覧を返す。

    算定できない組合せ(支持層の条件を満たさない等)は ``error`` に理由を
    格納し、行としては残す(なぜ使えないかを利用者に示すため)。

    Parameters
    ----------
    vertical_load:
        基礎全体に作用する鉛直力 (kN)。必要杭本数の算定に用いる。
    tip_treatment:
        中掘り杭の先端処理方式。
    wing_ratio:
        回転杭の羽根外径比。
    soil_cement_ratio:
        鋼管ソイルセメント杭のソイルセメント柱径 / 鋼管径。
    """
    targets = pile_types if pile_types is not None else list(PileType)
    rows: list[ComparisonRow] = []
    for pile_type in targets:
        for method in APPLICABLE_METHODS.get(pile_type, ()):
            for diameter in diameters:
                rows.append(
                    _evaluate(
                        profile=profile,
                        embedment=embedment,
                        length=length,
                        diameter=diameter,
                        pile_type=pile_type,
                        method=method,
                        vertical_load=vertical_load,
                        case=case,
                        support_type=support_type,
                        tip_treatment=tip_treatment,
                        wing_ratio=wing_ratio,
                        soil_cement_ratio=soil_cement_ratio,
                    )
                )
    return rows


def _evaluate(
    *,
    profile: SoilProfile,
    embedment: float,
    length: float,
    diameter: float,
    pile_type: PileType,
    method: ConstructionMethod,
    vertical_load: float,
    case: LoadCase,
    support_type: SupportType,
    tip_treatment: TipTreatment,
    wing_ratio: float,
    soil_cement_ratio: float,
) -> ComparisonRow:
    treatment = (
        tip_treatment if method == ConstructionMethod.INNER_DIGGING else None
    )
    pile = PileSpec(
        pile_type=pile_type,
        method=method,
        diameter=diameter,
        length=length,
        support_type=support_type,
        tip_treatment=treatment,
        wing_ratio=(
            wing_ratio if method == ConstructionMethod.ROTARY else None
        ),
        soil_cement_diameter=(
            soil_cement_ratio * diameter
            if method == ConstructionMethod.STEEL_PIPE_SOIL_CEMENT
            else None
        ),
        # 鋼管系杭は断面計算に板厚が要るが、支持力の比較では用いない
        wall_thickness=12.0,
    )
    try:
        bearing = compute_bearing_capacity(pile, profile, embedment)
    except (ValueError, NotImplementedError) as exc:
        return ComparisonRow(
            pile_type=pile_type,
            method=method,
            diameter=diameter,
            tip_treatment=treatment,
            error=str(exc),
        )
    ra = bearing.allowable_push(case)
    return ComparisonRow(
        pile_type=pile_type,
        method=method,
        diameter=diameter,
        tip_treatment=treatment,
        bearing=bearing,
        allowable_push=ra,
        allowable_pull=bearing.allowable_pull(case),
        required_piles=required_pile_count(vertical_load, ra),
    )
