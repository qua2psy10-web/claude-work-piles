"""設計プロジェクト(入力一式)のデータモデル。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.loads import FootingLoads
from core.models.pile import Footing, PileArrangement, PileSpec
from core.models.soil import SoilProfile
from core.standards import CZ_MAX, CZ_MIN, E0Method, GroundType


class SeismicConditions(BaseModel):
    """耐震設計の条件。

    .. note::
       地域別補正係数の上限は **1.20**(A1・B1 地域のタイプI)である。
       第28回まで 1.0 で頭打ちにしており、これらの地域を入力できず
       液状化判定を非安全側に評価していた。
    """

    ground_type: GroundType = GroundType.TYPE_II
    cz_type1: float = Field(
        default=1.0,
        ge=CZ_MIN,
        le=CZ_MAX,
        description="地域別補正係数 cIz(タイプI)。A1・B1 地域は 1.20",
    )
    cz_type2: float = Field(
        default=1.0,
        ge=CZ_MIN,
        le=CZ_MAX,
        description="地域別補正係数 cIIz(タイプII)。最大 1.00",
    )


class DesignProject(BaseModel):
    """プロジェクトファイルとして保存・読込する入力一式。"""

    name: str = "無題"
    soil_profile: SoilProfile
    seismic: SeismicConditions = SeismicConditions()
    e0_method: E0Method = Field(
        default=E0Method.N_VALUE,
        description="変形係数E0の推定方法(kHの換算係数αが決まる)",
    )
    pile: PileSpec | None = None
    arrangement: PileArrangement | None = None
    footing: Footing | None = None
    loads: list[FootingLoads] = Field(default_factory=list)
