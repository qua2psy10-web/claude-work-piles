"""設計プロジェクト(入力一式)のデータモデル。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.loads import FootingLoads
from core.models.pile import Footing, PileArrangement, PileSpec
from core.models.soil import SoilProfile
from core.standards import E0Method, GroundType


class SeismicConditions(BaseModel):
    ground_type: GroundType = GroundType.TYPE_II
    cz_type1: float = Field(
        default=1.0, gt=0, le=1.0, description="地域別補正係数 cIz(タイプI)"
    )
    cz_type2: float = Field(
        default=1.0, gt=0, le=1.0, description="地域別補正係数 cIIz(タイプII)"
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
