"""杭・フーチングのデータモデル。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PileType(str, Enum):
    STEEL_PIPE = "鋼管杭"
    STEEL_PIPE_SOIL_CEMENT = "鋼管ソイルセメント杭"
    CAST_IN_PLACE = "場所打ち杭"
    PHC = "PHC杭"
    SC = "SC杭"
    RC = "RC杭"
    H_STEEL = "H鋼杭"


class ConstructionMethod(str, Enum):
    DRIVEN = "打込み(打撃)"
    VIBRO = "バイブロハンマ"
    CAST_IN_PLACE = "場所打ち"
    INNER_DIGGING = "中掘り"
    PREBORING = "プレボーリング"
    STEEL_PIPE_SOIL_CEMENT = "鋼管ソイルセメント"
    ROTARY = "回転"


class PileSpec(BaseModel):
    pile_type: PileType
    method: ConstructionMethod
    diameter: float = Field(gt=0, description="杭径 (m)")
    length: float = Field(gt=0, description="杭長 (m)")
    wall_thickness: float | None = Field(
        default=None, gt=0, description="鋼管の板厚 (mm)。鋼管系杭のみ"
    )


class PileArrangement(BaseModel):
    """フーチング下面の杭配置(矩形配置)。"""

    nx: int = Field(ge=1, description="橋軸方向の列数")
    ny: int = Field(ge=1, description="橋軸直角方向の列数")
    spacing_x: float = Field(gt=0, description="橋軸方向の杭間隔 (m)")
    spacing_y: float = Field(gt=0, description="橋軸直角方向の杭間隔 (m)")

    @property
    def total_piles(self) -> int:
        return self.nx * self.ny


class Footing(BaseModel):
    width_x: float = Field(gt=0, description="橋軸方向幅 (m)")
    width_y: float = Field(gt=0, description="橋軸直角方向幅 (m)")
    height: float = Field(gt=0, description="厚さ (m)")
    embedment: float = Field(ge=0, description="根入れ深さ(地表面〜フーチング下面) (m)")
