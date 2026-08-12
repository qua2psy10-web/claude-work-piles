"""杭・フーチングのデータモデル。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from core.standards import TipTreatment


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


class SupportType(str, Enum):
    """杭の支持形式(道示Ⅳ 12.4)。押込み支持力の安全率が異なる。

    支持杭: 先端が良質な支持層に貫入している杭
    摩擦杭: 荷重の大部分を杭周面の摩擦抵抗で支持する杭
    """

    END_BEARING = "支持杭"
    FRICTION = "摩擦杭"


class HSection(BaseModel):
    """H形鋼の断面寸法 (mm)。"""

    height: float = Field(gt=0, description="せい H (mm)")
    width: float = Field(gt=0, description="フランジ幅 B (mm)")
    web_thickness: float = Field(gt=0, description="ウェブ厚 t1 (mm)")
    flange_thickness: float = Field(gt=0, description="フランジ厚 t2 (mm)")

    def validated(self) -> HSection:
        if self.height <= 2.0 * self.flange_thickness:
            raise ValueError(
                f"せい {self.height} mm がフランジ厚の2倍以下です"
            )
        if self.width <= self.web_thickness:
            raise ValueError(
                f"フランジ幅 {self.width} mm がウェブ厚以下です"
            )
        return self


class BendingAxis(str, Enum):
    """H鋼杭の曲げを受ける軸。"""

    STRONG = "強軸"
    WEAK = "弱軸"


class PileSpec(BaseModel):
    pile_type: PileType
    method: ConstructionMethod
    diameter: float = Field(gt=0, description="杭径 (m)")
    length: float = Field(gt=0, description="杭長 (m)")
    wall_thickness: float | None = Field(
        default=None, gt=0, description="鋼管の板厚 (mm)。鋼管系杭のみ"
    )
    support_type: SupportType = Field(
        default=SupportType.END_BEARING, description="支持形式(安全率が異なる)"
    )
    tip_treatment: TipTreatment | None = Field(
        default=None,
        description="中掘り杭の先端処理方式(qd の算定法が変わる)。他工法では未使用",
    )
    wing_ratio: float | None = Field(
        default=None,
        gt=0,
        description="回転杭の羽根外径/杭径(1.5 または 2.0)。qd と先端面積が変わる",
    )
    soil_cement_diameter: float | None = Field(
        default=None,
        gt=0,
        description="鋼管ソイルセメント杭のソイルセメント柱径 (m)。先端面積に用いる",
    )
    concrete_thickness: float | None = Field(
        default=None,
        gt=0,
        description="中空コンクリート杭(PHC・RC)およびSC杭のコンクリート部肉厚 (mm)",
    )
    concrete_young: float | None = Field(
        default=None,
        gt=0,
        description=(
            "コンクリートのヤング係数 Ec (kN/m2) を直接指定する場合の値。"
            "既製杭(PHC・SC)の標準である σck = 80 N/mm2 はヤング係数の表の"
            "範囲外のため、メーカーの断面性能表等から Ec を与えるための入口。"
            "指定するとヤング係数の表引き(σck → Ec)を上書きする"
        ),
    )
    h_section: HSection | None = Field(
        default=None, description="H鋼杭の断面寸法"
    )
    bending_axis: BendingAxis = Field(
        default=BendingAxis.WEAK,
        description="H鋼杭で曲げを受ける軸。既定は安全側の弱軸",
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
