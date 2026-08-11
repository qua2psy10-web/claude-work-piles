"""地盤(地層・地盤モデル)のデータモデル。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from core.standards import GAMMA_W


class SoilType(str, Enum):
    SAND = "砂質土"
    GRAVEL = "礫質土"
    CLAY = "粘性土"


class SoilLayer(BaseModel):
    """1つの地層。

    γt は地下水位以浅、γsat は地下水位以深の単位体積重量として応力計算に用いる。
    """

    name: str = ""
    soil_type: SoilType
    thickness: float = Field(gt=0, description="層厚 (m)")
    n_value: float = Field(ge=0, description="平均N値")
    gamma_t: float = Field(gt=0, description="湿潤単位体積重量 (kN/m3)")
    gamma_sat: float = Field(gt=0, description="飽和単位体積重量 (kN/m3)")
    fc: float | None = Field(default=None, ge=0, le=100, description="細粒分含有率 (%)")
    ip: float | None = Field(default=None, ge=0, description="塑性指数 IP")
    d50: float | None = Field(default=None, gt=0, description="平均粒径 D50 (mm)")
    d10: float | None = Field(default=None, gt=0, description="10%粒径 D10 (mm)")
    is_alluvial: bool = Field(default=True, description="沖積層か")
    cohesion: float | None = Field(default=None, ge=0, description="粘着力 c (kN/m2)")
    phi: float | None = Field(default=None, ge=0, description="せん断抵抗角 φ (度)")
    e0: float | None = Field(default=None, gt=0, description="変形係数 E0 (kN/m2)")


class SoilProfile(BaseModel):
    """地層の積み重ねと地下水位からなる地盤モデル。

    深度は地表面(設計地盤面)からの距離を正にとる。
    """

    layers: list[SoilLayer] = Field(min_length=1)
    gwl: float = Field(ge=0, description="地下水位の深さ (m)")

    @property
    def total_depth(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    def layer_boundaries(self) -> list[tuple[float, float, SoilLayer]]:
        """各層の (上端深度, 下端深度, 層) を返す。"""
        result = []
        top = 0.0
        for layer in self.layers:
            bottom = top + layer.thickness
            result.append((top, bottom, layer))
            top = bottom
        return result

    def layer_at(self, depth: float) -> SoilLayer:
        for top, bottom, layer in self.layer_boundaries():
            if top <= depth <= bottom:
                return layer
        raise ValueError(f"深度 {depth} m は地盤モデルの範囲外です")

    def stresses_at(self, depth: float) -> tuple[float, float]:
        """深度 depth における全上載圧 σv と有効上載圧 σ'v (kN/m2) を返す。"""
        if depth > self.total_depth:
            raise ValueError(f"深度 {depth} m は地盤モデルの範囲外です")
        sigma_v = 0.0
        for top, bottom, layer in self.layer_boundaries():
            seg_top = top
            seg_bottom = min(bottom, depth)
            if seg_bottom <= seg_top:
                break
            # 地下水位を境に γt / γsat を使い分ける
            above = max(0.0, min(seg_bottom, self.gwl) - seg_top)
            below = (seg_bottom - seg_top) - above
            sigma_v += above * layer.gamma_t + below * layer.gamma_sat
        u = GAMMA_W * max(0.0, depth - self.gwl)
        return sigma_v, sigma_v - u
