"""荷重のデータモデル。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LoadCase(str, Enum):
    PERMANENT = "常時"
    STORM = "暴風時"
    LEVEL1_EQ = "レベル1地震時"

    @property
    def is_seismic(self) -> bool:
        """地震の影響を考慮する荷重ケースか。

        液状化に伴う土質定数の低減 DE は**耐震設計上の扱い**であり、
        常時・暴風時の照査には適用しない(道示Ⅴ 8.2、提供資料で確認)。
        """
        return self is LoadCase.LEVEL1_EQ


class FootingLoads(BaseModel):
    """フーチング底面中心に作用する荷重(1荷重ケース分)。"""

    case: LoadCase
    v: float = Field(description="鉛直力 V (kN)")
    h: float = Field(default=0.0, description="水平力 H (kN)")
    m: float = Field(default=0.0, description="モーメント M (kN·m)")
