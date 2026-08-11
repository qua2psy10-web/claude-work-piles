"""荷重のデータモデル。"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LoadCase(str, Enum):
    PERMANENT = "常時"
    STORM = "暴風時"
    LEVEL1_EQ = "レベル1地震時"


class FootingLoads(BaseModel):
    """フーチング底面中心に作用する荷重(1荷重ケース分)。"""

    case: LoadCase
    v: float = Field(description="鉛直力 V (kN)")
    h: float = Field(default=0.0, description="水平力 H (kN)")
    m: float = Field(default=0.0, description="モーメント M (kN·m)")
