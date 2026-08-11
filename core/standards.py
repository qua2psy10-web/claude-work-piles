"""基準年度別の定数テーブル(H24年道示版)。

計算ロジック側では数値を直接書かず、必ず本モジュールの定数を参照する。
将来H29版(部分係数法)へ対応する際は、本モジュールと同じインタフェースの
定数テーブルを追加して差し替えられるようにする。

出典: 道路橋示方書・同解説 Ⅴ耐震設計編(平成24年3月)
"""
from __future__ import annotations

from enum import Enum


class GroundType(str, Enum):
    """耐震設計上の地盤種別(道示Ⅴ 4.5)"""

    TYPE_I = "I種"
    TYPE_II = "II種"
    TYPE_III = "III種"


class GroundMotionType(str, Enum):
    """レベル2地震動のタイプ(道示Ⅴ 2.2)

    タイプI: プレート境界型の大規模な地震
    タイプII: 内陸直下型地震
    """

    LEVEL2_TYPE1 = "レベル2タイプI"
    LEVEL2_TYPE2 = "レベル2タイプII"


# 水の単位体積重量 (kN/m3)
GAMMA_W = 9.8

# 液状化の判定に用いる地盤面における設計水平震度の標準値 khg0
# 道示Ⅴ(H24) 8.2.3。khg = cz × khg0(cz: 地域別補正係数)
KHG0_LIQUEFACTION: dict[GroundMotionType, dict[GroundType, float]] = {
    GroundMotionType.LEVEL2_TYPE1: {
        GroundType.TYPE_I: 0.50,
        GroundType.TYPE_II: 0.45,
        GroundType.TYPE_III: 0.40,
    },
    GroundMotionType.LEVEL2_TYPE2: {
        GroundType.TYPE_I: 0.80,
        GroundType.TYPE_II: 0.70,
        GroundType.TYPE_III: 0.60,
    },
}

# 土質定数の低減係数 DE(道示Ⅴ(H24) 表-8.2.1)
# キー: (FL区分, 深度区分, R区分) → DE
#   FL区分: 0: FL≦1/3, 1: 1/3<FL≦2/3, 2: 2/3<FL≦1
#   深度区分: 0: 0≦x≦10m, 1: 10m<x≦20m
#   R区分: 0: R≦0.3, 1: 0.3<R
DE_TABLE: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 0.0,
    (0, 0, 1): 1.0 / 6.0,
    (0, 1, 0): 1.0 / 3.0,
    (0, 1, 1): 1.0 / 3.0,
    (1, 0, 0): 1.0 / 3.0,
    (1, 0, 1): 2.0 / 3.0,
    (1, 1, 0): 2.0 / 3.0,
    (1, 1, 1): 2.0 / 3.0,
    (2, 0, 0): 2.0 / 3.0,
    (2, 0, 1): 1.0,
    (2, 1, 0): 1.0,
    (2, 1, 1): 1.0,
}

# 液状化判定の対象となる土層の条件(道示Ⅴ(H24) 8.2.2)
LIQUEFACTION_MAX_DEPTH = 20.0  # 地表面からの深さ (m)
LIQUEFACTION_MAX_GWL = 10.0  # 地下水位の地表面からの深さ (m)
LIQUEFACTION_MAX_FC = 35.0  # 細粒分含有率 (%)
LIQUEFACTION_MAX_IP = 15.0  # 塑性指数(FC>35%の場合の緩和条件)
LIQUEFACTION_MAX_D50 = 10.0  # 平均粒径 (mm)
LIQUEFACTION_MAX_D10 = 1.0  # 10%粒径 (mm)
