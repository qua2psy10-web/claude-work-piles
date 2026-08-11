"""杭種別の断面諸元(断面積・断面二次モーメント・ヤング係数)。

フェーズ1では場所打ち杭(RC円形断面、鉄筋無視の総断面)と
鋼管杭(円環断面、腐食代控除)を扱う。他杭種はフェーズ3で追加する。
"""
from __future__ import annotations

import math

from core.capacity.springs import PileSection
from core.models.pile import PileSpec, PileType
from core.standards import EC_CONCRETE, E_STEEL

# 鋼管杭の腐食代 (mm)(道示Ⅳ 12.10)
CORROSION_ALLOWANCE_MM = 1.0


def pile_section(
    pile: PileSpec, fck: int = 24, corrosion_mm: float = CORROSION_ALLOWANCE_MM
) -> PileSection:
    """杭の断面諸元を返す。

    Parameters
    ----------
    fck:
        場所打ち杭のコンクリート設計基準強度 (N/mm2)。
    corrosion_mm:
        鋼管杭の腐食代 (mm)。断面計算では板厚から控除する。
    """
    d = pile.diameter
    if pile.pile_type == PileType.CAST_IN_PLACE:
        if fck not in EC_CONCRETE:
            raise ValueError(
                f"σck={fck} のヤング係数が未定義です。"
                f"対応値: {sorted(EC_CONCRETE)}"
            )
        area = math.pi * d**2 / 4.0
        inertia = math.pi * d**4 / 64.0
        return PileSection(area=area, inertia=inertia, young=EC_CONCRETE[fck])

    if pile.pile_type in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        if pile.wall_thickness is None:
            raise ValueError("鋼管杭は板厚 wall_thickness の入力が必要です")
        t = (pile.wall_thickness - corrosion_mm) / 1000.0
        if t <= 0:
            raise ValueError(
                f"腐食代 {corrosion_mm} mm 控除後の板厚が 0 以下です"
            )
        d_in = d - 2.0 * t
        area = math.pi * (d**2 - d_in**2) / 4.0
        inertia = math.pi * (d**4 - d_in**4) / 64.0
        return PileSection(area=area, inertia=inertia, young=E_STEEL)

    raise NotImplementedError(
        f"{pile.pile_type.value}の断面計算は未実装です(フェーズ3で対応)"
    )
