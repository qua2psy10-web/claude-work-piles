"""杭種別の断面諸元(断面積・断面二次モーメント・ヤング係数)。

ここで扱うのは**幾何と材料力学のみ**であり、基準に固有の数値は
ヤング係数(:data:`core.standards.EC_CONCRETE`、:data:`E_STEEL`)と
腐食代のみである。したがって全杭種について算定できる。

一方、杭体の**応力度照査**に必要な許容応力度は杭種ごとに異なり、
未照合の杭種があるため :mod:`core.section.checks` 側で明示的にエラーとする。
"""
from __future__ import annotations

import math

from core.capacity.springs import PileSection
from core.models.pile import BendingAxis, HSection, PileSpec, PileType
from core.standards import EC_CONCRETE, E_STEEL

# 鋼管杭の腐食代 (mm)(道示Ⅳ 12.10)
CORROSION_ALLOWANCE_MM = 1.0

# 中空コンクリート杭(PHC・RC)およびSC杭
HOLLOW_CONCRETE_TYPES = (PileType.PHC, PileType.RC)
STEEL_TUBE_TYPES = (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT)


def _concrete_young(fck: int) -> float:
    if fck not in EC_CONCRETE:
        raise ValueError(
            f"σck={fck} のヤング係数が未定義です。対応値: {sorted(EC_CONCRETE)}"
            "(PHC杭の標準である高強度コンクリートは未照合です)"
        )
    return EC_CONCRETE[fck]


def hollow_circle(outer: float, thickness: float) -> tuple[float, float]:
    """中空円形断面の (断面積, 断面二次モーメント)。単位は m, m2, m4。"""
    inner = outer - 2.0 * thickness
    if inner <= 0:
        raise ValueError(
            f"肉厚 {thickness * 1000:.1f} mm が外径 {outer:.3f} m に対して"
            "大きすぎます(中空断面になりません)"
        )
    area = math.pi * (outer**2 - inner**2) / 4.0
    inertia = math.pi * (outer**4 - inner**4) / 64.0
    return area, inertia


def h_section_properties(
    section: HSection, axis: BendingAxis
) -> tuple[float, float]:
    """H形断面の (断面積, 断面二次モーメント)。単位は m2, m4。

    強軸: Ix =[B・H³ −(B − t1)(H − 2・t2)³]/ 12
    弱軸: Iy =[2・t2・B³ +(H − 2・t2)・t1³]/ 12
    """
    section.validated()
    h = section.height / 1000.0
    b = section.width / 1000.0
    tw = section.web_thickness / 1000.0
    tf = section.flange_thickness / 1000.0

    web_height = h - 2.0 * tf
    area = 2.0 * b * tf + web_height * tw
    if axis == BendingAxis.STRONG:
        inertia = (b * h**3 - (b - tw) * web_height**3) / 12.0
    else:
        inertia = (2.0 * tf * b**3 + web_height * tw**3) / 12.0
    return area, inertia


def pile_section(
    pile: PileSpec, fck: int = 24, corrosion_mm: float = CORROSION_ALLOWANCE_MM
) -> PileSection:
    """杭の断面諸元(バネ定数・断面力の算定に用いる A・I・E)を返す。

    Parameters
    ----------
    fck:
        コンクリート系杭の設計基準強度 (N/mm2)。
    corrosion_mm:
        鋼管杭の腐食代 (mm)。断面計算では板厚から控除する。

    Notes
    -----
    SC杭は鋼管とコンクリートの合成断面を、鋼を基準とした換算断面
    (コンクリート部を 1/n 倍、n = Es/Ec)として扱う。
    """
    d = pile.diameter

    if pile.pile_type == PileType.CAST_IN_PLACE:
        area = math.pi * d**2 / 4.0
        inertia = math.pi * d**4 / 64.0
        return PileSection(area=area, inertia=inertia, young=_concrete_young(fck))

    if pile.pile_type in STEEL_TUBE_TYPES:
        if pile.wall_thickness is None:
            raise ValueError("鋼管杭は板厚 wall_thickness の入力が必要です")
        t = (pile.wall_thickness - corrosion_mm) / 1000.0
        if t <= 0:
            raise ValueError(f"腐食代 {corrosion_mm} mm 控除後の板厚が 0 以下です")
        area, inertia = hollow_circle(d, t)
        return PileSection(area=area, inertia=inertia, young=E_STEEL)

    if pile.pile_type in HOLLOW_CONCRETE_TYPES:
        if pile.concrete_thickness is None:
            raise ValueError(
                f"{pile.pile_type.value}はコンクリート部の肉厚 "
                "concrete_thickness (mm) の入力が必要です"
            )
        area, inertia = hollow_circle(d, pile.concrete_thickness / 1000.0)
        return PileSection(area=area, inertia=inertia, young=_concrete_young(fck))

    if pile.pile_type == PileType.SC:
        return _sc_section(pile, fck, corrosion_mm)

    if pile.pile_type == PileType.H_STEEL:
        if pile.h_section is None:
            raise ValueError("H鋼杭は断面寸法 h_section の入力が必要です")
        area, inertia = h_section_properties(pile.h_section, pile.bending_axis)
        return PileSection(area=area, inertia=inertia, young=E_STEEL)

    raise NotImplementedError(f"{pile.pile_type.value}の断面計算は未実装です")


def _sc_section(pile: PileSpec, fck: int, corrosion_mm: float) -> PileSection:
    """SC杭(外側鋼管 + 内側コンクリート)の換算断面。

    鋼を基準とし、コンクリート部の寄与を 1/n(n = Es/Ec)倍して合算する。
    """
    if pile.wall_thickness is None:
        raise ValueError("SC杭は鋼管の板厚 wall_thickness の入力が必要です")
    if pile.concrete_thickness is None:
        raise ValueError(
            "SC杭はコンクリート部の肉厚 concrete_thickness (mm) の入力が必要です"
        )
    t_steel = (pile.wall_thickness - corrosion_mm) / 1000.0
    if t_steel <= 0:
        raise ValueError(f"腐食代 {corrosion_mm} mm 控除後の板厚が 0 以下です")

    steel_area, steel_inertia = hollow_circle(pile.diameter, t_steel)

    # コンクリートは鋼管の内側に位置する
    concrete_outer = pile.diameter - 2.0 * t_steel
    concrete_area, concrete_inertia = hollow_circle(
        concrete_outer, pile.concrete_thickness / 1000.0
    )

    ec = _concrete_young(fck)
    n = E_STEEL / ec
    return PileSection(
        area=steel_area + concrete_area / n,
        inertia=steel_inertia + concrete_inertia / n,
        young=E_STEEL,
    )
