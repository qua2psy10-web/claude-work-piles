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
from core.standards import EC_CONCRETE, EC_SC_PILE_CONCRETE, E_STEEL

# 鋼管杭の腐食代 (mm)(道示Ⅳ 12.10)
CORROSION_ALLOWANCE_MM = 1.0

# 中空コンクリート杭(PHC・RC)およびSC杭
HOLLOW_CONCRETE_TYPES = (PileType.PHC, PileType.RC)
STEEL_TUBE_TYPES = (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT)


def _concrete_young(fck: int, override: float | None = None) -> float:
    """コンクリートのヤング係数 Ec (kN/m2)。

    ``override`` が与えられればそれを用いる。既製杭(PHC・SC)の標準である
    σck = 80 N/mm² はヤング係数の表の範囲外であり、その値は道示ではなく
    JIS/業界側の資料(製品の断面性能表など)に拠るため、本ソフトでは
    **表に持たず利用者の入力に委ねる**。詳細は docs/VERIFICATION.md の
    「σck = 80 の Ec」を参照。
    """
    if override is not None:
        if override <= 0:
            raise ValueError("ヤング係数 Ec は正の値である必要があります")
        return override
    if fck not in EC_CONCRETE:
        raise ValueError(
            f"σck={fck} のヤング係数が未定義です。対応値: {sorted(EC_CONCRETE)}"
            "。既製杭の標準である σck = 80 N/mm² は表の範囲外のため、"
            "PileSpec.concrete_young にメーカーの断面性能表等の Ec を"
            "直接指定してください"
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
        return PileSection(
            area=area,
            inertia=inertia,
            young=_concrete_young(fck, pile.concrete_young),
        )

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
        return PileSection(
            area=area,
            inertia=inertia,
            young=_concrete_young(fck, pile.concrete_young),
        )

    if pile.pile_type == PileType.SC:
        return _sc_section(pile, corrosion_mm)

    if pile.pile_type == PileType.H_STEEL:
        if pile.h_section is None:
            raise ValueError("H鋼杭は断面寸法 h_section の入力が必要です")
        area, inertia = h_section_properties(pile.h_section, pile.bending_axis)
        return PileSection(area=area, inertia=inertia, young=E_STEEL)

    raise NotImplementedError(f"{pile.pile_type.value}の断面計算は未実装です")


def _sc_section(pile: PileSpec, corrosion_mm: float) -> PileSection:
    """SC杭(外側鋼管 + 内側コンクリート)の換算断面。

    SC杭の曲げ剛性 EI は**鋼管とコンクリートをともに考慮した合成断面**で
    評価する(鋼管のみで評価するのではない)。すなわち

        EI = Ec・Ic + Es・Is

    である。本実装は**鋼を基準**とした換算断面をとり、コンクリート部の寄与を
    1/n(n = Es/Ec)倍して合算する:

        E = Es,  I = Is + Ic/n = Is +(Ec/Es)・Ic
        → EI = Es・Is + Ec・Ic  ✓(上式と一致する)

    軸方向も同様に EA = Es・As + Ec・Ac となり、Kv = a・Ap・Ep/L と整合する。

    .. warning::
       コンクリート基準で整理された換算断面二次モーメント Ie(= Ic + n・Is)に
       Es を掛けると、鋼管の寄与を重複計上して EI を過大評価する。製品の
       断面性能表を使う場合は、その換算基準がどちらの材料かを必ず確認すること。

    ヤング係数は σck からは引かず、SC杭に対して定められた値
    (:data:`core.standards.EC_SC_PILE_CONCRETE`)を用いる。SC杭の
    コンクリートは σck = 80 N/mm² であり、道示Ⅲ 表-3.3.3(σck ≤ 60)の
    範囲外だからである。``PileSpec.concrete_young`` があればそちらを優先する。
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

    ec = pile.concrete_young or EC_SC_PILE_CONCRETE
    n = E_STEEL / ec
    return PileSection(
        area=steel_area + concrete_area / n,
        inertia=steel_inertia + concrete_inertia / n,
        young=E_STEEL,
    )
