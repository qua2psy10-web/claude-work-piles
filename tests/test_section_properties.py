"""杭種別の断面諸元のテスト(幾何・材料力学のみ)。"""
import math

import pytest

from core.capacity.section import (
    h_section_properties,
    hollow_circle,
    pile_section,
)
from core.models import (
    BendingAxis,
    ConstructionMethod,
    HSection,
    PileSpec,
    PileType,
)
from core.standards import EC_CONCRETE, E_STEEL


def test_hollow_circle_matches_formula():
    area, inertia = hollow_circle(1.0, 0.012)
    d_in = 1.0 - 0.024
    assert area == pytest.approx(math.pi * (1.0 - d_in**2) / 4)
    assert inertia == pytest.approx(math.pi * (1.0 - d_in**4) / 64)


def test_hollow_circle_rejects_excessive_thickness():
    with pytest.raises(ValueError, match="中空断面"):
        hollow_circle(0.6, 0.35)


def test_rc_pile_section():
    pile = PileSpec(
        pile_type=PileType.RC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.5,
        length=15.0,
        concrete_thickness=80.0,
    )
    section = pile_section(pile, fck=30)
    expected = hollow_circle(0.5, 0.08)
    assert section.area == pytest.approx(expected[0])
    assert section.inertia == pytest.approx(expected[1])
    assert section.young == EC_CONCRETE[30]


def test_unknown_concrete_grade_is_reported():
    """PHC杭の標準である高強度コンクリートは未照合のためエラーとする。"""
    pile = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
        concrete_thickness=90.0,
    )
    with pytest.raises(ValueError, match="未定義"):
        pile_section(pile, fck=80)


# --- SC杭 -------------------------------------------------------------------


def sc_pile() -> PileSpec:
    return PileSpec(
        pile_type=PileType.SC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
        wall_thickness=9.0,
        concrete_thickness=80.0,
    )


def test_sc_pile_is_transformed_section():
    """SC杭は鋼基準の換算断面(コンクリートを 1/n 倍)。"""
    pile = sc_pile()
    section = pile_section(pile, fck=30)
    t_steel = (9.0 - 1.0) / 1000.0  # 腐食代控除
    steel = hollow_circle(0.6, t_steel)
    concrete = hollow_circle(0.6 - 2 * t_steel, 0.08)
    n = E_STEEL / EC_CONCRETE[30]

    assert section.young == E_STEEL
    assert section.area == pytest.approx(steel[0] + concrete[0] / n)
    assert section.inertia == pytest.approx(steel[1] + concrete[1] / n)
    # 鋼管だけより硬い(コンクリートの寄与がある)
    assert section.inertia > steel[1]


def test_sc_pile_requires_both_thicknesses():
    with pytest.raises(ValueError, match="板厚"):
        pile_section(sc_pile().model_copy(update={"wall_thickness": None}))
    with pytest.raises(ValueError, match="肉厚"):
        pile_section(sc_pile().model_copy(update={"concrete_thickness": None}))


# --- H鋼杭 ------------------------------------------------------------------

# H-400×400×13×21(一般的な H形鋼)
H400 = HSection(height=400.0, width=400.0, web_thickness=13.0, flange_thickness=21.0)


def test_h_section_area_and_inertia():
    """H-400×400×13×21 の断面諸量を式どおり算定する。"""
    area_s, ix = h_section_properties(H400, BendingAxis.STRONG)
    area_w, iy = h_section_properties(H400, BendingAxis.WEAK)
    h, b, tw, tf = 0.4, 0.4, 0.013, 0.021
    web = h - 2 * tf

    # 断面積は軸によらない
    assert area_s == pytest.approx(area_w)
    assert area_s == pytest.approx(2 * b * tf + web * tw)
    assert ix == pytest.approx((b * h**3 - (b - tw) * web**3) / 12)
    assert iy == pytest.approx((2 * tf * b**3 + web * tw**3) / 12)
    # 強軸のほうが断面二次モーメントが大きい
    assert ix > iy


def test_h_pile_defaults_to_weak_axis():
    """既定は安全側の弱軸。"""
    pile = PileSpec(
        pile_type=PileType.H_STEEL,
        method=ConstructionMethod.DRIVEN,
        diameter=0.4,
        length=15.0,
        h_section=H400,
    )
    assert pile.bending_axis == BendingAxis.WEAK
    section = pile_section(pile)
    _, iy = h_section_properties(H400, BendingAxis.WEAK)
    assert section.inertia == pytest.approx(iy)
    assert section.young == E_STEEL

    strong = pile_section(pile.model_copy(update={"bending_axis": BendingAxis.STRONG}))
    assert strong.inertia > section.inertia


def test_h_pile_requires_section():
    pile = PileSpec(
        pile_type=PileType.H_STEEL,
        method=ConstructionMethod.DRIVEN,
        diameter=0.4,
        length=15.0,
    )
    with pytest.raises(ValueError, match="断面寸法"):
        pile_section(pile)


def test_h_section_validation():
    with pytest.raises(ValueError, match="せい"):
        h_section_properties(
            HSection(height=40.0, width=400.0, web_thickness=13.0, flange_thickness=21.0),
            BendingAxis.STRONG,
        )
    with pytest.raises(ValueError, match="フランジ幅"):
        h_section_properties(
            HSection(height=400.0, width=10.0, web_thickness=13.0, flange_thickness=21.0),
            BendingAxis.STRONG,
        )


def test_all_pile_types_have_section_properties():
    """全7杭種で断面諸元が算定できること(応力度照査とは別)。"""
    specs = {
        PileType.CAST_IN_PLACE: dict(method=ConstructionMethod.CAST_IN_PLACE),
        PileType.STEEL_PIPE: dict(
            method=ConstructionMethod.DRIVEN, wall_thickness=12.0
        ),
        PileType.STEEL_PIPE_SOIL_CEMENT: dict(
            method=ConstructionMethod.STEEL_PIPE_SOIL_CEMENT, wall_thickness=12.0
        ),
        PileType.PHC: dict(
            method=ConstructionMethod.DRIVEN, concrete_thickness=90.0
        ),
        PileType.RC: dict(
            method=ConstructionMethod.DRIVEN, concrete_thickness=80.0
        ),
        PileType.SC: dict(
            method=ConstructionMethod.DRIVEN,
            wall_thickness=9.0,
            concrete_thickness=80.0,
        ),
        PileType.H_STEEL: dict(
            method=ConstructionMethod.DRIVEN, h_section=H400
        ),
    }
    for pile_type, extra in specs.items():
        pile = PileSpec(
            pile_type=pile_type, diameter=0.6, length=20.0, **extra
        )
        section = pile_section(pile, fck=30)
        assert section.area > 0, pile_type
        assert section.inertia > 0, pile_type
        assert section.young > 0, pile_type
