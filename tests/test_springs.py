"""バネ定数・断面諸元のテスト(道示Ⅳ(H24) 9.5、12.6)。"""
import math

import pytest

from core.capacity.section import pile_section
from core.capacity.springs import axial_spring, lateral_springs
from core.models import (
    ConstructionMethod,
    LoadCase,
    PileSpec,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.standards import EC_CONCRETE, E_STEEL

CIP_PILE = PileSpec(
    pile_type=PileType.CAST_IN_PLACE,
    method=ConstructionMethod.CAST_IN_PLACE,
    diameter=1.0,
    length=20.0,
)


def sand_profile() -> SoilProfile:
    return SoilProfile(
        layers=[
            SoilLayer(
                name="As",
                soil_type=SoilType.SAND,
                thickness=30.0,
                n_value=10.0,
                gamma_t=18.0,
                gamma_sat=19.0,
            )
        ],
        gwl=2.0,
    )


def test_cast_in_place_section():
    section = pile_section(CIP_PILE, fck=24)
    assert section.area == pytest.approx(math.pi / 4)
    assert section.inertia == pytest.approx(math.pi / 64)
    assert section.young == EC_CONCRETE[24]


def test_steel_pipe_section_deducts_corrosion():
    pile = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=20.0,
        wall_thickness=12.0,
    )
    section = pile_section(pile, corrosion_mm=1.0)
    t = 0.011  # 12mm - 1mm腐食代
    d_in = 1.0 - 2 * t
    assert section.area == pytest.approx(math.pi * (1.0 - d_in**2) / 4)
    assert section.inertia == pytest.approx(math.pi * (1.0 - d_in**4) / 64)
    assert section.young == E_STEEL


def test_steel_pipe_requires_thickness():
    pile = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=20.0,
    )
    with pytest.raises(ValueError, match="板厚"):
        pile_section(pile)


def test_unimplemented_pile_type():
    pile = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
    )
    with pytest.raises(NotImplementedError):
        pile_section(pile)


def test_axial_spring_hand_calculation():
    """場所打ち杭 L/D = 20 → a = 0.031×20 − 0.15 = 0.47
    Kv = a・Ap・Ep/L = 0.47 × 0.785398 × 2.5e7 / 20
    """
    section = pile_section(CIP_PILE, fck=24)
    kv = axial_spring(CIP_PILE, section)
    expected = 0.47 * (math.pi / 4) * 2.5e7 / 20.0
    assert kv == pytest.approx(expected, rel=1e-6)


def test_lateral_springs_converge_and_satisfy_definitions():
    section = pile_section(CIP_PILE, fck=24)
    springs = lateral_springs(
        CIP_PILE, section, sand_profile(), embedment=2.0, case=LoadCase.PERMANENT
    )
    ei = section.ei
    # β の定義式を満たしていること
    assert springs.beta == pytest.approx(
        (springs.kh * CIP_PILE.diameter / (4 * ei)) ** 0.25, rel=1e-6
    )
    # BH = √(D/β)
    assert springs.bh == pytest.approx(math.sqrt(CIP_PILE.diameter / springs.beta))
    # K1〜K4 の関係
    assert springs.k1 == pytest.approx(4 * ei * springs.beta**3)
    assert springs.k2 == pytest.approx(-2 * ei * springs.beta**2)
    assert springs.k3 == springs.k2
    assert springs.k4 == pytest.approx(2 * ei * springs.beta)
    assert springs.is_semi_infinite
    # K1・K4 = 2・K2² より剛性マトリクスは常に正則(変位法が解ける条件)
    assert springs.k1 * springs.k4 == pytest.approx(2 * springs.k2**2, rel=1e-9)


def test_seismic_alpha_doubles_kh():
    section = pile_section(CIP_PILE, fck=24)
    normal = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.PERMANENT
    )
    seismic = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.LEVEL1_EQ
    )
    # α が 1→2 になるぶん kH が増え、杭は相対的に硬くなる
    assert seismic.kh > normal.kh
    assert seismic.beta > normal.beta
    assert seismic.k1 > normal.k1


def test_e0_from_n_value():
    section = pile_section(CIP_PILE, fck=24)
    springs = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.PERMANENT
    )
    assert springs.e0 == pytest.approx(2800.0 * 10.0)
