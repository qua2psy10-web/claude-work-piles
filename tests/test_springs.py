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
from core.standards import EC_CONCRETE, E_STEEL, E0Method

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


def test_hollow_concrete_pile_requires_thickness():
    """PHC・RC杭はコンクリート部の肉厚が必要。"""
    pile = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
    )
    with pytest.raises(ValueError, match="肉厚"):
        pile_section(pile)


def test_hollow_concrete_pile_section():
    """PHC杭 φ600・肉厚90mm の中空円形断面。"""
    pile = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
        concrete_thickness=90.0,
    )
    section = pile_section(pile, fck=24)
    d_in = 0.6 - 2 * 0.09
    assert section.area == pytest.approx(math.pi * (0.6**2 - d_in**2) / 4)
    assert section.inertia == pytest.approx(math.pi * (0.6**4 - d_in**4) / 64)
    assert section.young == EC_CONCRETE[24]
    # 中実断面より小さい
    assert section.area < math.pi * 0.6**2 / 4


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


def test_kh0_matches_reference_worked_example():
    """提供解説資料の計算例と突合する。

    E0 = 28,000 kN/m²(N=10 → 2800N)のとき
      常時  : kH0 =(1/0.3)× 1.0 × 28,000 ≒ 93,300 kN/m³
      地震時: kH0 =(1/0.3)× 2.0 × 28,000 ≒ 186,700 kN/m³
    kH は kH0 に (BH/0.3)^(−3/4) を乗じた値。
    """
    section = pile_section(CIP_PILE, fck=24)
    for case, expected_kh0 in (
        (LoadCase.PERMANENT, 93_333.3),
        (LoadCase.LEVEL1_EQ, 186_666.7),
    ):
        sp = lateral_springs(CIP_PILE, section, sand_profile(), 2.0, case)
        assert sp.e0 == pytest.approx(28_000.0)
        kh0 = sp.alpha * sp.e0 / 0.3
        assert kh0 == pytest.approx(expected_kh0, rel=1e-4)
        assert sp.kh == pytest.approx(kh0 * (sp.bh / 0.3) ** -0.75, rel=1e-9)


def test_alpha_depends_on_e0_method():
    """孔内水平載荷試験・室内試験では α = 4(常時)/ 8(地震時)。"""
    section = pile_section(CIP_PILE, fck=24)
    profile = sand_profile()
    for method, (a_normal, a_seismic) in (
        (E0Method.N_VALUE, (1.0, 2.0)),
        (E0Method.PLATE_LOADING, (1.0, 2.0)),
        (E0Method.BOREHOLE_LATERAL, (4.0, 8.0)),
        (E0Method.LAB_COMPRESSION, (4.0, 8.0)),
    ):
        normal = lateral_springs(
            CIP_PILE, section, profile, 2.0, LoadCase.PERMANENT, e0_method=method
        )
        seismic = lateral_springs(
            CIP_PILE, section, profile, 2.0, LoadCase.LEVEL1_EQ, e0_method=method
        )
        assert normal.alpha == a_normal
        assert seismic.alpha == a_seismic
        # α が4倍になれば kH も4倍(同じ BH で比較すれば)
        assert normal.alpha * normal.e0 / 0.3 == pytest.approx(
            normal.kh * (normal.bh / 0.3) ** 0.75
        )


def test_storm_uses_normal_alpha():
    """暴風時は常時と同じ α を用いる。"""
    section = pile_section(CIP_PILE, fck=24)
    normal = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.PERMANENT
    )
    storm = lateral_springs(CIP_PILE, section, sand_profile(), 2.0, LoadCase.STORM)
    assert storm.alpha == normal.alpha == 1.0
    assert storm.kh == pytest.approx(normal.kh)


def test_seismic_alpha_doubles_kh():
    """地震時の kH は常時の**ちょうど2倍**(BH が共通だから)。

    「2倍より大きい」を許す書き方をしていたために、BH まで地震時の α で
    反復し直す実装(kH が約 7% 過大)を見逃していた。第29回で修正。
    """
    section = pile_section(CIP_PILE, fck=24)
    normal = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.PERMANENT
    )
    seismic = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.LEVEL1_EQ
    )
    assert seismic.kh == pytest.approx(2.0 * normal.kh, rel=1e-12)
    # α が 1→2 になるぶん杭は相対的に硬くなる
    assert seismic.beta > normal.beta
    assert seismic.k1 > normal.k1


def test_bh_is_determined_under_the_permanent_condition():
    """換算載荷幅 BH は常時の kH で定め、地震時も同じ値を使う(道示Ⅳ 9.6)。"""
    section = pile_section(CIP_PILE, fck=24)
    profile = sand_profile()
    normal = lateral_springs(CIP_PILE, section, profile, 2.0, LoadCase.PERMANENT)
    cases = [
        lateral_springs(CIP_PILE, section, profile, 2.0, case)
        for case in (LoadCase.STORM, LoadCase.LEVEL1_EQ)
    ]
    for sp in cases:
        assert sp.bh == normal.bh
        assert sp.e0 == normal.e0
        # kH は BH を固定したまま α に正比例する
        assert sp.kh == pytest.approx(normal.kh * sp.alpha / normal.alpha, rel=1e-12)

    # β は最終の kH から求め直すので、α の 4乗根の比になる
    seismic = cases[-1]
    assert seismic.beta == pytest.approx(normal.beta * 2.0**0.25, rel=1e-12)
    assert seismic.beta == pytest.approx(
        (seismic.kh * CIP_PILE.diameter / (4 * section.ei)) ** 0.25, rel=1e-12
    )
    # BH は β の**常時の値**から決まるので、この関係は成り立たない
    assert seismic.bh != pytest.approx(math.sqrt(CIP_PILE.diameter / seismic.beta))


def test_bh_ignores_the_liquefaction_reduction():
    """DE は BH の決定には効かせない(BH は常時の条件で定めるため)。"""
    from core.soil.liquefaction import SoilReduction, assess_liquefaction
    from core.standards import GroundMotionType, GroundType
    from tests.test_soil_reduction import liquefiable_profile

    section = pile_section(CIP_PILE, fck=24)
    profile = liquefiable_profile()
    reduction = SoilReduction.from_assessment(
        assess_liquefaction(profile, GroundType.TYPE_II),
        GroundMotionType.LEVEL2_TYPE2,
    )

    plain = lateral_springs(CIP_PILE, section, profile, 2.0, LoadCase.LEVEL1_EQ)
    reduced = lateral_springs(
        CIP_PILE, section, profile, 2.0, LoadCase.LEVEL1_EQ, reduction=reduction
    )
    assert reduced.de < 1.0
    # BH は変わらない
    assert reduced.bh == plain.bh
    # kH には DE がそのまま乗る
    assert reduced.kh == pytest.approx(plain.kh * reduced.de, rel=1e-12)
    # β は低減後の kH から求め直すので下がる(最大曲げ位置が深くなる向き)
    assert reduced.beta < plain.beta


def test_e0_from_n_value():
    section = pile_section(CIP_PILE, fck=24)
    springs = lateral_springs(
        CIP_PILE, section, sand_profile(), 2.0, LoadCase.PERMANENT
    )
    assert springs.e0 == pytest.approx(2800.0 * 10.0)
