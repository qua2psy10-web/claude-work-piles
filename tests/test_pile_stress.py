"""杭体応力度照査・杭頭結合部・負の周面摩擦力のテスト。"""
import math

import pytest

from core.capacity.negative_friction import (
    compute_negative_friction,
    negative_friction_intensity,
)
from core.models import (
    ConstructionMethod,
    LoadCase,
    PileSpec,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.section.checks import MaterialSpec, check_section
from core.section.pile_head import check_pile_head, punching_shear_area
from core.section.rc import RebarLayout
from core.standards import SIGMA_A_STEEL, SIGMA_SA_REBAR, STRESS_INCREASE

CIP = PileSpec(
    pile_type=PileType.CAST_IN_PLACE,
    method=ConstructionMethod.CAST_IN_PLACE,
    diameter=1.0,
    length=20.0,
)
STEEL = PileSpec(
    pile_type=PileType.STEEL_PIPE,
    method=ConstructionMethod.DRIVEN,
    diameter=1.0,
    length=20.0,
    wall_thickness=12.0,
)
MATERIAL = MaterialSpec(
    fck=24, rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)
)


def test_steel_pipe_stress_hand_calculation():
    """σ = N/A ± M/Z。板厚12mm − 腐食代1mm = 11mm。"""
    t = 0.011
    d_in = 1.0 - 2 * t
    area = math.pi * (1.0 - d_in**2) / 4
    inertia = math.pi * (1.0 - d_in**4) / 64
    z = inertia / 0.5

    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=2000.0, moment=300.0
    )
    compression = next(c for c in result.checks if "圧縮" in c.name)
    expected = (2000.0 / area + 300.0 / z) / 1000.0
    assert compression.stress == pytest.approx(expected, rel=1e-9)
    assert compression.allowable == pytest.approx(SIGMA_A_STEEL["SKK400"])


def test_steel_pipe_tension_side():
    """曲げが軸圧縮を上回ると引張側が生じる。"""
    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=100.0, moment=2000.0
    )
    tension = next(c for c in result.checks if "引張" in c.name)
    assert tension.stress > 0


def test_no_tension_when_axial_dominates():
    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=5000.0, moment=50.0
    )
    tension = next(c for c in result.checks if "引張" in c.name)
    assert tension.stress == 0.0


def test_seismic_case_increases_allowable():
    normal = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=2000.0, moment=300.0
    )
    seismic = check_section(
        STEEL, MATERIAL, LoadCase.LEVEL1_EQ, depth=0.0, axial=2000.0, moment=300.0
    )
    assert seismic.checks[0].allowable == pytest.approx(
        normal.checks[0].allowable * STRESS_INCREASE["レベル1地震時"]
    )
    # 発生応力度は同じ
    assert seismic.checks[0].stress == pytest.approx(normal.checks[0].stress)


def test_cast_in_place_checks_concrete_and_rebar():
    result = check_section(
        CIP, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=1500.0, moment=800.0
    )
    names = [c.name for c in result.checks]
    assert "コンクリート圧縮応力度" in names
    assert "鉄筋引張応力度" in names
    assert result.rc_detail is not None
    # 許容値は水中施工を考慮して 0.8 倍(σck=24 → 8×0.8 = 6.4)
    concrete = next(c for c in result.checks if "コンクリート" in c.name)
    assert concrete.allowable == pytest.approx(6.4)
    rebar = next(c for c in result.checks if "鉄筋" in c.name)
    assert rebar.allowable == pytest.approx(SIGMA_SA_REBAR["SD345"])


def test_cast_in_place_requires_rebar():
    with pytest.raises(ValueError, match="鉄筋"):
        check_section(
            CIP, MaterialSpec(fck=24), LoadCase.PERMANENT, 0.0, 1500.0, 800.0
        )


def test_unimplemented_pile_type_raises():
    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
    )
    with pytest.raises(NotImplementedError):
        check_section(phc, MATERIAL, LoadCase.PERMANENT, 0.0, 1000.0, 100.0)


def test_stress_check_judgement():
    # 過大な断面力で NG になること
    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=50000.0, moment=5000.0
    )
    assert not result.all_ok


# --- 杭頭結合部 -------------------------------------------------------------


def test_punching_shear_area():
    # h = 1.5 − 0.1 = 1.4 → A = π(1.0+1.4)×1.4
    assert punching_shear_area(1.0, 1.5) == pytest.approx(math.pi * 2.4 * 1.4)


def test_punching_area_requires_thickness():
    with pytest.raises(ValueError, match="フーチング厚"):
        punching_shear_area(1.0, 0.05)


def test_pile_head_checks():
    result = check_pile_head(
        pile_diameter=1.0,
        footing_height=1.5,
        fck=24,
        case=LoadCase.PERMANENT,
        axial=2000.0,
        shear=200.0,
        moment=150.0,
    )
    names = [c.name for c in result.checks]
    assert "杭頭押抜きせん断応力度" in names
    assert "杭頭水平支圧応力度" in names
    tau = next(c for c in result.checks if "押抜き" in c.name)
    assert tau.stress == pytest.approx(2000.0 / result.punching_area / 1000.0)
    assert tau.allowable == pytest.approx(0.90)


def test_pile_head_uplift_uses_absolute_value():
    push = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 0.0, 0.0)
    pull = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, -2000.0, 0.0, 0.0)
    assert push.checks[0].stress == pytest.approx(pull.checks[0].stress)


# --- 負の周面摩擦力 ---------------------------------------------------------


def nf_profile() -> SoilProfile:
    return SoilProfile(
        layers=[
            SoilLayer(
                name="Ac", soil_type=SoilType.CLAY, thickness=10.0, n_value=3.0,
                gamma_t=15.0, gamma_sat=15.5, cohesion=30.0,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=40.0,
                gamma_t=19.0, gamma_sat=20.0,
            ),
        ],
        gwl=1.0,
    )


def test_negative_friction_intensity_clay_uses_cohesion():
    clay = nf_profile().layers[0]
    assert negative_friction_intensity(clay, 100.0) == 30.0


def test_negative_friction_intensity_sand_uses_effective_stress():
    sand = nf_profile().layers[1]
    assert negative_friction_intensity(sand, 100.0) == pytest.approx(30.0)


def test_negative_friction_neutral_point_at_soft_layer_bottom():
    result = compute_negative_friction(
        CIP, nf_profile(), embedment=1.0, dead_load=1000.0, ru=9000.0
    )
    # 圧密層(Ac, N=3)の下端 10m が中立点
    assert result.neutral_depth == pytest.approx(10.0)
    # NF = π×1.0×(10−1)×30 = 848.2 kN
    assert result.nf == pytest.approx(math.pi * 9.0 * 30.0, rel=1e-6)
    assert result.n_max == pytest.approx(1000.0 + result.nf)
    assert result.allowable == pytest.approx(9000.0 / 1.2)
    assert result.ok


def test_negative_friction_absent_without_soft_layer():
    profile = SoilProfile(
        layers=[
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=30.0, n_value=40.0,
                gamma_t=19.0, gamma_sat=20.0,
            )
        ],
        gwl=1.0,
    )
    result = compute_negative_friction(
        CIP, profile, embedment=1.0, dead_load=1000.0, ru=9000.0
    )
    assert result.nf == 0.0
    assert result.n_max == 1000.0


def test_negative_friction_ng_when_excessive():
    result = compute_negative_friction(
        CIP, nf_profile(), embedment=1.0, dead_load=8000.0, ru=5000.0
    )
    assert not result.ok
    assert result.judgement == "NG"
