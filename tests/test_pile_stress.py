"""杭体応力度照査・杭頭結合部・負の周面摩擦力のテスト。"""
import math

import pytest

from core.capacity.negative_friction import (
    compute_negative_friction,
    negative_friction_intensity,
)
from core.models import (
    ConstructionMethod,
    Footing,
    LoadCase,
    PileArrangement,
    PileSpec,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.section.checks import MaterialSpec, check_section
from core.section.pile_head import (
    check_pile_head,
    edge_distances,
    punching_shear_area,
)
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


def test_storm_case_uses_125_not_150():
    """暴風時の割増は 1.25(地震時 1.5 とは異なる)。"""
    normal = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=2000.0, moment=300.0
    )
    storm = check_section(
        STEEL, MATERIAL, LoadCase.STORM, depth=0.0, axial=2000.0, moment=300.0
    )
    seismic = check_section(
        STEEL, MATERIAL, LoadCase.LEVEL1_EQ, depth=0.0, axial=2000.0, moment=300.0
    )
    base = normal.checks[0].allowable
    assert storm.checks[0].allowable == pytest.approx(base * 1.25)
    assert seismic.checks[0].allowable == pytest.approx(base * 1.50)
    assert storm.checks[0].allowable < seismic.checks[0].allowable


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
    rc = PileSpec(
        pile_type=PileType.RC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
        concrete_thickness=90.0,
    )
    with pytest.raises(NotImplementedError):
        check_section(rc, MATERIAL, LoadCase.PERMANENT, 0.0, 1000.0, 100.0)


# --- PHC杭(全断面有効) ---------------------------------------------------

PHC = PileSpec(
    pile_type=PileType.PHC,
    method=ConstructionMethod.DRIVEN,
    diameter=0.6,
    length=20.0,
    concrete_thickness=90.0,
)


def _phc_section():
    """テストの期待値算定に用いる中空断面の A・Z。"""
    outer, inner = 0.6, 0.6 - 2 * 0.09
    area = math.pi * (outer**2 - inner**2) / 4.0
    inertia = math.pi * (outer**4 - inner**4) / 64.0
    return area, inertia / (outer / 2.0)


def test_phc_stress_matches_hand_calculation():
    area, z = _phc_section()
    axial, moment = 2000.0, 100.0
    result = check_section(PHC, MATERIAL, LoadCase.PERMANENT, 0.0, axial, moment)

    sigma_n = axial / area / 1000.0
    sigma_b = moment / z / 1000.0
    by_name = {c.name: c for c in result.checks}
    assert by_name["軸圧縮応力度"].stress == pytest.approx(sigma_n)
    assert by_name["曲げ圧縮応力度"].stress == pytest.approx(sigma_n + sigma_b)
    # 圧縮側が卓越しているため引張の照査は現れない
    assert "曲げ引張応力度" not in by_name
    assert by_name["軸圧縮応力度"].allowable == pytest.approx(23.0)
    assert by_name["曲げ圧縮応力度"].allowable == pytest.approx(27.0)
    assert result.all_ok


def test_phc_allowable_compression_is_increased_by_load_case():
    result = check_section(PHC, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 2000.0, 100.0)
    by_name = {c.name: c for c in result.checks}
    increase = STRESS_INCREASE[LoadCase.LEVEL1_EQ.value]
    assert by_name["曲げ圧縮応力度"].allowable == pytest.approx(27.0 * increase)
    assert by_name["軸圧縮応力度"].allowable == pytest.approx(23.0 * increase)


def test_phc_permanent_case_allows_no_tension():
    # 曲げが卓越して引張が生じるケース
    result = check_section(PHC, MATERIAL, LoadCase.PERMANENT, 0.0, 100.0, 400.0)
    tension = next(c for c in result.checks if c.name == "曲げ引張応力度")
    assert tension.stress > 0.0
    assert tension.allowable == 0.0
    assert tension.judgement == "NG"
    assert not result.all_ok


def test_phc_seismic_tension_allowable_depends_on_prestress():
    high = MaterialSpec(effective_prestress=8.0)
    mid = MaterialSpec(effective_prestress=5.0)
    low = MaterialSpec(effective_prestress=2.0)
    for material, expected in ((high, 5.0), (mid, 3.0), (low, 0.0)):
        result = check_section(
            PHC, material, LoadCase.LEVEL1_EQ, 0.0, 100.0, 400.0
        )
        tension = next(c for c in result.checks if c.name == "曲げ引張応力度")
        # 地震時の許容曲げ引張応力度には割増しを重ねない
        assert tension.allowable == pytest.approx(expected)


def test_phc_seismic_tension_requires_prestress_input():
    with pytest.raises(ValueError, match="有効プレストレス"):
        check_section(PHC, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 100.0, 400.0)
    # 引張が生じなければ σce の入力は不要
    check_section(PHC, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 2000.0, 100.0)


def test_phc_requires_concrete_thickness():
    thin = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
    )
    with pytest.raises(ValueError, match="肉厚"):
        check_section(thin, MATERIAL, LoadCase.PERMANENT, 0.0, 1000.0, 100.0)


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
    """押込み力に対する押抜きせん断と支圧を照査する(道示Ⅳ 12.9.3)。"""
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
    assert "杭頭支圧応力度" in names
    tau = next(c for c in result.checks if "押抜き" in c.name)
    assert tau.stress == pytest.approx(2000.0 / result.punching_area / 1000.0)
    assert tau.allowable == pytest.approx(0.90)
    # 支圧は軸力を杭断面積で除した値、許容値は安全側に σca を用いる
    bearing = next(c for c in result.checks if "支圧" in c.name)
    assert bearing.stress == pytest.approx(2000.0 / (math.pi / 4) / 1000.0)
    assert bearing.allowable == pytest.approx(8.0)


def test_punching_shear_allowable_is_not_increased():
    """杭頭結合部の τa3 には荷重組合せによる割増しを行わない(道示Ⅳ 4.2)。"""
    allowables = {}
    for case in (LoadCase.PERMANENT, LoadCase.STORM, LoadCase.LEVEL1_EQ):
        result = check_pile_head(1.0, 1.5, 24, case, 2000.0, 200.0, 150.0)
        tau = next(c for c in result.checks if "押抜き" in c.name)
        allowables[case] = tau.allowable
    # 地震時・暴風時も常時と同じ 0.90
    for case, allowable in allowables.items():
        assert allowable == pytest.approx(0.90), case

    # 一方、支圧は通常どおり割増しされる(原典未確認のため現状の扱い)
    normal = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 0.0, 0.0)
    seismic = check_pile_head(1.0, 1.5, 24, LoadCase.LEVEL1_EQ, 2000.0, 0.0, 0.0)
    n_b = next(c for c in normal.checks if "支圧" in c.name)
    s_b = next(c for c in seismic.checks if "支圧" in c.name)
    assert s_b.allowable == pytest.approx(n_b.allowable * 1.5)


def test_punching_shear_rejects_grade_outside_table():
    """τa3 の表は σck = 21〜30 のみ。範囲外は明示的にエラー。"""
    with pytest.raises(ValueError, match="範囲外"):
        check_pile_head(1.0, 1.5, 40, LoadCase.PERMANENT, 2000.0, 0.0, 0.0)


def test_removed_rebar_grade_is_rejected():
    """SD295 は H24 の道示Ⅳで削除されており選択できない。"""
    material = MaterialSpec(
        fck=24,
        rebar_grade="SD295",
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
    )
    with pytest.raises(ValueError, match="削除"):
        check_section(CIP, material, LoadCase.PERMANENT, 0.0, 1500.0, 800.0)


def test_unknown_rebar_grade_is_rejected():
    material = MaterialSpec(
        fck=24,
        rebar_grade="SD490",
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
    )
    with pytest.raises(ValueError, match="未対応"):
        check_section(CIP, material, LoadCase.PERMANENT, 0.0, 1500.0, 800.0)


def test_pile_head_bearing_ignores_uplift():
    """支圧は押込み時のみ。引抜き時は 0 とする。"""
    result = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, -2000.0, 0.0, 0.0)
    bearing = next(c for c in result.checks if "支圧" in c.name)
    assert bearing.stress == 0.0


def test_edge_distance_standard():
    """縁端距離が 1.0D 以上なら標準、未満なら水平押抜きせん断の照査が必要。"""
    arrangement = PileArrangement(nx=2, ny=3, spacing_x=2.5, spacing_y=2.5)
    # 橋軸方向: 幅8.0 → 8/2 − 1.25 = 2.75m、直角方向: 8/2 − 2.5 = 1.5m
    wide = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)
    edge = edge_distances(wide, arrangement, diameter=1.0)
    assert edge.edge_x == pytest.approx(2.75)
    assert edge.edge_y == pytest.approx(1.5)
    assert edge.minimum == pytest.approx(1.5)
    assert edge.required == pytest.approx(1.0)
    assert edge.is_standard
    assert not edge.needs_horizontal_punching_check

    # 幅を詰めると縁端距離が 1.0D を下回る
    narrow = Footing(width_x=8.0, width_y=5.5, height=1.5, embedment=2.0)
    edge = edge_distances(narrow, arrangement, diameter=1.0)
    assert edge.edge_y == pytest.approx(0.25)
    assert not edge.is_standard
    assert edge.needs_horizontal_punching_check


def test_edge_distance_scales_with_diameter():
    """必要縁端距離は杭径に比例する。"""
    arrangement = PileArrangement(nx=2, ny=2, spacing_x=3.0, spacing_y=3.0)
    footing = Footing(width_x=6.0, width_y=6.0, height=2.0, embedment=2.0)
    # 縁端距離 = 3.0 − 1.5 = 1.5m
    assert edge_distances(footing, arrangement, 1.0).is_standard
    assert not edge_distances(footing, arrangement, 2.0).is_standard


def test_check_pile_head_includes_edge_distance_when_given():
    arrangement = PileArrangement(nx=2, ny=3, spacing_x=2.5, spacing_y=2.5)
    footing = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)
    with_edge = check_pile_head(
        1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 200.0, 150.0,
        footing=footing, arrangement=arrangement,
    )
    assert with_edge.edge_distance is not None
    without = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 200.0, 150.0)
    assert without.edge_distance is None


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
