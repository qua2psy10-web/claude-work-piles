"""レベル2地震時照査(プッシュオーバー解析)のテスト。"""
import math

import pytest

from core.analysis.level2 import (
    AxialSpringModel,
    analyze_level2,
    pushover,
    yield_moment_steel_pipe,
)
from core.capacity.bearing import compute_bearing_capacity
from core.capacity.section import pile_section
from core.capacity.springs import PileSection, axial_spring, lateral_springs
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
from core.standards import SIGMA_Y_STEEL

ARRANGEMENT = PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5)

# 弾性域の挙動を手計算と突合するための素直なバネ値。
# 半無限長杭では K1・K4 = 2・K2² が成り立つ(test_springs で確認済み)。
K1, K2, K4 = 2.0e5, -1.0e5, 1.0e5
KV = 5.0e5


def spring(push=4000.0, pull=1500.0, kv=KV):
    return AxialSpringModel(kv=kv, push_limit=push, pull_limit=pull)


# --- バイリニアバネ ---------------------------------------------------------


def test_axial_spring_is_linear_within_limits():
    s = spring()
    assert s.reaction(0.001) == pytest.approx(500.0)
    assert s.reaction(-0.001) == pytest.approx(-500.0)
    assert s.tangent(0.001) == pytest.approx(KV)
    assert not s.is_plastic(0.001)


def test_axial_spring_saturates_at_limits():
    s = spring()
    # 押込み側: 4000 / 5e5 = 0.008 m で上限
    assert s.reaction(0.02) == pytest.approx(4000.0)
    assert s.tangent(0.02) == 0.0
    assert s.is_plastic(0.02)
    # 引抜き側: 1500 / 5e5 = 0.003 m
    assert s.reaction(-0.02) == pytest.approx(-1500.0)
    assert s.is_plastic(-0.02)


def test_axial_spring_from_bearing_uses_safety_factor_one():
    """上限値は許容応力度設計法の式で n = 1 とした値であること。"""
    profile = SoilProfile(
        layers=[
            SoilLayer(
                name="As", soil_type=SoilType.SAND, thickness=10.0, n_value=15.0,
                gamma_t=18.0, gamma_sat=19.0,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=45.0,
                gamma_t=19.0, gamma_sat=20.0,
            ),
        ],
        gwl=2.0,
    )
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    bearing = compute_bearing_capacity(pile, profile, 2.0)
    model = AxialSpringModel.from_bearing(KV, bearing)

    assert model.push_limit == pytest.approx(bearing.ru - bearing.w_pile)
    assert model.pull_limit == pytest.approx(
        bearing.skin_resistance + bearing.w_pile
    )
    # 常時の許容値より大きいこと(安全率で除していないため)
    assert model.push_limit > bearing.allowable_push(LoadCase.PERMANENT)
    assert model.pull_limit > bearing.allowable_pull(LoadCase.PERMANENT)


def test_axial_spring_rejects_invalid_input():
    with pytest.raises(ValueError):
        AxialSpringModel(kv=0.0, push_limit=1000.0, pull_limit=500.0)
    with pytest.raises(ValueError):
        AxialSpringModel(kv=KV, push_limit=-1.0, pull_limit=500.0)


# --- プッシュオーバー -------------------------------------------------------


def test_elastic_range_matches_linear_solution():
    """上限に達しないうちは線形の変位法と一致すること。"""
    from core.analysis.displacement import solve_stability

    s = spring(push=1.0e6, pull=1.0e6)  # 実質的に線形
    steps, _ = pushover(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=1.0, steps=10,
    )
    last = steps[-1]
    linear = solve_stability(
        ARRANGEMENT, kv=KV, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
    )
    assert last.u == pytest.approx(linear.u, rel=1e-6)
    assert last.v == pytest.approx(linear.v, rel=1e-6)
    assert last.theta == pytest.approx(linear.theta, rel=1e-6)
    assert last.plastic_axial == 0


def test_equilibrium_is_satisfied_at_every_step():
    s = spring()
    steps, _ = pushover(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=2.0, steps=40,
    )
    assert steps
    for step in steps:
        assert sum(r.axial for r in step.reactions) == pytest.approx(
            9000.0, rel=1e-6
        )
        assert sum(r.shear for r in step.reactions) == pytest.approx(
            step.h, rel=1e-6
        )
        moment = sum(
            r.moment + r.x * r.axial for r in step.reactions
        )
        assert moment == pytest.approx(step.m, rel=1e-6, abs=1e-6)


def test_displacement_increases_monotonically():
    s = spring()
    steps, _ = pushover(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=2.0, steps=40,
    )
    us = [s_.u for s_ in steps]
    assert us == sorted(us)


def test_yield_detected_when_push_limit_reached():
    # 押込み上限を低くして必ず降伏させる
    s = spring(push=1500.0, pull=1500.0)
    steps, yield_point = pushover(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=2.0, steps=60,
    )
    assert yield_point is not None
    assert "押込み" in yield_point.reason
    assert yield_point.step.max_axial == pytest.approx(1500.0, rel=1e-6)
    assert yield_point.displacement > 0


def test_no_yield_when_capacity_is_ample():
    s = spring(push=1.0e7, pull=1.0e7)
    _, yield_point = pushover(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=1.0, steps=20,
    )
    assert yield_point is None


def test_yield_by_pile_body_requires_all_piles():
    """杭体降伏による降伏判定は「全杭」が条件であること。"""
    s = spring(push=1.0e7, pull=1.0e7)
    # 直杭・杭頭剛結では全杭の杭頭モーメントが等しいため、
    # My を超えた時点で全杭が同時に降伏する
    steps, yield_point = pushover(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        yield_moment=100.0, max_factor=2.0, steps=60,
    )
    assert yield_point is not None
    assert "杭体" in yield_point.reason
    assert yield_point.step.yielded_piles == ARRANGEMENT.nx * ARRANGEMENT.ny
    assert abs(yield_point.step.reactions[0].moment) >= 100.0


def test_pushover_validates_arguments():
    s = spring()
    with pytest.raises(ValueError):
        pushover(ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
                 v_load=1.0, h_load=1.0, m_load=1.0, steps=0)
    with pytest.raises(ValueError):
        pushover(ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
                 v_load=1.0, h_load=1.0, m_load=1.0, max_factor=0.0)


# --- 照査 -------------------------------------------------------------------


def test_no_yield_case_skips_ductility_check():
    s = spring(push=1.0e7, pull=1.0e7)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        allowable_ductility=4.0, allowable_displacement=0.5,
    )
    assert not result.yielded
    assert result.response_ductility is None
    names = [c.name for c in result.checks]
    assert "基礎の降伏" in names
    assert "応答塑性率" not in names
    assert result.all_ok


def test_response_ductility_is_ratio_of_displacements():
    s = spring(push=1500.0, pull=1500.0)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        allowable_ductility=4.0, allowable_displacement=1.0,
    )
    assert result.yielded
    mu = result.response_ductility
    assert mu == pytest.approx(
        result.response.u / result.yield_point.displacement
    )
    assert mu >= 1.0
    check = next(c for c in result.checks if c.name == "応答塑性率")
    assert check.capacity == 4.0
    assert check.judgement == ("OK" if mu <= 4.0 else "NG")


def test_allowable_ductility_is_not_assumed_at_the_low_level_api():
    """低水準 API では μa を勝手に決めない(杭種・下部構造を知らないため)。"""
    s = spring(push=1500.0, pull=1500.0)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
    )
    assert result.allowable_ductility is None
    assert result.allowable_displacement is None
    assert not any(c.name == "応答塑性率" for c in result.checks)
    assert any("μa" in note for note in result.notes)


def test_footing_rotation_is_checked_against_the_standard_value():
    """フーチング底面の回転角は道示Ⅴ の 0.02 rad を既定で照査すること。"""
    from core.standards import ALLOWABLE_FOOTING_ROTATION

    assert ALLOWABLE_FOOTING_ROTATION == 0.02
    s = spring(push=1500.0, pull=1500.0)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
    )
    check = next(c for c in result.checks if "回転角" in c.name)
    assert check.capacity == pytest.approx(0.02)
    assert check.unit == "rad"
    # 応答値は回転角の絶対値(水平変位ではない)
    assert check.demand == pytest.approx(abs(result.response.theta))


def test_footing_rotation_check_can_fail():
    """回転角が 0.02 rad を超えれば NG になること。"""
    # 引抜き側の上限を極端に小さくすると回転が進む
    s = spring(push=1.0e7, pull=10.0, kv=1.0e4)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=200000.0,
    )
    assert result.response is not None
    check = next(c for c in result.checks if "回転角" in c.name)
    assert check.demand > 0.02
    assert check.judgement == "NG"
    assert not result.all_ok


# --- 許容塑性率の決定 -------------------------------------------------------


def test_allowable_ductility_by_structure_type():
    from core.analysis.level2 import allowable_ductility_for
    from core.standards import StructureType

    assert allowable_ductility_for(StructureType.PIER) == 4.0
    assert allowable_ductility_for(StructureType.ABUTMENT) == 3.0


def test_allowable_ductility_drops_for_cast_in_place_with_high_grade_rebar():
    from core.analysis.level2 import allowable_ductility_for
    from core.standards import StructureType

    cip = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    for grade in ("SD390", "SD490"):
        assert allowable_ductility_for(StructureType.PIER, cip, grade) == 2.0
        # 橋台は基礎の塑性化を考慮できない
        assert allowable_ductility_for(StructureType.ABUTMENT, cip, grade) is None
    # SD345 は通常の値
    assert allowable_ductility_for(StructureType.PIER, cip, "SD345") == 4.0


def test_high_grade_rebar_rule_applies_only_to_cast_in_place():
    from core.analysis.level2 import allowable_ductility_for
    from core.standards import StructureType

    assert allowable_ductility_for(StructureType.PIER, STEEL, "SD390") == 4.0


def test_run_level2_sets_allowable_ductility_automatically():
    from core.analysis.level2 import run_level2
    from core.standards import StructureType

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
        structure_type=StructureType.ABUTMENT,
    )
    assert result.allowable_ductility == 3.0
    assert any("μa = 3" in n for n in result.notes)


def test_run_level2_warns_when_plasticity_not_allowed():
    from core.analysis.level2 import run_level2
    from core.standards import StructureType

    cip = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    result = run_level2(
        cip, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
        structure_type=StructureType.ABUTMENT,
        rebar_grade="SD390",
    )
    assert result.allowable_ductility is None
    assert any("塑性化を考慮できない" in n for n in result.notes)


def test_limitations_are_reported():
    s = spring()
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
    )
    joined = "".join(result.notes)
    assert "pHU" in joined  # 水平地盤反力の非線形が未実装であること
    assert "M-φ" in joined


def test_unstable_before_design_load_has_no_response():
    """設計荷重に達する前に崩壊する場合、応答は None になること。"""
    # 支持力の上限が鉛直荷重の合計を下回ると釣合いが取れない
    s = spring(push=500.0, pull=100.0)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
    )
    assert result.response is None
    assert not result.yielded
    assert any("保有水平耐力" in note for note in result.notes)


# --- 降伏曲げモーメント -----------------------------------------------------


STEEL = PileSpec(
    pile_type=PileType.STEEL_PIPE,
    method=ConstructionMethod.DRIVEN,
    diameter=1.0,
    length=20.0,
    wall_thickness=12.0,
)


def test_yield_moment_steel_pipe_hand_calculation():
    section = pile_section(STEEL)
    z = section.inertia / 0.5
    axial = 2000.0
    expected = (
        SIGMA_Y_STEEL["SKK400"] - axial / section.area / 1000.0
    ) * z * 1000.0
    assert yield_moment_steel_pipe(STEEL, section, axial) == pytest.approx(expected)


def test_yield_moment_decreases_with_axial_force():
    section = pile_section(STEEL)
    assert yield_moment_steel_pipe(STEEL, section, 5000.0) < yield_moment_steel_pipe(
        STEEL, section, 0.0
    )


def test_yield_moment_higher_grade_is_larger():
    section = pile_section(STEEL)
    assert yield_moment_steel_pipe(
        STEEL, section, 2000.0, steel_grade="SKK490"
    ) > yield_moment_steel_pipe(STEEL, section, 2000.0, steel_grade="SKK400")


def test_yield_moment_rejects_non_steel_pipe():
    cip = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=20.0,
    )
    with pytest.raises(ValueError, match="鋼管杭"):
        yield_moment_steel_pipe(cip, pile_section(cip), 1000.0)


def test_yield_moment_rejects_unknown_grade():
    section = pile_section(STEEL)
    with pytest.raises(ValueError, match="降伏点"):
        yield_moment_steel_pipe(STEEL, section, 1000.0, steel_grade="SS400")


def test_yield_moment_rejects_axial_beyond_yield():
    section = pile_section(STEEL)
    huge = SIGMA_Y_STEEL["SKK400"] * section.area * 1000.0
    with pytest.raises(ValueError, match="降伏点"):
        yield_moment_steel_pipe(STEEL, section, huge)


def test_design_load_is_always_sampled():
    """λ = 1 が分割の刻みに乗らなくても応答が得られること。"""
    from core.analysis.level2 import _load_factors

    # 3.0 / 100 = 0.03 刻み → λ = 1.0 はちょうどには現れない
    factors = _load_factors(3.0, 100)
    assert any(abs(f - 1.0) < 1e-12 for f in factors)
    assert factors == sorted(factors)

    s = spring(push=1.0e7, pull=1.0e7)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=3.0, steps=100,
    )
    assert result.response is not None
    assert result.response.factor == pytest.approx(1.0)
    assert result.response.h == pytest.approx(2000.0)


def test_max_factor_below_one_has_no_response():
    """λmax < 1 なら設計荷重に達しないため応答は得られない。"""
    s = spring(push=1.0e7, pull=1.0e7)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
        max_factor=0.5, steps=10,
    )
    assert result.response is None


# --- 一括実行 ---------------------------------------------------------------


def sample_ground():
    return SoilProfile(
        layers=[
            SoilLayer(
                name="As", soil_type=SoilType.SAND, thickness=10.0, n_value=15.0,
                gamma_t=18.0, gamma_sat=19.0,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=45.0,
                gamma_t=19.0, gamma_sat=20.0,
            ),
        ],
        gwl=2.0,
    )


FOOTING = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)


def test_run_level2_auto_yield_moment_for_steel_pipe():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
        allowable_ductility=4.0, allowable_displacement=0.3,
    )
    assert result.steps
    assert any("Mp" in n for n in result.notes)
    assert result.response is not None
    assert result.response.h == pytest.approx(3000.0)


def test_run_level2_notes_missing_yield_moment_for_other_types():
    from core.analysis.level2 import run_level2

    cip = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    result = run_level2(
        cip, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert any("My が未入力" in n for n in result.notes)


def test_run_level2_uses_bearing_capacity_limits():
    """自動算定された上限値が支持力計算と一致すること。"""
    from core.analysis.level2 import AxialSpringModel, run_level2

    profile = sample_ground()
    bearing = compute_bearing_capacity(STEEL, profile, FOOTING.embedment)
    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, profile,
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    limit = bearing.ru - bearing.w_pile
    assert all(
        r.axial <= limit + 1e-6 for step in result.steps for r in step.reactions
    )


# --- 全塑性モーメント Mp ----------------------------------------------------


def test_plastic_moment_without_axial_force():
    """N = 0 のとき Mp0 = 4・σy・t・r²(薄肉厳密解)。"""
    from core.analysis.level2 import plastic_moment_steel_pipe

    t = 0.011  # 12mm − 腐食代 1mm
    r = (1.0 - t) / 2.0
    expected = 4.0 * SIGMA_Y_STEEL["SKK400"] * t * r**2 * 1000.0
    assert plastic_moment_steel_pipe(STEEL, 0.0) == pytest.approx(expected)


def test_plastic_moment_agrees_with_exact_hollow_section():
    """薄肉近似は中実解 Zp =(D³−d³)/6 と 1% 以内で一致すること。"""
    from core.analysis.level2 import (
        plastic_moment_steel_pipe,
        plastic_section_modulus_hollow,
    )

    t = 0.011
    exact = plastic_section_modulus_hollow(1.0, t) * SIGMA_Y_STEEL["SKK400"] * 1000.0
    assert plastic_moment_steel_pipe(STEEL, 0.0) == pytest.approx(exact, rel=0.01)


def test_plastic_moment_exceeds_first_yield_moment():
    """全塑性モーメントは最外縁降伏モーメントより大きいこと。"""
    from core.analysis.level2 import plastic_moment_steel_pipe

    section = pile_section(STEEL)
    for axial in (0.0, 1000.0, 3000.0):
        mp = plastic_moment_steel_pipe(STEEL, axial)
        my = yield_moment_steel_pipe(STEEL, section, axial)
        assert mp > my
    # 形状係数 Zp/Z は円環で 4/π ≒ 1.27 程度
    assert plastic_moment_steel_pipe(STEEL, 0.0) / yield_moment_steel_pipe(
        STEEL, section, 0.0
    ) == pytest.approx(4.0 / math.pi, rel=0.02)


def test_plastic_moment_decreases_with_axial_force():
    from core.analysis.level2 import plastic_moment_steel_pipe

    values = [plastic_moment_steel_pipe(STEEL, n) for n in (0.0, 2000.0, 5000.0)]
    assert values[0] > values[1] > values[2]


def test_plastic_moment_vanishes_at_squash_load():
    """全塑性軸力に近づくと曲げ耐力が 0 に近づくこと。"""
    from core.analysis.level2 import plastic_moment_steel_pipe

    t = 0.011
    r = (1.0 - t) / 2.0
    squash = SIGMA_Y_STEEL["SKK400"] * 2 * math.pi * r * t * 1000.0
    assert plastic_moment_steel_pipe(STEEL, squash * 0.999) == pytest.approx(
        0.0, abs=squash * 1e-3
    )
    with pytest.raises(ValueError, match="全塑性軸力"):
        plastic_moment_steel_pipe(STEEL, squash)


def test_plastic_moment_accounts_for_corrosion():
    from core.analysis.level2 import plastic_moment_steel_pipe

    assert plastic_moment_steel_pipe(STEEL, 0.0, corrosion_mm=3.0) < (
        plastic_moment_steel_pipe(STEEL, 0.0, corrosion_mm=0.0)
    )


def test_plastic_moment_rejects_non_steel_pipe():
    from core.analysis.level2 import plastic_moment_steel_pipe

    cip = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=20.0,
    )
    with pytest.raises(ValueError, match="鋼管杭"):
        plastic_moment_steel_pipe(cip, 1000.0)


def test_run_level2_uses_plastic_moment_for_steel_pipe():
    """鋼管杭の杭体降伏判定は Mp を用いること(最外縁降伏ではない)。"""
    from core.analysis.level2 import plastic_moment_steel_pipe, run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    mean_axial = 9000.0 / 9
    mp = plastic_moment_steel_pipe(STEEL, mean_axial)
    assert any(f"Mp = {mp:.0f}" in n for n in result.notes)
    assert any("バイリニア" in n for n in result.notes)


# --- pHU との突合診断 -------------------------------------------------------


def ground_with_kep(k_ep=3.0):
    return SoilProfile(
        layers=[
            SoilLayer(
                name="As", soil_type=SoilType.SAND, thickness=10.0, n_value=15.0,
                gamma_t=18.0, gamma_sat=19.0, k_ep=k_ep,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=45.0,
                gamma_t=19.0, gamma_sat=20.0, k_ep=k_ep,
            ),
        ],
        gwl=2.0,
    )


def test_diagnosis_skipped_without_kep():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert result.soil_reaction is None
    assert any("KEP が未入力" in n for n in result.notes)


def test_diagnosis_runs_when_kep_is_given():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert result.soil_reaction is not None
    assert result.soil_reaction.points
    # 3列配置なので最前列以外(1/2 が効く側)で判定する
    assert result.soil_reaction.front_row is False


def test_diagnosis_flags_exceedance_as_unsafe_in_elastic_mode():
    """弾性解析を選んだ場合、pHU 超過は非安全側である旨が注記されること。"""
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=0.01),
        v_load=9000.0, h_load=3000.0, m_load=12000.0, use_bnwf=False,
    )
    assert result.soil_reaction is not None
    assert not result.soil_reaction.ok
    assert result.soil_reaction.exceeded_depth_range is not None
    assert any("非安全側" in n for n in result.notes)


def test_bnwf_mode_does_not_call_the_exceedance_unsafe():
    """分布バネモデルで解いた場合、超過は解析に反映済みなので非安全側ではない。"""
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=0.01),
        v_load=9000.0, h_load=3000.0, m_load=12000.0, steps=20,
    )
    assert not result.soil_reaction.ok
    assert not any("非安全側" in n for n in result.notes)
    assert any("考慮している" in n for n in result.notes)


def test_diagnosis_reports_ok_when_within_limit():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=50.0),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert result.soil_reaction is not None
    assert result.soil_reaction.ok
    assert result.soil_reaction.max_ratio < 1.0
    assert any("上限値 pHU 以下" in n for n in result.notes)


def test_diagnosis_depths_are_measured_from_ground_surface():
    """診断の深さは杭頭からではなく地表面からであること。"""
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    shallowest = min(p.depth for p in result.soil_reaction.points)
    assert shallowest == pytest.approx(FOOTING.embedment)


# --- BNWF の統合 ------------------------------------------------------------


def test_bnwf_with_huge_limits_converges_to_elastic_pushover():
    """pHU を十分大きくすると、従来の K1〜K4 による解析に収束すること。

    BNWF は離散化した数値解なので厳密一致はしない。分割を細かくすると
    弾性解(Chang の式に基づく K1〜K4)との差が 2次で減ることを確認する。
    """
    from core.analysis.level2 import run_level2

    kwargs = dict(
        v_load=9000.0, h_load=3000.0, m_load=12000.0, max_factor=1.0, steps=10,
    )
    elastic = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(), use_bnwf=False, **kwargs
    )
    errors = []
    for n in (100, 200, 400):
        # KEP を極端に大きくすれば pHU は事実上無限大
        bnwf = run_level2(
            STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=1.0e6),
            bnwf_elements=n, **kwargs
        )
        assert bnwf.response.plastic_ground_nodes == 0
        errors.append(
            abs(
                bnwf.response.reactions[0].moment
                / elastic.response.reactions[0].moment
                - 1.0
            )
        )
        finest = bnwf

    assert errors[0] < 0.02
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse / 3.0  # 2次収束
    # 最も細かい分割では実用上一致する
    assert finest.response.u == pytest.approx(elastic.response.u, rel=1e-4)
    assert finest.response.theta == pytest.approx(elastic.response.theta, rel=1e-4)
    assert finest.response.reactions[0].moment == pytest.approx(
        elastic.response.reactions[0].moment, rel=1e-3
    )


def test_bnwf_is_used_when_kep_is_available():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0, steps=20,
    )
    assert any("分割)で解析し" in n for n in result.notes)
    from core.analysis.level2 import LIMITATION_ELASTIC_GROUND

    assert LIMITATION_ELASTIC_GROUND not in result.notes


def test_falls_back_to_elastic_springs_without_kep():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0, steps=20,
    )
    from core.analysis.level2 import LIMITATION_ELASTIC_GROUND

    assert LIMITATION_ELASTIC_GROUND in result.notes
    assert not any("分割)で解析し" in n for n in result.notes)
    assert all(s.plastic_ground_nodes == 0 for s in result.steps)


def test_ground_plasticity_softens_the_foundation():
    """地盤が塑性化すると、同じ水平力に対する変位が弾性解析より大きくなる。"""
    from core.analysis.level2 import run_level2

    kwargs = dict(
        v_load=9000.0, h_load=3000.0, m_load=12000.0, max_factor=1.0, steps=20,
    )
    elastic = run_level2(
        STEEL, ARRANGEMENT, FOOTING, sample_ground(), use_bnwf=False, **kwargs
    )
    # 小さい KEP → pHU が小さく、浅部の地盤が塑性化する
    plastic = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=0.3), **kwargs
    )
    assert plastic.response.plastic_ground_nodes > 0
    assert plastic.response.u > elastic.response.u


def test_front_row_and_back_rows_differ_in_sand():
    """砂質地盤では最前列以外の pHU が 1/2 なので、杭頭反力に差が出ること。"""
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=0.3),
        v_load=9000.0, h_load=3000.0, m_load=12000.0, max_factor=1.0, steps=20,
    )
    assert result.response.plastic_ground_nodes > 0
    shears = {round(r.shear, 6) for r in result.response.reactions}
    assert len(shears) == 2  # 最前列と、それ以外
    # 最前列(x が最大)のほうが大きな水平力を負担する
    front = max(result.response.reactions, key=lambda r: r.x)
    back = min(result.response.reactions, key=lambda r: r.x)
    assert abs(front.shear) > abs(back.shear)


def test_equilibrium_holds_with_bnwf():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(k_ep=0.3),
        v_load=9000.0, h_load=3000.0, m_load=12000.0, max_factor=1.0, steps=20,
    )
    for step in result.steps:
        assert sum(r.shear for r in step.reactions) == pytest.approx(
            step.h, rel=1e-6, abs=1e-6
        )
        assert sum(r.axial for r in step.reactions) == pytest.approx(
            9000.0, rel=1e-6
        )
        assert sum(
            r.moment + r.x * r.axial for r in step.reactions
        ) == pytest.approx(step.m, rel=1e-6, abs=1e-6)


# --- 群杭の補正係数 μ -------------------------------------------------------


NARROW = PileArrangement(nx=3, ny=3, spacing_x=2.0, spacing_y=2.0)


def test_group_correction_is_skipped_for_the_distributed_spring_model():
    """基礎地盤の非線形性を考慮する場合、μ による補正は考慮しない。"""
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, NARROW, FOOTING, ground_with_kep(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert result.group_factor == 1.0
    note = next(n for n in result.notes if "補正係数 μ" in n)
    assert "乗じていない" in note


def test_group_correction_applies_on_the_elastic_path():
    """KEP が無く杭頭バネの弾性解析に落ちる場合は μ を乗じる。"""
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, NARROW, FOOTING, sample_ground(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert result.group_factor == pytest.approx(0.90)
    assert any("kH に乗じている" in n for n in result.notes)


def test_no_group_note_when_spacing_is_wide_enough():
    from core.analysis.level2 import run_level2

    result = run_level2(
        STEEL, ARRANGEMENT, FOOTING, ground_with_kep(),
        v_load=9000.0, h_load=3000.0, m_load=12000.0,
    )
    assert result.group_factor == 1.0
    # 群杭の補正に関する実行時の注記(制限事項の一覧とは別)が出ないこと
    assert not any("乗じ" in n for n in result.notes)
