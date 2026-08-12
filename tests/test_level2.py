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


def test_allowable_values_are_not_assumed():
    """許容塑性率・許容変位は未入力なら照査しない(既定値を置かない)。"""
    s = spring(push=1500.0, pull=1500.0)
    result = analyze_level2(
        ARRANGEMENT, s, k1=K1, k2=K2, k4=K4,
        v_load=9000.0, h_load=2000.0, m_load=8000.0,
    )
    assert result.allowable_ductility is None
    assert result.allowable_displacement is None
    assert result.checks == []
    assert any("μa" in note for note in result.notes)


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
    assert any("降伏曲げモーメント" in n for n in result.notes)
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
