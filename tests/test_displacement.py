"""変位法のテスト(道示Ⅳ(H24) 12.6)。"""
import pytest

from core.analysis.displacement import pile_x_coordinates, solve_stability
from core.analysis.stability import allowable_displacement
from core.models import PileArrangement

ARR_2X3 = PileArrangement(nx=2, ny=3, spacing_x=2.5, spacing_y=2.5)


def test_x_coordinates_symmetric():
    xs = pile_x_coordinates(ARR_2X3)
    assert len(xs) == 6
    assert sorted(set(xs)) == [-1.25, 1.25]
    assert sum(xs) == pytest.approx(0.0)


def test_vertical_load_only_gives_uniform_axial():
    result = solve_stability(
        ARR_2X3, kv=1.0e5, k1=2.0e5, k2=-1.0e5, k4=1.0e5,
        v_load=6000.0, h_load=0.0, m_load=0.0,
    )
    assert result.u == pytest.approx(0.0)
    assert result.theta == pytest.approx(0.0)
    # 6本で等分
    for r in result.reactions:
        assert r.axial == pytest.approx(1000.0)
        assert r.shear == pytest.approx(0.0)


def test_moment_produces_antisymmetric_axial_forces():
    result = solve_stability(
        ARR_2X3, kv=1.0e5, k1=2.0e5, k2=-1.0e5, k4=1.0e5,
        v_load=0.0, h_load=0.0, m_load=3000.0,
    )
    # 正のモーメントで +x 側が押込み、−x 側が引抜き
    front = [r.axial for r in result.reactions if r.x > 0]
    back = [r.axial for r in result.reactions if r.x < 0]
    assert all(a > 0 for a in front)
    assert all(a < 0 for a in back)
    assert sum(front) + sum(back) == pytest.approx(0.0, abs=1e-6)


def test_equilibrium_is_satisfied():
    """求めた反力がフーチングの釣合いを満たすこと。"""
    v_load, h_load, m_load = 8000.0, 900.0, 2500.0
    kv, k1, k2, k4 = 2.5e5, 4.0e4, -3.0e4, 6.0e4
    result = solve_stability(
        ARR_2X3, kv=kv, k1=k1, k2=k2, k4=k4,
        v_load=v_load, h_load=h_load, m_load=m_load,
    )
    assert sum(r.axial for r in result.reactions) == pytest.approx(v_load, rel=1e-8)
    assert sum(r.shear for r in result.reactions) == pytest.approx(h_load, rel=1e-8)
    total_m = sum(r.moment + r.axial * r.x for r in result.reactions)
    assert total_m == pytest.approx(m_load, rel=1e-8)


def test_single_pile_row_has_no_axial_from_moment():
    """1列配置ではモーメントを軸力偶で受けられず、杭頭モーメントで受ける。"""
    single = PileArrangement(nx=1, ny=3, spacing_x=2.5, spacing_y=2.5)
    result = solve_stability(
        single, kv=1.0e5, k1=2.0e5, k2=-1.0e5, k4=1.0e5,
        v_load=0.0, h_load=0.0, m_load=3000.0,
    )
    assert all(r.axial == pytest.approx(0.0) for r in result.reactions)
    assert sum(r.moment for r in result.reactions) == pytest.approx(3000.0)


def test_allowable_displacement_rule():
    assert allowable_displacement(1.0) == pytest.approx(0.015)
    assert allowable_displacement(1.4) == pytest.approx(0.015)
    # 1.5m以上は杭径の1%
    assert allowable_displacement(1.5) == pytest.approx(0.015)
    assert allowable_displacement(2.0) == pytest.approx(0.020)
