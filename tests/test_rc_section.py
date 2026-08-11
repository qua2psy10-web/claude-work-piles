"""円形RC断面の応力度算定のテスト。"""
import math

import pytest

from core.section.rc import RebarLayout, _concrete_integrals, analyze_circular_rc

EC = 2.5e7  # kN/m2(σck=24)
N_RATIO = 8.0  # E_REBAR/EC = 2.0e8/2.5e7
D = 1.0
REBAR = RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)


def test_rebar_geometry():
    assert REBAR.bar_area == pytest.approx(math.pi * 0.025**2 / 4)
    assert REBAR.total_area == pytest.approx(24 * math.pi * 0.025**2 / 4)
    assert REBAR.radius(D) == pytest.approx(0.375)
    ys = REBAR.positions(D)
    assert len(ys) == 24
    assert max(ys) == pytest.approx(0.375)
    assert sum(ys) == pytest.approx(0.0, abs=1e-12)


def test_concrete_integrals_full_section():
    """中立軸を断面外に置くと、全断面積・断面二次モーメントに一致する。"""
    radius = 0.5
    y_n = -radius
    s_area, s_moment = _concrete_integrals(radius, y_n, divisions=2000)
    # ∫(y+R)·b dy = R·A (Aは円の面積、∫y·b dy = 0 のため)
    area = math.pi * radius**2
    assert s_area == pytest.approx(radius * area, rel=1e-4)
    # ∫(y+R)·y·b dy = I(∫y²b dy)
    inertia = math.pi * radius**4 / 4.0
    assert s_moment == pytest.approx(inertia, rel=1e-4)


def test_uncracked_section_matches_hand_calculation():
    """全断面圧縮のケースを換算断面の手計算と突合する。

    N=1500 kN, M=100 kN·m, D=1.0m, 24-D25 (かぶり125mm), n=8
      A  = πD²/4 + (n−1)·As = 0.785398 + 7×0.0117810 = 0.867865 m²
      I  = πD⁴/64 + (n−1)·ΣAs·y² = 0.0490874 + 7×0.00082849 = 0.0548868 m⁴
      σ  = N/A ± M·y/I
    """
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=100.0)
    assert result.fully_compressed

    area = math.pi * D**2 / 4 + (N_RATIO - 1) * REBAR.total_area
    sum_y2 = sum(y**2 for y in REBAR.positions(D)) * REBAR.bar_area
    inertia = math.pi * D**4 / 64 + (N_RATIO - 1) * sum_y2
    sigma_c = (1500.0 / area + 100.0 * 0.5 / inertia) / 1000.0
    assert result.sigma_c == pytest.approx(sigma_c, rel=2e-3)

    # 最外縁鉄筋(y=0.375)の圧縮応力度 = n × その位置のコンクリート応力度
    sigma_c_at_bar = (1500.0 / area + 100.0 * 0.375 / inertia) / 1000.0
    assert result.sigma_s_compression == pytest.approx(
        N_RATIO * sigma_c_at_bar, rel=2e-3
    )
    assert result.sigma_s_tension == 0.0


def test_cracked_section_equilibrium():
    """ひび割れ断面で、求めた応力状態が N・M の釣合いを満たすこと。"""
    axial, moment = 1500.0, 800.0
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=axial, moment=moment)
    assert not result.fully_compressed

    # 応力分布から軸力・モーメントを積分し直して照合する
    radius = D / 2
    y_n = result.neutral_axis_y
    k = result.curvature
    s_area, s_moment = _concrete_integrals(radius, y_n, divisions=4000)
    n_axial = EC * k * s_area
    n_moment = EC * k * s_moment
    for y in result_positions():
        ratio = N_RATIO if y <= y_n else N_RATIO - 1.0
        force = ratio * REBAR.bar_area * EC * k * (y - y_n)
        n_axial += force
        n_moment += force * y
    assert n_axial == pytest.approx(axial, rel=1e-3)
    assert n_moment == pytest.approx(moment, rel=1e-3)


def result_positions():
    return REBAR.positions(D)


def test_larger_moment_reduces_compression_depth():
    depths = []
    for moment in (200.0, 600.0, 1200.0):
        r = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=moment)
        depths.append(r.compression_depth)
    assert depths[0] > depths[1] > depths[2]


def test_moment_sign_does_not_matter():
    plus = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=800.0)
    minus = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=-800.0)
    assert plus.sigma_c == pytest.approx(minus.sigma_c)
    assert plus.sigma_s_tension == pytest.approx(minus.sigma_s_tension)


def test_pure_axial_gives_uniform_stress():
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=0.0)
    area = math.pi * D**2 / 4 + (N_RATIO - 1) * REBAR.total_area
    assert result.sigma_c == pytest.approx(1500.0 / area / 1000.0, rel=2e-3)
    assert result.sigma_s_tension == 0.0


def test_tension_axial_not_supported():
    with pytest.raises(NotImplementedError):
        analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=-100.0, moment=500.0)


def test_excessive_cover_rejected():
    bad = RebarLayout(count=12, diameter_mm=25.0, cover_mm=600.0)
    with pytest.raises(ValueError, match="かぶり"):
        bad.radius(D)
