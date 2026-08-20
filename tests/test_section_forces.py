"""杭体断面力分布のテスト(Chang の式)。"""
import math

import pytest

from core.analysis.section_forces import (
    chang_coefficients,
    distribution,
    evaluate_at,
    extremum_depths,
)

EI = 1.2272e6  # kN·m2(場所打ち杭 D=1.0m、σck=24 相当)
BETA = 0.2613  # 1/m


def test_head_boundary_conditions_are_reproduced():
    """杭頭で与えた H0・M0 が断面力として再現されること。"""
    h0, m0 = 50.0, -95.7
    p = evaluate_at(EI, BETA, h0, m0, 0.0)
    assert p.shear == pytest.approx(h0, rel=1e-9)
    assert p.moment == pytest.approx(m0, rel=1e-9)


def test_fixed_head_matches_spring_definitions():
    """杭頭剛結(回転角0)では A = B となり K1・K3 と整合すること。

    K1 = 4EIβ³、K3 = −2EIβ² より H0 = K1·y0、M0 = K3·y0。
    """
    y0 = 0.001
    h0 = 4 * EI * BETA**3 * y0
    m0 = -2 * EI * BETA**2 * y0
    a, b = chang_coefficients(EI, BETA, h0, m0)
    assert a == pytest.approx(y0, rel=1e-9)
    assert b == pytest.approx(y0, rel=1e-9)
    # 杭頭変位が y0 に一致する
    assert evaluate_at(EI, BETA, h0, m0, 0.0).displacement == pytest.approx(y0)


def test_fixed_head_underground_peak_is_0208_of_head_moment():
    """杭頭剛結の地中部最大曲げモーメントは杭頭の約 0.208 倍(βx = π/2)。"""
    y0 = 0.001
    h0 = 4 * EI * BETA**3 * y0
    m0 = -2 * EI * BETA**2 * y0
    dist = distribution(EI, BETA, h0, m0, length=20.0)
    peak = dist.max_underground_moment
    assert peak.depth == pytest.approx(math.pi / (2 * BETA), rel=1e-6)
    # e^(-π/2) = 0.2079
    assert peak.moment / abs(m0) == pytest.approx(math.exp(-math.pi / 2), rel=1e-6)
    # 杭頭とは逆符号
    assert peak.moment * m0 < 0


def test_moment_decays_with_depth():
    h0, m0 = 100.0, -200.0
    dist = distribution(EI, BETA, h0, m0, length=25.0)
    deep = [p for p in dist.points if p.depth > 20.0]
    # 深部では杭頭モーメントの 1% 程度まで減衰する
    assert deep and all(abs(p.moment) < 0.02 * abs(m0) for p in deep)


def test_max_moment_is_at_head_for_fixed_head():
    y0 = 0.001
    h0 = 4 * EI * BETA**3 * y0
    m0 = -2 * EI * BETA**2 * y0
    dist = distribution(EI, BETA, h0, m0, length=20.0)
    assert dist.max_moment.depth == pytest.approx(0.0)


def test_shear_equals_head_value_at_top():
    dist = distribution(EI, BETA, 80.0, -150.0, length=20.0)
    assert dist.points[0].shear == pytest.approx(80.0, rel=1e-9)


def test_distribution_covers_full_length():
    dist = distribution(EI, BETA, 50.0, -95.0, length=18.0, pitch=0.5)
    assert dist.points[0].depth == 0.0
    assert dist.points[-1].depth == pytest.approx(18.0)
    assert all(
        dist.points[i].depth <= dist.points[i + 1].depth
        for i in range(len(dist.points) - 1)
    )


def test_extrema_have_zero_shear():
    """極値点として抽出した深さでは、せん断力が 0 になっていること。"""
    dist = distribution(EI, BETA, 50.0, 10.7, length=20.0)
    extrema = [p for p in dist.points if p.is_extremum]
    assert extrema
    for p in extrema:
        assert p.shear == pytest.approx(0.0, abs=1e-6)


def test_extremum_found_when_head_moment_is_small():
    """杭頭モーメントが小さい場合、地中部に杭頭を上回るピークが生じる。

    杭頭 H0=50 kN, M0=10.7 kN·m のケース。tan βx =(A+B)/(A−B) より
    βx = 0.7325 → 深さ 2.80 m でモーメントが極大となる。
    """
    h0, m0 = 50.0, 10.7
    depths = extremum_depths(EI, BETA, h0, m0, length=20.0)
    assert depths[0] == pytest.approx(2.80, rel=1e-2)

    dist = distribution(EI, BETA, h0, m0, length=20.0)
    peak = dist.max_underground_moment
    assert peak.depth == pytest.approx(depths[0])
    # 杭頭モーメントを大きく上回る
    assert abs(peak.moment) > abs(m0) * 5
    assert peak.moment == pytest.approx(68.8, rel=1e-2)


def test_extremum_depths_are_periodic():
    depths = extremum_depths(EI, BETA, 50.0, -95.7, length=30.0)
    assert len(depths) >= 2
    for a, b in zip(depths, depths[1:]):
        assert b - a == pytest.approx(math.pi / BETA, rel=1e-9)
