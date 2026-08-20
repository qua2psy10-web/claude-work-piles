"""分布バネモデル(BNWF)のテスト。

検証の軸は「地盤バネが弾性のとき、Chang の式(半無限長)の杭頭バネ
K1〜K4 に収束すること」である。
"""
import math

import numpy as np
import pytest

from core.analysis.bnwf import PileLateralModel, tributary_lengths
from core.analysis.section_forces import distribution

EI = 1.0e6  # kN·m2
DIAMETER = 1.0
KH = 5.0e4  # kN/m3
LENGTH = 30.0
BETA = (KH * DIAMETER / (4.0 * EI)) ** 0.25


def chang_head_stiffness():
    """Chang の式による杭頭バネ [[K1, K2], [K3, K4]]。"""
    k1 = 4.0 * EI * BETA**3
    k2 = -2.0 * EI * BETA**2
    k4 = 2.0 * EI * BETA
    return np.array([[k1, k2], [k2, k4]])


def model(n_elements=200, limits=None, length=LENGTH):
    return PileLateralModel(
        EI, DIAMETER, length, KH, limits=limits, n_elements=n_elements
    )


# --- 分担長 -----------------------------------------------------------------


def test_tributary_lengths_sum_to_pile_length():
    lengths = tributary_lengths(20.0, 40)
    assert lengths.sum() == pytest.approx(20.0)
    # 両端は要素長の 1/2
    assert lengths[0] == pytest.approx(lengths[1] / 2.0)
    assert lengths[-1] == pytest.approx(lengths[1] / 2.0)


# --- 弾性: Chang の式との一致 ------------------------------------------------


def test_semi_infinite_pile_reproduces_chang_head_springs():
    """βL ≫ 3 の弾性杭で、縮約剛性が K1〜K4 に一致すること。"""
    assert BETA * LENGTH > 3.0
    computed = model().head_stiffness()
    expected = chang_head_stiffness()
    assert computed == pytest.approx(expected, rel=1.0e-3)


def test_head_stiffness_converges_with_refinement():
    """分割を細かくすると Chang の式への誤差が単調に減ること。"""
    expected = chang_head_stiffness()
    errors = []
    for n in (25, 50, 100, 200):
        computed = model(n_elements=n).head_stiffness()
        errors.append(float(np.max(np.abs(computed - expected) / np.abs(expected))))
    assert errors == sorted(errors, reverse=True)
    # 2次収束(分割を倍にすると誤差は約 1/4)
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse / 3.0


def test_head_stiffness_is_symmetric_and_positive_definite():
    k = model().head_stiffness()
    assert k[0, 1] == pytest.approx(k[1, 0], rel=1e-9)
    assert k[0, 0] > 0 and k[1, 1] > 0
    assert np.linalg.det(k) > 0


def test_coupling_terms_are_negative():
    """K2 = K3 = −2EIβ² は負(変位法の符号規約)。"""
    k = model().head_stiffness()
    assert k[0, 1] < 0
    assert k[0, 1] == pytest.approx(-2.0 * EI * BETA**2, rel=1e-3)


# --- 弾性: 杭頭反力と断面力分布 ---------------------------------------------


def test_fixed_head_response_matches_chang():
    """杭頭剛結(θ = 0)で H = K1・u、M = K3・u となること。"""
    u = 0.005
    response = model().solve(u_head=u, theta_head=0.0)
    assert response.shear == pytest.approx(4.0 * EI * BETA**3 * u, rel=1e-3)
    assert response.moment == pytest.approx(-2.0 * EI * BETA**2 * u, rel=1e-3)
    assert response.plastic_nodes == 0


def test_rotation_only_response_matches_chang():
    """u = 0、θ = 1 で H = K2、M = K4 となること。"""
    theta = 0.001
    response = model().solve(u_head=0.0, theta_head=theta)
    assert response.shear == pytest.approx(-2.0 * EI * BETA**2 * theta, rel=1e-3)
    assert response.moment == pytest.approx(2.0 * EI * BETA * theta, rel=1e-3)


def test_displacement_profile_matches_chang_analytic_solution():
    """弾性域の変位分布が Chang の式と一致すること。"""
    u = 0.004
    response = model(n_elements=300).solve(u_head=u, theta_head=0.0)
    analytic = distribution(
        ei=EI, beta=BETA, h0=response.shear, m0=response.moment, length=LENGTH
    )
    depths = np.array([p.depth for p in analytic.points])
    values = np.array([p.displacement for p in analytic.points])
    computed = np.interp(depths, np.linspace(0.0, LENGTH, 301), response.displacements)
    # 杭頭変位で正規化した誤差
    assert np.max(np.abs(computed - values)) < 0.01 * u


def test_displacement_decays_with_depth():
    response = model().solve(u_head=0.005, theta_head=0.0)
    y = response.displacements
    assert y[0] == pytest.approx(0.005)
    # 深部ではほぼ 0(βL ≒ 10 なので e^(−βL) のオーダー)
    assert abs(y[-1]) < 1e-3 * abs(y[0])
    assert abs(y[-1]) < abs(y[len(y) // 2])


def test_linear_scaling_in_elastic_range():
    m = model()
    small = m.solve(u_head=0.001, theta_head=0.0)
    large = m.solve(u_head=0.002, theta_head=0.0)
    assert large.shear == pytest.approx(2.0 * small.shear, rel=1e-6)
    assert large.moment == pytest.approx(2.0 * small.moment, rel=1e-6)


# --- 塑性: pHU による頭打ち --------------------------------------------------


def uniform_limits(value, n_elements=200):
    return np.full(n_elements + 1, value)


def test_ground_yields_when_limit_is_exceeded():
    limits = uniform_limits(50.0)  # kN/m2、かなり小さい
    response = model(limits=limits).solve(u_head=0.02, theta_head=0.0)
    assert response.plastic_nodes > 0
    assert response.yielded_ground


def test_high_limit_behaves_like_elastic_model():
    """上限を十分大きくすると弾性モデルと一致すること。"""
    elastic = model().solve(u_head=0.005, theta_head=0.0)
    capped = model(limits=uniform_limits(1.0e9)).solve(u_head=0.005, theta_head=0.0)
    assert capped.shear == pytest.approx(elastic.shear, rel=1e-9)
    assert capped.moment == pytest.approx(elastic.moment, rel=1e-9)
    assert capped.plastic_nodes == 0


def test_plasticity_softens_the_head_stiffness():
    """地盤が塑性化すると杭頭剛性が下がること。"""
    limits = uniform_limits(80.0)
    m = model(limits=limits)
    elastic_k = m.head_stiffness()[0, 0]
    response = m.solve(u_head=0.03, theta_head=0.0)
    assert response.plastic_nodes > 0
    assert response.tangent[0, 0] < elastic_k


def test_plastic_model_gives_lower_shear_than_elastic():
    """上限を設けると同じ変位に対する杭頭水平力が小さくなる(非安全側の解消)。"""
    u = 0.03
    elastic = model().solve(u_head=u, theta_head=0.0)
    plastic = model(limits=uniform_limits(80.0)).solve(u_head=u, theta_head=0.0)
    assert plastic.plastic_nodes > 0
    assert abs(plastic.shear) < abs(elastic.shear)


def test_reaction_is_capped_at_the_limit():
    """弾性反力が上限を超える節点では、実際の反力が pHU で頭打ちになること。"""
    limit = 60.0
    n = 200
    m = model(n_elements=n, limits=uniform_limits(limit, n))
    response = m.solve(u_head=0.05, theta_head=0.0)

    elastic = KH * np.abs(response.displacements)
    assert elastic.max() > limit  # 上限がなければ超えている状況
    actual = np.minimum(elastic, limit)
    assert actual.max() == pytest.approx(limit)
    # 頭打ちになった節点数がモデルの報告と一致すること
    assert int(np.count_nonzero(elastic >= limit)) == response.plastic_nodes


def test_equilibrium_of_the_pile():
    """杭頭水平力と地盤反力の総和が釣り合うこと。"""
    n = 200
    m = model(n_elements=n, limits=uniform_limits(120.0, n))
    response = m.solve(u_head=0.02, theta_head=0.0)
    tributary = tributary_lengths(LENGTH, n)
    resistance = np.clip(
        KH * response.displacements, -120.0, 120.0
    ) * DIAMETER * tributary
    assert float(resistance.sum()) == pytest.approx(response.shear, rel=1e-6)


# --- 入力チェック -----------------------------------------------------------


def test_rejects_invalid_discretisation():
    with pytest.raises(ValueError):
        PileLateralModel(EI, DIAMETER, LENGTH, KH, n_elements=1)


def test_rejects_non_positive_properties():
    with pytest.raises(ValueError):
        PileLateralModel(0.0, DIAMETER, LENGTH, KH)
    with pytest.raises(ValueError):
        PileLateralModel(EI, DIAMETER, LENGTH, 0.0)


def test_rejects_limit_length_mismatch():
    with pytest.raises(ValueError, match="要素数"):
        PileLateralModel(EI, DIAMETER, LENGTH, KH, limits=np.ones(5), n_elements=50)


def test_rejects_non_positive_limits():
    with pytest.raises(ValueError, match="上限値"):
        PileLateralModel(EI, DIAMETER, LENGTH, KH, limits=np.zeros(51), n_elements=50)
