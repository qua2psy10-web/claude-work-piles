"""杭体の M-φ 骨格曲線と、それによる曲げ剛性低下のテスト(第54回)。"""
import numpy as np
import pytest

from core.analysis.bnwf import PileLateralModel
from core.section.moment_curvature import MomentCurvature

# フォーラムエイト UC-1 計算書サンプル Kui_10 の 6.6.1「M－φ」より、
# 区間2(PHC杭 B種、φ600・t=90、PC鋼材14.08cm²、σce=8.0)の死荷重時軸力
# (N=445.6kN)のトリリニア。
KUI10_PHC_B = MomentCurvature.trilinear(
    cracking_curvature=0.0012680, cracking_moment=257.3,
    yield_curvature=0.0055255, yield_moment=456.4,
    ultimate_curvature=0.0184293, ultimate_moment=543.9,
)


def test_trilinear_passes_through_its_own_break_points():
    """折れ点そのものを与えれば、その値がそのまま返ること。"""
    mc = KUI10_PHC_B
    assert mc.moment_at(0.0012680) == pytest.approx(257.3)
    assert mc.moment_at(0.0055255) == pytest.approx(456.4)
    assert mc.moment_at(0.0184293) == pytest.approx(543.9)
    assert mc.curvature_at(257.3) == pytest.approx(0.0012680)
    assert mc.curvature_at(456.4) == pytest.approx(0.0055255)
    assert mc.curvature_at(543.9) == pytest.approx(0.0184293)


def test_break_point_accessors_match_kui10():
    mc = KUI10_PHC_B
    assert mc.yield_moment == pytest.approx(456.4)
    assert mc.yield_curvature == pytest.approx(0.0055255)
    assert mc.ultimate_moment == pytest.approx(543.9)
    assert mc.ultimate_curvature == pytest.approx(0.0184293)
    # 初期剛性 = Mc/φc
    assert mc.initial_ei == pytest.approx(257.3 / 0.0012680)


def test_moment_and_curvature_are_mutual_inverses():
    mc = KUI10_PHC_B
    for phi in np.linspace(1.0e-5, mc.ultimate_curvature, 25):
        assert mc.curvature_at(mc.moment_at(phi)) == pytest.approx(phi, rel=1e-9)


def test_secant_stiffness_decreases_monotonically():
    """割線剛性は曲率とともに単調に減少すること(剛性低下)。"""
    mc = KUI10_PHC_B
    curvatures = np.linspace(1.0e-5, mc.ultimate_curvature, 40)
    stiffness = [mc.secant_ei_at(float(p)) for p in curvatures]
    assert stiffness[0] == pytest.approx(mc.initial_ei, rel=1e-6)
    assert all(a >= b - 1e-9 for a, b in zip(stiffness, stiffness[1:]))
    # 終局点の割線剛性 Mu/φu は初期剛性 Mc/φc の 1/5 未満まで落ちる
    # (Kui_10 の区間2 では 29,513 / 202,918 ≒ 1/6.9)
    assert stiffness[-1] < mc.initial_ei / 5.0


def test_secant_ei_agrees_whether_indexed_by_moment_or_curvature():
    mc = KUI10_PHC_B
    for phi in np.linspace(1.0e-5, mc.ultimate_curvature, 20):
        moment = mc.moment_at(float(phi))
        assert mc.secant_ei(moment) == pytest.approx(
            mc.secant_ei_at(float(phi)), rel=1e-9
        )


def test_beyond_ultimate_the_moment_is_capped():
    """終局曲率を超えると完全塑性(モーメントは Mu で頭打ち)。"""
    mc = KUI10_PHC_B
    assert mc.moment_at(0.05) == pytest.approx(543.9)
    assert mc.moment_at(1.0) == pytest.approx(543.9)
    assert not mc.exceeds_ultimate(543.9)
    assert mc.exceeds_ultimate(543.91)


def test_symmetric_in_sign():
    mc = KUI10_PHC_B
    assert mc.moment_at(-0.0055255) == pytest.approx(-456.4)
    assert mc.curvature_at(-456.4) == pytest.approx(-0.0055255)
    assert mc.secant_ei(-456.4) == pytest.approx(mc.secant_ei(456.4))


def test_yielded_flags_at_the_yield_moment():
    mc = KUI10_PHC_B
    assert not mc.yielded(456.3)
    assert mc.yielded(456.4)
    assert mc.yielded(-500.0)


def test_bilinear_treats_the_first_point_as_yield():
    """バイリニア(鋼管杭)は1点目が降伏、2点目が全塑性。"""
    mc = MomentCurvature.bilinear(
        yield_curvature=0.0026821, yield_moment=3192.5,
        ultimate_curvature=0.0039703, ultimate_moment=4725.9,
    )
    assert mc.yield_moment == pytest.approx(3192.5)
    assert mc.ultimate_moment == pytest.approx(4725.9)
    assert mc.initial_ei == pytest.approx(3192.5 / 0.0026821)


def test_rejects_non_monotonic_points():
    with pytest.raises(ValueError, match="曲率"):
        MomentCurvature(((0.002, 100.0), (0.001, 200.0)))
    with pytest.raises(ValueError, match="モーメント"):
        MomentCurvature(((0.001, 200.0), (0.002, 100.0)))
    with pytest.raises(ValueError, match="曲率"):
        MomentCurvature(((0.0, 100.0),))
    with pytest.raises(ValueError, match="折れ点"):
        MomentCurvature(())


# --- BNWF への組み込み -------------------------------------------------------

_COMMON = dict(diameter=0.6, length=15.0, kh=5000.0, n_elements=40)


def test_small_displacement_matches_the_elastic_model():
    """ひび割れ前(M < Mc)は M-φ を与えても弾性解と一致すること。"""
    mc = KUI10_PHC_B
    elastic = PileLateralModel(ei=mc.initial_ei, **_COMMON)
    nonlinear = PileLateralModel(ei=mc.initial_ei, moment_curvature=mc, **_COMMON)

    e = elastic.solve(0.001, 0.0)
    n = nonlinear.solve(0.001, 0.0)
    assert abs(n.moment) < mc.points[0][1]  # ひび割れモーメント未満
    assert n.shear == pytest.approx(e.shear, rel=1e-6)
    assert n.moment == pytest.approx(e.moment, rel=1e-6)
    assert n.plastic_hinges == 0
    assert not n.yielded_body


def test_large_displacement_softens_relative_to_the_elastic_model():
    """変位が大きくなると、剛性低下により杭頭反力・モーメントが弾性解を
    下回り、モーメントは終局モーメント付近で頭打ちになること。"""
    mc = KUI10_PHC_B
    elastic = PileLateralModel(ei=mc.initial_ei, **_COMMON)
    nonlinear = PileLateralModel(ei=mc.initial_ei, moment_curvature=mc, **_COMMON)

    e = elastic.solve(0.10, 0.0)
    n = nonlinear.solve(0.10, 0.0)
    assert abs(n.moment) < abs(e.moment)
    assert abs(n.shear) < abs(e.shear)
    # 骨格曲線の上限を(数値誤差の範囲を除いて)超えない
    assert abs(n.moment) <= mc.ultimate_moment * 1.01
    assert n.plastic_hinges > 0
    assert n.yielded_body


def test_head_moment_never_exceeds_the_skeleton_curve():
    """どの変位でも杭頭モーメントが Mu を超えないこと。"""
    mc = KUI10_PHC_B
    nonlinear = PileLateralModel(ei=mc.initial_ei, moment_curvature=mc, **_COMMON)
    for u in (0.005, 0.02, 0.05, 0.1, 0.2):
        response = nonlinear.solve(u, 0.0)
        assert abs(response.moment) <= mc.ultimate_moment * 1.01


def test_plastic_hinges_grow_with_displacement():
    mc = KUI10_PHC_B
    nonlinear = PileLateralModel(ei=mc.initial_ei, moment_curvature=mc, **_COMMON)
    counts = [nonlinear.solve(u, 0.0).plastic_hinges for u in (0.01, 0.05, 0.1, 0.2)]
    assert counts[0] == 0
    assert all(a <= b for a, b in zip(counts, counts[1:]))
    assert counts[-1] > counts[0]


def test_element_curvature_is_independent_of_the_stiffness_used():
    """曲率は変位場のみで決まり、EI に依存しないこと(実装の要)。

    同じ変位を与えた2つのモデル(EI が 10 倍違う)に、同一の節点変位を
    直接セットして曲率を比べる。
    """
    soft = PileLateralModel(ei=1.0e5, **_COMMON)
    stiff = PileLateralModel(ei=1.0e6, **_COMMON)
    state = np.linspace(0.0, 0.01, 2 * (_COMMON["n_elements"] + 1))
    assert np.allclose(
        soft.element_curvatures(state), stiff.element_curvatures(state)
    )


def test_ultimate_curvature_is_flagged_when_exceeded():
    """終局曲率を超えたら結果に立つこと(照査で使える診断)。"""
    mc = KUI10_PHC_B
    nonlinear = PileLateralModel(ei=mc.initial_ei, moment_curvature=mc, **_COMMON)
    assert not nonlinear.solve(0.001, 0.0).exceeds_ultimate_curvature
    assert nonlinear.solve(0.30, 0.0).exceeds_ultimate_curvature


def test_head_stiffness_stays_elastic_after_a_nonlinear_solve():
    """head_stiffness() は剛性低下後も弾性値を返すこと(比較の基準として)。"""
    mc = KUI10_PHC_B
    nonlinear = PileLateralModel(ei=mc.initial_ei, moment_curvature=mc, **_COMMON)
    before = nonlinear.head_stiffness().copy()
    nonlinear.solve(0.20, 0.0)
    assert np.allclose(nonlinear.head_stiffness(), before)
