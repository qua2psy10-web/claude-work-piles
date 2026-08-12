"""液状化に伴う土質定数の低減 DE の反映(道示Ⅴ(H24) 8.2.4)のテスト。"""
import pytest

from core.analysis.stability import analyze
from core.capacity.section import pile_section
from core.capacity.springs import lateral_springs
from core.models import (
    ConstructionMethod,
    Footing,
    FootingLoads,
    LoadCase,
    PileArrangement,
    PileSpec,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.soil.liquefaction import SoilReduction, assess_liquefaction
from core.standards import GroundMotionType, GroundType

MOTION = GroundMotionType.LEVEL2_TYPE2


def liquefiable_profile(n_value: float = 10.0, fc: float = 25.0):
    """浅部に液状化する砂層をもつ地盤。

    既定(N = 10、FC = 25%)では DE = 2/3 となり部分的に低減される。
    N を下げると DE = 0(完全液状化)になる。
    """
    return SoilProfile(
        layers=[
            SoilLayer(
                name="As1", soil_type=SoilType.SAND, thickness=10.0,
                n_value=n_value,
                gamma_t=18.0, gamma_sat=19.0, fc=fc, d50=0.3, d10=0.08,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=45.0,
                gamma_t=19.0, gamma_sat=20.0, fc=8.0, d50=0.5, d10=0.1,
                is_alluvial=False,
            ),
        ],
        gwl=1.0,
    )


PILE = PileSpec(
    pile_type=PileType.CAST_IN_PLACE,
    method=ConstructionMethod.CAST_IN_PLACE,
    diameter=1.0,
    length=20.0,
)
ARRANGEMENT = PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5)
FOOTING = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)


def sample_reduction(**kwargs):
    assessment = assess_liquefaction(
        liquefiable_profile(**kwargs), GroundType.TYPE_II,
        cz_type1=1.0, cz_type2=1.0,
    )
    return SoilReduction.from_assessment(assessment, MOTION)


def fully_liquefied_reduction():
    """杭頭直下が完全液状化(DE = 0)する条件。"""
    return sample_reduction(n_value=6.0, fc=5.0)


# --- SoilReduction ----------------------------------------------------------


def test_reduction_is_derived_from_the_assessment():
    reduction = sample_reduction()
    assert reduction.motion_type == MOTION
    assert reduction.has_reduction
    span = reduction.reduced_depth_range()
    assert span is not None
    # 緩い砂層(0〜10 m、地下水位 1 m 以深)が低減される
    assert span[0] < 10.0


def test_factor_at_returns_one_outside_the_judged_range():
    reduction = sample_reduction()
    # 判定対象は地表面から 20 m まで
    assert reduction.factor_at(25.0) == 1.0
    assert 0.0 <= reduction.factor_at(5.0) <= 1.0


def test_mean_factor_is_thickness_weighted():
    reduction = SoilReduction(
        segments=((0.0, 2.0, 1.0), (2.0, 8.0, 1.0 / 3.0), (8.0, 20.0, 2.0 / 3.0)),
        motion_type=MOTION,
    )
    assert reduction.mean_factor(2.0, 8.0) == pytest.approx(1.0 / 3.0)
    expected = (2 * 1.0 + 6 * (1 / 3) + 2 * (2 / 3)) / 10
    assert reduction.mean_factor(0.0, 10.0) == pytest.approx(expected)


def test_mean_factor_does_not_reduce_beyond_the_judged_depth():
    """判定範囲(20 m)より深い区間は低減しない。"""
    reduction = SoilReduction(
        segments=((0.0, 20.0, 0.5),), motion_type=MOTION
    )
    # 0〜40 m のうち低減されるのは前半だけ
    assert reduction.mean_factor(0.0, 40.0) == pytest.approx((20 * 0.5 + 20 * 1.0) / 40)


def test_no_reduction_when_nothing_liquefies():
    dense = SoilProfile(
        layers=[
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=30.0, n_value=45.0,
                gamma_t=19.0, gamma_sat=20.0, fc=8.0, d50=0.5, d10=0.1,
                is_alluvial=False,
            )
        ],
        gwl=1.0,
    )
    assessment = assess_liquefaction(dense, GroundType.TYPE_II)
    reduction = SoilReduction.from_assessment(assessment, MOTION)
    assert not reduction.has_reduction
    assert reduction.reduced_depth_range() is None
    assert reduction.mean_factor(0.0, 20.0) == pytest.approx(1.0)


# --- kH への反映 ------------------------------------------------------------


def springs_with(reduction, profile=None):
    profile = profile if profile is not None else liquefiable_profile()
    section = pile_section(PILE)
    return lateral_springs(
        PILE, section, profile, FOOTING.embedment, LoadCase.LEVEL1_EQ,
        reduction=reduction,
    )


def test_reduction_lowers_kh_and_beta():
    plain = springs_with(None)
    reduced = springs_with(sample_reduction())

    assert reduced.de < 1.0
    assert reduced.kh < plain.kh
    # kH が下がれば β も下がる(特性長が伸びる)
    assert reduced.beta < plain.beta
    # 杭頭バネもすべて小さくなる
    assert reduced.k1 < plain.k1
    assert abs(reduced.k2) < abs(plain.k2)
    assert reduced.k4 < plain.k4


def test_de_is_recorded_on_the_result():
    plain = springs_with(None)
    assert plain.de == 1.0
    assert 0.0 <= springs_with(sample_reduction()).de < 1.0


def test_reduction_is_applied_inside_the_convergence_loop():
    """DE を後から掛けるのではなく収束計算の内側で効かせていること。

    後から kH に掛けるだけなら β は変わらない。β が変わることで、
    地中部最大曲げモーメントの位置が深部へ移動する。
    """
    from core.analysis.section_forces import distribution

    section = pile_section(PILE)
    plain = springs_with(None)
    reduced = springs_with(sample_reduction())

    # 単純に DE を後掛けした場合の β(変化しない)と比較する
    assert reduced.beta < plain.beta

    peak_plain = distribution(
        ei=section.ei, beta=plain.beta, h0=200.0, m0=-300.0, length=PILE.length
    ).max_underground_moment
    peak_reduced = distribution(
        ei=section.ei, beta=reduced.beta, h0=200.0, m0=-300.0, length=PILE.length
    ).max_underground_moment
    assert peak_reduced.depth > peak_plain.depth


# --- 安定計算への反映 -------------------------------------------------------


def stability(reduction, profile=None):
    loads = [FootingLoads(case=LoadCase.LEVEL1_EQ, v=9000.0, h=2000.0, m=8000.0)]
    return analyze(
        PILE, ARRANGEMENT, FOOTING,
        profile if profile is not None else liquefiable_profile(), loads,
        reduction=reduction,
    )


def test_full_liquefaction_below_the_footing_is_rejected_with_guidance():
    """杭頭直下が全て DE = 0 なら Chang の式は適用できず、理由を示すこと。"""
    with pytest.raises(ValueError, match="分布バネモデル"):
        springs_with(fully_liquefied_reduction())


def test_liquefaction_increases_displacement_and_head_moment():
    """低減を反映すると水平変位と杭頭モーメントが増える(従来は過小評価)。

    杭頭水平力は釣合いから決まり(全杭が同一なので H / 本数)、バネの
    大小によらない。増えるのは変位と杭頭モーメントである。
    """
    plain = stability(None).cases[0]
    reduced = stability(sample_reduction()).cases[0]

    assert reduced.result.u > plain.result.u
    assert abs(reduced.critical_pile.moment) > abs(plain.critical_pile.moment)


def test_liquefaction_moves_the_maximum_moment_deeper():
    """地盤反力の低下により地中部最大曲げモーメントの位置が深部へ移る。

    VERIFICATION.md 第7回に記録した挙動(提供資料の指摘)を再現する。
    """
    plain = stability(None).cases[0]
    reduced = stability(sample_reduction()).cases[0]

    assert reduced.springs.beta < plain.springs.beta
    assert (
        reduced.forces.max_underground_moment.depth
        > plain.forces.max_underground_moment.depth
    )


def test_stability_records_a_note_about_the_reduction():
    report = stability(sample_reduction())
    assert any("液状化による土質定数の低減" in n for n in report.notes)
    # レベル1に適用することの適否は利用者判断である旨を明示する
    assert any("利用者が判断" in n for n in report.notes)


def test_no_note_without_reduction():
    assert not any(
        "液状化" in n for n in stability(None).notes
    )


# --- レベル2への反映 --------------------------------------------------------


def test_level2_applies_reduction_per_node_with_bnwf():
    from core.analysis.level2 import run_level2

    # 杭頭直下が完全液状化する条件。分布バネモデルなら節点ごとに扱えるため
    # 解けるが、杭頭バネ K1〜K4 では解けない(上のテスト)。
    profile = liquefiable_profile(n_value=6.0, fc=5.0)
    with_kep = SoilProfile(
        layers=[layer.model_copy(update={"k_ep": 3.0}) for layer in profile.layers],
        gwl=profile.gwl,
    )
    assessment = assess_liquefaction(with_kep, GroundType.TYPE_II)
    reduction = SoilReduction.from_assessment(assessment, MOTION)

    kwargs = dict(
        v_load=9000.0, h_load=3000.0, m_load=12000.0, max_factor=1.0, steps=10,
    )
    plain = run_level2(PILE, ARRANGEMENT, FOOTING, with_kep, **kwargs)
    reduced = run_level2(
        PILE, ARRANGEMENT, FOOTING, with_kep, reduction=reduction, **kwargs
    )

    assert reduced.response.u > plain.response.u
    assert any("節点ごとに乗じている" in n for n in reduced.notes)


# --- 支持力への反映 ----------------------------------------------------------


def test_skin_friction_is_reduced_in_liquefied_layers():
    """液状化層の周面摩擦力度に DE を乗じること(道示Ⅴ 8.2.4)。"""
    from core.capacity.bearing import compute_bearing_capacity

    profile = liquefiable_profile()
    plain = compute_bearing_capacity(PILE, profile, FOOTING.embedment)
    reduced = compute_bearing_capacity(
        PILE, profile, FOOTING.embedment, reduction=sample_reduction()
    )

    assert reduced.skin_resistance < plain.skin_resistance
    assert reduced.ru < plain.ru
    # 低減前の値も保持している(内訳の提示に用いる)
    assert reduced.skin_resistance_unreduced == pytest.approx(plain.skin_resistance)
    assert reduced.has_reduced_skin
    assert not plain.has_reduced_skin
    # 先端支持力は低減しない
    assert reduced.tip_resistance == pytest.approx(plain.tip_resistance)


def test_reduction_is_applied_only_to_the_liquefied_layer():
    """低減されるのは液状化と判定された層だけであること。"""
    from core.capacity.bearing import compute_bearing_capacity

    bc = compute_bearing_capacity(
        PILE, liquefiable_profile(), FOOTING.embedment,
        reduction=sample_reduction(),
    )
    by_name = {s.layer_name: s for s in bc.skin_segments}
    assert by_name["As1"].is_reduced          # 浅部の緩い砂層
    assert not by_name["Ds"].is_reduced       # 支持層(N=45、洪積層)
    assert by_name["Ds"].de == 1.0
    # f 自体は低減前の値のまま保持し、f_design が低減後
    seg = by_name["As1"]
    assert seg.f_design == pytest.approx(seg.f * seg.de)
    assert seg.f_design < seg.f


def test_reduced_skin_force_equals_the_exact_integral():
    """f が層内一定なので、DE の層厚加重平均を乗じた値は ∫f・DE dz に一致する。"""
    import math

    from core.capacity.bearing import compute_bearing_capacity

    reduction = sample_reduction()
    bc = compute_bearing_capacity(
        PILE, liquefiable_profile(), FOOTING.embedment, reduction=reduction
    )
    perimeter = math.pi * PILE.diameter
    top = FOOTING.embedment
    for seg in bc.skin_segments:
        bottom = min(top + seg.length, bc.skin_bottom_depth)
        # 区間を細かく分割した数値積分と比較する
        n = 2000
        dz = (bottom - top) / n
        exact = sum(
            reduction.factor_at(top + (i + 0.5) * dz) * seg.f * perimeter * dz
            for i in range(n)
        )
        assert seg.force == pytest.approx(exact, rel=1e-6)
        top = bottom


def test_fully_liquefied_layer_contributes_no_skin_friction():
    """DE = 0 の層は周面摩擦を全く負担しないこと。"""
    from core.capacity.bearing import compute_bearing_capacity

    bc = compute_bearing_capacity(
        PILE, liquefiable_profile(n_value=6.0, fc=5.0), FOOTING.embedment,
        reduction=fully_liquefied_reduction(),
    )
    shallow = next(s for s in bc.skin_segments if s.layer_name == "As1")
    assert shallow.de == 0.0
    assert shallow.force == 0.0
    assert shallow.f > 0.0  # f 自体は算定されている


def test_stability_reports_the_bearing_reduction():
    """安定計算が低減の内訳と、許容支持力の低下を出すこと。"""
    profile = liquefiable_profile()
    loads = [FootingLoads(case=LoadCase.LEVEL1_EQ, v=9000.0, h=1500.0, m=4000.0)]
    kwargs = dict(pile=PILE, arrangement=ARRANGEMENT, footing=FOOTING,
                  profile=profile, loads=loads)
    plain = analyze(**kwargs)
    reduced = analyze(**kwargs, reduction=sample_reduction())

    seismic = reduced.bearing_seismic
    assert seismic is not None
    assert seismic.ru < plain.bearing.ru
    case = LoadCase.LEVEL1_EQ
    assert seismic.allowable_push(case) < plain.bearing.allowable_push(case)
    # 引抜き抵抗は周面摩擦力のみなので、より強く効く
    push_ratio = seismic.allowable_push(case) / plain.bearing.allowable_push(case)
    pull_ratio = seismic.allowable_pull(case) / plain.bearing.allowable_pull(case)
    assert pull_ratio < push_ratio < 1.0
    assert any("周面摩擦力度の低減内訳" in n for n in reduced.notes)


def test_reduction_does_not_apply_to_permanent_or_storm_cases():
    """DE は耐震設計上の扱い。常時・暴風時の照査には適用しないこと。"""
    profile = liquefiable_profile()
    loads = [
        FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=500.0, m=1500.0),
        FootingLoads(case=LoadCase.STORM, v=9000.0, h=1200.0, m=3000.0),
        FootingLoads(case=LoadCase.LEVEL1_EQ, v=9000.0, h=1500.0, m=4000.0),
    ]
    kwargs = dict(pile=PILE, arrangement=ARRANGEMENT, footing=FOOTING,
                  profile=profile, loads=loads)
    plain = analyze(**kwargs)
    reduced = analyze(**kwargs, reduction=sample_reduction())

    # 常時用の支持力は低減されない
    assert not reduced.bearing.has_reduced_skin
    assert reduced.bearing.ru == pytest.approx(plain.bearing.ru)

    by_case = {c.loads.case: c for c in reduced.cases}
    plain_by_case = {c.loads.case: c for c in plain.cases}
    for case in (LoadCase.PERMANENT, LoadCase.STORM):
        # kH も低減されないので、変位・断面力は低減なしの結果と完全に一致する
        assert by_case[case].springs.de == 1.0
        assert by_case[case].springs.kh == pytest.approx(
            plain_by_case[case].springs.kh
        )
        assert by_case[case].result.u == pytest.approx(plain_by_case[case].result.u)
        assert reduced.bearing_for(case) is reduced.bearing
    # 地震時だけが低減される
    assert by_case[LoadCase.LEVEL1_EQ].springs.de < 1.0
    assert by_case[LoadCase.LEVEL1_EQ].result.u > plain_by_case[LoadCase.LEVEL1_EQ].result.u
    assert reduced.bearing_for(LoadCase.LEVEL1_EQ) is reduced.bearing_seismic
    assert any("常時・暴風時の照査には適用していない" in n for n in reduced.notes)


def test_load_case_seismic_flag():
    assert LoadCase.LEVEL1_EQ.is_seismic
    assert not LoadCase.PERMANENT.is_seismic
    assert not LoadCase.STORM.is_seismic


def test_liquefying_bearing_stratum_is_flagged():
    """支持層が液状化する場合、qd を低減していない旨を警告すること。"""
    from core.capacity.bearing import compute_bearing_capacity

    # 全層が緩い砂で、杭先端まで液状化判定の範囲(20 m)に入る配置
    profile = SoilProfile(
        layers=[
            SoilLayer(
                name="As", soil_type=SoilType.SAND, thickness=30.0, n_value=12.0,
                gamma_t=18.0, gamma_sat=19.0, fc=5.0, d50=0.3, d10=0.08,
            ),
        ],
        gwl=1.0,
    )
    assessment = assess_liquefaction(profile, GroundType.TYPE_II)
    # 打込み杭は支持層の最低N値の制約がないため、この地盤でも算定できる
    short = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=15.0,
        wall_thickness=12.0,
    )
    bc = compute_bearing_capacity(
        short, profile, 2.0,
        reduction=SoilReduction.from_assessment(assessment, MOTION),
    )
    assert bc.tip_zone_liquefies
    assert bc.tip_de < 1.0

    report = analyze(short, ARRANGEMENT, FOOTING, profile,
                     [FootingLoads(case=LoadCase.LEVEL1_EQ, v=5000.0, h=500.0, m=1000.0)],
                     reduction=SoilReduction.from_assessment(assessment, MOTION))
    assert any("先端支持力度 qd は低減して" in n for n in report.notes)
