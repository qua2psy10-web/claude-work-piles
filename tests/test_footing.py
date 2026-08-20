"""底版(フーチング)本体の許容応力度法照査のテスト(第58回)。

期待値はフォーラムエイト UC-1「基礎の設計」計算書サンプル **Kui_8** の
7章「底版許容応力度法照査」による(σck=24、SD345、φ1200 場所打ち杭、
14.4m × 14.4m × 厚2.5m のフーチング)。
"""
import math

import pytest

from core.models import LoadCase
from core.section.footing import (
    _stress_block_factors,
    cdc_factor,
    cds_factor,
    check_footing_flexure,
    check_footing_shear,
    cpt_factor_footing,
    cracking_moment_rectangular,
    required_rebar_area,
    singly_reinforced_stress,
    ultimate_moment_singly_reinforced,
)

D25 = 506.7e-6  # D25 の公称断面積 (m2)


# --- 単鉄筋長方形断面の応力度 -------------------------------------------------


def test_neutral_axis_satisfies_its_defining_equation():
    """中立軸が b・x²/2 = n・As・(d − x) を満たすこと(式の自己検証)。"""
    b, d, area, n = 12.0, 2.3, 47629.8e-6, 15.0
    result = singly_reinforced_stress(b, d, area, 12963.75, n)
    x = result.neutral_axis
    assert b * x * x / 2.0 == pytest.approx(n * area * (d - x), rel=1e-12)
    assert result.lever_arm == pytest.approx(d - x / 3.0)


def test_flexural_stress_matches_kui8_bottom_tension():
    """Kui_8 7.5(1) 橋軸方向 柱左前面・下側引張(常時)。

    M=12963.75 kN·m、b=12.0m、d=2.3m、As=47629.8mm²(D25@125)に対し
    計算例は x=467.2mm、σc=2.16、σs=126.93 N/mm²。
    """
    result = singly_reinforced_stress(12.0, 2.3, 47629.8e-6, 12963.75)
    assert result.neutral_axis * 1000.0 == pytest.approx(467.2, abs=0.1)
    assert result.sigma_c == pytest.approx(2.16, abs=0.005)
    assert result.sigma_s == pytest.approx(126.93, abs=0.01)


def test_flexural_stress_matches_kui8_top_tension():
    """同 上側引張(地震時)。M=-4463.27、b=8.78m、d=2.39m、As=18241.2mm²。

    計算例は x=356.1mm、σc=1.26、σs=107.72。モーメントの符号は結果に
    影響しない(絶対値で扱う)。
    """
    result = singly_reinforced_stress(8.78, 2.39, 18241.2e-6, -4463.27)
    assert result.neutral_axis * 1000.0 == pytest.approx(356.1, abs=0.2)
    assert result.sigma_c == pytest.approx(1.26, abs=0.005)
    assert result.sigma_s == pytest.approx(107.72, abs=0.02)
    # 符号を反転しても同じ
    positive = singly_reinforced_stress(8.78, 2.39, 18241.2e-6, 4463.27)
    assert positive.sigma_s == pytest.approx(result.sigma_s)


def test_required_rebar_area_makes_the_steel_stress_equal_the_allowable():
    """必要鉄筋量は σs がちょうど許容値になる量(自己整合の検証)。"""
    b, d, moment, sigma_sa = 12.0, 2.3, 12963.75, 180.0
    area = required_rebar_area(b, d, moment, sigma_sa)
    assert singly_reinforced_stress(b, d, area, moment).sigma_s == pytest.approx(
        sigma_sa, rel=1e-9
    )
    # Kui_8 の計算例は 33226 mm²
    assert area * 1.0e6 == pytest.approx(33226, rel=1e-4)


def test_required_rebar_area_matches_kui8_top_tension():
    area = required_rebar_area(8.78, 2.39, 4463.27, 300.0)
    assert area * 1.0e6 == pytest.approx(6421, rel=1e-3)


def test_required_rebar_area_is_zero_without_moment():
    assert required_rebar_area(12.0, 2.3, 0.0, 180.0) == 0.0


def test_singly_reinforced_rejects_degenerate_input():
    with pytest.raises(ValueError, match="部材幅・有効高"):
        singly_reinforced_stress(0.0, 2.3, 1.0e-3, 100.0)
    with pytest.raises(ValueError, match="鉄筋量"):
        singly_reinforced_stress(12.0, 2.3, 0.0, 100.0)


# --- ひび割れ/終局モーメント -------------------------------------------------


def test_cracking_moment_matches_kui8_and_confirms_the_tensile_strength():
    """Mc = σbt・b・h²/6 で σbt = 0.23・σck^(2/3)。

    Kui_8 の最小鉄筋量照査は b=12.0m で Mc=23920.96、b=8.78m で
    Mc=17502.17 kN·m(いずれも h=2.5m、σck=24)。この2点から逆算した
    σbt はどちらも 1.913676 で、0.23・24^(2/3) = 1.9136758 と**7桁で
    一致**する。第54回に M-φ のひび割れモーメントから同じ式を確認しようと
    して換算断面の丸めのため 1.5% 合わなかったものが、長方形断面(丸めの
    無い断面諸元)で厳密に裏付けられた。
    """
    assert cracking_moment_rectangular(12.0, 2.5, 24) == pytest.approx(
        23920.96, abs=0.01
    )
    assert cracking_moment_rectangular(8.78, 2.5, 24) == pytest.approx(
        17502.17, abs=0.01
    )
    # 逆算した σbt が 0.23・σck^(2/3) と一致すること
    z = 12.0 * 2.5**2 / 6.0
    assert 23920.96 / z / 1000.0 == pytest.approx(
        0.23 * 24 ** (2.0 / 3.0), rel=1e-6
    )


def test_cracking_moment_scales_with_width_and_squared_height():
    base = cracking_moment_rectangular(10.0, 2.0, 24)
    assert cracking_moment_rectangular(20.0, 2.0, 24) == pytest.approx(2 * base)
    assert cracking_moment_rectangular(10.0, 4.0, 24) == pytest.approx(4 * base)


def test_stress_block_factors_match_the_hand_integration():
    """道示Ⅲ の放物線＋矩形分布を積分した α・β。"""
    alpha, beta = _stress_block_factors()
    assert alpha == pytest.approx(1.0 - (0.002 / 0.0035) / 3.0)
    assert alpha == pytest.approx(0.809524, abs=1e-6)
    assert beta == pytest.approx(0.415966, abs=1e-6)
    # β = 1 −(1/2 − r²/12)/α を厳密な分数で確かめる
    r = 0.002 / 0.0035
    assert beta == pytest.approx(1.0 - (0.5 - r * r / 12.0) / alpha, rel=1e-12)


def test_ultimate_moment_matches_kui8_both_sections():
    """Mu の計算例との一致(0.004% 以内)。

    等価応力矩形ブロック(0.85σck × 0.8x)では 0.04% ずれるのに対し、
    放物線を含む分布では 0.004% まで詰まる。
    """
    assert ultimate_moment_singly_reinforced(
        12.0, 2.3, 47629.8e-6, 24, "SD345"
    ) == pytest.approx(37228.61, rel=5e-5)
    assert ultimate_moment_singly_reinforced(
        8.78, 2.39, 18241.2e-6, 24, "SD345"
    ) == pytest.approx(14927.40, rel=5e-5)


def test_ultimate_moment_rejects_unknown_rebar_grade():
    with pytest.raises(ValueError, match="降伏点"):
        ultimate_moment_singly_reinforced(12.0, 2.3, 1.0e-3, 24, "SD000")


# --- 曲げ照査(最小鉄筋量を含む) ---------------------------------------------


def test_check_footing_flexure_bottom_tension_case():
    result = check_footing_flexure(
        moment=12963.75, width=12.0, height=2.5, effective_depth=2.3,
        rebar_area=47629.8e-6, fck=24, case=LoadCase.PERMANENT,
        sigma_sa=180.0, label="柱左前面 下側引張",
    )
    by_name = {c.name: c for c in result.checks}
    assert by_name["底版コンクリート曲げ圧縮応力度"].stress == pytest.approx(2.16, abs=0.005)
    assert by_name["底版コンクリート曲げ圧縮応力度"].allowable == pytest.approx(8.0)
    assert by_name["底版鉄筋引張応力度"].stress == pytest.approx(126.93, abs=0.01)
    assert by_name["底版鉄筋引張応力度"].allowable == pytest.approx(180.0)
    assert result.rebar_area_per_m == pytest.approx(3969.2, abs=0.1)
    assert result.all_ok


def test_check_footing_flexure_applies_the_seismic_increase_to_concrete():
    """地震時はコンクリートの許容曲げ圧縮応力度が 1.5 倍(8.0 → 12.0)。"""
    result = check_footing_flexure(
        moment=-4463.27, width=8.78, height=2.5, effective_depth=2.39,
        rebar_area=18241.2e-6, fck=24, case=LoadCase.LEVEL1_EQ,
        sigma_sa=300.0,
    )
    by_name = {c.name: c for c in result.checks}
    assert by_name["底版コンクリート曲げ圧縮応力度"].allowable == pytest.approx(12.0)
    assert result.rebar_area_per_m == pytest.approx(2077.6, abs=0.1)
    assert result.all_ok


def test_minimum_rebar_uses_the_mu_ge_mc_branch_when_it_holds():
    """下側引張ケースは Mu(37229)≧ Mc(23921)で1つ目の条件を満たす。"""
    result = check_footing_flexure(
        moment=12963.75, width=12.0, height=2.5, effective_depth=2.3,
        rebar_area=47629.8e-6, fck=24, case=LoadCase.PERMANENT, sigma_sa=180.0,
    )
    assert result.ultimate_moment > result.cracking_moment
    assert result.min_rebar_ok
    assert "Mu=" in result.min_rebar_note and "≧ Mc=" in result.min_rebar_note


def test_minimum_rebar_falls_back_to_the_1_7m_branch():
    """上側引張ケースは Mu(14927)< Mc(17502)だが 1.7M(7588)≦ Mc なので OK。

    この2段構えの判定(Mu≧Mc **または** 1.7M≦Mc)が効いていることを、
    Kui_8 が実際に「OK」としているケースで固定する。
    """
    result = check_footing_flexure(
        moment=-4463.27, width=8.78, height=2.5, effective_depth=2.39,
        rebar_area=18241.2e-6, fck=24, case=LoadCase.LEVEL1_EQ, sigma_sa=300.0,
    )
    assert result.ultimate_moment < result.cracking_moment
    assert 1.7 * abs(result.moment) <= result.cracking_moment
    assert result.min_rebar_ok
    assert "1.7M=" in result.min_rebar_note


def test_minimum_rebar_fails_when_the_area_per_metre_is_too_small():
    """鉄筋量が 500mm²/m 未満なら、強度側を満たしても NG になること。"""
    result = check_footing_flexure(
        moment=50.0, width=12.0, height=2.5, effective_depth=2.3,
        rebar_area=12.0 * 400.0e-6, fck=24, case=LoadCase.PERMANENT,
        sigma_sa=180.0,
    )
    assert result.rebar_area_per_m == pytest.approx(400.0)
    assert not result.min_rebar_ok
    assert not result.all_ok
    assert "< 500" in result.min_rebar_note


def test_check_footing_flexure_rejects_unsupported_fck():
    with pytest.raises(ValueError, match="σca"):
        check_footing_flexure(
            moment=100.0, width=12.0, height=2.5, effective_depth=2.3,
            rebar_area=1.0e-3, fck=80, case=LoadCase.PERMANENT, sigma_sa=180.0,
        )


# --- せん断照査 ---------------------------------------------------------------


def test_cdc_factor_interpolates_the_table():
    assert cdc_factor(0.5, 1.0) == pytest.approx(6.4)
    assert cdc_factor(2.5, 1.0) == pytest.approx(1.0)
    # 範囲外は端の値で頭打ち
    assert cdc_factor(0.1, 1.0) == pytest.approx(6.4)
    assert cdc_factor(10.0, 1.0) == pytest.approx(1.0)
    # Kui_8 の2点
    assert cdc_factor(1.65, 2.3) == pytest.approx(5.357, abs=0.001)
    assert cdc_factor(3.0, 2.39) == pytest.approx(3.234, abs=0.001)


def test_cds_factor_matches_kui8_and_is_capped_at_one():
    assert cds_factor(1.65, 2.3) == pytest.approx(0.287, abs=0.001)
    assert cds_factor(3.0, 2.39) == pytest.approx(0.502, abs=0.001)
    # a/d' ≧ 2.5 で 1.0 に頭打ち(cdc が割増しなしになる点と整合)
    assert cds_factor(2.5, 1.0) == pytest.approx(1.0)
    assert cds_factor(10.0, 1.0) == pytest.approx(1.0)


def test_cpt_factor_footing_extrapolates_below_the_table_minimum():
    """底版の cpt は pt < 0.1% で線形外挿する(Kui_8 で確認、安全側)。"""
    from core.section.shear import cpt_factor

    assert cpt_factor_footing(0.0854) == pytest.approx(0.671, abs=0.001)
    # 表の範囲内では杭体と同じ
    assert cpt_factor_footing(0.1744) == pytest.approx(cpt_factor(0.1744))
    assert cpt_factor_footing(0.5) == pytest.approx(cpt_factor(0.5))
    # 杭体は従来どおり下限で頭打ち(据え置き)
    assert cpt_factor(0.0854) == pytest.approx(0.7)
    # 外挿は必ず小さい側 = 安全側
    assert cpt_factor_footing(0.05) < 0.7
    # 負にはならない
    assert cpt_factor_footing(0.0) >= 0.0


def test_shear_matches_kui8_left_overhang_permanent():
    """Kui_8 7.6(1) 橋軸方向 左張出し部・下側引張(常時)。

    S=9132.81kN、b=14.4m、d=2.3m、a=1.65m、d'=2.3m、D25@125×114本。
    計算例は Ce=0.805、Cpt=0.849、Cdc=5.357、τm=0.276、τa=0.842、
    τa2=1.700、Sca=27881.11kN。
    """
    result = check_footing_shear(
        shear=9132.81, width=14.4, effective_depth=2.3,
        rebar_area=114 * D25, shear_span=1.65, column_face_depth=2.3,
        fck=24, case=LoadCase.PERMANENT, label="左張出し部",
    )
    assert result.pt == pytest.approx(0.1744, abs=0.0005)
    assert result.ce == pytest.approx(0.805, abs=0.001)
    assert result.cpt == pytest.approx(0.849, abs=0.001)
    assert result.cdc == pytest.approx(5.357, abs=0.001)
    assert result.cds == pytest.approx(0.287, abs=0.001)
    assert result.tau_m == pytest.approx(0.276, abs=0.001)
    assert result.tau_a == pytest.approx(0.842, abs=0.001)
    assert result.tau_a2 == pytest.approx(1.700, abs=0.001)
    assert result.concrete_shear == pytest.approx(27881.11, abs=1.0)
    assert not result.needs_stirrup
    assert result.stirrup_shear == 0.0
    assert result.all_ok


def test_shear_matches_kui8_left_overhang_seismic():
    """同 上側引張(地震時)。τa1 に代えて τc=0.35 を用い、pt<0.1% を外挿。

    計算例は Ce=0.791、Cpt=0.671、Cdc=3.234、τm=0.053、τa=0.601、
    τa2=2.550、Sca=20684.45kN。
    """
    result = check_footing_shear(
        shear=-1833.66, width=14.4, effective_depth=2.39,
        rebar_area=58 * D25, shear_span=3.0, column_face_depth=2.39,
        fck=24, case=LoadCase.LEVEL1_EQ,
    )
    assert result.pt == pytest.approx(0.0854, abs=0.0005)
    assert result.ce == pytest.approx(0.791, abs=0.001)
    assert result.cpt == pytest.approx(0.671, abs=0.001)
    assert result.cdc == pytest.approx(3.234, abs=0.001)
    assert result.cds == pytest.approx(0.502, abs=0.001)
    assert result.tau_m == pytest.approx(0.053, abs=0.001)
    assert result.tau_a == pytest.approx(0.601, abs=0.001)
    assert result.tau_a2 == pytest.approx(2.550, abs=0.001)
    assert result.concrete_shear == pytest.approx(20684.45, abs=1.0)
    assert result.seismic
    assert not result.needs_stirrup
    assert result.all_ok


def test_seismic_uses_tau_c_instead_of_scaling_tau_a1():
    """地震時は τa1×1.5 ではなく τc を用いる(杭体の照査と同じ扱い)。"""
    from core.standards import TAU_A1_CONCRETE, TAU_C_CONCRETE

    common = dict(
        shear=1000.0, width=14.4, effective_depth=2.3, rebar_area=114 * D25,
        shear_span=1.65, column_face_depth=2.3, fck=24,
    )
    static = check_footing_shear(case=LoadCase.PERMANENT, **common)
    seismic = check_footing_shear(case=LoadCase.LEVEL1_EQ, **common)
    # τa の比は基本値の比に等しい(補正係数は同じ)
    assert seismic.tau_a / static.tau_a == pytest.approx(
        TAU_C_CONCRETE[24] / TAU_A1_CONCRETE[24], rel=1e-9
    )


def test_shear_flags_a_stirrup_requirement_without_inventing_the_amount():
    """τm > τa なら斜引張鉄筋が必要と判定し、必要量は算定しないこと。"""
    result = check_footing_shear(
        shear=40000.0, width=14.4, effective_depth=2.3,
        rebar_area=114 * D25, shear_span=1.65, column_face_depth=2.3,
        fck=24, case=LoadCase.PERMANENT,
    )
    assert result.needs_stirrup
    assert result.stirrup_shear == pytest.approx(40000.0 - result.concrete_shear)
    assert any("未確認" in n for n in result.notes)
    # τa2 を超えていなければ照査自体は OK(鉄筋を入れれば成立する)
    assert result.tau_m < result.tau_a2
    assert result.all_ok


def test_shear_is_ng_only_when_tau_a2_is_exceeded():
    """τa2 を超えると鉄筋を増やしても解決しないので NG。"""
    result = check_footing_shear(
        shear=90000.0, width=14.4, effective_depth=2.3,
        rebar_area=114 * D25, shear_span=1.65, column_face_depth=2.3,
        fck=24, case=LoadCase.PERMANENT,
    )
    assert result.tau_m > result.tau_a2
    assert not result.all_ok


def test_shear_uses_the_absolute_value_of_the_force():
    common = dict(
        width=14.4, effective_depth=2.3, rebar_area=114 * D25,
        shear_span=1.65, column_face_depth=2.3, fck=24,
        case=LoadCase.PERMANENT,
    )
    assert check_footing_shear(shear=5000.0, **common).tau_m == pytest.approx(
        check_footing_shear(shear=-5000.0, **common).tau_m
    )


def test_shear_rejects_unsupported_fck_and_zero_depth():
    with pytest.raises(ValueError, match="許容せん断応力度"):
        check_footing_shear(
            shear=1000.0, width=14.4, effective_depth=2.3,
            rebar_area=1.0e-3, shear_span=1.65, column_face_depth=2.3,
            fck=80, case=LoadCase.PERMANENT,
        )
    with pytest.raises(ValueError, match="柱前面での有効高"):
        cdc_factor(1.65, 0.0)
