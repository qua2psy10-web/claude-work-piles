"""液状化判定のテスト。

期待値は道示Ⅴ(H24) 8.2 の式による手計算値。
検証ケース: 砂質土 N=10, FC=5%, γt=18, γsat=19, 地下水位1.0m, II種地盤, cz=1.0
深度5mにて
  σv = 94, σ'v = 54.8
  N1 = 170×10/124.8 = 13.622, Na = N1 (FC<10%), RL = 0.0882√(Na/1.7) = 0.24967
  rd = 0.925
  タイプI : khg=0.45, L = 0.925×0.45×94/54.8 = 0.71401, cw=1.0 → FL = 0.3497
  タイプII: khg=0.70, L = 1.11068, cw = 3.3RL+0.67 = 1.49390,
            R = 0.37298 → FL = 0.3358
  DE: FL∈(1/3, 2/3], 深度≦10m → タイプI(R≦0.3)=1/3, タイプII(R>0.3)=2/3
"""
import pytest

from core.models import SoilLayer, SoilProfile, SoilType
from core.soil.liquefaction import (
    assess_liquefaction,
    cw_value,
    evaluate_at,
    na_sand,
    reduction_factor_de,
    rl_value,
)
from core.standards import GroundMotionType, GroundType


def sand_profile(gwl: float = 1.0, **layer_kw) -> SoilProfile:
    defaults = dict(
        name="As",
        soil_type=SoilType.SAND,
        thickness=10.0,
        n_value=10,
        gamma_t=18.0,
        gamma_sat=19.0,
        fc=5.0,
        d50=0.3,
        d10=0.05,
    )
    defaults.update(layer_kw)
    return SoilProfile(layers=[SoilLayer(**defaults)], gwl=gwl)


def test_fl_hand_calculation():
    result = evaluate_at(sand_profile(), 5.0, GroundType.TYPE_II)
    assert result.is_target
    assert result.n1 == pytest.approx(13.622, rel=1e-3)
    assert result.na == pytest.approx(13.622, rel=1e-3)
    assert result.rl == pytest.approx(0.24967, rel=1e-3)
    assert result.l_type1 == pytest.approx(0.71401, rel=1e-3)
    assert result.fl_type1 == pytest.approx(0.3497, rel=1e-3)
    assert result.r_type2 == pytest.approx(0.37298, rel=1e-3)
    assert result.fl_type2 == pytest.approx(0.3358, rel=1e-3)
    assert result.de_type1 == pytest.approx(1.0 / 3.0)
    assert result.de_type2 == pytest.approx(2.0 / 3.0)


def test_na_sand_fc_correction():
    assert na_sand(10.0, 5.0) == pytest.approx(10.0)
    # FC=20%: c1=(20+40)/50=1.2, c2=(20-10)/18=0.5556
    assert na_sand(10.0, 20.0) == pytest.approx(12.556, rel=1e-3)
    # FC=70%: c1=70/20-1=2.5, c2=(70-10)/18=3.333
    assert na_sand(10.0, 70.0) == pytest.approx(28.333, rel=1e-3)


def test_rl_high_na_term():
    # Na≧14 では第2項が加算される
    assert rl_value(20.0) == pytest.approx(
        0.0882 * (20.0 / 1.7) ** 0.5 + 1.6e-6 * 6.0**4.5, rel=1e-6
    )


def test_cw():
    assert cw_value(0.05, GroundMotionType.LEVEL2_TYPE2) == 1.0
    assert cw_value(0.2, GroundMotionType.LEVEL2_TYPE2) == pytest.approx(1.33)
    assert cw_value(0.5, GroundMotionType.LEVEL2_TYPE2) == 2.0
    assert cw_value(0.5, GroundMotionType.LEVEL2_TYPE1) == 1.0


def test_de_table():
    assert reduction_factor_de(0.2, 5.0, 0.2) == 0.0
    assert reduction_factor_de(0.2, 5.0, 0.4) == pytest.approx(1 / 6)
    assert reduction_factor_de(0.2, 15.0, 0.2) == pytest.approx(1 / 3)
    assert reduction_factor_de(0.5, 5.0, 0.4) == pytest.approx(2 / 3)
    assert reduction_factor_de(0.9, 5.0, 0.4) == 1.0
    assert reduction_factor_de(1.5, 5.0, 0.2) == 1.0  # FL>1 は低減しない


def test_screening_clay_excluded():
    profile = sand_profile(soil_type=SoilType.CLAY)
    result = evaluate_at(profile, 5.0, GroundType.TYPE_II)
    assert not result.is_target
    assert "粘性土" in result.excluded_reason


def test_screening_above_gwl():
    result = evaluate_at(sand_profile(gwl=6.0), 5.0, GroundType.TYPE_II)
    assert not result.is_target
    assert "地下水位以浅" in result.excluded_reason


def test_screening_deep_gwl():
    # 地下水位が10mより深い場合は全層対象外
    profile = sand_profile(gwl=12.0, thickness=20.0)
    assessment = assess_liquefaction(profile, GroundType.TYPE_II)
    assert all(not s.is_target for s in assessment.slices)


def test_screening_depth_limit():
    profile = sand_profile(thickness=25.0)
    assessment = assess_liquefaction(profile, GroundType.TYPE_II)
    deep = [s for s in assessment.slices if s.depth_mid > 20.0]
    assert deep and all(not s.is_target for s in deep)


def test_screening_fine_content():
    # FC>35% かつ IP 未入力 → 対象外
    result = evaluate_at(sand_profile(fc=40.0), 5.0, GroundType.TYPE_II)
    assert not result.is_target
    # FC>35% でも IP≦15 なら対象
    result = evaluate_at(sand_profile(fc=40.0, ip=10.0), 5.0, GroundType.TYPE_II)
    assert result.is_target


def test_assess_overall():
    assessment = assess_liquefaction(sand_profile(), GroundType.TYPE_II)
    assert assessment.liquefiable_type1
    assert assessment.liquefiable_type2
    # 1mピッチで10スライス
    assert len(assessment.slices) == 10
