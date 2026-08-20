"""水平地盤反力度の上限値 pHU(道示Ⅳ 12.10)のテスト。"""
import math

import pytest

from core.capacity.lateral_limit import (
    alpha_p,
    eta_p_alpha_p,
    p_hu,
    passive_pressure,
    spring_force_limit,
)
from core.models import SoilLayer, SoilProfile, SoilType


def layer(soil_type, n_value, **kwargs):
    defaults = dict(
        name="L", thickness=10.0, gamma_t=18.0, gamma_sat=19.0, k_ep=3.0
    )
    defaults.update(kwargs)
    return SoilLayer(soil_type=soil_type, n_value=n_value, **defaults)


# --- αp ---------------------------------------------------------------------


def test_alpha_p_by_soil_type():
    assert alpha_p(layer(SoilType.SAND, 20.0)) == 3.0
    assert alpha_p(layer(SoilType.GRAVEL, 20.0)) == 3.0
    assert alpha_p(layer(SoilType.CLAY, 20.0)) == 1.5


def test_soft_clay_uses_alpha_p_one():
    """N ≤ 2 の軟弱な粘性土は αp = 1.0。"""
    assert alpha_p(layer(SoilType.CLAY, 2.0)) == 1.0
    assert alpha_p(layer(SoilType.CLAY, 1.0)) == 1.0
    # 境界のすぐ上は通常値
    assert alpha_p(layer(SoilType.CLAY, 2.1)) == 1.5
    # 砂質土には N 値の特例はない
    assert alpha_p(layer(SoilType.SAND, 1.0)) == 3.0


# --- ηp・αp -----------------------------------------------------------------


def test_group_effect_in_sand_uses_spacing_ratio():
    """砂質地盤では ηp・αp = min(s/D, αp)。"""
    sand = layer(SoilType.SAND, 20.0)
    # s/D = 2.2 < 3.0 → 2.2
    assert eta_p_alpha_p(sand, diameter=1.0, spacing_perpendicular=2.2) == 2.2
    # s/D = 4.0 > 3.0 → αp = 3.0 で頭打ち
    assert eta_p_alpha_p(sand, diameter=1.0, spacing_perpendicular=4.0) == 3.0


def test_group_effect_not_applied_in_clay():
    """粘性土地盤では ηp = 1.0(間隔によらない)。"""
    clay = layer(SoilType.CLAY, 20.0)
    for spacing in (1.5, 2.2, 6.0):
        assert eta_p_alpha_p(clay, 1.0, spacing) == 1.5


def test_group_effect_rejects_zero_diameter():
    with pytest.raises(ValueError):
        eta_p_alpha_p(layer(SoilType.SAND, 20.0), 0.0, 2.0)


# --- pU ---------------------------------------------------------------------


def test_passive_pressure_matches_formula():
    """pU = KEP・σ'v + 2c・√KEP。"""
    profile = SoilProfile(
        layers=[layer(SoilType.CLAY, 10.0, cohesion=30.0, k_ep=2.0)], gwl=100.0
    )
    depth = 5.0
    _, sigma_v_eff = profile.stresses_at(depth)
    expected = 2.0 * sigma_v_eff + 2.0 * 30.0 * math.sqrt(2.0)
    assert passive_pressure(profile, depth) == pytest.approx(expected)


def test_passive_pressure_uses_effective_stress_below_water():
    """地下水位以深は有効応力を用いること(全応力ではない)。"""
    profile = SoilProfile(
        layers=[layer(SoilType.SAND, 20.0, k_ep=3.0)], gwl=0.0
    )
    dry = SoilProfile(
        layers=[layer(SoilType.SAND, 20.0, k_ep=3.0)], gwl=100.0
    )
    assert passive_pressure(profile, 5.0) < passive_pressure(dry, 5.0)


def test_passive_pressure_requires_k_ep():
    profile = SoilProfile(
        layers=[layer(SoilType.SAND, 20.0, k_ep=None, name="As1")], gwl=2.0
    )
    with pytest.raises(ValueError, match="KEP"):
        passive_pressure(profile, 5.0)


# --- pHU --------------------------------------------------------------------


def test_p_hu_combines_factors():
    profile = SoilProfile(layers=[layer(SoilType.SAND, 20.0, k_ep=3.0)], gwl=100.0)
    pu = passive_pressure(profile, 5.0)
    value = p_hu(profile, 5.0, diameter=1.0, spacing_perpendicular=2.2)
    assert value == pytest.approx(2.2 * pu)


def test_non_front_row_is_halved_in_sand():
    profile = SoilProfile(layers=[layer(SoilType.SAND, 20.0)], gwl=100.0)
    front = p_hu(profile, 5.0, 1.0, 2.5, front_row=True)
    back = p_hu(profile, 5.0, 1.0, 2.5, front_row=False)
    assert back == pytest.approx(front * 0.5)


def test_non_front_row_not_halved_in_clay():
    """1/2 の規定は資料上、砂質地盤にのみ記載されている。"""
    profile = SoilProfile(layers=[layer(SoilType.CLAY, 10.0)], gwl=100.0)
    front = p_hu(profile, 5.0, 1.0, 2.5, front_row=True)
    back = p_hu(profile, 5.0, 1.0, 2.5, front_row=False)
    assert back == pytest.approx(front)


def test_p_hu_increases_with_depth():
    profile = SoilProfile(layers=[layer(SoilType.SAND, 20.0)], gwl=100.0)
    values = [p_hu(profile, d, 1.0, 2.5) for d in (1.0, 3.0, 5.0, 9.0)]
    assert values == sorted(values)


# --- 反力度 → 力 -------------------------------------------------------------


def test_spring_force_limit_converts_stress_to_force():
    """R_max = pHU × D × Δz。反力度と力の取り違えを防ぐ。"""
    profile = SoilProfile(layers=[layer(SoilType.SAND, 20.0)], gwl=100.0)
    stress = p_hu(profile, 5.0, 1.2, 3.0)
    force = spring_force_limit(profile, 5.0, 1.2, 3.0, tributary_length=0.5)
    assert force == pytest.approx(stress * 1.2 * 0.5)


def test_spring_force_limit_rejects_zero_length():
    profile = SoilProfile(layers=[layer(SoilType.SAND, 20.0)], gwl=100.0)
    with pytest.raises(ValueError):
        spring_force_limit(profile, 5.0, 1.0, 2.5, tributary_length=0.0)
