"""軸方向支持力のテスト(道示Ⅳ(H24) 12.4)。"""
import math

import pytest

from core.capacity.bearing import (
    compute_bearing_capacity,
    skin_friction_intensity,
    tip_resistance_intensity,
)
from core.models import (
    ConstructionMethod,
    LoadCase,
    PileSpec,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
    SupportType,
)


def profile_two_layers() -> SoilProfile:
    return SoilProfile(
        layers=[
            SoilLayer(
                name="Ac",
                soil_type=SoilType.CLAY,
                thickness=10.0,
                n_value=4.0,
                gamma_t=16.0,
                gamma_sat=16.5,
                cohesion=40.0,
            ),
            SoilLayer(
                name="Ds",
                soil_type=SoilType.SAND,
                thickness=15.0,
                n_value=40.0,
                gamma_t=19.0,
                gamma_sat=20.0,
            ),
        ],
        gwl=2.0,
    )


def test_qd_cast_in_place_sand_is_constant():
    layer = profile_two_layers().layers[1]
    qd = tip_resistance_intensity(ConstructionMethod.CAST_IN_PLACE, layer, 40.0)
    assert qd == 3000.0


def test_qd_driven_proportional_to_n_with_cap():
    layer = profile_two_layers().layers[1]
    assert tip_resistance_intensity(ConstructionMethod.DRIVEN, layer, 40.0) == 5200.0
    # 上限 6500 で頭打ち
    assert tip_resistance_intensity(ConstructionMethod.DRIVEN, layer, 60.0) == 6500.0


def test_qd_cast_in_place_clay_uses_qu():
    layer = profile_two_layers().layers[0]  # c=40 → qu=80 → 3qu=240
    qd = tip_resistance_intensity(ConstructionMethod.CAST_IN_PLACE, layer, 4.0)
    assert qd == pytest.approx(240.0)


def test_qd_rejects_unsupported_soil():
    layer = profile_two_layers().layers[0]  # 粘性土
    with pytest.raises(ValueError, match="支持層"):
        tip_resistance_intensity(ConstructionMethod.DRIVEN, layer, 4.0)


def test_skin_friction_caps():
    sand = SoilLayer(
        soil_type=SoilType.SAND, thickness=1.0, n_value=50.0,
        gamma_t=19.0, gamma_sat=20.0,
    )
    # 場所打ち: f=5N=250 → 上限200
    assert skin_friction_intensity(ConstructionMethod.CAST_IN_PLACE, sand) == 200.0
    # 打込み: f=2N=100 → 上限100
    assert skin_friction_intensity(ConstructionMethod.DRIVEN, sand) == 100.0

    clay = SoilLayer(
        soil_type=SoilType.CLAY, thickness=1.0, n_value=4.0,
        gamma_t=16.0, gamma_sat=16.5, cohesion=200.0,
    )
    # f=c=200 → 上限150
    assert skin_friction_intensity(ConstructionMethod.CAST_IN_PLACE, clay) == 150.0


def test_skin_friction_cement_methods():
    """中掘り・プレボーリングは砂質土 3N(≦150)、粘性土 c(≦100)。"""
    sand = SoilLayer(
        soil_type=SoilType.SAND, thickness=1.0, n_value=20.0,
        gamma_t=19.0, gamma_sat=20.0,
    )
    clay = SoilLayer(
        soil_type=SoilType.CLAY, thickness=1.0, n_value=8.0,
        gamma_t=16.0, gamma_sat=16.5, cohesion=80.0,
    )
    for method in (ConstructionMethod.INNER_DIGGING, ConstructionMethod.PREBORING):
        assert skin_friction_intensity(method, sand) == pytest.approx(60.0)
        assert skin_friction_intensity(method, clay) == pytest.approx(80.0)

    # 上限で頭打ち
    hard_sand = sand.model_copy(update={"n_value": 80.0})
    stiff_clay = clay.model_copy(update={"cohesion": 200.0})
    assert skin_friction_intensity(ConstructionMethod.INNER_DIGGING, hard_sand) == 150.0
    assert skin_friction_intensity(ConstructionMethod.INNER_DIGGING, stiff_clay) == 100.0


def test_soil_cement_keeps_separate_values():
    """鋼管ソイルセメントは未照合のため中掘り系と異なる値のまま。"""
    sand = SoilLayer(
        soil_type=SoilType.SAND, thickness=1.0, n_value=10.0,
        gamma_t=19.0, gamma_sat=20.0,
    )
    assert skin_friction_intensity(
        ConstructionMethod.STEEL_PIPE_SOIL_CEMENT, sand
    ) == pytest.approx(100.0)
    assert skin_friction_intensity(
        ConstructionMethod.INNER_DIGGING, sand
    ) == pytest.approx(30.0)


def test_skin_friction_clay_without_c_uses_10n():
    clay = SoilLayer(
        soil_type=SoilType.CLAY, thickness=1.0, n_value=4.0,
        gamma_t=16.0, gamma_sat=16.5,
    )
    assert skin_friction_intensity(ConstructionMethod.CAST_IN_PLACE, clay) == 40.0


def test_bearing_capacity_hand_calculation():
    """場所打ち杭 D=1.0m, 杭頭深さ2m, 杭長18m(先端深度20m)

    先端は Ds層(砂質土) → qd = 3000, A = π/4 = 0.785398
      先端支持力 = 2356.19 kN
    周面: 先端から 1D = 1.0m 手前(深さ19m)までを計上する(道示Ⅳ 12.4.1)
      Ac層 2〜10m (8m, f = c = 40), Ds層 10〜19m (9m, f = 5N = 200 → 上限200)
      U = π = 3.141593
      Ac: 3.141593×8×40 = 1005.31
      Ds: 3.141593×9×200 = 5654.87
      計 6660.18 kN
    Ru = 2356.19 + 6660.18 = 9016.37 kN
    """
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    bc = compute_bearing_capacity(pile, profile_two_layers(), embedment=2.0)
    assert bc.qd == 3000.0
    assert bc.tip_area == pytest.approx(math.pi / 4)
    assert bc.tip_resistance == pytest.approx(2356.194, rel=1e-4)
    assert bc.skin_resistance == pytest.approx(6660.18, rel=1e-4)
    assert bc.ru == pytest.approx(9016.37, rel=1e-4)
    assert [s.layer_name for s in bc.skin_segments] == ["Ac", "Ds"]
    assert bc.skin_bottom_depth == pytest.approx(19.0)
    assert bc.tip_zone_excluded


def test_tip_zone_exclusion_can_be_disabled():
    """1D 除外を無効にすると先端まで全長で周面摩擦を計上する。"""
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    profile = profile_two_layers()
    with_rule = compute_bearing_capacity(pile, profile, embedment=2.0)
    without = compute_bearing_capacity(
        pile, profile, embedment=2.0, exclude_tip_zone=False
    )
    # 除外分 = π×1.0m×200 = 628.32 kN
    assert without.skin_resistance - with_rule.skin_resistance == pytest.approx(
        math.pi * 1.0 * 200.0, rel=1e-6
    )
    assert without.skin_bottom_depth == pytest.approx(20.0)
    assert not without.tip_zone_excluded


def test_tip_zone_exclusion_clamped_for_short_pile():
    """杭長が 1D 以下でも周面摩擦の下端が杭頭より上に行かない。"""
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=2.0,
        length=1.5,
    )
    bc = compute_bearing_capacity(pile, profile_two_layers(), embedment=2.0)
    assert bc.skin_bottom_depth == pytest.approx(2.0)
    assert bc.skin_resistance == 0.0


def test_average_n_near_tip():
    """N値は先端から上下 1D の範囲の層厚加重平均を用いる。"""
    profile = profile_two_layers()  # Ac(N=4) 0〜10m, Ds(N=40) 10〜25m
    pile = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=8.5,
        wall_thickness=12.0,
    )
    # 先端 10.5m、範囲 9.5〜11.5m → Ac 0.5m(N=4) + Ds 1.0m(N=40)... の加重平均
    bc = compute_bearing_capacity(pile, profile, embedment=2.0)
    expected_n = (4.0 * 0.5 + 40.0 * 1.5) / 2.0
    assert bc.n_tip == pytest.approx(expected_n)
    # qd = 130N(上限 6500)
    assert bc.qd == pytest.approx(min(130.0 * expected_n, 6500.0))


def test_average_n_clamped_at_profile_bottom():
    """先端+1D が地盤モデルを超える場合は範囲を切り詰める。"""
    profile = profile_two_layers()  # 全深度 25m
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=23.0,
    )
    bc = compute_bearing_capacity(pile, profile, embedment=2.0)
    # 先端 25m、範囲 24〜25m はすべて Ds層
    assert bc.n_tip == pytest.approx(40.0)


def test_explicit_n_tip_overrides_average():
    pile = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=18.0,
        wall_thickness=12.0,
    )
    bc = compute_bearing_capacity(
        pile, profile_two_layers(), embedment=2.0, n_tip=25.0
    )
    assert bc.n_tip == 25.0
    assert bc.qd == pytest.approx(130.0 * 25.0)


def test_allowable_capacity_safety_factors():
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    bc = compute_bearing_capacity(pile, profile_two_layers(), embedment=2.0)
    assert bc.support_type == SupportType.END_BEARING
    ra_normal = bc.allowable_push(LoadCase.PERMANENT)
    ra_eq = bc.allowable_push(LoadCase.LEVEL1_EQ)
    # 地震時は安全率が小さいぶん許容値が大きい
    assert ra_eq > ra_normal
    # Ra = (Ru - Ws)/n + Ws - W、支持杭の常時は n = 3
    assert bc.safety_factor_push(LoadCase.PERMANENT) == 3.0
    expected = (bc.ru - bc.w_soil) / 3.0 + bc.w_soil - bc.w_pile
    assert ra_normal == pytest.approx(expected)
    # 引抜きは周面摩擦のみ、常時は n = 6
    assert bc.safety_factor_pull(LoadCase.PERMANENT) == 6.0
    assert bc.allowable_pull(LoadCase.PERMANENT) == pytest.approx(
        bc.skin_resistance / 6.0 + bc.w_pile
    )


def test_friction_pile_uses_stricter_safety_factor():
    """摩擦杭は押込みの安全率が大きく、許容支持力が小さくなる。"""
    profile = profile_two_layers()
    common = dict(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    end_bearing = compute_bearing_capacity(
        PileSpec(**common, support_type=SupportType.END_BEARING), profile, 2.0
    )
    friction = compute_bearing_capacity(
        PileSpec(**common, support_type=SupportType.FRICTION), profile, 2.0
    )
    # 極限支持力は同じ、安全率だけが変わる
    assert friction.ru == pytest.approx(end_bearing.ru)
    assert friction.safety_factor_push(LoadCase.PERMANENT) == 4.0
    assert friction.safety_factor_push(LoadCase.LEVEL1_EQ) == 3.0
    assert friction.allowable_push(LoadCase.PERMANENT) < end_bearing.allowable_push(
        LoadCase.PERMANENT
    )
    # 引抜きは支持形式によらない
    assert friction.allowable_pull(LoadCase.PERMANENT) == pytest.approx(
        end_bearing.allowable_pull(LoadCase.PERMANENT)
    )


def test_pile_tip_beyond_profile_raises():
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=30.0,
    )
    with pytest.raises(ValueError, match="地盤モデル"):
        compute_bearing_capacity(pile, profile_two_layers(), embedment=2.0)
