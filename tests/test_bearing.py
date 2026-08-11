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
    周面: Ac層 2〜10m (8m, f = c = 40), Ds層 10〜20m (10m, f = 5N = 200 → 上限200)
      U = π = 3.141593
      Ac: 3.141593×8×40 = 1005.31
      Ds: 3.141593×10×200 = 6283.19
      計 7288.50 kN
    Ru = 2356.19 + 7288.50 = 9644.69 kN
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
    assert bc.skin_resistance == pytest.approx(7288.50, rel=1e-4)
    assert bc.ru == pytest.approx(9644.69, rel=1e-4)
    assert [s.layer_name for s in bc.skin_segments] == ["Ac", "Ds"]


def test_allowable_capacity_safety_factors():
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    bc = compute_bearing_capacity(pile, profile_two_layers(), embedment=2.0)
    ra_normal = bc.allowable_push(LoadCase.PERMANENT)
    ra_eq = bc.allowable_push(LoadCase.LEVEL1_EQ)
    # 地震時は安全率が小さいぶん許容値が大きい
    assert ra_eq > ra_normal
    # Ra = (Ru - Ws)/n + Ws - W
    expected = (bc.ru - bc.w_soil) / 3.0 + bc.w_soil - bc.w_pile
    assert ra_normal == pytest.approx(expected)
    # 引抜きは周面摩擦のみ
    assert bc.allowable_pull(LoadCase.PERMANENT) == pytest.approx(
        bc.skin_resistance / 6.0 + bc.w_pile
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
