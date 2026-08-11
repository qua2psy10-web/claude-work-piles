"""杭種・工法の比較表のテスト。"""
import pytest

from core.analysis.comparison import (
    APPLICABLE_METHODS,
    compare,
    required_pile_count,
)
from core.models import (
    ConstructionMethod,
    LoadCase,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.standards import TipTreatment


def profile() -> SoilProfile:
    return SoilProfile(
        layers=[
            SoilLayer(
                name="Ac", soil_type=SoilType.CLAY, thickness=10.0, n_value=4.0,
                gamma_t=16.0, gamma_sat=16.5, cohesion=40.0,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=40.0,
                gamma_t=19.0, gamma_sat=20.0,
            ),
        ],
        gwl=2.0,
    )


def test_required_pile_count():
    assert required_pile_count(9000.0, 3000.0) == 3
    assert required_pile_count(9001.0, 3000.0) == 4  # 切り上げ
    assert required_pile_count(9000.0, 0.0) is None
    assert required_pile_count(9000.0, -100.0) is None


def test_applicable_methods_cover_all_pile_types():
    assert set(APPLICABLE_METHODS) == set(PileType)
    for methods in APPLICABLE_METHODS.values():
        assert methods
    # 場所打ち杭は場所打ち工法のみ
    assert APPLICABLE_METHODS[PileType.CAST_IN_PLACE] == (
        ConstructionMethod.CAST_IN_PLACE,
    )
    # 回転杭工法は鋼管杭のみに適用
    rotary_types = [
        t for t, ms in APPLICABLE_METHODS.items() if ConstructionMethod.ROTARY in ms
    ]
    assert rotary_types == [PileType.STEEL_PIPE]


def test_compare_returns_row_per_combination():
    rows = compare(
        profile(), embedment=2.0, length=20.0,
        diameters=[0.8, 1.0], vertical_load=9000.0,
    )
    expected = sum(len(ms) for ms in APPLICABLE_METHODS.values()) * 2
    assert len(rows) == expected
    # 杭径ごとに行がある
    assert {r.diameter for r in rows} == {0.8, 1.0}


def test_compare_computes_capacity_and_pile_count():
    rows = compare(
        profile(), embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
    )
    cip = next(
        r for r in rows
        if r.pile_type == PileType.CAST_IN_PLACE and r.ok
    )
    assert cip.ru > 0
    assert cip.allowable_push > 0
    assert cip.allowable_pull > 0
    assert cip.required_piles == required_pile_count(9000.0, cip.allowable_push)


def test_larger_diameter_gives_more_capacity():
    rows = compare(
        profile(), embedment=2.0, length=20.0,
        diameters=[0.8, 1.2], vertical_load=9000.0,
        pile_types=[PileType.CAST_IN_PLACE],
    )
    small, large = sorted((r for r in rows if r.ok), key=lambda r: r.diameter)
    assert large.ru > small.ru
    assert large.required_piles <= small.required_piles


def test_h_steel_is_reported_as_unavailable():
    """H鋼杭は円形断面で表せないため、理由付きで算定不可とする。"""
    rows = compare(
        profile(), embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
        pile_types=[PileType.H_STEEL],
    )
    assert rows
    for row in rows:
        assert not row.ok
        assert "H鋼杭" in row.error
        assert row.required_piles is None


def test_unsuitable_bearing_layer_is_reported_with_reason():
    """支持層の条件を満たさない組合せは理由を添えて残す。"""
    soft = SoilProfile(
        layers=[
            SoilLayer(
                name="As", soil_type=SoilType.SAND, thickness=30.0, n_value=10.0,
                gamma_t=18.0, gamma_sat=19.0,
            )
        ],
        gwl=2.0,
    )
    rows = compare(
        soft, embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
        pile_types=[PileType.CAST_IN_PLACE],
    )
    # 場所打ち杭は N ≧ 30 が必要なので N=10 では算定できない
    assert all(not r.ok for r in rows)
    assert all("N ≧ 30" in r.error for r in rows)


def test_soil_cement_uses_column_diameter():
    """鋼管ソイルセメント杭は柱径(既定 1.4D)で支持力が決まる。"""
    rows = compare(
        profile(), embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
        pile_types=[PileType.STEEL_PIPE_SOIL_CEMENT, PileType.STEEL_PIPE],
    )
    sc = next(r for r in rows if r.pile_type == PileType.STEEL_PIPE_SOIL_CEMENT)
    driven = next(
        r for r in rows if r.method == ConstructionMethod.DRIVEN
    )
    # 柱径が杭径より大きいぶん先端面積・周長が大きい
    assert sc.tip_area > driven.tip_area
    assert sc.ru > driven.ru


def test_rotary_wing_ratio_affects_result():
    common = dict(
        profile=profile(), embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
        pile_types=[PileType.STEEL_PIPE],
    )
    r15 = next(
        r for r in compare(**common, wing_ratio=1.5)
        if r.method == ConstructionMethod.ROTARY
    )
    r20 = next(
        r for r in compare(**common, wing_ratio=2.0)
        if r.method == ConstructionMethod.ROTARY
    )
    # 羽根が大きいほど qd は下がるが面積は増える
    assert r20.bearing.qd < r15.bearing.qd
    assert r20.tip_area > r15.tip_area


def test_tip_treatment_is_only_set_for_inner_digging():
    rows = compare(
        profile(), embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
        tip_treatment=TipTreatment.CONCRETE,
    )
    for row in rows:
        if row.method == ConstructionMethod.INNER_DIGGING:
            assert row.tip_treatment == TipTreatment.CONCRETE
        else:
            assert row.tip_treatment is None


def test_load_case_affects_allowable():
    common = dict(
        profile=profile(), embedment=2.0, length=20.0,
        diameters=[1.0], vertical_load=9000.0,
        pile_types=[PileType.CAST_IN_PLACE],
    )
    normal = next(r for r in compare(**common, case=LoadCase.PERMANENT) if r.ok)
    seismic = next(r for r in compare(**common, case=LoadCase.LEVEL1_EQ) if r.ok)
    # 地震時は安全率が小さく許容値が大きい → 必要本数が減る
    assert seismic.allowable_push > normal.allowable_push
    assert seismic.required_piles <= normal.required_piles
    assert seismic.ru == pytest.approx(normal.ru)
