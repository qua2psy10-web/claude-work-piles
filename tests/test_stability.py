"""安定計算の統合テスト。"""
import pytest

from core.analysis.stability import analyze
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


def sample_inputs():
    profile = SoilProfile(
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
    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.0,
        length=18.0,
    )
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5)
    footing = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)
    return pile, arrangement, footing, profile


def test_analyze_runs_all_cases():
    pile, arrangement, footing, profile = sample_inputs()
    loads = [
        FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=300.0, m=1500.0),
        FootingLoads(case=LoadCase.LEVEL1_EQ, v=9000.0, h=2000.0, m=8000.0),
    ]
    report = analyze(pile, arrangement, footing, profile, loads)

    assert len(report.cases) == 2
    for case in report.cases:
        # 釣合いが取れていること
        assert sum(r.axial for r in case.result.reactions) == pytest.approx(
            case.loads.v, rel=1e-8
        )
        assert sum(r.shear for r in case.result.reactions) == pytest.approx(
            case.loads.h, rel=1e-8
        )
        assert {c.name for c in case.checks} >= {"押込み支持力", "水平変位"}

    # 地震時のほうが水平力が大きいので変位も大きい
    assert report.cases[1].result.u > report.cases[0].result.u


def test_uplift_check_added_only_when_tension_occurs():
    pile, arrangement, footing, profile = sample_inputs()
    # 大きなモーメントで引抜きを発生させる
    loads = [FootingLoads(case=LoadCase.LEVEL1_EQ, v=1000.0, h=500.0, m=40000.0)]
    report = analyze(pile, arrangement, footing, profile, loads)
    case = report.cases[0]
    assert case.result.min_axial < 0
    assert any(c.name == "引抜き抵抗力" for c in case.checks)

    # 鉛直荷重のみなら引抜きは生じない
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=0.0, m=0.0)]
    report = analyze(pile, arrangement, footing, profile, loads)
    assert not any(c.name == "引抜き抵抗力" for c in report.cases[0].checks)


def test_check_judgement():
    pile, arrangement, footing, profile = sample_inputs()
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=300.0, m=1500.0)]
    report = analyze(pile, arrangement, footing, profile, loads)
    push = next(c for c in report.cases[0].checks if c.name == "押込み支持力")
    assert push.ratio == pytest.approx(abs(push.demand) / push.capacity)
    assert push.judgement == ("OK" if push.ratio <= 1.0 else "NG")


def test_excessive_load_is_ng():
    pile, arrangement, footing, profile = sample_inputs()
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=200000.0, h=0.0, m=0.0)]
    report = analyze(pile, arrangement, footing, profile, loads)
    assert not report.all_ok
    push = next(c for c in report.cases[0].checks if c.name == "押込み支持力")
    assert push.judgement == "NG"
