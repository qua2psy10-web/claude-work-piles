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


def test_unimplemented_stress_check_is_skipped_with_note():
    """応力度照査が未実装の杭種でも、支持力・変位の照査は行い注記を残す。"""
    from core.section.checks import MaterialSpec
    from core.section.rc import RebarLayout

    _, arrangement, footing, profile = sample_inputs()
    rc = PileSpec(
        pile_type=PileType.RC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=18.0,
        concrete_thickness=90.0,
    )
    material = MaterialSpec(
        fck=30, rebar=RebarLayout(count=12, diameter_mm=25.0, cover_mm=60.0)
    )
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=3000.0, h=200.0, m=800.0)]
    report = analyze(rc, arrangement, footing, profile, loads, fck=30, material=material)

    case = report.cases[0]
    # 支持力・変位の照査は行われている
    assert {c.name for c in case.checks} >= {"押込み支持力", "水平変位"}
    # 杭体の応力度照査は省略され、理由が注記される
    assert case.stress_head is None
    assert case.stress_max is None
    assert report.notes
    assert any("RC杭" in note for note in report.notes)
    # 杭頭結合部の照査は杭種によらず行われる
    assert case.pile_head is not None


def test_phc_stress_check_runs_with_young_modulus_note():
    """PHC杭は応力度照査を行うが、σck=80 の Ec が未照合である旨を注記する。"""
    from core.section.checks import MaterialSpec

    _, arrangement, footing, profile = sample_inputs()
    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=18.0,
        concrete_thickness=90.0,
    )
    material = MaterialSpec(fck=30, effective_prestress=8.0)
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=3000.0, h=200.0, m=800.0)]
    report = analyze(phc, arrangement, footing, profile, loads, fck=30, material=material)

    case = report.cases[0]
    assert case.stress_head is not None
    assert case.stress_max is not None
    assert {c.name for c in case.stress_head.checks} >= {
        "軸圧縮応力度", "曲げ圧縮応力度"
    }
    assert any("表-3.3.3" in note for note in report.notes)


def test_phc_with_a_given_young_modulus_changes_the_stiffness_and_the_note():
    """Ec を直接入力すると EI が変わり、注記も「入力値を使った」旨に変わる。"""
    from core.section.checks import MaterialSpec

    _, arrangement, footing, profile = sample_inputs()
    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=18.0,
        concrete_thickness=90.0,
    )
    material = MaterialSpec(fck=30, effective_prestress=8.0)
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=3000.0, h=200.0, m=800.0)]
    kwargs = dict(fck=30, material=material)
    base = analyze(phc, arrangement, footing, profile, loads, **kwargs)
    stiff = analyze(
        phc.model_copy(update={"concrete_young": 4.0e7}),
        arrangement, footing, profile, loads, **kwargs,
    )

    # 剛性が上がると杭頭変位は小さくなる
    assert stiff.cases[0].result.u < base.cases[0].result.u
    # 注記は「範囲外」の警告から「入力値を用いた」旨に変わる
    assert any("表-3.3.3" in n for n in base.notes)
    assert not any("表-3.3.3" in n for n in stiff.notes)
    assert any("入力されたヤング係数" in n for n in stiff.notes)
    # 許容応力度は既製杭の規定値なので Ec には依存しない
    assert [c.allowable for c in stiff.cases[0].stress_head.checks] == [
        c.allowable for c in base.cases[0].stress_head.checks
    ]


def test_implemented_pile_type_has_no_notes():
    pile, arrangement, footing, profile = sample_inputs()
    from core.section.checks import MaterialSpec
    from core.section.rc import RebarLayout

    material = MaterialSpec(
        fck=24, rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)
    )
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=300.0, m=1500.0)]
    report = analyze(pile, arrangement, footing, profile, loads, material=material)
    assert report.notes == []
    assert report.cases[0].stress_head is not None


def test_excessive_load_is_ng():
    pile, arrangement, footing, profile = sample_inputs()
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=200000.0, h=0.0, m=0.0)]
    report = analyze(pile, arrangement, footing, profile, loads)
    assert not report.all_ok
    push = next(c for c in report.cases[0].checks if c.name == "押込み支持力")
    assert push.judgement == "NG"
