"""軸方向鉄筋量の最小・最大の照査(道示Ⅳ(H24) 7.3)のテスト。

期待値は原典(スキャン)の式から手計算で導いている。
docs/VERIFICATION.md 第25回を参照。
"""
import math

import pytest

from core.models import ConstructionMethod, PileSpec, PileType
from core.section.detailing import (
    bending_tensile_strength,
    check_rebar_detailing,
    cracking_moment,
    required_concrete_area,
)
from core.section.rc import RebarLayout
from core.standards import (
    MAX_TENSILE_REBAR_RATIO,
    MAX_TOTAL_REBAR_RATIO,
    MIN_REBAR_RATIO_AXIAL,
    SIGMA_CAG_CONCRETE,
    SIGMA_CA_REBAR,
)

CIP = PileSpec(
    pile_type=PileType.CAST_IN_PLACE,
    method=ConstructionMethod.CAST_IN_PLACE,
    diameter=1.0,
    length=20.0,
)
REBAR = RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)


# --- 個別の式 ---------------------------------------------------------------


def test_bending_tensile_strength():
    """σbt = 0.23・σck^(2/3)(式(解 7.3.1))。"""
    assert bending_tensile_strength(24) == pytest.approx(0.23 * 24 ** (2 / 3))
    assert bending_tensile_strength(24) == pytest.approx(1.9137, abs=1e-4)
    # 強度が上がれば引張強度も上がる
    assert bending_tensile_strength(30) > bending_tensile_strength(24)


def test_cracking_moment_hand_calculation():
    """Mc = Zc(σbt + N/Ac)。D=1.0m、σck=24、N=1500kN。

      Zc = πD³/32 = 9.8175e7 mm³、Ac = πD²/4 = 785,398 mm²
      σbt = 1.9137、N/Ac = 1.5e6/785398 = 1.9099
      Mc = 9.8175e7 × 3.8236 = 3.754e8 N·mm = 375.4 kN·m
    """
    zc = math.pi * 1000.0**3 / 32.0
    ac = math.pi * 1000.0**2 / 4.0
    expected = zc * (bending_tensile_strength(24) + 1.5e6 / ac) / 1.0e6

    assert cracking_moment(1.0, 24, 1500.0) == pytest.approx(expected)
    assert cracking_moment(1.0, 24, 1500.0) == pytest.approx(375.4, abs=0.1)
    # 軸圧縮が大きいほどひび割れにくい
    assert cracking_moment(1.0, 24, 3000.0) > cracking_moment(1.0, 24, 1500.0)
    # 引張軸力では小さくなる
    assert cracking_moment(1.0, 24, -500.0) < cracking_moment(1.0, 24, 0.0)


def test_required_concrete_area_hand_calculation():
    """A′1 = Na/(0.008σsa + σca)、A′2 = Nu/(0.008σsy + 0.85σck)。

      SD345・σck=24 なら σsa = 200、σca = 6.5
      A′1 = 1.5e6 /(0.008×200 + 6.5) = 1.5e6/8.1 = 185,185 mm²
      A′2 = 3.0e6 /(0.008×345 + 0.85×24) = 3.0e6/23.16 = 129,534 mm²
    """
    a1, a2 = required_concrete_area(24, "SD345", 1500.0, 3000.0)
    assert a1 == pytest.approx(1.5e6 / (0.008 * 200.0 + 6.5))
    assert a1 == pytest.approx(185185.2, abs=1.0)
    assert a2 == pytest.approx(3.0e6 / (0.008 * 345.0 + 0.85 * 24.0))
    assert a2 == pytest.approx(129533.7, abs=1.0)
    # 定数どおりであること
    assert SIGMA_CA_REBAR["SD345"] == 200.0
    assert SIGMA_CAG_CONCRETE[24] == 6.5


def test_required_concrete_area_skips_a2_without_nu():
    a1, a2 = required_concrete_area(24, "SD345", 1500.0)
    assert a2 is None
    assert a1 > 0


def test_required_concrete_area_rejects_unknown_material():
    with pytest.raises(ValueError, match="許容圧縮応力度"):
        required_concrete_area(24, "SD500", 1500.0)
    with pytest.raises(ValueError, match="許容軸圧縮応力度"):
        required_concrete_area(50, "SD345", 1500.0)


# --- 照査 -------------------------------------------------------------------


def _result(axial=1500.0, moment=800.0, nu=3000.0, rebar=REBAR):
    return check_rebar_detailing(
        CIP, rebar, 24, "SD345",
        axial_allowable=axial, moment=moment, axial_ultimate=nu,
    )


def test_minimum_rebar_uses_the_larger_of_a1_and_a2():
    r = _result()
    assert r.required_concrete_area == pytest.approx(max(r.a1, r.a2))
    assert r.required_min_area == pytest.approx(
        MIN_REBAR_RATIO_AXIAL * r.required_concrete_area
    )
    # 24-D25 は十分に足りている。D25 の**公称断面積**は 506.7 mm²
    # (π・25²/4 = 490.9 mm² ではない。JIS G 3112。第34回に修正)
    assert r.provided_area == pytest.approx(24 * 506.7)
    check = next(c for c in r.checks if "最小鉄筋量" in c.name)
    assert check.kind == "min" and check.ok


def test_sparse_reinforcement_fails_the_minimum():
    """配筋が薄すぎると最小鉄筋量で NG になること。"""
    sparse = RebarLayout(count=6, diameter_mm=13.0, cover_mm=125.0)
    r = _result(axial=8000.0, rebar=sparse)
    check = next(c for c in r.checks if "最小鉄筋量" in c.name)
    assert not check.ok
    assert not r.all_ok
    # 比は「必要 / 配置」で 1 を超える
    assert check.ratio == pytest.approx(r.required_min_area / r.provided_area)


def test_maximum_rebar_ratios():
    """引張鉄筋は有効断面積の 2%、全鉄筋は全断面積の 6% 以下。"""
    r = _result()
    tensile = next(c for c in r.checks if "引張鉄筋/有効断面積" in c.name)
    total = next(c for c in r.checks if "全鉄筋/全断面積" in c.name)
    assert tensile.limit == MAX_TENSILE_REBAR_RATIO * 100.0
    assert total.limit == MAX_TOTAL_REBAR_RATIO * 100.0
    assert tensile.value == pytest.approx(100.0 * r.tensile_area / r.effective_area)
    assert total.value == pytest.approx(100.0 * r.provided_area / r.gross_area)
    assert tensile.ok and total.ok


def test_overcrowded_reinforcement_fails_the_maximum():
    dense = RebarLayout(count=60, diameter_mm=51.0, cover_mm=125.0)
    r = _result(rebar=dense)
    total = next(c for c in r.checks if "全鉄筋/全断面積" in c.name)
    assert not total.ok
    assert not r.all_ok


def test_crack_control_rebar_checks():
    """表面 1m あたり 500mm² 以上、中心間隔 300mm 以下。"""
    r = _result()
    area = next(c for c in r.checks if "表面1mあたり" in c.name)
    spacing = next(c for c in r.checks if "中心間隔" in c.name)
    # 中心間隔は鉄筋円の円周 ÷ 本数
    assert spacing.value == pytest.approx(
        2 * math.pi * 375.0 / 24
    )
    assert spacing.value == pytest.approx(98.17, abs=0.01)
    # 表面 1m あたりは部材表面 πD で割る
    assert area.value == pytest.approx(r.provided_area / (math.pi * 1.0))
    assert area.ok and spacing.ok


def test_wide_spacing_fails_the_crack_control_check():
    sparse = RebarLayout(count=6, diameter_mm=25.0, cover_mm=125.0)
    r = _result(rebar=sparse)
    spacing = next(c for c in r.checks if "中心間隔" in c.name)
    assert spacing.value > 300.0
    assert not spacing.ok


# --- ひび割れ曲げモーメントのただし書き --------------------------------------


def test_crack_moment_exemption_applies_when_1_7m_is_below_mc():
    """1.7M ≤ Mc なら曲げの最小鉄筋量の規定によらなくてよい。"""
    small = _result(moment=100.0)  # 1.7×100 = 170 ≤ Mc = 375
    assert small.crack_moment_exempt
    assert any("ただし書きにより適用しない" in n for n in small.notes)

    large = _result(moment=800.0)  # 1.7×800 = 1360 > Mc
    assert not large.crack_moment_exempt
    assert any("別途必要" in n for n in large.notes)
    assert any("最大抵抗曲げモーメントの算定は未実装" in n for n in large.notes)


def test_missing_ultimate_axial_is_noted():
    r = _result(nu=None)
    assert r.a2 is None
    assert any("A′2" in n and "未入力" in n for n in r.notes)


def test_non_cast_in_place_is_rejected():
    steel = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=20.0,
        wall_thickness=12.0,
    )
    with pytest.raises(ValueError, match="未実装"):
        check_rebar_detailing(steel, REBAR, 24, "SD345", 1500.0, 800.0)


# --- 安定計算・計算書への反映 ------------------------------------------------


def _profile():
    from core.models import SoilLayer, SoilProfile, SoilType

    return SoilProfile(
        layers=[
            SoilLayer(name="As", soil_type=SoilType.SAND, thickness=10.0,
                      n_value=15.0, gamma_t=18.0, gamma_sat=19.0),
            SoilLayer(name="Ds", soil_type=SoilType.SAND, thickness=25.0,
                      n_value=45.0, gamma_t=19.0, gamma_sat=20.0),
        ],
        gwl=2.0,
    )


def _report(rebar=REBAR, level2_axial=None):
    from core.analysis.stability import analyze
    from core.models import Footing, FootingLoads, LoadCase, PileArrangement
    from core.section.checks import MaterialSpec

    return analyze(
        CIP,
        PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5),
        Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0),
        _profile(),
        [
            FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=600.0, m=1500.0),
            FootingLoads(case=LoadCase.LEVEL1_EQ, v=9000.0, h=3000.0, m=8000.0),
        ],
        fck=24,
        material=MaterialSpec(fck=24, rebar=rebar),
        level2_axial=level2_axial,
    )


def test_stability_runs_the_detailing_check():
    """安定計算が鉄筋量の照査を行い、Na に全ケースの最大軸力を使うこと。"""
    report = _report()
    dt = report.rebar_detailing
    assert dt is not None
    expected_na = max(c.result.max_axial for c in report.cases)
    a1, _ = required_concrete_area(24, "SD345", expected_na)
    assert dt.a1 == pytest.approx(a1)
    assert dt.a2 is None  # level2_axial 未指定


def test_stability_uses_the_level2_axial_when_given():
    report = _report(level2_axial=6000.0)
    dt = report.rebar_detailing
    assert dt.a2 is not None
    assert not any("A′2" in n and "未入力" in n for n in dt.notes)


def test_detailing_failure_makes_the_whole_report_ng():
    """鉄筋量が NG なら総合判定も NG になること。"""
    sparse = RebarLayout(count=6, diameter_mm=13.0, cover_mm=125.0)
    report = _report(rebar=sparse)
    assert not report.rebar_detailing.all_ok
    assert not report.all_ok


def test_steel_pipe_pile_has_no_detailing_result():
    from core.analysis.stability import analyze
    from core.models import Footing, FootingLoads, LoadCase, PileArrangement
    from core.section.checks import MaterialSpec

    steel = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.0,
        length=20.0,
        wall_thickness=12.0,
    )
    report = analyze(
        steel,
        PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5),
        Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0),
        _profile(),
        [FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=600.0, m=1500.0)],
        fck=24,
        material=MaterialSpec(fck=24, rebar=REBAR),
    )
    assert report.rebar_detailing is None


def test_markdown_report_includes_the_detailing_section():
    from core.models import DesignProject, Footing, FootingLoads, LoadCase, PileArrangement
    from core.report.markdown import build_report

    report = _report()
    project = DesignProject(
        pile=CIP,
        arrangement=PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5),
        footing=Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0),
        soil_profile=_profile(),
        loads=[FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=600.0, m=1500.0)],
    )
    text = build_report(project, report)
    assert "軸方向鉄筋量の照査" in text
    assert "A′1" in text
    assert "ひび割れ曲げモーメント Mc" in text
    # 章番号が重複していないこと。「4. 安定計算」だけは荷重ケースごとに
    # 繰り返すのが設計なので、それ以外で重複がないことを確かめる
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    numbers = [h.split(".")[0].removeprefix("## ") for h in headings]
    others = [n for n in numbers if n != "4"]
    assert len(others) == len(set(others)), headings
    assert numbers.count("5") == 1
