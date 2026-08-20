"""入力チェックのテスト。

主旨は「間違った答えが OK と表示されない」こと。物理的に成立しない入力で
:func:`core.analysis.stability.analyze` が黙って結果を返さないことを確かめる。
"""
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
from core.section.checks import MaterialSpec
from core.section.rc import RebarLayout, StirrupLayout
from core.validation import (
    InvalidInputError,
    Severity,
    ValidationIssue,
    raise_on_error,
    validate_inputs,
)


def sample_inputs():
    """健全な入力(警告もエラーも出ない)。"""
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


def sample_loads():
    return [FootingLoads(case=LoadCase.PERMANENT, v=9000.0, h=300.0, m=1500.0)]


def issues_for(**overrides):
    """既定の健全な入力に差分を当てて検査する。"""
    pile, arrangement, footing, profile = sample_inputs()
    args = dict(
        pile=pile,
        arrangement=arrangement,
        footing=footing,
        profile=profile,
        loads=sample_loads(),
    )
    args.update(overrides)
    return validate_inputs(**args)


def fields(issues, severity=None):
    return {
        i.field for i in issues if severity is None or i.severity is severity
    }


# --- 健全な入力 -----------------------------------------------------------


def test_sound_input_has_no_issues():
    assert issues_for() == []


def test_sound_input_with_materials_has_no_issues():
    material = MaterialSpec(
        fck=24,
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
        stirrup=StirrupLayout(diameter_mm=16.0, spacing_mm=150.0),
    )
    assert issues_for(material=material) == []


# --- 杭配置(エラー) ------------------------------------------------------


@pytest.mark.parametrize("spacing", [0.5, 1.0])
def test_overlapping_piles_are_an_error(spacing):
    """中心間隔が杭径以下なら杭どうしが重なる(1.0 m は接触)。"""
    arrangement = PileArrangement(
        nx=3, ny=3, spacing_x=spacing, spacing_y=2.5
    )
    issues = issues_for(arrangement=arrangement)
    assert fields(issues, Severity.ERROR) == {"杭中心間隔(橋軸方向)"}


def test_single_row_ignores_spacing():
    """1列なら間隔は結果に影響しないので、狭くても指摘しない。"""
    arrangement = PileArrangement(nx=1, ny=3, spacing_x=0.1, spacing_y=2.5)
    footing = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)
    assert issues_for(arrangement=arrangement, footing=footing) == []


def test_piles_outside_the_footing_are_an_error():
    """3×3@4.0 m は 8 m 幅のフーチングからはみ出す。"""
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=4.0, spacing_y=4.0)
    issues = issues_for(arrangement=arrangement)
    assert fields(issues, Severity.ERROR) == {
        "フーチング幅(橋軸方向)", "フーチング幅(直角方向)"
    }


def test_pile_exactly_flush_with_the_footing_edge_is_accepted():
    """縁端 = 半径ちょうどは「はみ出していない」ので通す。"""
    # 3×3@3.5 → 縁端 = 8/2 − 3.5 = 0.5 m = 半径。杭の外面がフーチング側面と一致
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=3.5, spacing_y=3.5)
    assert issues_for(arrangement=arrangement) == []


# --- 杭配置(警告) --------------------------------------------------------


def test_close_spacing_warns_about_group_pile_effects():
    """2.5D 未満は「群杭影響を考慮する境界」であって最小寸法ではない。

    第27回では「最小値の規定」と誤って説明していた(第28回で訂正)。
    """
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=2.0, spacing_y=2.5)
    issues = issues_for(arrangement=arrangement)
    assert len(issues) == 1
    (issue,) = issues
    assert issue.severity is Severity.WARNING
    assert issue.field == "杭中心間隔(橋軸方向)"
    assert "群杭" in issue.message
    assert "最小" not in issue.message
    # 実装した分と、していない分の両方を伝えること
    assert "μ" in issue.remedy
    assert "未実装" in issue.remedy


def test_spacing_at_the_threshold_is_clean():
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=2.5, spacing_y=2.5)
    assert issues_for(arrangement=arrangement) == []


# --- 断面・配筋 ------------------------------------------------------------


def test_cover_larger_than_the_radius_is_an_error():
    material = MaterialSpec(
        fck=24, rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=500.0)
    )
    issues = issues_for(material=material)
    assert fields(issues, Severity.ERROR) == {"かぶり"}


def test_too_many_bars_to_fit_is_an_error():
    """かぶり 125 mm・径 1.0 m の鉄筋円(周長 2356 mm)に D25 を 200 本。"""
    material = MaterialSpec(
        fck=24, rebar=RebarLayout(count=200, diameter_mm=25.0, cover_mm=125.0)
    )
    issues = issues_for(material=material)
    assert fields(issues, Severity.ERROR) == {"軸方向鉄筋"}


def test_tight_but_feasible_bar_spacing_is_a_warning():
    """あきが鉄筋径未満(中心間隔が径の 2 倍未満)なら警告にとどめる。"""
    # 周長 2356 mm / 60 本 = 39.3 mm 間隔。D25 に対し あき 14 mm
    material = MaterialSpec(
        fck=24, rebar=RebarLayout(count=60, diameter_mm=25.0, cover_mm=125.0)
    )
    issues = issues_for(material=material)
    assert fields(issues, Severity.WARNING) == {"軸方向鉄筋"}
    assert not [i for i in issues if i.is_error]


def test_rebar_geometry_is_not_checked_for_piles_without_axial_rebar():
    """PHC杭の PC鋼材は RebarLayout の円形配置ではないので対象外。"""
    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=18.0,
        concrete_thickness=90.0,
    )
    material = MaterialSpec(
        fck=30, rebar=RebarLayout(count=200, diameter_mm=25.0, cover_mm=500.0)
    )
    assert issues_for(pile=phc, material=material) == []


def _rc_pile(concrete_thickness=90.0):
    return PileSpec(
        pile_type=PileType.RC,
        method=ConstructionMethod.PREBORING,
        diameter=0.6,
        length=18.0,
        concrete_thickness=concrete_thickness,
    )


def test_rc_pile_rebar_must_stay_inside_the_concrete_wall():
    """中空のRC杭では、鉄筋が肉厚の外(中空部)に出ることをエラーにする。

    外径 600 mm・肉厚 90 mm なので中空部の半径は 210 mm。かぶり 250 mm では
    鉄筋円の半径が 50 mm となり、コンクリートが存在しない位置になる。
    """
    material = MaterialSpec(
        fck=30, rebar=RebarLayout(count=12, diameter_mm=16.0, cover_mm=250.0)
    )
    issues = issues_for(pile=_rc_pile(), material=material)
    assert fields(issues, Severity.ERROR) == {"軸方向鉄筋"}
    assert "中空部" in issues[0].message


def test_rc_pile_with_rebar_in_the_wall_passes():
    material = MaterialSpec(
        fck=30, rebar=RebarLayout(count=12, diameter_mm=16.0, cover_mm=40.0)
    )
    assert issues_for(pile=_rc_pile(), material=material) == []


def test_wide_stirrup_spacing_is_a_warning():
    material = MaterialSpec(
        fck=24,
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
        # 有効高の目安 0.75 × 1000 = 750 mm を超える
        stirrup=StirrupLayout(diameter_mm=16.0, spacing_mm=800.0),
    )
    issues = issues_for(material=material)
    assert fields(issues, Severity.WARNING) == {"帯鉄筋の間隔"}
    assert "7.10" in issues[0].remedy


# --- 地盤 ------------------------------------------------------------------


def test_pile_tip_below_the_soil_model_is_an_error():
    pile, *_ = sample_inputs()
    deep = pile.model_copy(update={"length": 40.0})  # 2 + 40 > 30
    issues = issues_for(pile=deep)
    assert fields(issues, Severity.ERROR) == {"杭長"}


def test_footing_below_the_soil_model_is_an_error():
    footing = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=35.0)
    issues = issues_for(footing=footing)
    assert fields(issues, Severity.ERROR) == {"根入れ深さ"}


def test_pile_tip_exactly_at_the_bottom_is_accepted():
    pile, *_ = sample_inputs()
    exact = pile.model_copy(update={"length": 28.0})  # 2 + 28 = 30
    assert issues_for(pile=exact) == []


def test_water_table_below_the_soil_model_is_a_warning():
    profile = sample_inputs()[3]
    dry = profile.model_copy(update={"gwl": 50.0})
    issues = issues_for(profile=dry)
    assert fields(issues, Severity.WARNING) == {"地下水位"}


# --- 荷重 ------------------------------------------------------------------


def test_no_load_case_is_an_error():
    issues = issues_for(loads=[])
    assert fields(issues, Severity.ERROR) == {"荷重"}


def test_duplicate_load_cases_are_a_warning():
    loads = sample_loads() * 2
    issues = issues_for(loads=loads)
    assert fields(issues, Severity.WARNING) == {"荷重ケース"}


def test_non_positive_vertical_load_is_a_warning():
    loads = [FootingLoads(case=LoadCase.PERMANENT, v=0.0, h=300.0, m=1500.0)]
    issues = issues_for(loads=loads)
    assert "鉛直力(常時)" in fields(issues, Severity.WARNING)


# --- 送出の作法 ------------------------------------------------------------


def test_raise_on_error_returns_warnings_when_there_is_no_error():
    warning = ValidationIssue(Severity.WARNING, "そこ", "ゆるい")
    assert raise_on_error([warning]) == [warning]


def test_raise_on_error_raises_and_carries_every_error():
    errors = [
        ValidationIssue(Severity.ERROR, "あ", "だめ", "こうして"),
        ValidationIssue(Severity.ERROR, "い", "これも"),
    ]
    with pytest.raises(InvalidInputError) as exc:
        raise_on_error([*errors, ValidationIssue(Severity.WARNING, "う", "ゆるい")])
    assert exc.value.issues == errors
    text = str(exc.value)
    # 場所・内容・対処がすべて読めること
    assert "あ" in text and "だめ" in text and "こうして" in text
    assert "い" in text and "これも" in text
    # 警告は例外には含めない
    assert "ゆるい" not in text


def test_issue_str_reads_as_one_line():
    issue = ValidationIssue(Severity.ERROR, "杭長", "長すぎます", "短くして")
    assert str(issue) == "[エラー] 杭長: 長すぎます — 短くして"
    assert str(ValidationIssue(Severity.WARNING, "杭長", "長い")) == "[警告] 杭長: 長い"


# --- 安定計算との結合 ------------------------------------------------------


def test_analyze_refuses_overlapping_piles():
    """これまでは総合判定「OK」を返していた入力。"""
    pile, _, footing, profile = sample_inputs()
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=0.5, spacing_y=0.5)
    with pytest.raises(InvalidInputError) as exc:
        analyze(pile, arrangement, footing, profile, sample_loads())
    assert all(i.is_error for i in exc.value.issues)
    assert "重なり" in str(exc.value)


def test_analyze_refuses_piles_outside_the_footing():
    pile, _, footing, profile = sample_inputs()
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=4.0, spacing_y=4.0)
    with pytest.raises(InvalidInputError):
        analyze(pile, arrangement, footing, profile, sample_loads())


def test_analyze_continues_on_warnings_and_records_them():
    pile, _, footing, profile = sample_inputs()
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=2.0, spacing_y=2.0)
    report = analyze(pile, arrangement, footing, profile, sample_loads())

    assert len(report.warnings) == 2  # 橋軸方向・直角方向
    assert all(not w.is_error for w in report.warnings)
    # 計算書にも残ること
    assert any("⚠" in note and "杭中心間隔" in note for note in report.notes)
    # 計算そのものは通常どおり行われる
    assert report.cases[0].checks


def test_analyze_has_no_warnings_for_sound_input():
    pile, arrangement, footing, profile = sample_inputs()
    report = analyze(pile, arrangement, footing, profile, sample_loads())
    assert report.warnings == []
    assert report.notes == []


def test_analyze_validates_the_section_when_materials_are_given():
    pile, arrangement, footing, profile = sample_inputs()
    material = MaterialSpec(
        fck=24, rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=600.0)
    )
    with pytest.raises(InvalidInputError) as exc:
        analyze(
            pile, arrangement, footing, profile, sample_loads(), material=material
        )
    assert {i.field for i in exc.value.issues} == {"かぶり"}


# --- 群杭の補正係数 μ ------------------------------------------------------


def wide_footing():
    return Footing(width_x=12.0, width_y=12.0, height=1.5, embedment=2.0)


def test_group_correction_lowers_kh_and_raises_displacement():
    """L < 2.5D では kH に μ が効き、変位が大きく出る(安全側)。"""
    pile, _, _, profile = sample_inputs()
    footing = wide_footing()
    reports = {}
    for spacing in (2.5, 2.0):
        arrangement = PileArrangement(
            nx=3, ny=3, spacing_x=spacing, spacing_y=spacing
        )
        reports[spacing] = analyze(
            pile, arrangement, footing, profile, sample_loads()
        )

    base, tight = reports[2.5], reports[2.0]
    assert base.cases[0].springs.group_factor == 1.0
    assert tight.cases[0].springs.group_factor == pytest.approx(0.90)

    # μ は**収束計算の内側**で kH に乗るので、kH は μ 倍ちょうどにはならない。
    # kH = C・D^(-3/8)・β^(3/8)・μ と β = (kH・D/4EI)^(1/4) を連立すると
    # kH ∝ μ^(32/29) となる。この指数を固定して、μ を収束後に外から
    # 掛けるだけの実装に戻っていないことを担保する。
    ratio = tight.cases[0].springs.kh / base.cases[0].springs.kh
    assert ratio == pytest.approx(0.90 ** (32 / 29), rel=1e-6)
    assert ratio < 0.90  # 単純に μ 倍するより下がる

    # kH が下がるので変位は増える
    assert tight.cases[0].result.u > base.cases[0].result.u
    # β も下がる(地中部最大曲げモーメントの位置が深くなる向き)
    assert tight.cases[0].springs.beta < base.cases[0].springs.beta


def test_group_correction_uses_the_narrower_direction():
    """方向で間隔が違えば、狭いほうで μ を決める(安全側)。"""
    pile, _, _, profile = sample_inputs()
    footing = wide_footing()
    mixed = analyze(
        pile,
        PileArrangement(nx=3, ny=3, spacing_x=2.0, spacing_y=4.0),
        footing, profile, sample_loads(),
    )
    both_narrow = analyze(
        pile,
        PileArrangement(nx=3, ny=3, spacing_x=2.0, spacing_y=2.0),
        footing, profile, sample_loads(),
    )
    assert mixed.cases[0].springs.group_factor == pytest.approx(
        both_narrow.cases[0].springs.group_factor
    )


def test_group_correction_is_explained_including_what_is_missing():
    """μ を乗じたことと、支持力側が未実装であることの両方を注記する。"""
    pile, _, _, profile = sample_inputs()
    arrangement = PileArrangement(nx=3, ny=3, spacing_x=2.0, spacing_y=2.0)
    report = analyze(pile, arrangement, wide_footing(), profile, sample_loads())
    note = next(n for n in report.notes if "補正係数" in n)
    assert "μ" in note
    assert "仮想ケーソン" in note and "未実装" in note
    # H24 の条文そのものではないことも伝える
    assert "令和7年改訂版" in note


def test_no_group_note_at_or_above_the_threshold():
    pile, arrangement, footing, profile = sample_inputs()
    report = analyze(pile, arrangement, footing, profile, sample_loads())
    assert report.cases[0].springs.group_factor == 1.0
    assert report.notes == []
