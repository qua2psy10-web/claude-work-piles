"""GUI のスモークテスト(Streamlit AppTest でボタン操作まで実行する)。"""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.main import DEFAULT_LAYERS, DEFAULT_LOADS, df_from_loads, layers_from_df, loads_from_df
from core.models import LoadCase

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


def run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_exception():
    at = run_app()
    assert not at.exception


def test_liquefaction_button_produces_result():
    at = run_app()
    button = next(b for b in at.button if "液状化判定" in b.label)
    button.click().run()
    assert not at.exception
    # デフォルト地層(As1 N=10, FC=5%)は液状化するため警告が出る
    assert any("液状化の可能性あり" in e.value for e in at.error)


def test_stability_button_produces_result():
    at = run_app()
    button = next(b for b in at.button if "安定計算" in b.label)
    button.click().run()
    assert not at.exception
    assert not any("計算エラー" in e.value for e in at.error)
    # 照査結果(OK/NG)がいずれか表示される
    messages = [e.value for e in at.error] + [s.value for s in at.success]
    assert any("OK" in m or "NG" in m for m in messages)
    # 支持力・杭頭反力の表が出ている
    assert len(at.dataframe) >= 2
    # 支持力3件 + 荷重ケースごとに kH・β・Kv・変位・杭頭M・地中部最大M
    assert len(at.metric) >= 12
    labels = [m.label for m in at.metric]
    assert "地中部最大モーメント" in labels
    # 応力度照査・杭頭結合部の表が描画されている
    texts = [m.value for m in at.markdown]
    assert any("応力度照査" in t for t in texts)
    assert any("杭頭結合部" in t for t in texts)


def test_comparison_button_produces_table():
    at = run_app()
    button = next(b for b in at.button if "比較表" in b.label)
    button.click().run()
    assert not at.exception
    # 最小本数の組合せが提示される
    assert any("必要本数が最小" in s.value for s in at.success)
    # H鋼杭など算定できない組合せも行として残る
    assert len(at.dataframe) >= 1


def test_report_download_buttons_exist():
    at = run_app()
    labels = [b.label for b in at.download_button]
    assert any("プロジェクト保存" in label for label in labels)
    assert any("Markdown" in label for label in labels)
    assert any("Excel" in label for label in labels)


def test_report_picks_up_analysis_results():
    """安定計算を実行すると結果がセッションに保持され、計算書に反映される。"""
    at = run_app()
    assert "report" not in at.session_state

    next(b for b in at.button if "安定計算" in b.label).click().run()
    assert not at.exception
    report = at.session_state["report"]
    assert report is not None
    assert len(report.cases) == 2

    from core.report.markdown import build_report

    text = build_report(at.session_state["project"], report)
    assert "杭の軸方向支持力" in text
    assert "総合判定" in text


def test_default_layers_roundtrip():
    layers = layers_from_df(DEFAULT_LAYERS)
    assert len(layers) == len(DEFAULT_LAYERS)
    assert layers[0].name == "B"
    assert layers[2].ip == pytest.approx(30.0)
    # D50 未入力の粘性土は None のまま
    assert layers[2].d50 is None


def test_loads_roundtrip():
    loads = loads_from_df(DEFAULT_LOADS)
    assert [load.case for load in loads] == [LoadCase.PERMANENT, LoadCase.LEVEL1_EQ]
    restored = loads_from_df(df_from_loads(loads))
    assert restored == loads


def test_loads_skips_blank_rows():
    df = pd.concat(
        [DEFAULT_LOADS, pd.DataFrame([{"荷重ケース": "常時", "V(kN)": None}])],
        ignore_index=True,
    )
    assert len(loads_from_df(df)) == 2


def test_young_modulus_can_be_entered_for_precast_piles():
    """σck = 80 の Ec は表にないため、直接入力できること(既定は表引き)。"""
    at = run_app()
    field = next(
        n for n in at.number_input if "ヤング係数" in n.label
    )
    assert field.value == 0.0  # 0 = 未入力(σck から表引き)
    assert at.session_state["project"].pile.concrete_young is None

    field.set_value(40000.0).run()
    assert not at.exception
    # 入力単位は N/mm²、内部は kN/m²(1 N/mm² = 1000 kN/m²)
    assert at.session_state["project"].pile.concrete_young == pytest.approx(4.0e7)


def test_level2_button_produces_result():
    at = run_app()
    button = next(b for b in at.button if "レベル2照査" in b.label)
    button.click().run()
    assert not at.exception
    assert not any("計算エラー" in e.value for e in at.error)
    labels = [m.label for m in at.metric]
    assert "応答変位 δr" in labels
    assert "降伏変位 δy" in labels
    # 制限事項が必ず提示される
    texts = [m.value for m in at.markdown]
    assert any("非線形" in t or "pHU" in t for t in texts)


def test_level2_derives_allowable_ductility_and_checks_rotation():
    """μa は下部構造の種別から自動設定し、回転角は 0.02 rad で照査すること。"""
    at = run_app()
    button = next(b for b in at.button if "レベル2照査" in b.label)
    button.click().run()
    assert not at.exception
    result = at.session_state["level2"]
    # 既定は橋脚 → μa = 4
    assert result.allowable_ductility == 4.0
    assert result.allowable_rotation == 0.02
    assert any("回転角" in c.name for c in result.checks)
    # 自動設定した旨が注記される
    assert any("μa = 4" in n for n in result.notes)


def test_liquefaction_reduction_is_opt_in_and_needs_the_assessment():
    """DE の反映はチェックボックスで明示的に選ぶ(既定では反映しない)。"""
    at = run_app()
    checkbox = next(
        c for c in at.checkbox if "液状化による土質定数の低減" in c.label
    )
    assert checkbox.value is False
    # 液状化判定を実行していないうちは選択できない
    assert checkbox.disabled

    # 判定を実行すると選択できるようになる
    next(b for b in at.button if "液状化判定" in b.label).click().run()
    at2 = at
    checkbox = next(
        c for c in at2.checkbox if "液状化による土質定数の低減" in c.label
    )
    assert not checkbox.disabled


def test_stability_reflects_the_reduction_when_enabled():
    at = run_app()
    next(b for b in at.button if "液状化判定" in b.label).click().run()
    next(
        c for c in at.checkbox if "液状化による土質定数の低減" in c.label
    ).set_value(True).run()
    next(b for b in at.button if "安定計算を実行" in b.label).click().run()
    assert not at.exception

    report = at.session_state["report"]
    assert any("液状化による土質定数の低減" in n for n in report.notes)

    # DE は耐震設計上の扱い。常時には効かず、地震時だけに効く
    by_case = {c.loads.case: c for c in report.cases}
    assert by_case[LoadCase.PERMANENT].springs.de == 1.0
    assert by_case[LoadCase.LEVEL1_EQ].springs.de < 1.0

    # 支持力側にも効く(kH だけでなく周面摩擦力度も低減される)
    assert not report.bearing.has_reduced_skin  # 常時用は低減なし
    seismic = report.bearing_seismic
    assert seismic is not None and seismic.has_reduced_skin
    assert seismic.skin_resistance < report.bearing.skin_resistance
    assert report.bearing_for(LoadCase.PERMANENT) is report.bearing
    assert report.bearing_for(LoadCase.LEVEL1_EQ) is seismic
    assert any("周面摩擦力度の低減内訳" in n for n in report.notes)


def test_stability_without_the_reduction_keeps_full_skin_friction():
    """既定(低減なし)では周面摩擦力が満額であること。"""
    at = run_app()
    next(b for b in at.button if "安定計算を実行" in b.label).click().run()
    report = at.session_state["report"]
    assert not report.bearing.has_reduced_skin
    assert report.bearing_seismic is None
    assert all(s.de == 1.0 for s in report.bearing.skin_segments)


def test_impossible_arrangement_is_refused_instead_of_reported_as_ok():
    """杭が重なる配置は、結果を出さずに止める。

    以前は杭径 1.0 m を 0.5 m 間隔にしても総合判定「OK」を返していた。
    """
    at = run_app()
    next(n for n in at.number_input if n.label == "杭間隔 (m)").set_value(0.5).run()
    next(b for b in at.button if "安定計算を実行" in b.label).click().run()

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("入力が物理的に成立しません" in e for e in errors)
    assert any("杭中心間隔" in e and "重なり" in e for e in errors)
    # 結果は一切表示しない
    assert not any("全ケース OK" in s.value for s in at.success)
    assert "report" not in at.session_state


def test_questionable_arrangement_is_computed_with_a_warning():
    """2.5D 未満は計算を続け、注記として画面に出す。"""
    at = run_app()
    next(n for n in at.number_input if n.label == "杭間隔 (m)").set_value(2.0).run()
    next(b for b in at.button if "安定計算を実行" in b.label).click().run()

    assert not at.exception
    assert at.session_state["report"].warnings
    assert any("杭中心間隔" in w.value for w in at.warning)


def test_region_selector_sets_the_correction_factors():
    """地域区分を選ぶと cIz・cIIz が表から入る。A1 は 1.20(1.0 上限ではない)。"""
    at = run_app()
    selector = next(s for s in at.selectbox if "地域区分" in s.label)
    assert selector.value == "A2"
    seismic = at.session_state["project"].seismic
    assert (seismic.cz_type1, seismic.cz_type2) == (1.00, 1.00)

    selector.set_value("A1").run()
    assert not at.exception
    seismic = at.session_state["project"].seismic
    assert seismic.cz_type1 == 1.20
    assert seismic.cz_type2 == 1.00


def test_region_selector_falls_back_to_manual_entry():
    at = run_app()
    next(s for s in at.selectbox if "地域区分" in s.label).set_value(
        "(直接入力)"
    ).run()
    assert not at.exception
    assert any("cIz" in n.label for n in at.number_input)


# --- M-φ 骨格曲線の入力欄(第54回) -------------------------------------------


def test_parse_moment_curvature_reads_break_points():
    from app.main import parse_moment_curvature

    mc = parse_moment_curvature(
        "0.0012680, 257.3\n0.0055255, 456.4\n0.0184293, 543.9"
    )
    assert mc is not None
    assert len(mc.points) == 3
    assert mc.yield_moment == 456.4
    assert mc.ultimate_moment == 543.9


def test_parse_moment_curvature_treats_blank_as_elastic():
    from app.main import parse_moment_curvature

    assert parse_moment_curvature("") is None
    assert parse_moment_curvature("   \n\n  ") is None
    assert parse_moment_curvature("# コメントだけ\n") is None


def test_parse_moment_curvature_accepts_tabs_and_extra_whitespace():
    from app.main import parse_moment_curvature

    mc = parse_moment_curvature("  0.001\t100  \n 0.002 ,  200 \n")
    assert mc is not None
    assert mc.points == ((0.001, 100.0), (0.002, 200.0))


def test_parse_moment_curvature_reports_the_offending_line():
    import pytest as _pytest
    from app.main import parse_moment_curvature

    with _pytest.raises(ValueError, match="2 行目"):
        parse_moment_curvature("0.001, 100\n0.002")
    with _pytest.raises(ValueError, match="数値として読めません"):
        parse_moment_curvature("0.001, たくさん")


# --- 底版照査タブ(第58回) ---------------------------------------------------


def test_footing_button_produces_result():
    """底版照査タブの既定値(Kui_8 の計算例)で例外なく結果が出ること。"""
    at = run_app()
    button = next(b for b in at.button if "底版照査を実行" in b.label)
    button.click().run()
    assert not at.exception
    # 曲げ・せん断・最小鉄筋量の3ブロックが描画される
    markdown = " ".join(m.value for m in at.markdown)
    assert "曲げ応力度照査" in markdown
    assert "せん断応力度照査" in markdown
    assert "最小鉄筋量" in markdown


def test_footing_tab_reports_input_errors_without_crashing():
    """有効高を部材高より大きくするなど不整合な入力でもエラー表示で止まること。"""
    at = run_app()
    # 鉄筋量を 0 にすると singly_reinforced_stress が ValueError を送出する
    rebar = next(n for n in at.number_input if n.label.startswith("引張主鉄筋量"))
    rebar.set_value(0.0).run()
    button = next(b for b in at.button if "底版照査を実行" in b.label)
    button.click().run()
    assert not at.exception
    assert any("計算エラー" in e.value for e in at.error)
