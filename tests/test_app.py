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
    assert report.cases[0].springs.de < 1.0
