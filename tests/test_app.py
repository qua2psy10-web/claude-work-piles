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
    # kH・β・Kv・変位のメトリクスが出ている
    assert len(at.metric) >= 4


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
