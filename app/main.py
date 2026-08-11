"""杭基礎の設計(H24年道示版) Streamlit GUI。

起動: streamlit run app/main.py
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from core.models import (
    ConstructionMethod,
    DesignProject,
    Footing,
    PileArrangement,
    PileSpec,
    PileType,
    SeismicConditions,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.soil.liquefaction import assess_liquefaction
from core.standards import GroundType

LAYER_COLUMNS = {
    "層名": "name",
    "土質": "soil_type",
    "層厚(m)": "thickness",
    "N値": "n_value",
    "γt(kN/m3)": "gamma_t",
    "γsat(kN/m3)": "gamma_sat",
    "FC(%)": "fc",
    "IP": "ip",
    "D50(mm)": "d50",
    "D10(mm)": "d10",
    "沖積層": "is_alluvial",
}

DEFAULT_LAYERS = pd.DataFrame(
    [
        {"層名": "B", "土質": "砂質土", "層厚(m)": 2.0, "N値": 5.0,
         "γt(kN/m3)": 17.0, "γsat(kN/m3)": 18.0, "FC(%)": 15.0, "IP": None,
         "D50(mm)": 0.35, "D10(mm)": 0.05, "沖積層": True},
        {"層名": "As1", "土質": "砂質土", "層厚(m)": 6.0, "N値": 10.0,
         "γt(kN/m3)": 18.0, "γsat(kN/m3)": 19.0, "FC(%)": 5.0, "IP": None,
         "D50(mm)": 0.30, "D10(mm)": 0.08, "沖積層": True},
        {"層名": "Ac1", "土質": "粘性土", "層厚(m)": 5.0, "N値": 4.0,
         "γt(kN/m3)": 16.0, "γsat(kN/m3)": 16.5, "FC(%)": 80.0, "IP": 30.0,
         "D50(mm)": None, "D10(mm)": None, "沖積層": True},
        {"層名": "Ds1", "土質": "砂質土", "層厚(m)": 12.0, "N値": 35.0,
         "γt(kN/m3)": 19.0, "γsat(kN/m3)": 20.0, "FC(%)": 8.0, "IP": None,
         "D50(mm)": 0.50, "D10(mm)": 0.10, "沖積層": False},
    ]
)


def _opt(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def layers_from_df(df: pd.DataFrame) -> list[SoilLayer]:
    layers = []
    for i, row in df.iterrows():
        if pd.isna(row.get("層厚(m)")):
            continue
        layers.append(
            SoilLayer(
                name=str(row.get("層名") or f"層{i + 1}"),
                soil_type=SoilType(row["土質"]),
                thickness=float(row["層厚(m)"]),
                n_value=float(row["N値"]),
                gamma_t=float(row["γt(kN/m3)"]),
                gamma_sat=float(row["γsat(kN/m3)"]),
                fc=_opt(row.get("FC(%)")),
                ip=_opt(row.get("IP")),
                d50=_opt(row.get("D50(mm)")),
                d10=_opt(row.get("D10(mm)")),
                is_alluvial=bool(row.get("沖積層", True)),
            )
        )
    return layers


def df_from_layers(layers: list[SoilLayer]) -> pd.DataFrame:
    rows = []
    for layer in layers:
        rows.append(
            {
                "層名": layer.name,
                "土質": layer.soil_type.value,
                "層厚(m)": layer.thickness,
                "N値": layer.n_value,
                "γt(kN/m3)": layer.gamma_t,
                "γsat(kN/m3)": layer.gamma_sat,
                "FC(%)": layer.fc,
                "IP": layer.ip,
                "D50(mm)": layer.d50,
                "D10(mm)": layer.d10,
                "沖積層": layer.is_alluvial,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="杭基礎の設計(H24年道示版)", layout="wide")
    st.title("杭基礎の設計(H24年道示版)")
    st.caption(
        "道路橋示方書・同解説 Ⅳ下部構造編/Ⅴ耐震設計編(平成24年3月)準拠 — "
        "フェーズ0: 地盤入力と液状化判定"
    )

    if "load_nonce" not in st.session_state:
        st.session_state.load_nonce = 0
    loaded: DesignProject | None = st.session_state.get("loaded_project")

    with st.sidebar:
        st.header("設計条件")
        uploaded = st.file_uploader("プロジェクト読込(JSON)", type=["json"])
        if uploaded is not None:
            file_key = (uploaded.name, uploaded.size)
            if st.session_state.get("loaded_file_key") != file_key:
                try:
                    project = DesignProject.model_validate_json(uploaded.getvalue())
                except Exception as exc:  # noqa: BLE001
                    st.error(f"読込エラー: {exc}")
                else:
                    st.session_state.loaded_project = project
                    st.session_state.loaded_file_key = file_key
                    st.session_state.load_nonce += 1
                    st.rerun()

        nonce = st.session_state.load_nonce
        name = st.text_input(
            "工事名", value=(loaded.name if loaded else "無題"), key=f"name_{nonce}"
        )
        ground_type = st.selectbox(
            "地盤種別(道示Ⅴ 4.5)",
            [g.value for g in GroundType],
            index=(
                [g for g in GroundType].index(loaded.seismic.ground_type)
                if loaded
                else 1
            ),
            key=f"gt_{nonce}",
        )
        cz1 = st.number_input(
            "地域別補正係数 cIz(タイプI)",
            0.1, 1.0,
            value=(loaded.seismic.cz_type1 if loaded else 1.0),
            step=0.05, key=f"cz1_{nonce}",
        )
        cz2 = st.number_input(
            "地域別補正係数 cIIz(タイプII)",
            0.1, 1.0,
            value=(loaded.seismic.cz_type2 if loaded else 1.0),
            step=0.05, key=f"cz2_{nonce}",
        )
        gwl = st.number_input(
            "地下水位の深さ (m)",
            0.0, 99.0,
            value=(loaded.soil_profile.gwl if loaded else 1.5),
            step=0.1, key=f"gwl_{nonce}",
        )

    tab_soil, tab_pile, tab_liq = st.tabs(["地盤", "杭・フーチング", "液状化判定"])

    with tab_soil:
        st.subheader("地層データ")
        st.caption("行の追加・削除は表の左端から。深度は上から順に積み上げ。")
        base_df = df_from_layers(loaded.soil_profile.layers) if loaded else DEFAULT_LAYERS
        layers_df = st.data_editor(
            base_df,
            num_rows="dynamic",
            key=f"layers_{nonce}",
            column_config={
                "土質": st.column_config.SelectboxColumn(
                    "土質", options=[t.value for t in SoilType], required=True
                ),
                "沖積層": st.column_config.CheckboxColumn("沖積層", default=True),
            },
            width="stretch",
        )

    with tab_pile:
        st.subheader("杭・フーチング(フェーズ1の安定計算で使用)")
        col1, col2, col3 = st.columns(3)
        with col1:
            pile_type = st.selectbox(
                "杭種", [t.value for t in PileType],
                index=(
                    [t for t in PileType].index(loaded.pile.pile_type)
                    if loaded and loaded.pile else 2
                ),
                key=f"pt_{nonce}",
            )
            method = st.selectbox(
                "施工工法", [m.value for m in ConstructionMethod],
                index=(
                    [m for m in ConstructionMethod].index(loaded.pile.method)
                    if loaded and loaded.pile else 2
                ),
                key=f"pm_{nonce}",
            )
        with col2:
            diameter = st.number_input(
                "杭径 (m)", 0.1, 5.0,
                value=(loaded.pile.diameter if loaded and loaded.pile else 1.0),
                step=0.1, key=f"pd_{nonce}",
            )
            length = st.number_input(
                "杭長 (m)", 1.0, 100.0,
                value=(loaded.pile.length if loaded and loaded.pile else 20.0),
                step=0.5, key=f"pl_{nonce}",
            )
        with col3:
            nx = st.number_input(
                "橋軸方向 列数", 1, 20,
                value=(loaded.arrangement.nx if loaded and loaded.arrangement else 2),
                key=f"nx_{nonce}",
            )
            ny = st.number_input(
                "直角方向 列数", 1, 20,
                value=(loaded.arrangement.ny if loaded and loaded.arrangement else 3),
                key=f"ny_{nonce}",
            )
            spacing = st.number_input(
                "杭間隔 (m)", 0.5, 20.0,
                value=(
                    loaded.arrangement.spacing_x
                    if loaded and loaded.arrangement else 2.5
                ),
                step=0.25, key=f"sp_{nonce}",
            )
        st.info(f"杭本数: {int(nx) * int(ny)} 本(安定計算はフェーズ1で実装予定)")

    with tab_liq:
        st.subheader("液状化判定(道示Ⅴ(H24) 8.2 — レベル2地震動)")
        pitch = st.select_slider(
            "判定ピッチ (m)", options=[0.5, 1.0, 2.0], value=1.0, key=f"pitch_{nonce}"
        )
        try:
            layers = layers_from_df(layers_df)
            profile = SoilProfile(layers=layers, gwl=gwl)
        except Exception as exc:  # noqa: BLE001
            st.error(f"地層データにエラーがあります: {exc}")
            profile = None

        if profile is not None and st.button("液状化判定を実行", type="primary"):
            assessment = assess_liquefaction(
                profile, GroundType(ground_type), cz1, cz2, pitch=pitch
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if assessment.liquefiable_type1:
                    st.error("タイプI: 液状化の可能性あり(FL≦1 の層あり)")
                else:
                    st.success("タイプI: 液状化の可能性なし")
            with col_b:
                if assessment.liquefiable_type2:
                    st.error("タイプII: 液状化の可能性あり(FL≦1 の層あり)")
                else:
                    st.success("タイプII: 液状化の可能性なし")

            rows = []
            for s in assessment.slices:
                rows.append(
                    {
                        "深度 (m)": f"{s.depth_top:.1f}〜{s.depth_bottom:.1f}",
                        "層名": s.layer_name,
                        "土質": s.soil_type.value,
                        "対象": "○" if s.is_target else "—",
                        "対象外理由": s.excluded_reason or "",
                        "σv": _round(s.sigma_v),
                        "σ'v": _round(s.sigma_v_eff),
                        "Na": _round(s.na),
                        "RL": _round(s.rl, 4),
                        "FL(タイプI)": _round(s.fl_type1, 3),
                        "DE(タイプI)": _round(s.de_type1, 3),
                        "FL(タイプII)": _round(s.fl_type2, 3),
                        "DE(タイプII)": _round(s.de_type2, 3),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", height=480)
            st.caption(
                "σv, σ'v: kN/m²。DE は FL≦1 の層のみ低減(道示Ⅴ 表-8.2.1)。"
            )

    # プロジェクト保存
    with st.sidebar:
        st.divider()
        try:
            project = DesignProject(
                name=name,
                soil_profile=SoilProfile(layers=layers_from_df(layers_df), gwl=gwl),
                seismic=SeismicConditions(
                    ground_type=GroundType(ground_type), cz_type1=cz1, cz_type2=cz2
                ),
                pile=PileSpec(
                    pile_type=PileType(pile_type),
                    method=ConstructionMethod(method),
                    diameter=diameter,
                    length=length,
                ),
                arrangement=PileArrangement(
                    nx=int(nx), ny=int(ny), spacing_x=spacing, spacing_y=spacing
                ),
            )
            st.download_button(
                "プロジェクト保存(JSON)",
                data=project.model_dump_json(indent=2),
                file_name=f"{name or 'project'}.json",
                mime="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"保存不可(入力エラー): {exc}")


def _round(value: float | None, ndigits: int = 1) -> float | str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return round(value, ndigits)


main()
