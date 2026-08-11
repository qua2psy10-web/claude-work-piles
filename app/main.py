"""杭基礎の設計(H24年道示版) Streamlit GUI。

起動: streamlit run app/main.py
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from core.analysis.stability import StabilityReport, analyze
from core.models import (
    ConstructionMethod,
    DesignProject,
    Footing,
    FootingLoads,
    LoadCase,
    PileArrangement,
    PileSpec,
    PileType,
    SeismicConditions,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.report.excel import build_workbook
from core.report.markdown import build_report
from core.section.checks import MaterialSpec
from core.section.rc import RebarLayout
from core.soil.liquefaction import assess_liquefaction
from core.standards import (
    EC_CONCRETE,
    SIGMA_A_STEEL,
    SIGMA_SA_REBAR,
    GroundType,
)

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


DEFAULT_LOADS = pd.DataFrame(
    [
        {"荷重ケース": "常時", "V(kN)": 9000.0, "H(kN)": 300.0, "M(kN·m)": 1500.0},
        {"荷重ケース": "レベル1地震時", "V(kN)": 9000.0, "H(kN)": 2000.0,
         "M(kN·m)": 8000.0},
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


def loads_from_df(df: pd.DataFrame) -> list[FootingLoads]:
    loads = []
    for _, row in df.iterrows():
        if pd.isna(row.get("V(kN)")):
            continue
        loads.append(
            FootingLoads(
                case=LoadCase(row["荷重ケース"]),
                v=float(row["V(kN)"]),
                h=float(row.get("H(kN)") or 0.0),
                m=float(row.get("M(kN·m)") or 0.0),
            )
        )
    return loads


def df_from_loads(loads: list[FootingLoads]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "荷重ケース": load.case.value,
                "V(kN)": load.v,
                "H(kN)": load.h,
                "M(kN·m)": load.m,
            }
            for load in loads
        ]
    )


def _render_stability(report: StabilityReport) -> None:
    """安定計算結果を画面に表示する。"""
    bc = report.bearing
    if report.all_ok:
        st.success("全ケース OK")
    else:
        st.error("NG の照査項目があります")

    with st.expander("軸方向支持力(道示Ⅳ 12.4)", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("極限支持力 Ru", f"{bc.ru:,.0f} kN")
        c2.metric("先端支持力 qd·A", f"{bc.tip_resistance:,.0f} kN")
        c3.metric("周面摩擦力 U·ΣLf", f"{bc.skin_resistance:,.0f} kN")
        st.caption(
            f"先端付近の平均N値(±1D) = {bc.n_tip:.1f}、"
            f"qd = {bc.qd:,.0f} kN/m², Ap = {bc.tip_area:.4f} m², "
            f"W = {bc.w_pile:,.0f} kN, Ws = {bc.w_soil:,.0f} kN"
        )
        if bc.tip_zone_excluded:
            st.caption(
                f"周面摩擦は杭先端から 1D 手前(深さ {bc.skin_bottom_depth:.2f} m)"
                "までを計上(道示Ⅳ 12.4.1 の重複計上排除)"
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "層名": s.layer_name,
                        "土質": s.soil_type.value,
                        "長さ (m)": round(s.length, 2),
                        "f (kN/m²)": round(s.f, 1),
                        "U·L·f (kN)": round(s.force, 1),
                    }
                    for s in bc.skin_segments
                ]
            ),
            width="stretch",
        )

    for case in report.cases:
        label = case.loads.case.value
        header = f"{label} — {'OK' if case.all_ok else 'NG'}"
        with st.expander(header, expanded=True):
            sp = case.springs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("kH", f"{sp.kh:,.0f} kN/m³")
            c2.metric("β", f"{sp.beta:.4f} 1/m")
            c3.metric("Kv", f"{case.kv:,.0f} kN/m")
            c4.metric("水平変位 δ", f"{case.result.u * 1000:.2f} mm")
            st.caption(
                f"E0 = {sp.e0:,.0f} kN/m², BH = {sp.bh:.3f} m, "
                f"βL = {sp.beta_le:.2f}"
                + ("(半無限長)" if sp.is_semi_infinite else "(**有限長: 要注意**)")
                + f", 収束 {sp.iterations} 回"
            )
            if not sp.is_semi_infinite:
                st.warning(
                    "βL < 3 のため半無限長の杭の式は適用範囲外です。"
                    "有限長の杭の K1〜K4 は未実装のため、結果は参考値です。"
                )

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "照査項目": c.name,
                            "作用値": f"{c.demand:,.4g} {c.unit}",
                            "制限値": f"{c.capacity:,.4g} {c.unit}",
                            "比": round(c.ratio, 3),
                            "判定": c.judgement,
                        }
                        for c in case.checks
                    ]
                ),
                width="stretch",
            )

            st.markdown("**杭頭反力**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "杭": r.index,
                            "x (m)": round(r.x, 3),
                            "軸力 (kN)": round(r.axial, 1),
                            "水平力 (kN)": round(r.shear, 1),
                            "杭頭モーメント (kN·m)": round(r.moment, 1),
                        }
                        for r in case.result.reactions
                    ]
                ),
                width="stretch",
            )

            _render_section_forces(case)
            _render_stress_checks(case)

    if report.negative_friction is not None:
        nf = report.negative_friction
        with st.expander(
            f"負の周面摩擦力の検討 — {nf.judgement}", expanded=True
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("中立点深さ", f"{nf.neutral_depth:.2f} m")
            c2.metric("NF", f"{nf.nf:,.0f} kN")
            c3.metric("最大軸力 Nmax", f"{nf.n_max:,.0f} kN")
            st.caption(
                f"死荷重軸力 {nf.dead_load:,.0f} kN + NF {nf.nf:,.0f} kN = "
                f"{nf.n_max:,.0f} kN ≦ Ru/1.2 = {nf.allowable:,.0f} kN "
                f"(比 {nf.ratio:.3f}) → {nf.judgement}"
            )
            if nf.segments:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "層名": s.layer_name,
                                "区間 (m)": f"{s.depth_top:.2f}〜{s.depth_bottom:.2f}",
                                "fn (kN/m²)": round(s.fn, 1),
                                "NF (kN)": round(s.force, 1),
                            }
                            for s in nf.segments
                        ]
                    ),
                    width="stretch",
                )


def _render_section_forces(case) -> None:
    """杭体の断面力分布を図表で表示する。"""
    if case.forces is None:
        return
    peak = case.forces.max_underground_moment
    st.markdown("**杭体の断面力(Chang の式)**")
    c1, c2 = st.columns(2)
    c1.metric("杭頭モーメント", f"{case.critical_pile.moment:,.1f} kN·m")
    c2.metric(
        "地中部最大モーメント",
        f"{peak.moment:,.1f} kN·m",
        help=f"深さ {peak.depth:.2f} m",
    )
    chart = pd.DataFrame(
        {
            "深さ (m)": [p.depth for p in case.forces.points],
            "曲げモーメント (kN·m)": [p.moment for p in case.forces.points],
            "せん断力 (kN)": [p.shear for p in case.forces.points],
        }
    ).set_index("深さ (m)")
    st.line_chart(chart)


def _render_stress_checks(case) -> None:
    """杭体・杭頭結合部の応力度照査結果を表示する。"""
    rows = []
    for label, stress in (
        ("杭頭", case.stress_head),
        ("地中部最大曲げ", case.stress_max),
    ):
        if stress is None:
            continue
        for c in stress.checks:
            rows.append(
                {
                    "位置": f"{label}(深さ {stress.depth:.2f} m)",
                    "照査項目": c.name,
                    "応力度 (N/mm²)": round(c.stress, 2),
                    "許容値 (N/mm²)": round(c.allowable, 2),
                    "比": round(c.ratio, 3),
                    "判定": c.judgement,
                }
            )
    if rows:
        st.markdown("**杭体の応力度照査(道示Ⅳ 12.10)**")
        st.dataframe(pd.DataFrame(rows), width="stretch")
        if case.stress_head is not None and case.stress_head.rc_detail is not None:
            d = case.stress_head.rc_detail
            st.caption(
                "杭頭断面: "
                + (
                    "全断面圧縮"
                    if d.fully_compressed
                    else f"中立軸深さ x = {d.compression_depth:.3f} m"
                )
                + f"、σc = {d.sigma_c:.2f} N/mm²、"
                f"σs(引張) = {d.sigma_s_tension:.1f} N/mm²"
            )

    if case.pile_head is not None:
        st.markdown("**杭頭結合部の照査(道示Ⅳ 12.9)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "照査項目": c.name,
                        "応力度 (N/mm²)": round(c.stress, 3),
                        "許容値 (N/mm²)": round(c.allowable, 3),
                        "比": round(c.ratio, 3),
                        "判定": c.judgement,
                    }
                    for c in case.pile_head.checks
                ]
            ),
            width="stretch",
        )


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

    tab_soil, tab_pile, tab_load, tab_liq, tab_stab = st.tabs(
        ["地盤", "杭・フーチング", "荷重", "液状化判定", "安定計算"]
    )

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
        st.subheader("杭・フーチング")
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
            thickness = st.number_input(
                "板厚 (mm)", 0.0, 100.0,
                value=(
                    loaded.pile.wall_thickness
                    if loaded and loaded.pile and loaded.pile.wall_thickness
                    else 12.0
                ),
                step=1.0, key=f"pw_{nonce}",
                help="鋼管系杭のみ使用。腐食代1mmを控除して断面計算する",
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
        st.info(f"杭本数: {int(nx) * int(ny)} 本")

        st.markdown("**フーチング**")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            fw_x = st.number_input(
                "橋軸方向幅 (m)", 0.5, 50.0,
                value=(loaded.footing.width_x if loaded and loaded.footing else 8.0),
                step=0.5, key=f"fwx_{nonce}",
            )
        with fcol2:
            fw_y = st.number_input(
                "直角方向幅 (m)", 0.5, 50.0,
                value=(loaded.footing.width_y if loaded and loaded.footing else 8.0),
                step=0.5, key=f"fwy_{nonce}",
            )
        with fcol3:
            fh = st.number_input(
                "厚さ (m)", 0.3, 10.0,
                value=(loaded.footing.height if loaded and loaded.footing else 1.5),
                step=0.1, key=f"fh_{nonce}",
            )
        with fcol4:
            embedment = st.number_input(
                "根入れ深さ (m)", 0.0, 30.0,
                value=(loaded.footing.embedment if loaded and loaded.footing else 2.0),
                step=0.5, key=f"fe_{nonce}",
                help="地表面からフーチング下面(杭頭)までの深さ",
            )
        st.markdown("**材料・配筋**")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            fck = st.selectbox(
                "σck (N/mm²)",
                sorted(EC_CONCRETE), index=1, key=f"fck_{nonce}",
                help="場所打ち杭のコンクリート設計基準強度",
            )
        with mcol2:
            rebar_grade = st.selectbox(
                "鉄筋材質", list(SIGMA_SA_REBAR), index=1, key=f"rg_{nonce}"
            )
            steel_grade = st.selectbox(
                "鋼材材質", list(SIGMA_A_STEEL), index=0, key=f"sg_{nonce}"
            )
        with mcol3:
            rebar_count = st.number_input(
                "軸方向鉄筋 本数", 4, 200, value=24, key=f"rn_{nonce}"
            )
            rebar_dia = st.number_input(
                "鉄筋径 (mm)", 10.0, 60.0, value=25.0, step=1.0, key=f"rd2_{nonce}"
            )
        with mcol4:
            rebar_cover = st.number_input(
                "かぶり (mm)", 30.0, 500.0, value=125.0, step=5.0, key=f"rc_{nonce}",
                help="断面縁から鉄筋中心までの距離",
            )
            use_nf = st.checkbox(
                "負の周面摩擦力を検討", value=False, key=f"nf_{nonce}",
                help="圧密沈下層(N値10以下の粘性土)を自動判定する",
            )

    with tab_load:
        st.subheader("荷重(フーチング底面中心に作用する値)")
        st.caption(
            "V: 鉛直力(下向き正)、H: 水平力(橋軸方向)、M: モーメント。"
            "不要な荷重ケースは行を削除する。"
        )
        base_loads = (
            df_from_loads(loaded.loads)
            if loaded and loaded.loads
            else DEFAULT_LOADS
        )
        loads_df = st.data_editor(
            base_loads,
            num_rows="dynamic",
            key=f"loads_{nonce}",
            column_config={
                "荷重ケース": st.column_config.SelectboxColumn(
                    "荷重ケース", options=[c.value for c in LoadCase], required=True
                ),
            },
            width="stretch",
        )

    try:
        profile = SoilProfile(layers=layers_from_df(layers_df), gwl=gwl)
        profile_error = None
    except Exception as exc:  # noqa: BLE001
        profile = None
        profile_error = str(exc)

    with tab_liq:
        st.subheader("液状化判定(道示Ⅴ(H24) 8.2 — レベル2地震動)")
        pitch = st.select_slider(
            "判定ピッチ (m)", options=[0.5, 1.0, 2.0], value=1.0, key=f"pitch_{nonce}"
        )
        if profile is None:
            st.error(f"地層データにエラーがあります: {profile_error}")

        if profile is not None and st.button("液状化判定を実行", type="primary"):
            assessment = assess_liquefaction(
                profile, GroundType(ground_type), cz1, cz2, pitch=pitch
            )
            st.session_state.liquefaction = assessment
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

    pile_spec = PileSpec(
        pile_type=PileType(pile_type),
        method=ConstructionMethod(method),
        diameter=diameter,
        length=length,
        wall_thickness=thickness if thickness > 0 else None,
    )
    arrangement_spec = PileArrangement(
        nx=int(nx), ny=int(ny), spacing_x=spacing, spacing_y=spacing
    )
    footing_spec = Footing(
        width_x=fw_x, width_y=fw_y, height=fh, embedment=embedment
    )

    material_spec = MaterialSpec(
        fck=int(fck),
        rebar_grade=rebar_grade,
        steel_grade=steel_grade,
        rebar=RebarLayout(
            count=int(rebar_count),
            diameter_mm=rebar_dia,
            cover_mm=rebar_cover,
        ),
    )

    with tab_stab:
        st.subheader("安定計算・断面照査(道示Ⅳ(H24) 12.6、12.9、12.10)")
        st.caption(
            "直杭・杭頭剛結、フーチング剛体を仮定。杭種は場所打ち杭・鋼管杭に対応。"
        )
        if profile is None:
            st.error(f"地層データにエラーがあります: {profile_error}")
        elif st.button("安定計算を実行", type="primary"):
            try:
                report = analyze(
                    pile_spec,
                    arrangement_spec,
                    footing_spec,
                    profile,
                    loads_from_df(loads_df),
                    fck=int(fck),
                    material=material_spec,
                    check_negative_friction=use_nf,
                )
            except (ValueError, NotImplementedError, RuntimeError) as exc:
                st.error(f"計算エラー: {exc}")
            else:
                st.session_state.report = report
                _render_stability(report)
        elif st.session_state.get("report") is not None:
            _render_stability(st.session_state.report)

    # プロジェクト保存・計算書出力
    with st.sidebar:
        st.divider()
        project = None
        try:
            project = DesignProject(
                name=name,
                soil_profile=SoilProfile(layers=layers_from_df(layers_df), gwl=gwl),
                seismic=SeismicConditions(
                    ground_type=GroundType(ground_type), cz_type1=cz1, cz_type2=cz2
                ),
                pile=pile_spec,
                arrangement=arrangement_spec,
                footing=footing_spec,
                loads=loads_from_df(loads_df),
            )
            st.session_state.project = project
            st.download_button(
                "プロジェクト保存(JSON)",
                data=project.model_dump_json(indent=2),
                file_name=f"{name or 'project'}.json",
                mime="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"保存不可(入力エラー): {exc}")

        if project is not None:
            st.divider()
            st.caption("計算書")
            report = st.session_state.get("report")
            liq = st.session_state.get("liquefaction")
            if report is None:
                st.caption("安定計算を実行すると照査結果が計算書に含まれます。")
            try:
                st.download_button(
                    "計算書(Markdown)",
                    data=build_report(project, report, liq),
                    file_name=f"{name or 'report'}.md",
                    mime="text/markdown",
                )
                st.download_button(
                    "計算書(Excel)",
                    data=build_workbook(project, report, liq),
                    file_name=f"{name or 'report'}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"計算書を生成できません: {exc}")


def _round(value: float | None, ndigits: int = 1) -> float | str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return round(value, ndigits)


main()
