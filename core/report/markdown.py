"""設計計算書(Markdown)の生成。

式・代入値・判定(OK/NG)を明記した計算書を組み立てる。
"""
from __future__ import annotations

from core.analysis.level2 import Level2Result
from core.analysis.stability import CaseResult, StabilityReport
from core.models.project import DesignProject
from core.soil.liquefaction import LiquefactionAssessment
from core.standards import NF_SAFETY_FACTOR


def _table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(該当なし)\n"
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def build_report(
    project: DesignProject,
    report: StabilityReport | None = None,
    liquefaction: LiquefactionAssessment | None = None,
    level2: Level2Result | None = None,
) -> str:
    """設計計算書を Markdown で生成する。"""
    parts: list[str] = []
    parts.append("# 杭基礎設計計算書\n")
    parts.append(f"**工事名**: {project.name}\n")
    parts.append(
        "**適用基準**: 道路橋示方書・同解説 Ⅳ下部構造編/Ⅴ耐震設計編"
        "(平成24年3月)\n"
    )
    parts.append(_design_conditions(project))
    if liquefaction is not None:
        parts.append(_liquefaction_section(liquefaction))
    if report is not None:
        parts.append(_bearing_section(report))
        for case in report.cases:
            parts.append(_case_section(case))
        if report.rebar_detailing is not None:
            parts.append(_detailing_section(report))
        if report.negative_friction is not None:
            parts.append(_nf_section(report))
        parts.append(_summary_section(report))
    if level2 is not None:
        parts.append(_level2_section(level2))
    parts.append(
        "\n---\n\n"
        "本計算書は道示H24に基づく実装により作成したものであるが、"
        "一部の係数・許容値は原典との照合が未完了である。"
        "実務利用にあたっては `docs/VERIFICATION.md` の照合を完了すること。\n"
    )
    return "\n".join(parts)


def _design_conditions(project: DesignProject) -> str:
    s = ["## 1. 設計条件\n", "### 1.1 地層構成\n"]
    rows = []
    top = 0.0
    for layer in project.soil_profile.layers:
        bottom = top + layer.thickness
        rows.append(
            [
                layer.name,
                layer.soil_type.value,
                f"{top:.1f}〜{bottom:.1f}",
                _num(layer.thickness, 1),
                _num(layer.n_value, 1),
                _num(layer.gamma_t, 1),
                _num(layer.gamma_sat, 1),
                "" if layer.cohesion is None else _num(layer.cohesion, 1),
            ]
        )
        top = bottom
    s.append(
        _table(
            ["層名", "土質", "深度(m)", "層厚(m)", "N値", "γt", "γsat", "c(kN/m²)"],
            rows,
        )
    )
    s.append(f"\n地下水位: 地表面下 {project.soil_profile.gwl:.1f} m\n")
    s.append(f"\n変形係数 E0 の推定方法: {project.e0_method.value}\n")
    s.append(
        f"\n地盤種別: {project.seismic.ground_type.value}、"
        f"地域別補正係数 cIz={project.seismic.cz_type1}、"
        f"cIIz={project.seismic.cz_type2}\n"
    )

    if project.pile is not None:
        s.append("\n### 1.2 杭諸元\n")
        rows = [
            ["杭種", project.pile.pile_type.value],
            ["施工工法", project.pile.method.value],
            ["杭径 D", f"{project.pile.diameter:.3f} m"],
            ["杭長 L", f"{project.pile.length:.2f} m"],
        ]
        if project.pile.wall_thickness is not None:
            rows.append(["板厚 t", f"{project.pile.wall_thickness:.1f} mm"])
        if project.arrangement is not None:
            rows.append(
                [
                    "杭配置",
                    f"{project.arrangement.nx}×{project.arrangement.ny} = "
                    f"{project.arrangement.total_piles} 本",
                ]
            )
            rows.append(["杭間隔", f"{project.arrangement.spacing_x:.2f} m"])
        if project.footing is not None:
            rows.append(
                [
                    "フーチング",
                    f"{project.footing.width_x:.1f}×{project.footing.width_y:.1f}"
                    f"×{project.footing.height:.1f} m",
                ]
            )
            rows.append(["根入れ深さ", f"{project.footing.embedment:.2f} m"])
        s.append(_table(["項目", "値"], rows))

    if project.loads:
        s.append("\n### 1.3 荷重\n")
        s.append(
            _table(
                ["荷重ケース", "V (kN)", "H (kN)", "M (kN·m)"],
                [
                    [
                        load.case.value,
                        _num(load.v, 1),
                        _num(load.h, 1),
                        _num(load.m, 1),
                    ]
                    for load in project.loads
                ],
            )
        )
    return "".join(s)


def _liquefaction_section(assessment: LiquefactionAssessment) -> str:
    s = ["\n## 2. 液状化の判定(道示Ⅴ 8.2)\n"]
    s.append("FL = R / L,  R = cw・RL,  L = rd・khg・σv/σ'v\n\n")
    for label, flag in (
        ("タイプI", assessment.liquefiable_type1),
        ("タイプII", assessment.liquefiable_type2),
    ):
        s.append(
            f"- {label}: {'液状化の可能性あり(FL≦1の層あり)' if flag else '液状化の可能性なし'}\n"
        )
    rows = []
    for sl in assessment.slices:
        if not sl.is_target:
            rows.append(
                [
                    f"{sl.depth_top:.1f}〜{sl.depth_bottom:.1f}",
                    sl.layer_name,
                    "対象外",
                    sl.excluded_reason or "",
                    "", "", "", "",
                ]
            )
        else:
            rows.append(
                [
                    f"{sl.depth_top:.1f}〜{sl.depth_bottom:.1f}",
                    sl.layer_name,
                    "○",
                    "",
                    _num(sl.na, 1),
                    _num(sl.rl, 3),
                    f"{sl.fl_type1:.3f} / {_num(sl.de_type1, 3)}",
                    f"{sl.fl_type2:.3f} / {_num(sl.de_type2, 3)}",
                ]
            )
    s.append(
        "\n"
        + _table(
            [
                "深度(m)", "層名", "対象", "対象外理由", "Na", "RL",
                "FL/DE(タイプI)", "FL/DE(タイプII)",
            ],
            rows,
        )
    )
    return "".join(s)


def _detailing_section(report: StabilityReport) -> str:
    """軸方向鉄筋量の構造細目照査(道示Ⅳ 7.3)。"""
    dt = report.rebar_detailing
    s = ["\n## 5. 軸方向鉄筋量の照査(道示Ⅳ 7.3)\n"]
    s.append(
        "\n応力度の照査とは別に、**配筋そのものが足りているか・過密でないか**を"
        "確認する。趣旨はひび割れとともに耐力が減じて急激に破壊することの防止。\n\n"
        f"- 配置鉄筋量 = {_num(dt.provided_area, 0)} mm²"
        f"(うち図心より引張側 {_num(dt.tensile_area, 0)} mm²)\n"
        f"- 全断面積 = {_num(dt.gross_area, 0)} mm²、"
        f"有効断面積 b・d = {_num(dt.effective_area, 0)} mm²\n"
        f"- A′1 = Na /(0.008・σsa + σca) = {_num(dt.a1, 0)} mm²\n"
    )
    if dt.a2 is not None:
        s.append(
            f"- A′2 = Nu /(0.008・σsy + 0.85・σck) = {_num(dt.a2, 0)} mm²\n"
        )
    s.append(
        f"- A′ = {_num(dt.required_concrete_area, 0)} mm² → "
        f"必要最小鉄筋量 = 0.8% × A′ = {_num(dt.required_min_area, 0)} mm²\n"
        f"- ひび割れ曲げモーメント Mc = Zc(σbt + N/Ac) = "
        f"{_num(dt.cracking_moment, 1)} kN·m(σbt = 0.23・σck^(2/3) = "
        f"{dt.sigma_bt:.3f} N/mm²)\n"
    )
    s.append(
        "\n"
        + _table(
            ["照査項目", "値", "制限値", "単位", "比", "判定"],
            [
                [
                    c.name,
                    _num(c.value, 2),
                    ("≥ " if c.kind == "min" else "≤ ") + _num(c.limit, 2),
                    c.unit,
                    f"{c.ratio:.3f}",
                    c.judgement,
                ]
                for c in dt.checks
            ],
        )
    )
    for note in dt.notes:
        s.append(f"\n> {note}\n")
    return "".join(s)


def _bearing_section(report: StabilityReport) -> str:
    bc = report.bearing
    s = ["\n## 3. 杭の軸方向支持力(道示Ⅳ 12.4)\n"]
    s.append("Ru = qd・A + U・Σ Li・fi\n\n")
    s.append(
        f"- 先端付近の平均N値(先端±1D) = {_num(bc.n_tip, 1)}\n"
        f"- 先端支持力度 qd = {_num(bc.qd, 0)} kN/m²、"
        f"先端面積 Ap = {bc.tip_area:.4f} m²\n"
        f"- 先端支持力 qd・Ap = {_num(bc.tip_resistance, 0)} kN\n"
    )
    if bc.tip_zone_excluded:
        s.append(
            f"- 周面摩擦は杭先端から 1D 手前(深さ "
            f"{bc.skin_bottom_depth:.2f} m)までを計上(道示Ⅳ 12.4.1)\n"
        )
    s.append(_skin_table(bc))
    s.append(
        f"\n- 周面摩擦力 U・ΣLi・fi = {_num(bc.skin_resistance, 0)} kN\n"
        f"- **極限支持力 Ru = {_num(bc.ru, 0)} kN**\n"
        f"- 杭の有効重量 W = {_num(bc.w_pile, 0)} kN、"
        f"置換土の有効重量 Ws = {_num(bc.w_soil, 0)} kN\n"
    )

    seismic = report.bearing_seismic
    if seismic is not None:
        lost = seismic.skin_resistance_unreduced - seismic.skin_resistance
        s.append(
            "\n### 液状化を考慮する地震時(道示Ⅴ 8.2)\n\n"
            "液状化すると判定された層の最大周面摩擦力度に土質定数の低減係数を"
            "乗じる: **f′i = DE,i × fi**。\n"
            "先端支持力度 qd は低減しない。"
            "**この低減は耐震設計上の扱いであり、常時・暴風時には適用しない。**\n"
        )
        s.append(_skin_table(seismic))
        s.append(
            f"\n- 周面摩擦力 U・ΣLi・f′i = {_num(seismic.skin_resistance, 0)} kN"
            f"(低減前 {_num(seismic.skin_resistance_unreduced, 0)} kN、"
            f"**{_num(lost, 0)} kN の減少**)\n"
            f"- **極限支持力 Ru = {_num(seismic.ru, 0)} kN**"
            f"(低減前 {_num(bc.ru, 0)} kN)\n"
        )
        if seismic.tip_zone_liquefies:
            s.append(
                f"- ⚠ 杭先端付近(先端±1D)が液状化すると判定されている"
                f"(DE = {seismic.tip_de:.2f})。**先端支持力度 qd は低減して"
                "いない**ため、支持層の設定を確認すること\n"
            )

    s.append(
        f"\n支持形式: **{bc.support_type.value}**"
        "(押込みの安全率が支持形式により異なる)\n"
    )
    s.append("\nRa =(1/n)(Ru − Ws)+ Ws − W、Pa =(1/n)・Ruf + W\n\n")
    rows = []
    for case in report.cases:
        load_case = case.loads.case
        cap = report.bearing_for(load_case)
        rows.append(
            [
                load_case.value,
                _num(cap.ru, 0),
                _num(cap.safety_factor_push(load_case), 1),
                _num(cap.allowable_push(load_case), 0),
                _num(cap.safety_factor_pull(load_case), 1),
                _num(cap.allowable_pull(load_case), 0),
            ]
        )
    s.append(
        _table(
            ["荷重ケース", "Ru (kN)", "n(押込み)", "Ra (kN)",
             "n(引抜き)", "Pa (kN)"],
            rows,
        )
    )
    return "".join(s)


def _skin_table(bc) -> str:
    """周面摩擦力の内訳表。液状化による低減がある場合のみ DE の列を出す。"""
    reduced = bc.has_reduced_skin
    header = ["層名", "土質", "長さ(m)", "f (kN/m²)"]
    if reduced:
        header += ["DE", "f′ = f·DE (kN/m²)"]
    header += ["U·L·f (kN)"]
    return "\n" + _table(
        header,
        [
            [
                seg.layer_name,
                seg.soil_type.value,
                _num(seg.length, 2),
                _num(seg.f, 1),
            ]
            + ([_num(seg.de, 2), _num(seg.f_design, 1)] if reduced else [])
            + [_num(seg.force, 1)]
            for seg in bc.skin_segments
        ],
    )


def _case_section(case: CaseResult) -> str:
    sp = case.springs
    s = [f"\n## 4. 安定計算 — {case.loads.case.value}(道示Ⅳ 12.6)\n"]
    s.append("### 4.1 バネ定数\n")
    s.append(
        _table(
            ["項目", "値"],
            [
                ["変形係数 E0", f"{_num(sp.e0, 0)} kN/m²"],
                ["換算係数 α", f"{sp.alpha:g}"],
                ["換算載荷幅 BH", f"{sp.bh:.3f} m"],
                ["水平方向地盤反力係数 kH", f"{_num(sp.kh, 0)} kN/m³"],
                *(
                    [["液状化による低減係数 DE", f"{sp.de:.3f}"]]
                    if sp.de < 1.0
                    else []
                ),
                ["特性値 β", f"{sp.beta:.4f} 1/m"],
                ["βL", f"{sp.beta_le:.2f}" + ("(半無限長)" if sp.is_semi_infinite else "(**適用範囲外**)")],
                ["軸方向バネ Kv", f"{_num(case.kv, 0)} kN/m"],
                ["K1", f"{_num(sp.k1, 0)} kN/m"],
                ["K2 = K3", f"{_num(sp.k2, 0)} kN/rad"],
                ["K4", f"{_num(sp.k4, 0)} kN·m/rad"],
            ],
        )
    )
    s.append("\n### 4.2 フーチング変位と杭頭反力\n")
    s.append(
        f"- 水平変位 δ = {case.result.u * 1000:.2f} mm\n"
        f"- 鉛直変位 = {case.result.v * 1000:.2f} mm\n"
        f"- 回転角 θ = {case.result.theta:.3e} rad\n\n"
    )
    s.append(
        _table(
            ["杭", "x (m)", "軸力 (kN)", "水平力 (kN)", "杭頭モーメント (kN·m)"],
            [
                [
                    str(r.index),
                    f"{r.x:.3f}",
                    _num(r.axial, 1),
                    _num(r.shear, 1),
                    _num(r.moment, 1),
                ]
                for r in case.result.reactions
            ],
        )
    )
    s.append("\n### 4.3 安定照査\n")
    s.append(
        _table(
            ["照査項目", "作用値", "制限値", "比", "判定"],
            [
                [
                    c.name,
                    f"{c.demand:,.4g} {c.unit}",
                    f"{c.capacity:,.4g} {c.unit}",
                    f"{c.ratio:.3f}",
                    c.judgement,
                ]
                for c in case.checks
            ],
        )
    )

    if case.forces is not None:
        peak = case.forces.max_underground_moment
        s.append("\n### 4.4 杭体の断面力(Chang の式)\n")
        s.append(
            f"- 杭頭曲げモーメント = {_num(case.critical_pile.moment, 1)} kN·m\n"
            f"- 地中部最大曲げモーメント = {_num(peak.moment, 1)} kN·m"
            f"(深さ {peak.depth:.2f} m)\n"
        )

    stress_rows = []
    for label, stress in (
        ("杭頭", case.stress_head),
        ("地中部最大曲げ", case.stress_max),
    ):
        if stress is None:
            continue
        for c in stress.checks:
            stress_rows.append(
                [
                    f"{label}(深さ {stress.depth:.2f} m)",
                    c.name,
                    _num(c.stress, 2),
                    _num(c.allowable, 2),
                    f"{c.ratio:.3f}",
                    c.judgement,
                ]
            )
    if stress_rows:
        s.append("\n### 4.5 杭体の応力度照査(道示Ⅳ 12.10)\n")
        s.append(
            _table(
                ["位置", "照査項目", "応力度 (N/mm²)", "許容値 (N/mm²)", "比", "判定"],
                stress_rows,
            )
        )

    if case.shear is not None:
        sh = case.shear
        s.append("\n### 4.6 杭体のせん断照査(道示Ⅳ 5.1.3)\n")
        s.append(
            "\nτm = Sh /(b・d)。杭は等断面なので Sh = S(有効高の変化の項は 0)。\n"
            "円形断面の b・d は面積の等しい正方形断面に置換えて求める"
            "(道示Ⅳ 図-解4.2.2)。\n\n"
            f"- 照査断面: 深さ {sh.depth:.2f} m(せん断力最大)、"
            f"S = {_num(sh.shear, 1)} kN、M = {_num(sh.moment, 1)} kN·m、"
            f"N = {_num(sh.axial, 1)} kN\n"
            f"- 換算幅 b = {sh.width * 1000:.1f} mm、"
            f"有効高 d = {sh.effective_depth * 1000:.1f} mm\n"
            f"- 平均せん断応力度 τm = {sh.tau_m:.3f} N/mm²\n"
            f"- 補正係数: ce = {sh.ce:.3f}(有効高)、"
            f"cpt = {sh.cpt:.3f}(pt = {sh.pt:.3f}%)、"
            f"cN = {sh.cn:.3f}(軸方向圧縮力)\n"
        )
        if sh.seismic:
            s.append(
                "- 地震時のため、τa1 に割増係数 1.50 を乗じる代わりに"
                "表-5.2.1 の τc を用いている\n"
            )
        s.append(
            "\n"
            + _table(
                ["照査項目", "τm (N/mm²)", "許容値 (N/mm²)", "比", "判定"],
                [
                    [
                        c.name,
                        _num(c.stress, 3),
                        _num(c.allowable, 3),
                        f"{c.ratio:.3f}",
                        c.judgement,
                    ]
                    for c in sh.checks
                ],
            )
        )
        if sh.stirrup is not None:
            st_check = sh.stirrup
            s.append(
                "\n"
                + _table(
                    ["照査項目", "必要 (mm²/mm)", "配置 (mm²/mm)", "比", "判定"],
                    [[
                        "斜引張鉄筋量 Aw/s",
                        _num(st_check.required, 4),
                        _num(st_check.provided, 4),
                        f"{st_check.ratio:.3f}",
                        st_check.judgement,
                    ]],
                )
            )
            s.append(
                f"\nσsa = {st_check.sigma_sa:.0f} N/mm²(表-4.3.1 の「上記以外」。"
                f"軸方向鉄筋とは区分が異なる)、θ = {st_check.angle_deg:.0f}°\n"
            )
        if sh.needs_stirrup and sh.stirrup is None:
            aw = sh.required_aw_per_spacing
            s.append(
                f"\n> **斜引張鉄筋が必要**: τm = {sh.tau_m:.3f} が"
                f" τa1 = {sh.tau_a1:.3f} N/mm² を超えている。\n"
                f"> コンクリートが負担できるせん断力 Sca = τa1・b・d = "
                f"{_num(sh.concrete_shear_capacity, 1)} kN。\n"
            )
            if aw is not None:
                s.append(
                    f"> 帯鉄筋(θ = 90°、σsa = {sh.stirrup_sigma_sa:.0f} N/mm²)"
                    f"として必要量は **Aw/s = {aw:.4f} mm²/mm**"
                    f"(間隔 150mm なら Aw = {aw * 150:.0f} mm²)。\n"
                )
        else:
            s.append(
                f"\n> コンクリートのみでせん断力を負担できる"
                f"(τm = {sh.tau_m:.3f} ≤ τa1 = {sh.tau_a1:.3f} N/mm²)。"
                "ただし構造細目上の最小帯鉄筋量は別途確認すること。\n"
            )

    if case.pile_head is not None:
        s.append("\n### 4.7 杭頭結合部の照査(道示Ⅳ 12.9.3)\n")
        s.append(
            _table(
                ["照査項目", "応力度 (N/mm²)", "許容値 (N/mm²)", "比", "判定"],
                [
                    [
                        c.name,
                        _num(c.stress, 3),
                        _num(c.allowable, 3),
                        f"{c.ratio:.3f}",
                        c.judgement,
                    ]
                    for c in case.pile_head.checks
                ],
            )
        )
        edge = case.pile_head.edge_distance
        if edge is not None:
            s.append(
                f"\n最外周杭の縁端距離: 橋軸方向 {edge.edge_x:.2f} m、"
                f"直角方向 {edge.edge_y:.2f} m(標準 1.0D = {edge.required:.2f} m)\n"
            )
            if edge.needs_horizontal_punching_check:
                s.append(
                    f"\n> **注意**: 縁端距離 {edge.minimum:.2f} m が標準の 1.0D を"
                    "下回るため、フーチングの水平方向押抜きせん断の照査が"
                    "必要である(レベル2地震動まで)。本ソフトでは未実装。\n"
                )
        s.append(
            "\n> 杭頭補強鉄筋の応力度・定着長、仮想RC断面の照査は未実装である。\n"
        )
    return "".join(s)


def _nf_section(report: StabilityReport) -> str:
    nf = report.negative_friction
    s = ["\n## 6. 負の周面摩擦力の検討(道示Ⅳ 12.4.3)\n"]
    s.append(f"中立点: 地表面下 {nf.neutral_depth:.2f} m\n\n")
    s.append(
        _table(
            ["層名", "区間 (m)", "fn (kN/m²)", "NF (kN)"],
            [
                [
                    seg.layer_name,
                    f"{seg.depth_top:.2f}〜{seg.depth_bottom:.2f}",
                    _num(seg.fn, 1),
                    _num(seg.force, 1),
                ]
                for seg in nf.segments
            ],
        )
    )
    s.append(
        f"\n- 負の周面摩擦力 NF = {_num(nf.nf, 1)} kN\n"
        f"- 死荷重による軸力 = {_num(nf.dead_load, 1)} kN\n"
        f"- 最大軸力 Nmax = {_num(nf.n_max, 1)} kN\n"
        f"- 許容値 Ru/{NF_SAFETY_FACTOR} = {_num(nf.allowable, 1)} kN\n"
        f"- 比 = {nf.ratio:.3f} → **{nf.judgement}**\n"
    )
    return "".join(s)


def _summary_section(report: StabilityReport) -> str:
    s = ["\n## 7. 総括\n"]
    rows = []
    for case in report.cases:
        rows.append([case.loads.case.value, "OK" if case.all_ok else "NG"])
    if report.negative_friction is not None:
        rows.append(["負の周面摩擦力", report.negative_friction.judgement])
    s.append(_table(["項目", "判定"], rows))
    s.append(
        f"\n**総合判定: {'OK' if report.all_ok else 'NG'}**\n"
    )
    if report.notes:
        s.append("\n### 省略した照査\n")
        for note in report.notes:
            s.append(f"- {note}\n")
    return "".join(s)


def _level2_section(result: Level2Result) -> str:
    """レベル2地震時の照査(道示Ⅴ 地震時保有水平耐力法)。"""
    s = ["\n## 8. レベル2地震時の照査(道示Ⅴ(H24))\n"]
    s.append(
        "水平力を漸増させるプッシュオーバー解析により基礎の降伏点を求め、"
        "応答塑性率を照査する。\n"
    )

    if result.response is None:
        s.append(
            "\n> **設計レベル2荷重に達する前に釣合いが保てなくなった。**"
            "基礎が保有水平耐力に達していると考えられる。\n"
        )
    else:
        rows = [
            ["応答変位 δr", f"{_num(result.response.u * 1000, 2)} mm"],
            ["応答水平力 H", f"{_num(result.response.h, 1)} kN"],
        ]
        if result.yield_point is not None:
            rows.append(
                ["降伏変位 δy", f"{_num(result.yield_point.displacement * 1000, 2)} mm"]
            )
            rows.append(
                [
                    "降伏水平力 Hy",
                    f"{_num(result.yield_point.horizontal_force, 1)} kN",
                ]
            )
            rows.append(["降伏の理由", result.yield_point.reason])
        else:
            rows.append(["基礎の降伏", "設計荷重の範囲では降伏しない"])
        mu = result.response_ductility
        if mu is not None:
            rows.append(["応答塑性率 μr = δr/δy", f"{mu:.2f}"])
        s.append("\n### 8.1 応答値\n")
        s.append(_table(["項目", "値"], rows))

    if result.shear_capacity is not None:
        cap = result.shear_capacity
        s.append("\n### 8.2 杭体のせん断耐力(道示Ⅳ 5.2.3)\n")
        s.append(
            "\nPs = Sc + Ss、Sc = cc・ce・cpt・cN・τc・b・d、"
            "Ss = Aw・σsy・d・(sinθ + cosθ)/(1.15 s)\n\n"
            f"- cc = {cap.cc:g}(橋台及び基礎は 1)、ce = {cap.ce:.3f}、"
            f"cpt = {cap.cpt:.3f}(pt = {cap.pt:.3f}%)、cN = {cap.cn:.3f}\n"
            f"- τc = {cap.tau_c:.2f} N/mm²(表-5.2.1)、"
            f"b = {cap.width * 1000:,.0f} mm、d = {cap.effective_depth * 1000:,.0f} mm\n"
            f"- **Sc = {_num(cap.sc, 1)} kN**、**Ss = {_num(cap.ss, 1)} kN**、"
            f"斜引張破壊に対する耐力 **Sus = {_num(cap.total, 1)} kN**\n"
            f"- ウェブコンクリートの圧壊に対する耐力 "
            f"**Suc = τmax・bw・d = {cap.tau_max:.1f}×{cap.width * 1000:,.0f}×"
            f"{cap.effective_depth * 1000:,.0f} = {_num(cap.web_crushing_capacity, 1)}"
            " kN**(道示Ⅲ 4.3.4、表-4.3.2。RC部材なので Sp = 0)\n"
        )
        if cap.sigma_sy is not None:
            s.append(
                f"- 斜引張鉄筋の降伏点 σsy = {cap.sigma_sy:.0f} N/mm²"
                "(345 N/mm² で頭打ち)\n"
            )
        else:
            s.append("- 帯鉄筋が未入力のため Ss = 0(コンクリートのみ)\n")
        if cap.web_crushing_governs:
            s.append(
                f"\n> **斜め圧縮破壊が支配している**(Suc = "
                f"{_num(cap.web_crushing_capacity, 0)} < Sus = "
                f"{_num(cap.total, 0)} kN)。斜引張鉄筋を増やしても耐力は"
                "伸びない。断面を大きくする等の対応が必要である。\n"
            )

    if result.checks:
        s.append("\n### 8.3 照査結果\n")
        s.append(
            _table(
                ["照査項目", "応答値", "制限値", "単位", "比", "判定"],
                [
                    [
                        c.name,
                        _num(c.demand, 3),
                        _num(c.capacity, 3),
                        c.unit,
                        f"{c.ratio:.3f}",
                        c.judgement,
                    ]
                    for c in result.checks
                ],
            )
        )
        s.append(f"\n**判定: {'OK' if result.all_ok else 'NG'}**\n")

    sr = result.soil_reaction
    if sr is not None:
        s.append("\n### 8.4 水平地盤反力度と上限値 pHU の突合(診断)\n")
        s.append(
            "判定に用いた杭: "
            + ("最前列" if sr.front_row else "最前列以外(砂質地盤で pHU が 1/2)")
            + "\n"
        )
        if sr.ok:
            s.append(
                f"\n地盤反力度は上限値 pHU 以下(最大で pHU の "
                f"{sr.max_ratio * 100:.0f}%)。\n"
            )
        else:
            top, bottom = sr.exceeded_depth_range
            s.append(
                f"\n> **深さ {top:.1f}〜{bottom:.1f} m で pHU を超過"
                f"(最大 {sr.max_ratio * 100:.0f}%)。**"
                "本解析は水平地盤バネを弾性としているため、この区間の"
                "地盤抵抗を過大に評価しており、結果は非安全側である。\n"
            )

    s.append("\n### 8.5 この解析の制限事項\n")
    for note in result.notes:
        s.append(f"- {note}\n")
    return "".join(s)
