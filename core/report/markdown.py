"""設計計算書(Markdown)の生成。

式・代入値・判定(OK/NG)を明記した計算書を組み立てる。
"""
from __future__ import annotations

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
        if report.negative_friction is not None:
            parts.append(_nf_section(report))
        parts.append(_summary_section(report))
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
    s.append(
        "\n"
        + _table(
            ["層名", "土質", "長さ(m)", "f (kN/m²)", "U·L·f (kN)"],
            [
                [
                    seg.layer_name,
                    seg.soil_type.value,
                    _num(seg.length, 2),
                    _num(seg.f, 1),
                    _num(seg.force, 1),
                ]
                for seg in bc.skin_segments
            ],
        )
    )
    s.append(
        f"\n- 周面摩擦力 U・ΣLi・fi = {_num(bc.skin_resistance, 0)} kN\n"
        f"- **極限支持力 Ru = {_num(bc.ru, 0)} kN**\n"
        f"- 杭の有効重量 W = {_num(bc.w_pile, 0)} kN、"
        f"置換土の有効重量 Ws = {_num(bc.w_soil, 0)} kN\n"
    )
    s.append(
        f"\n支持形式: **{bc.support_type.value}**"
        "(押込みの安全率が支持形式により異なる)\n"
    )
    s.append("\nRa =(1/n)(Ru − Ws)+ Ws − W、Pa =(1/n)・Ruf + W\n\n")
    rows = []
    for case in report.cases:
        rows.append(
            [
                case.loads.case.value,
                _num(bc.safety_factor_push(case.loads.case), 1),
                _num(bc.allowable_push(case.loads.case), 0),
                _num(bc.safety_factor_pull(case.loads.case), 1),
                _num(bc.allowable_pull(case.loads.case), 0),
            ]
        )
    s.append(
        _table(
            ["荷重ケース", "n(押込み)", "Ra (kN)", "n(引抜き)", "Pa (kN)"], rows
        )
    )
    return "".join(s)


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

    if case.pile_head is not None:
        s.append("\n### 4.6 杭頭結合部の照査(道示Ⅳ 12.9)\n")
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
    return "".join(s)


def _nf_section(report: StabilityReport) -> str:
    nf = report.negative_friction
    s = ["\n## 5. 負の周面摩擦力の検討(道示Ⅳ 12.4.3)\n"]
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
    s = ["\n## 6. 総括\n"]
    rows = []
    for case in report.cases:
        rows.append([case.loads.case.value, "OK" if case.all_ok else "NG"])
    if report.negative_friction is not None:
        rows.append(["負の周面摩擦力", report.negative_friction.judgement])
    s.append(_table(["項目", "判定"], rows))
    s.append(
        f"\n**総合判定: {'OK' if report.all_ok else 'NG'}**\n"
    )
    return "".join(s)
