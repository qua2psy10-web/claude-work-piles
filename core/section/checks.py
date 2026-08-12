"""杭種別の杭体応力度照査(道示Ⅳ(H24) 12.10)。

フェーズ2では場所打ち杭(円形RC断面)と鋼管杭(円環断面)に対応する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.capacity.section import CORROSION_ALLOWANCE_MM
from core.models.loads import LoadCase
from core.models.pile import PileSpec, PileType
from core.section.rc import RebarLayout, RcStressResult, analyze_circular_rc
from core.standards import (
    CIP_CONCRETE_REDUCTION,
    E_REBAR,
    EC_CONCRETE,
    REMOVED_REBAR_GRADES,
    SIGMA_A_STEEL,
    SIGMA_CA_CONCRETE,
    SIGMA_SA_REBAR,
    STRESS_INCREASE,
)


# 応力度照査が未実装の杭種と、その理由(未照合の基準値)。
# 断面諸元(A・I・E)は core.capacity.section で全杭種算定できるため、
# 支持力・バネ定数・変位法・断面力分布は利用できる。
UNVERIFIED_ALLOWABLES: dict[PileType, str] = {
    PileType.PHC: (
        "高強度コンクリートのヤング係数・許容応力度、および有効プレストレス"
    ),
    PileType.SC: "鋼管とコンクリートの合成断面に対する許容応力度",
    PileType.RC: "既製RC杭のコンクリート・鉄筋の許容応力度",
    PileType.H_STEEL: "H形鋼(SS材・SM材)の許容応力度",
}


@dataclass(frozen=True)
class StressCheck:
    """1つの応力度照査項目。"""

    name: str
    stress: float  # 発生応力度 (N/mm2)
    allowable: float  # 許容応力度 (N/mm2、割増後)

    @property
    def ratio(self) -> float:
        return abs(self.stress) / self.allowable if self.allowable else math.inf

    @property
    def ok(self) -> bool:
        return self.ratio <= 1.0

    @property
    def judgement(self) -> str:
        return "OK" if self.ok else "NG"


@dataclass(frozen=True)
class PileStressResult:
    """杭体1断面の照査結果。"""

    depth: float  # 照査位置(杭頭からの深さ) (m)
    axial: float  # 軸力 (kN)
    moment: float  # 曲げモーメント (kN·m)
    checks: list[StressCheck] = field(default_factory=list)
    rc_detail: RcStressResult | None = None

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


@dataclass(frozen=True)
class MaterialSpec:
    """杭体の材料条件。"""

    fck: int = 24  # コンクリート設計基準強度 (N/mm2)
    rebar_grade: str = "SD345"
    steel_grade: str = "SKK400"
    rebar: RebarLayout | None = None  # 場所打ち杭の軸方向鉄筋
    corrosion_mm: float = CORROSION_ALLOWANCE_MM


def check_section(
    pile: PileSpec,
    material: MaterialSpec,
    case: LoadCase,
    depth: float,
    axial: float,
    moment: float,
) -> PileStressResult:
    """1断面の応力度照査を行う。

    ``axial`` は圧縮正 (kN)、``moment`` は曲げモーメント (kN·m)。
    """
    increase = STRESS_INCREASE[case.value]
    if pile.pile_type == PileType.CAST_IN_PLACE:
        return _check_cast_in_place(
            pile, material, increase, depth, axial, moment
        )
    if pile.pile_type in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        return _check_steel_pipe(pile, material, increase, depth, axial, moment)
    raise NotImplementedError(
        f"{pile.pile_type.value}の応力度照査は未実装です。"
        f"{UNVERIFIED_ALLOWABLES.get(pile.pile_type, '許容応力度')}が"
        "未照合のためです(断面諸元の算定と安定計算は可能です)"
    )


def _check_cast_in_place(
    pile: PileSpec,
    material: MaterialSpec,
    increase: float,
    depth: float,
    axial: float,
    moment: float,
) -> PileStressResult:
    if material.rebar is None:
        raise ValueError("場所打ち杭の照査には軸方向鉄筋の入力が必要です")
    if material.fck not in EC_CONCRETE:
        raise ValueError(f"σck={material.fck} は未対応です")
    ec = EC_CONCRETE[material.fck]
    n_ratio = E_REBAR / ec
    detail = analyze_circular_rc(
        diameter=pile.diameter,
        rebar=material.rebar,
        ec=ec,
        n_ratio=n_ratio,
        axial=axial,
        moment=moment,
    )
    # 場所打ち杭は水中施工を考慮してコンクリートの許容応力度を低減する
    sigma_ca = SIGMA_CA_CONCRETE[material.fck] * CIP_CONCRETE_REDUCTION * increase
    if material.rebar_grade in REMOVED_REBAR_GRADES:
        raise ValueError(
            f"{material.rebar_grade} は H24 の道示Ⅳ下部構造編で鉄筋の種類から"
            "削除されており、許容引張応力度が規定されていません。"
            f"対応材質: {sorted(SIGMA_SA_REBAR)}"
        )
    if material.rebar_grade not in SIGMA_SA_REBAR:
        raise ValueError(
            f"鉄筋材質 {material.rebar_grade} は未対応です。"
            f"対応材質: {sorted(SIGMA_SA_REBAR)}"
        )
    sigma_sa = SIGMA_SA_REBAR[material.rebar_grade] * increase
    checks = [
        StressCheck("コンクリート圧縮応力度", detail.sigma_c, sigma_ca),
        StressCheck("鉄筋引張応力度", detail.sigma_s_tension, sigma_sa),
    ]
    return PileStressResult(
        depth=depth, axial=axial, moment=moment, checks=checks, rc_detail=detail
    )


def _check_steel_pipe(
    pile: PileSpec,
    material: MaterialSpec,
    increase: float,
    depth: float,
    axial: float,
    moment: float,
) -> PileStressResult:
    if pile.wall_thickness is None:
        raise ValueError("鋼管杭の照査には板厚の入力が必要です")
    t = (pile.wall_thickness - material.corrosion_mm) / 1000.0
    if t <= 0:
        raise ValueError("腐食代控除後の板厚が 0 以下です")
    d_out = pile.diameter
    d_in = d_out - 2.0 * t
    area = math.pi * (d_out**2 - d_in**2) / 4.0
    inertia = math.pi * (d_out**4 - d_in**4) / 64.0
    section_modulus = inertia / (d_out / 2.0)

    # kN, m → N/mm2 は 1/1000
    sigma_axial = axial / area / 1000.0
    sigma_bending = abs(moment) / section_modulus / 1000.0
    sigma_max = sigma_axial + sigma_bending  # 圧縮側
    sigma_min = sigma_axial - sigma_bending  # 引張側(負なら引張)
    sigma_a = SIGMA_A_STEEL[material.steel_grade] * increase
    checks = [
        StressCheck("鋼管圧縮応力度", sigma_max, sigma_a),
        StressCheck("鋼管引張応力度", abs(min(0.0, sigma_min)), sigma_a),
    ]
    return PileStressResult(
        depth=depth, axial=axial, moment=moment, checks=checks
    )
