"""杭種別の杭体応力度照査(道示Ⅳ(H24) 12.10)。

対応杭種は場所打ち杭(円形RC断面)・鋼管杭(円環断面)・
鋼管ソイルセメント杭(鋼管部で照査)・PHC杭(非ひび割れ中空断面)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.capacity.section import CORROSION_ALLOWANCE_MM, hollow_circle
from core.models.loads import LoadCase
from core.models.pile import PileSpec, PileType
from core.section.rc import RebarLayout, RcStressResult, analyze_circular_rc
from core.standards import (
    EC_CONCRETE,
    PHC_BENDING_TENSION_BY_PRESTRESS,
    PRECAST_CONCRETE_ALLOWABLE,
    REBAR_GRADES,
    REMOVED_REBAR_GRADES,
    SIGMA_A_STEEL,
    SIGMA_SA_REBAR_SEISMIC,
    SIGMA_SA_REBAR_STATIC,
    STRESS_INCREASE,
    UNDERWATER_CONCRETE_ALLOWABLE,
    YOUNG_MODULUS_RATIO_RC,
    RebarMember,
)


# 応力度照査が未実装の杭種と、その理由。
# 断面諸元(A・I・E)は core.capacity.section で全杭種算定できるため、
# 支持力・バネ定数・変位法・断面力分布は利用できる。
UNIMPLEMENTED_STRESS_CHECK: dict[PileType, str] = {
    PileType.SC: (
        "鋼管とコンクリートの合成断面に対する応力度分担のモデル化と、"
        "SC杭の鋼管部の許容応力度が未照合"
    ),
    PileType.RC: (
        "中空断面のひび割れ断面解析(軸方向鉄筋の配置入力)が未実装"
        "(コンクリートの許容応力度は照合済み)"
    ),
    PileType.H_STEEL: (
        "H形鋼杭は断面・座屈・曲げ圧縮等の照査条件があり、"
        "許容応力度の値のみでは断面照査を行えない"
    ),
}


# 許容値が 0 の照査で「応力なし」とみなす閾値 (N/mm2)。
# 数値誤差(kN·m → N/mm2 換算で 1e-9 オーダー)を吸収するためのもの。
_ZERO_STRESS_TOL = 1.0e-9


@dataclass(frozen=True)
class StressCheck:
    """1つの応力度照査項目。"""

    name: str
    stress: float  # 発生応力度 (N/mm2)
    allowable: float  # 許容応力度 (N/mm2、割増後)

    @property
    def ratio(self) -> float:
        if self.allowable > 0.0:
            return abs(self.stress) / self.allowable
        # 許容値 0(PHC杭の常時の曲げ引張など)は「応力を生じさせないこと」を
        # 要求する。応力が 0 なら OK、少しでも生じれば NG とする。
        return 0.0 if abs(self.stress) <= _ZERO_STRESS_TOL else math.inf

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
    # PHC杭の有効プレストレス σce (N/mm2)。地震時の許容曲げ引張応力度が
    # この値で決まるため、PHC杭に引張が生じる地震時の照査では必須。
    effective_prestress: float | None = None


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
            pile, material, increase, depth, axial, moment, case
        )
    if pile.pile_type in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        return _check_steel_pipe(pile, material, increase, depth, axial, moment)
    if pile.pile_type == PileType.PHC:
        return _check_phc(pile, material, increase, case, depth, axial, moment)
    raise NotImplementedError(
        f"{pile.pile_type.value}の応力度照査は未実装です。理由: "
        f"{UNIMPLEMENTED_STRESS_CHECK.get(pile.pile_type, '許容応力度が未照合')}"
        "(断面諸元の算定と安定計算は可能です)"
    )


def phc_bending_tension_allowable(
    case: LoadCase, effective_prestress: float | None
) -> float:
    """PHC杭の許容曲げ引張応力度 (N/mm2)。

    常時は引張を許さない(0)。地震時のみ有効プレストレス σce に応じて
    :data:`core.standards.PHC_BENDING_TENSION_BY_PRESTRESS` の値を用いる。

    暴風時は原典に地震時のような規定を確認できていないため、安全側に常時と
    同じ扱い(0)とする。
    """
    if case != LoadCase.LEVEL1_EQ:
        return PRECAST_CONCRETE_ALLOWABLE["PHC杭"].bending_tension or 0.0
    if effective_prestress is None:
        raise ValueError(
            "PHC杭に曲げ引張応力度が生じる地震時の照査には、有効プレストレス "
            "σce (N/mm2) の入力が必要です(MaterialSpec.effective_prestress)"
        )
    for threshold, allowable in PHC_BENDING_TENSION_BY_PRESTRESS:
        if effective_prestress >= threshold:
            return allowable
    # σce が表の下限(3.9)未満のときは規定がないため引張を許さない
    return 0.0


def _check_phc(
    pile: PileSpec,
    material: MaterialSpec,
    increase: float,
    case: LoadCase,
    depth: float,
    axial: float,
    moment: float,
) -> PileStressResult:
    """PHC杭の応力度照査(全断面有効)。

    PHC杭はプレストレスによりひび割れを生じさせない設計とするため、
    中空円形の全断面を有効として σ = N/A ± M/Z で照査する。
    """
    if pile.concrete_thickness is None:
        raise ValueError(
            "PHC杭の照査にはコンクリート部の肉厚 concrete_thickness (mm) の"
            "入力が必要です"
        )
    area, inertia = hollow_circle(pile.diameter, pile.concrete_thickness / 1000.0)
    section_modulus = inertia / (pile.diameter / 2.0)

    # kN, m → N/mm2 は 1/1000
    sigma_axial = axial / area / 1000.0  # 圧縮正
    sigma_bending = abs(moment) / section_modulus / 1000.0
    sigma_compression = sigma_axial + sigma_bending
    sigma_tension = max(0.0, sigma_bending - sigma_axial)

    allow = PRECAST_CONCRETE_ALLOWABLE["PHC杭"]
    checks = [
        StressCheck(
            "軸圧縮応力度", sigma_axial, allow.axial_compression * increase
        ),
        StressCheck(
            "曲げ圧縮応力度", sigma_compression, allow.bending_compression * increase
        ),
    ]
    if sigma_tension > _ZERO_STRESS_TOL:
        # 許容曲げ引張応力度は荷重の組合せごとに直接与えられる値であり、
        # 割増し係数を重ねて乗じない。
        checks.append(
            StressCheck(
                "曲げ引張応力度",
                sigma_tension,
                phc_bending_tension_allowable(case, material.effective_prestress),
            )
        )
    return PileStressResult(
        depth=depth, axial=axial, moment=moment, checks=checks
    )


def rebar_tension_allowable(
    grade: str,
    case: LoadCase,
    underwater: bool,
    increase: float,
    axial_rebar: bool = True,
) -> float:
    """鉄筋の許容引張応力度 (N/mm2)(道示Ⅳ(H24) 4.3、表-4.3.1)。

    表は荷重の組合せの区分ごとに**基本値**を与えており、その基本値に
    表-4.1.1 の割増係数を乗じる。

    * 衝突荷重又は地震の影響を**含まない**組合せ(常時・暴風時)
        一般の部材 180、水中又は地下水位以下に設ける部材 160
    * **含む**組合せ(地震時)
        軸方向鉄筋 200(SD345)/ 230(SD390)/ 290(SD490)

    したがって、たとえば SD345 の軸方向鉄筋は
    常時 180、暴風時 180×1.25 = 225、レベル1地震時 200×1.50 = 300 となる。

    Parameters
    ----------
    underwater:
        水中又は地下水位以下に設ける部材か。場所打ち杭は水中施工であり、
        地下水位以下にもなるため真とする。地震時の区分にはこの区別がない。
    axial_rebar:
        軸方向鉄筋か。地震時の基本値が軸方向鉄筋(200/230/290)と
        それ以外(一律 200)で異なるため、**斜引張鉄筋・帯鉄筋では偽**に
        すること。地震を含まない組合せではこの区別はない。
    """
    if grade in REMOVED_REBAR_GRADES:
        raise ValueError(
            f"{grade} は H24 の道示Ⅳ下部構造編で鉄筋の種類から削除されており、"
            f"許容引張応力度が規定されていません。対応材質: {list(REBAR_GRADES)}"
        )
    if case.is_seismic:
        key = (
            RebarMember.AXIAL.value
            if axial_rebar
            else RebarMember.OTHER_SEISMIC.value
        )
        table = SIGMA_SA_REBAR_SEISMIC[key]
    else:
        key = (
            RebarMember.UNDERWATER.value if underwater else RebarMember.GENERAL.value
        )
        table = SIGMA_SA_REBAR_STATIC[key]
    if grade not in table:
        raise ValueError(
            f"鉄筋材質 {grade} は未対応です。対応材質: {sorted(table)}"
        )
    return table[grade] * increase


def _check_cast_in_place(
    pile: PileSpec,
    material: MaterialSpec,
    increase: float,
    depth: float,
    axial: float,
    moment: float,
    case: LoadCase,
) -> PileStressResult:
    if material.rebar is None:
        raise ValueError("場所打ち杭の照査には軸方向鉄筋の入力が必要です")
    if material.fck not in EC_CONCRETE:
        raise ValueError(f"σck={material.fck} は未対応です")
    # 場所打ち杭は水中施工。許容応力度は表-4.2.5(水中コンクリートの設計基準
    # 強度で引く)による。**0.8 倍の低減は道示Ⅳ に存在しない**
    if material.fck not in UNDERWATER_CONCRETE_ALLOWABLE:
        raise ValueError(
            f"水中で施工する場所打ち杭の σck={material.fck} は道示Ⅳ 表-4.2.5 に"
            f"規定がありません。対応値: {sorted(UNDERWATER_CONCRETE_ALLOWABLE)}"
            "(呼び強度 30/36/40 に対する水中コンクリートの設計基準強度)"
        )
    ec = EC_CONCRETE[material.fck]
    # ヤング係数比は Es/Ec ではなく一定値 15(道示Ⅳ 5.1.2(3))
    detail = analyze_circular_rc(
        diameter=pile.diameter,
        rebar=material.rebar,
        ec=ec,
        n_ratio=YOUNG_MODULUS_RATIO_RC,
        axial=axial,
        moment=moment,
    )
    sigma_ca = (
        UNDERWATER_CONCRETE_ALLOWABLE[material.fck].bending_compression * increase
    )
    sigma_sa = rebar_tension_allowable(
        material.rebar_grade, case, underwater=True, increase=increase
    )
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
