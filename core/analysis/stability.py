"""安定計算のオーケストレーション(支持力・バネ定数・変位法・照査)。"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.analysis.displacement import PileReaction, StabilityResult, solve_stability
from core.analysis.section_forces import SectionForceDistribution, distribution
from core.capacity.bearing import BearingCapacity, compute_bearing_capacity
from core.capacity.negative_friction import (
    NegativeFrictionResult,
    compute_negative_friction,
)
from core.capacity.section import pile_section
from core.capacity.springs import LateralSprings, PileSection, axial_spring, lateral_springs
from core.models.loads import FootingLoads, LoadCase
from core.models.pile import Footing, PileArrangement, PileSpec, PileType
from core.models.soil import SoilProfile
from core.soil.liquefaction import SoilReduction
from core.section.checks import MaterialSpec, PileStressResult, check_section
from core.section.shear import ShearResult, check_shear
from core.section.pile_head import PileHeadResult, check_pile_head
from core.standards import (
    ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD,
    ALLOWABLE_DISPLACEMENT_MM,
    ALLOWABLE_DISPLACEMENT_RATIO,
    EC_SC_PILE_CONCRETE,
    E0Method,
)


def _check_max_shear(
    pile: PileSpec,
    material: MaterialSpec,
    case: LoadCase,
    forces: SectionForceDistribution,
    axial: float,
) -> ShearResult | None:
    """せん断力が最大となる断面のせん断照査(道示Ⅳ 5.1.3)。

    場所打ち杭のみ対応。他杭種は ``None`` を返す(注記は応力度照査側で出る)。
    """
    if pile.pile_type != PileType.CAST_IN_PLACE or material.rebar is None:
        return None
    peak = forces.max_shear
    return check_shear(
        pile,
        material.rebar,
        material.fck,
        case,
        depth=peak.depth,
        shear=peak.shear,
        moment=peak.moment,
        axial=axial,
        rebar_grade=material.rebar_grade,
        stirrup=material.stirrup,
    )


def allowable_displacement(diameter: float) -> float:
    """杭基礎の許容水平変位 (m)(道示Ⅳ 9.6)。

    杭径 1.5 m 未満は 15 mm、1.5 m 以上は杭径の 1%。
    """
    if diameter < ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD:
        return ALLOWABLE_DISPLACEMENT_MM / 1000.0
    return ALLOWABLE_DISPLACEMENT_RATIO * diameter


@dataclass(frozen=True)
class Check:
    name: str
    demand: float
    capacity: float
    unit: str

    @property
    def ratio(self) -> float:
        return abs(self.demand) / self.capacity if self.capacity else float("inf")

    @property
    def ok(self) -> bool:
        return self.ratio <= 1.0

    @property
    def judgement(self) -> str:
        return "OK" if self.ok else "NG"


@dataclass(frozen=True)
class CaseResult:
    loads: FootingLoads
    springs: LateralSprings
    kv: float
    result: StabilityResult
    checks: list[Check]
    critical_pile: PileReaction | None = None  # 照査対象とした最大反力の杭
    forces: SectionForceDistribution | None = None  # 杭体の断面力分布
    stress_head: PileStressResult | None = None  # 杭頭断面の応力度
    stress_max: PileStressResult | None = None  # 地中部最大曲げ断面の応力度
    pile_head: PileHeadResult | None = None  # 杭頭結合部
    shear: ShearResult | None = None  # せん断照査(場所打ち杭のみ)

    @property
    def all_ok(self) -> bool:
        if not all(c.ok for c in self.checks):
            return False
        for stress in (self.stress_head, self.stress_max):
            if stress is not None and not stress.all_ok:
                return False
        if self.pile_head is not None and not self.pile_head.all_ok:
            return False
        if self.shear is not None and not self.shear.all_ok:
            return False
        return True


@dataclass(frozen=True)
class StabilityReport:
    section: PileSection
    bearing: BearingCapacity  # 常時・暴風時に用いる支持力(低減なし)
    cases: list[CaseResult]
    negative_friction: NegativeFrictionResult | None = None
    notes: list[str] = field(default_factory=list)  # 省略した照査などの注記
    # 液状化を考慮する地震時の支持力(周面摩擦力度を DE で低減)。
    # 低減がない場合は None で、全ケースが :attr:`bearing` を用いる。
    bearing_seismic: BearingCapacity | None = None

    def bearing_for(self, case: LoadCase) -> BearingCapacity:
        """荷重ケースに適用する支持力。

        DE による低減は**耐震設計上の扱い**であり、常時・暴風時には
        適用しない(道示Ⅴ 8.2)。
        """
        if case.is_seismic and self.bearing_seismic is not None:
            return self.bearing_seismic
        return self.bearing

    @property
    def all_ok(self) -> bool:
        if self.negative_friction is not None and not self.negative_friction.ok:
            return False
        return all(c.all_ok for c in self.cases)


def analyze(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    loads: list[FootingLoads],
    fck: int = 24,
    material: MaterialSpec | None = None,
    check_negative_friction: bool = False,
    e0_method: E0Method = E0Method.N_VALUE,
    reduction: SoilReduction | None = None,
) -> StabilityReport:
    """全荷重ケースについて安定計算・断面照査・杭頭結合部の照査を行う。

    Parameters
    ----------
    material:
        杭体の材料条件。省略時は断面照査・杭頭結合部の照査を行わない。
    check_negative_friction:
        負の周面摩擦力(NF)を検討するか。常時の杭頭最大軸力を死荷重とみなす。
    e0_method:
        変形係数 E0 の推定方法。kH の換算係数 α がこれにより決まる。
    reduction:
        液状化に伴う土質定数の低減係数 DE(道示Ⅴ 8.2.4)。与えると
        **水平方向地盤反力係数 kH と最大周面摩擦力度 f** に乗じる
        (f′i = DE,i × fi)。

        **DE による低減は耐震設計上の扱いであり、常時・暴風時の照査には
        適用しない**。地震時のケースにのみ低減後の値を用い、常時・暴風時は
        低減前の値で照査する(:meth:`StabilityReport.bearing_for`)。

        なお DE 自体はレベル2地震動に対する液状化判定から得られる値である。
        レベル1地震時の照査に用いることの適否は利用者の判断となる
        (レベル1地震動に対する液状化判定は未実装。注記を出す)。
    """
    section = pile_section(pile, fck=fck)
    # 常時・暴風時は低減なし。DE は耐震設計上の扱いなので地震時のみ低減する
    bearing = compute_bearing_capacity(pile, profile, footing.embedment)
    bearing_seismic: BearingCapacity | None = None
    if reduction is not None and reduction.has_reduction:
        bearing_seismic = compute_bearing_capacity(
            pile, profile, footing.embedment, reduction=reduction
        )
    kv = axial_spring(pile, section)
    delta_a = allowable_displacement(pile.diameter)

    notes: list[str] = []
    if pile.pile_type in (PileType.PHC, PileType.SC):
        if pile.concrete_young is not None:
            notes.append(
                f"{pile.pile_type.value}の断面剛性 EI には、入力されたヤング係数 "
                f"Ec = {pile.concrete_young / 1.0e7:.2f}×10⁴ N/mm² を用いている。"
                "許容応力度は既製コンクリート杭として規定された値を用いており、"
                "σck の入力値には依存しない。"
            )
        elif pile.pile_type == PileType.SC:
            notes.append(
                "SC杭の断面剛性は鋼管とコンクリートの合成断面で評価している"
                f"(EI = Ec・Ic + Es・Is、Ec = {EC_SC_PILE_CONCRETE / 1.0e7:.1f}"
                "×10⁴ N/mm²、H24版の値)。σck の入力値には依存しない。"
                "H29版では Ec = 4.0×10⁴ N/mm² に改定されているため、"
                "製品の断面性能表を併用する場合は版の整合を確認すること。"
            )
        else:
            notes.append(
                "PHC杭の標準である σck = 80 N/mm² は道示Ⅲ 表-3.3.3"
                "(σck = 21〜60)の範囲外であり、ヤング係数が規定されていない。"
                "断面剛性 EI には入力した σck の Ec を用いているため"
                "(断面力・変位に影響する)、メーカーの断面性能表等の Ec を"
                "直接入力することを推奨する。許容応力度は既製コンクリート杭として"
                "規定された値を用いており、σck の入力値には依存しない。"
            )
    if reduction is not None and reduction.has_reduction:
        span = reduction.reduced_depth_range()
        notes.append(
            f"液状化による土質定数の低減を kH と周面摩擦力度 f に反映している"
            f"(低減区間: 深さ {span[0]:.1f}〜{span[1]:.1f} m、"
            f"{reduction.motion_type.value})。**DE による低減は耐震設計上の"
            "扱いであり、常時・暴風時の照査には適用していない**"
            "(道示Ⅴ 8.2)。また DE はレベル2地震動に対する液状化判定から"
            "得た値であり、レベル1地震時の照査に用いることの適否は利用者が"
            "判断すること(レベル1地震動に対する液状化判定は未実装)。"
        )
        if bearing_seismic is not None and bearing_seismic.has_reduced_skin:
            lost = (
                bearing_seismic.skin_resistance_unreduced
                - bearing_seismic.skin_resistance
            )
            ratio = lost / bearing.ru if bearing.ru > 0 else 0.0
            reduced = ", ".join(
                f"{s.layer_name}: f {s.f:.0f} → {s.f_design:.0f} kN/m²"
                f"(DE={s.de:.2f})"
                for s in bearing_seismic.skin_segments
                if s.is_reduced
            )
            notes.append(
                f"地震時の周面摩擦力度の低減内訳 — {reduced}。"
                f"極限支持力 Ru は {bearing.ru:.0f} → {bearing_seismic.ru:.0f} kN"
                f"({lost:.0f} kN、{ratio * 100:.1f}% の減少)。"
                "引抜き抵抗は周面摩擦力のみで決まるため、押込みより強く効く。"
            )
        if bearing_seismic is not None and bearing_seismic.tip_zone_liquefies:
            notes.append(
                f"⚠ 杭先端付近(先端±1D)が液状化すると判定されている"
                f"(DE={bearing_seismic.tip_de:.2f})。**先端支持力度 qd は"
                "低減していない**(支持層は液状化しない良質層であることが"
                "前提のため)。支持層の設定が適切か、杭長を見直す必要がないかを"
                "確認すること。"
            )
    cases: list[CaseResult] = []
    for load in loads:
        # DE による低減は耐震設計上の扱い。常時・暴風時には適用しない
        case_reduction = reduction if load.case.is_seismic else None
        case_bearing = (
            bearing_seismic
            if load.case.is_seismic and bearing_seismic is not None
            else bearing
        )
        springs = lateral_springs(
            pile, section, profile, footing.embedment, load.case,
            e0_method=e0_method, reduction=case_reduction,
        )
        result = solve_stability(
            arrangement,
            kv=kv,
            k1=springs.k1,
            k2=springs.k2,
            k4=springs.k4,
            v_load=load.v,
            h_load=load.h,
            m_load=load.m,
        )
        checks = [
            Check(
                name="押込み支持力",
                demand=result.max_axial,
                capacity=case_bearing.allowable_push(load.case),
                unit="kN",
            ),
            Check(
                name="水平変位",
                demand=result.u,
                capacity=delta_a,
                unit="m",
            ),
        ]
        # 引抜きが生じる場合のみ引抜き照査を行う
        if result.min_axial < 0:
            checks.append(
                Check(
                    name="引抜き抵抗力",
                    demand=-result.min_axial,
                    capacity=case_bearing.allowable_pull(load.case),
                    unit="kN",
                )
            )
        # 最も厳しい杭(押込み軸力が最大の杭)を代表断面として照査する
        critical = max(result.reactions, key=lambda r: r.axial)
        forces = distribution(
            ei=section.ei,
            beta=springs.beta,
            h0=critical.shear,
            m0=critical.moment,
            length=pile.length,
        )
        stress_head = stress_max = head_result = None
        shear_result = None
        if material is not None:
            try:
                stress_head = check_section(
                    pile, material, load.case, 0.0, critical.axial, critical.moment
                )
                peak = forces.max_underground_moment
                stress_max = check_section(
                    pile, material, load.case, peak.depth, critical.axial, peak.moment
                )
                shear_result = _check_max_shear(
                    pile, material, load.case, forces, critical.axial
                )
            except NotImplementedError as exc:
                # 杭体の応力度照査が未実装の杭種でも、支持力・変位の照査は
                # 有効なので計算を続け、省略した旨を注記として残す。
                stress_head = stress_max = None
                if str(exc) not in notes:
                    notes.append(str(exc))
            head_result = check_pile_head(
                pile_diameter=pile.diameter,
                footing_height=footing.height,
                fck=fck,
                case=load.case,
                axial=critical.axial,
                shear=critical.shear,
                moment=critical.moment,
                footing=footing,
                arrangement=arrangement,
            )
        cases.append(
            CaseResult(
                loads=load,
                springs=springs,
                kv=kv,
                result=result,
                checks=checks,
                critical_pile=critical,
                forces=forces,
                stress_head=stress_head,
                stress_max=stress_max,
                pile_head=head_result,
                shear=shear_result,
            )
        )

    nf = None
    if check_negative_friction:
        permanent = [c for c in cases if c.loads.case == LoadCase.PERMANENT]
        if not permanent:
            raise ValueError(
                "負の周面摩擦力の検討には常時の荷重ケースが必要です"
            )
        dead_load = permanent[0].result.max_axial
        nf = compute_negative_friction(
            pile, profile, footing.embedment, dead_load, bearing.ru
        )

    return StabilityReport(
        section=section,
        bearing=bearing,
        cases=cases,
        negative_friction=nf,
        notes=notes,
        bearing_seismic=bearing_seismic,
    )
