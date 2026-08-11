"""安定計算のオーケストレーション(支持力・バネ定数・変位法・照査)。"""
from __future__ import annotations

from dataclasses import dataclass

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
from core.models.pile import Footing, PileArrangement, PileSpec
from core.models.soil import SoilProfile
from core.section.checks import MaterialSpec, PileStressResult, check_section
from core.section.pile_head import PileHeadResult, check_pile_head
from core.standards import (
    ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD,
    ALLOWABLE_DISPLACEMENT_MM,
    ALLOWABLE_DISPLACEMENT_RATIO,
    E0Method,
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

    @property
    def all_ok(self) -> bool:
        if not all(c.ok for c in self.checks):
            return False
        for stress in (self.stress_head, self.stress_max):
            if stress is not None and not stress.all_ok:
                return False
        if self.pile_head is not None and not self.pile_head.all_ok:
            return False
        return True


@dataclass(frozen=True)
class StabilityReport:
    section: PileSection
    bearing: BearingCapacity
    cases: list[CaseResult]
    negative_friction: NegativeFrictionResult | None = None

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
    """
    section = pile_section(pile, fck=fck)
    bearing = compute_bearing_capacity(pile, profile, footing.embedment)
    kv = axial_spring(pile, section)
    delta_a = allowable_displacement(pile.diameter)

    cases: list[CaseResult] = []
    for load in loads:
        springs = lateral_springs(
            pile, section, profile, footing.embedment, load.case, e0_method=e0_method
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
                capacity=bearing.allowable_push(load.case),
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
                    capacity=bearing.allowable_pull(load.case),
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
        if material is not None:
            stress_head = check_section(
                pile, material, load.case, 0.0, critical.axial, critical.moment
            )
            peak = forces.max_underground_moment
            stress_max = check_section(
                pile, material, load.case, peak.depth, critical.axial, peak.moment
            )
            head_result = check_pile_head(
                pile_diameter=pile.diameter,
                footing_height=footing.height,
                fck=fck,
                case=load.case,
                axial=critical.axial,
                shear=critical.shear,
                moment=critical.moment,
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
        section=section, bearing=bearing, cases=cases, negative_friction=nf
    )
