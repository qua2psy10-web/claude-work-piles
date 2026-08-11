"""安定計算のオーケストレーション(支持力・バネ定数・変位法・照査)。"""
from __future__ import annotations

from dataclasses import dataclass

from core.analysis.displacement import StabilityResult, solve_stability
from core.capacity.bearing import BearingCapacity, compute_bearing_capacity
from core.capacity.section import pile_section
from core.capacity.springs import LateralSprings, PileSection, axial_spring, lateral_springs
from core.models.loads import FootingLoads
from core.models.pile import Footing, PileArrangement, PileSpec
from core.models.soil import SoilProfile
from core.standards import (
    ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD,
    ALLOWABLE_DISPLACEMENT_MM,
    ALLOWABLE_DISPLACEMENT_RATIO,
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

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


@dataclass(frozen=True)
class StabilityReport:
    section: PileSection
    bearing: BearingCapacity
    cases: list[CaseResult]

    @property
    def all_ok(self) -> bool:
        return all(c.all_ok for c in self.cases)


def analyze(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    loads: list[FootingLoads],
    fck: int = 24,
) -> StabilityReport:
    """全荷重ケースについて安定計算と照査を行う。"""
    section = pile_section(pile, fck=fck)
    bearing = compute_bearing_capacity(pile, profile, footing.embedment)
    kv = axial_spring(pile, section)
    delta_a = allowable_displacement(pile.diameter)

    cases: list[CaseResult] = []
    for load in loads:
        springs = lateral_springs(pile, section, profile, footing.embedment, load.case)
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
        cases.append(
            CaseResult(loads=load, springs=springs, kv=kv, result=result, checks=checks)
        )

    return StabilityReport(section=section, bearing=bearing, cases=cases)
