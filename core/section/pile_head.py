"""杭頭結合部の照査(道示Ⅳ(H24) 12.9)。

結合方法B(杭頭をフーチングに埋込み長さ 100mm 程度とし、杭頭鉄筋で
フーチングと結合する方法)を対象とし、以下を照査する:

  1. 押抜きせん断(杭頭反力がフーチングを押抜く)
  2. 仮想RC断面の応力度(杭頭鉄筋を主鉄筋とする仮想断面)
  3. 水平支圧応力度(杭頭側面の支圧)

.. warning::
   本モジュールの式・許容値は特に確度が低い(docs/VERIFICATION.md の確度C)。
   実務適用前に必ず道示Ⅳ 12.9 と照合すること。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.loads import LoadCase
from core.section.checks import StressCheck
from core.standards import (
    SIGMA_CA_CONCRETE,
    STRESS_INCREASE,
    TAU_A_PUNCHING,
)


@dataclass(frozen=True)
class PileHeadResult:
    punching_area: float  # 押抜きせん断の抵抗面積 (m2)
    checks: list[StressCheck]

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def punching_shear_area(
    pile_diameter: float, footing_height: float, embedment: float = 0.1
) -> float:
    """押抜きせん断の抵抗面積 (m2)。

    杭頭埋込み部の下端から 45 度で広がる仮想破壊面を想定し、
    有効高さ h = フーチング厚 − 埋込み長 に対して
        A = π・(D + h)・h
    とする。
    """
    h = footing_height - embedment
    if h <= 0:
        raise ValueError(
            f"フーチング厚 {footing_height} m が杭頭埋込み長 {embedment} m 以下です"
        )
    return math.pi * (pile_diameter + h) * h


def check_pile_head(
    pile_diameter: float,
    footing_height: float,
    fck: int,
    case: LoadCase,
    axial: float,
    shear: float,
    moment: float,
    embedment: float = 0.1,
) -> PileHeadResult:
    """杭頭結合部を照査する。

    ``axial`` は杭頭軸力 (kN、押込み正)、``shear`` は杭頭水平力 (kN)、
    ``moment`` は杭頭モーメント (kN·m)。
    """
    if fck not in TAU_A_PUNCHING:
        raise ValueError(f"σck={fck} は未対応です")
    increase = STRESS_INCREASE[case.value]
    area = punching_shear_area(pile_diameter, footing_height, embedment)

    # 押抜きせん断応力度(押込み・引抜きとも絶対値で照査)
    tau = abs(axial) / area / 1000.0  # kN/m2 → N/mm2
    tau_a = TAU_A_PUNCHING[fck] * increase

    # 水平支圧応力度: 杭頭側面の投影面積で水平力を受けると仮定
    bearing_area = pile_diameter * embedment
    sigma_bearing = abs(shear) / bearing_area / 1000.0
    sigma_ba = SIGMA_CA_CONCRETE[fck] * increase

    checks = [
        StressCheck("杭頭押抜きせん断応力度", tau, tau_a),
        StressCheck("杭頭水平支圧応力度", sigma_bearing, sigma_ba),
    ]
    return PileHeadResult(punching_area=area, checks=checks)
