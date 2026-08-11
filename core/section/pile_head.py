"""杭頭結合部の照査(道示Ⅳ(H24) 12.9.3「杭とフーチングの接合部」)。

杭とフーチングの接合部は原則として剛結とし、接合部に生じる力に対して
安全であることを照査する。H24 の標準は従来「方法B」と呼ばれた方法で、
杭のフーチングへの埋込み長を最小限(目安 100mm)にとどめ、主として
杭頭補強鉄筋によって杭頭曲げモーメントに抵抗する。

作用と抵抗機構(道示Ⅳ 12.9.3):

===========  ==========================================  ====================
作用          主な抵抗の考え方                              確認の焦点
===========  ==========================================  ====================
押込み力      フーチングコンクリートの支圧・押抜きせん断      杭頭周辺の局部破壊
引抜き力      杭頭補強鉄筋などの引張抵抗                    補強鉄筋・定着部
水平力・M     補強鉄筋、仮想RC断面、水平押抜きせん断         縁端部を含む破壊
===========  ==========================================  ====================

本モジュールの実装範囲:

* 押込み力に対する **押抜きせん断** と **支圧** — 実装済み
* 縁端距離の確認と、水平方向押抜きせん断照査の要否判定 — 実装済み
* 杭頭補強鉄筋の応力度・定着長、仮想RC断面の照査 — **未実装**

.. warning::
   照査式・許容値は原典未照合の項目を含む(docs/VERIFICATION.md 参照)。
   特に杭頭補強鉄筋と仮想RC断面が未実装であるため、**本モジュールだけで
   杭頭結合部の安全性を確認したことにはならない**。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.loads import LoadCase
from core.models.pile import Footing, PileArrangement
from core.section.checks import StressCheck
from core.standards import (
    SIGMA_CA_CONCRETE,
    STRESS_INCREASE,
    TAU_A_PUNCHING,
)

# 方法B の標準的な杭頭埋込み長 (m)(道示Ⅳ 12.9.3)
STANDARD_EMBEDMENT = 0.1

# 最外周杭の縁端距離の標準値(杭径 D の倍数)。これを下回る場合は
# フーチングの水平方向押抜きせん断の照査が必要(レベル2地震動まで)。
STANDARD_EDGE_DISTANCE_RATIO = 1.0


@dataclass(frozen=True)
class EdgeDistance:
    """最外周杭のフーチング縁端距離。"""

    edge_x: float  # 橋軸方向 (m)
    edge_y: float  # 橋軸直角方向 (m)
    diameter: float  # 杭径 (m)

    @property
    def minimum(self) -> float:
        return min(self.edge_x, self.edge_y)

    @property
    def required(self) -> float:
        return STANDARD_EDGE_DISTANCE_RATIO * self.diameter

    @property
    def is_standard(self) -> bool:
        """標準値(1.0D)以上か。"""
        return self.minimum >= self.required

    @property
    def needs_horizontal_punching_check(self) -> bool:
        """フーチングの水平方向押抜きせん断の照査が必要か。"""
        return not self.is_standard


@dataclass(frozen=True)
class PileHeadResult:
    punching_area: float  # 押抜きせん断の抵抗面積 (m2)
    checks: list[StressCheck]
    edge_distance: EdgeDistance | None = None

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def edge_distances(
    footing: Footing, arrangement: PileArrangement, diameter: float
) -> EdgeDistance:
    """最外周杭の中心からフーチング縁端までの距離を求める。"""
    max_x = (arrangement.nx - 1) / 2.0 * arrangement.spacing_x
    max_y = (arrangement.ny - 1) / 2.0 * arrangement.spacing_y
    return EdgeDistance(
        edge_x=footing.width_x / 2.0 - max_x,
        edge_y=footing.width_y / 2.0 - max_y,
        diameter=diameter,
    )


def punching_shear_area(
    pile_diameter: float,
    footing_height: float,
    embedment: float = STANDARD_EMBEDMENT,
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
    embedment: float = STANDARD_EMBEDMENT,
    footing: Footing | None = None,
    arrangement: PileArrangement | None = None,
) -> PileHeadResult:
    """杭頭結合部を照査する。

    ``axial`` は杭頭軸力 (kN、押込み正)、``shear`` は杭頭水平力 (kN)、
    ``moment`` は杭頭モーメント (kN·m)。

    ``footing`` と ``arrangement`` を与えると縁端距離も評価する。

    .. note::
       ``shear`` と ``moment`` は現時点で照査に用いていない。これらに対する
       抵抗は杭頭補強鉄筋・仮想RC断面が担うが、いずれも未実装のため。
    """
    if fck not in TAU_A_PUNCHING:
        raise ValueError(
            f"σck={fck} は許容押抜きせん断応力度 τa3 の表(道示Ⅳ 表4.2.1、"
            f"σck = {sorted(TAU_A_PUNCHING)})の範囲外です。"
            "適用する設計条件・発注者基準を別途確認してください"
        )
    increase = STRESS_INCREASE[case.value]
    area = punching_shear_area(pile_diameter, footing_height, embedment)
    pile_area = math.pi * pile_diameter**2 / 4.0

    # 押込み力に対する押抜きせん断(引抜き時も絶対値で照査)。
    # 杭頭結合部では水平力・曲げモーメントが同時に作用し得るため、
    # 荷重の組合せによる τa3 の割増しは行わない(地震時も表の値のまま)。
    tau = abs(axial) / area / 1000.0  # kN/m2 → N/mm2
    tau_a = TAU_A_PUNCHING[fck]

    # 押込み力に対する支圧。コンクリートの許容支圧応力度は拘束効果により
    # 曲げ圧縮より大きく採れるが、安全側に σca を用いる。
    sigma_bearing = max(0.0, axial) / pile_area / 1000.0
    sigma_ba = SIGMA_CA_CONCRETE[fck] * increase

    checks = [
        StressCheck("杭頭押抜きせん断応力度", tau, tau_a),
        StressCheck("杭頭支圧応力度", sigma_bearing, sigma_ba),
    ]
    # .. note::
    #    支圧については割増しの扱いが原典で未確認のため、通常どおり
    #    荷重ケース別の割増しを適用している(押抜きせん断のみ割増しなし)。
    edge = (
        edge_distances(footing, arrangement, pile_diameter)
        if footing is not None and arrangement is not None
        else None
    )
    return PileHeadResult(punching_area=area, checks=checks, edge_distance=edge)
