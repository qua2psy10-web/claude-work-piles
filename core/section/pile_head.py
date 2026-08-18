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
* 引抜き力に対する **押抜きせん断**(方法A、専用の抵抗厚さ ht を使用) — 実装済み
* 水平力・モーメントに対する **水平支圧応力度**(方法B は PH のみ、方法A は
  M も加味)— 実装済み
* フーチング端部の杭に対する **水平方向押抜きせん断** — 関数として実装済み
  (:func:`horizontal_edge_punching_shear`。フーチング有効厚さ h' は
  利用者が与える必要があるため :func:`check_pile_head` には自動配線していない)
* 縁端距離の確認と、水平方向押抜きせん断照査の要否判定 — 実装済み
* 杭頭補強鉄筋の応力度・定着長、仮想RC断面の照査 — **未実装**
  (docs/VERIFICATION.md 第43回に、公式の再現と数値一致を確認した記録がある。
  ``core.section.rc.analyze_circular_rc`` がそのまま転用できる見込みだが、
  引張軸力を受ける断面の未実装(第31回来の既知の制限)と鉄筋許容応力度の
  入力方法の設計が残っている)

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
    PULL_OUT_RESISTANCE_THICKNESS,
    SIGMA_CVA_PILE_HEAD_BEARING,
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


def horizontal_bearing_stress(
    shear: float,
    diameter: float,
    embedment: float,
    moment: float = 0.0,
) -> float:
    """フーチングコンクリートの水平支圧応力度 σch (N/mm2)(道示Ⅳ 12.9.3)。

        σch = PH/(D・L) + 6・M/(D・L²)

    ``moment`` を省略(0.0)すると PH のみの式になる。

    .. important::
       **方法B(H24 標準、埋込み長 100mm)はモーメント項を含まない式**が
       計算例で使われている(モーメント抵抗は仮想RC断面が負担するため)。
       **方法A(杭径相当を埋め込む剛結)はモーメント項を含む式**が使われて
       いる。方法Aとして評価したい場合のみ ``moment`` を渡すこと。

    出典: フォーラムエイト UC-1「基礎の設計」計算書サンプル Kui_5 の
    6.2(3)(第43回)。既設鋼管杭(方法A、L=D=0.6m)PH=100.3kN,
    M=90.0kN·m, D=0.6m → σch=2.78 N/mm²(本式 2.7786)、増し杭
    (方法B、L=0.1m)PH=167.1kN, D=1.0m(モーメント省略)→
    σch=1.67 N/mm²(本式 1.671)と一致確認済み。確度C(他社製品の出力
    からσck=24の例のみで確認。原典は未照合)。
    """
    if diameter <= 0 or embedment <= 0:
        raise ValueError("杭径・埋込み長は正の値である必要があります")
    return (
        abs(shear) / (diameter * embedment)
        + 6.0 * abs(moment) / (diameter * embedment**2)
    ) / 1000.0


def horizontal_edge_punching_shear(
    shear: float,
    diameter: float,
    embedment: float,
    effective_thickness: float,
) -> float:
    """フーチング端部の杭に対する水平方向の押抜きせん断応力度 τh (N/mm2)。

        τh = PH / (h'・(2・L + D + 2・h'))

    ``effective_thickness`` は h'(水平方向の押抜きせん断力に抵抗する
    フーチングの有効厚さ)。垂直方向の押抜きせん断に用いる h とは別の値で、
    出典の計算例でも導出式は示されず利用者が与える値として扱われている
    ため、本関数でも呼び出し側が明示的に与える設計とした。

    最外周杭のフーチング縁端距離が標準値(1.0D)以上であれば本照査は不要
    (:attr:`EdgeDistance.needs_horizontal_punching_check` を参照)。

    出典: Kui_5 6.2(3)2)(第43回)。既設鋼管杭(L=D=0.6m)PH=100.3kN,
    h'=2.45m, D=0.6m → τh=0.006 N/mm²(本式 0.00611)、増し杭
    (L=0.1m)PH=167.1kN, h'=2.45m, D=1.0m → τh=0.011 N/mm²
    (本式 0.01118)と一致確認済み。確度C。
    """
    if diameter <= 0 or embedment <= 0 or effective_thickness <= 0:
        raise ValueError("杭径・埋込み長・有効厚さは正の値である必要があります")
    denom = effective_thickness * (
        2.0 * embedment + diameter + 2.0 * effective_thickness
    )
    return abs(shear) / denom / 1000.0


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
    include_moment_in_bearing: bool = False,
) -> PileHeadResult:
    """杭頭結合部を照査する。

    ``axial`` は杭頭軸力 (kN、押込み正)、``shear`` は杭頭水平力 (kN)、
    ``moment`` は杭頭モーメント (kN·m)。

    ``footing`` と ``arrangement`` を与えると縁端距離も評価する。

    .. note::
       ``moment`` は既定では水平支圧応力度の算定に用いない(方法B の式に
       合わせている。:func:`horizontal_bearing_stress` の説明を参照)。
       方法Aとして評価したい場合は ``include_moment_in_bearing=True`` を
       指定すること。杭頭補強鉄筋の応力度・定着長・仮想RC断面の照査は
       本関数の対象外(未実装。docs/VERIFICATION.md 第43回を参照)。
    """
    if fck not in TAU_A_PUNCHING:
        raise ValueError(
            f"σck={fck} は許容押抜きせん断応力度 τa3 の表(道示Ⅳ 表4.2.1、"
            f"σck = {sorted(TAU_A_PUNCHING)})の範囲外です。"
            "適用する設計条件・発注者基準を別途確認してください"
        )
    if fck not in SIGMA_CVA_PILE_HEAD_BEARING:
        raise ValueError(
            f"σck={fck} は杭頭支圧応力度 σcva の表(σck = "
            f"{sorted(SIGMA_CVA_PILE_HEAD_BEARING)})の範囲外です。"
            "適用する設計条件・発注者基準を別途確認してください"
        )
    increase = STRESS_INCREASE[case.value]
    pile_area = math.pi * pile_diameter**2 / 4.0

    # 押込み力に対する押抜きせん断。有効高さ h はフーチング厚から埋込み長を
    # 差し引いた値。
    # 引抜き力に対する押抜きせん断は**専用の抵抗厚さ ht(道示Ⅳ 12.9.3、
    # 標準100mm)**を使う、押込み側とは別の仮想破壊面(Kui_5 6.2、第43回)。
    # 従来は押込み側の面積を引抜き時にも流用しており、ht(通常はフーチング厚
    # より薄い)より過大な面積となって応力度を過小評価していた(非安全側)。
    if axial >= 0.0:
        area = punching_shear_area(pile_diameter, footing_height, embedment)
        tau = axial / area / 1000.0  # kN/m2 → N/mm2
    else:
        area = punching_shear_area(
            pile_diameter, PULL_OUT_RESISTANCE_THICKNESS, embedment=0.0
        )
        tau = abs(axial) / area / 1000.0
    # 杭頭結合部では水平力・曲げモーメントが同時に作用し得るため、
    # 荷重の組合せによる τa3 の割増しは行わない(地震時も表の値のまま)。
    tau_a = TAU_A_PUNCHING[fck]

    # 押込み力に対する垂直支圧。許容値は SIGMA_CVA_PILE_HEAD_BEARING
    # (第43回。曲げ圧縮の SIGMA_CA_CONCRETE とは別表で、σck=24 で
    # 7.20 対 8.00 と 11% 小さい)。
    sigma_bearing = max(0.0, axial) / pile_area / 1000.0
    sigma_ba = SIGMA_CVA_PILE_HEAD_BEARING[fck] * increase

    # 水平力・モーメントに対する水平支圧。許容値は垂直支圧と同じ表
    # (Kui_5 で σcva = σcha を確認済み)。
    sigma_ch = horizontal_bearing_stress(
        shear,
        pile_diameter,
        embedment,
        moment=moment if include_moment_in_bearing else 0.0,
    )
    sigma_cha = sigma_ba

    checks = [
        StressCheck("杭頭押抜きせん断応力度", tau, tau_a),
        StressCheck("杭頭支圧応力度", sigma_bearing, sigma_ba),
        StressCheck("杭頭水平支圧応力度", sigma_ch, sigma_cha),
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
