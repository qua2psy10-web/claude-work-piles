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
* 杭頭補強鉄筋の**定着長**(コンクリートへの埋込み) — 実装済み
  (:func:`anchorage_length`)
* 杭頭補強鉄筋の**溶接長**(鋼管杭への溶接定着) — 実装済み
  (:func:`weld_length`。第52回)
* **仮想RC断面の照査** — 実装済み(:func:`virtual_rc_section_check`。
  ``core.section.rc.analyze_circular_section`` をそのまま転用している。
  半径の異なる複数の鉄筋環(鋼管杭の外周溶接鉄筋+中詰め補強鉄筋)にも
  対応(第52回)。H24 で削除済みの鉄筋材質(SD295 等)には未対応。
  net引張軸力でもモーメントが卓越すれば部分圧縮ゾーンを解析可能
  (第51回で対応)だが、
  モーメントを伴わない純引張(M=0, N<0)は引き続き未対応)

.. warning::
   照査式・許容値は原典未照合の項目を含む(docs/VERIFICATION.md 参照)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.loads import LoadCase
from core.models.pile import Footing, PileArrangement
from core.section.checks import StressCheck, rebar_tension_allowable
from core.section.rc import RcStressResult, RebarLayout, analyze_circular_section
from core.standards import (
    EC_CONCRETE,
    PULL_OUT_RESISTANCE_THICKNESS,
    REBAR_NOMINAL_AREA,
    SIGMA_CA_CONCRETE,
    SIGMA_CVA_PILE_HEAD_BEARING,
    SIGMA_SA_REBAR_COMPRESSION,
    STRESS_INCREASE,
    YOUNG_MODULUS_RATIO_RC,
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


@dataclass(frozen=True)
class AnchorageLength:
    """杭頭補強鉄筋の定着長(道示Ⅳ 12.9.3)。"""

    lo: float  # 鉄筋の定着長 Lo (mm)
    required: float  # 必要埋込み長 L = Lo + 10・d (mm)


def anchorage_length(
    sigma_sa: float, tau_oa: float, bar_diameter_mm: float
) -> AnchorageLength:
    """杭頭補強鉄筋の定着長を求める。

        Lo = σsa・Ast / (τoa・u)
        L  ≧ Lo + 10・d

    ``Ast``(鉄筋1本の公称断面積)・``u``(同公称周長)は
    ``REBAR_NOMINAL_AREA`` から算定する(:attr:`core.section.rc.RebarLayout.
    bar_perimeter_mm` と同じ式。表にない呼び径は幾何学的な値で代用)。

    出典: フォーラムエイト UC-1 計算書サンプル Kui_4・Kui_5 の 6.4
    「杭頭補強鉄筋の定着長」(第43・45回)。

    - D22(SD295、σsa=180、τoa=1.6)→ Lo=622.1(計算例 622)、
      L=842.1(計算例 842)
    - D35(SD345、σsa=200、τoa=1.6)→ Lo=1087.0(計算例 1087)、
      L=1437.0(計算例 1437)

    確度C(他社製品の出力からの2点のみ。原典は未照合)。
    """
    if sigma_sa <= 0 or tau_oa <= 0 or bar_diameter_mm <= 0:
        raise ValueError("許容応力度・付着応力度・鉄筋径は正の値である必要があります")
    nominal_area = REBAR_NOMINAL_AREA.get(bar_diameter_mm)
    if nominal_area is None:
        ast = math.pi * bar_diameter_mm**2 / 4.0
        u = math.pi * bar_diameter_mm
    else:
        ast = nominal_area
        u = round(2.0 * math.sqrt(math.pi * ast))
    lo = sigma_sa * ast / (tau_oa * u)
    return AnchorageLength(lo=lo, required=lo + 10.0 * bar_diameter_mm)


def weld_length(
    sigma_sa: float, tau_sa: float, bar_diameter_mm: float, leg_size_mm: float
) -> float:
    """杭頭補強鉄筋(鋼管杭に溶接で定着する場合)の必要すみ肉溶接長を求める。

        Ls = σsa・Ast / (2・0.7・τsa・λ)

    鉄筋が負担する引張力 σsa・Ast を、鉄筋周囲**両側**のすみ肉溶接
    (有効のど厚 0.7・λ)のせん断抵抗で受け持たせる。鋼管杭は場所打ち杭・
    RC/PHC杭のようにコンクリートへの定着(:func:`anchorage_length`)が
    使えないため、方法B の鋼管杭ではこちらを用いる。

    ``Ast``(鉄筋1本の公称断面積)は ``REBAR_NOMINAL_AREA`` から算定する
    (表にない呼び径は幾何学的な値で代用)。``sigma_sa``(鉄筋の許容引張
    応力度)・``tau_sa``(すみ肉溶接の許容せん断応力度)は利用者が与える
    (本ソフトは溶接の許容応力度表を持たない)。

    出典: フォーラムエイト UC-1 計算書サンプル Kui_9 の 6.5
    「杭頭補強鉄筋溶接部のせん断応力度による溶接長」(第52回)。
    D29(Ast=642.4、σsa=200、τsa=94.5)で、脚長λ=6,7,8,9(mm)に対し
    Ls=162,139,121,108(mm)(計算例と一致、4点とも誤差1mm以内)。

    確度C(他社製品の出力から4点のみ。原典は未照合)。
    """
    if sigma_sa <= 0 or tau_sa <= 0 or bar_diameter_mm <= 0 or leg_size_mm <= 0:
        raise ValueError(
            "許容応力度・すみ肉溶接の許容せん断応力度・鉄筋径・脚長は"
            "正の値である必要があります"
        )
    nominal_area = REBAR_NOMINAL_AREA.get(bar_diameter_mm)
    ast = (
        nominal_area
        if nominal_area is not None
        else math.pi * bar_diameter_mm**2 / 4.0
    )
    return round(sigma_sa * ast / (2.0 * 0.7 * tau_sa * leg_size_mm))


@dataclass(frozen=True)
class VirtualRcSectionResult:
    """杭頭の仮想鉄筋コンクリート断面の照査結果。"""

    detail: RcStressResult
    checks: list[StressCheck]

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def virtual_rc_section_check(
    virtual_diameter: float,
    rebar: RebarLayout | list[RebarLayout],
    fck: int,
    rebar_grade: str,
    case: LoadCase,
    axial: float,
    moment: float,
) -> VirtualRcSectionResult:
    """杭頭の仮想鉄筋コンクリート断面を照査する(方法B、道示Ⅳ 12.9.3)。

    ``virtual_diameter`` は仮想RC断面の直径 Do(実際の杭径より大きい。
    利用者が与える設計値)。``rebar`` はその断面に配置する補強鉄筋
    ――単一の :class:`RebarLayout` のほか、**半径の異なる複数の鉄筋環**
    (鋼管杭で「杭外周溶接鉄筋」と「中詰め補強鉄筋」を併用する場合など)を
    ``list[RebarLayout]`` として渡せる(第52回、Kui_9 の6.3で確認)。

    フーチングコンクリート(水中施工ではない)として ``SIGMA_CA_CONCRETE``
    を、鉄筋圧縮側は場所打ち杭の照査と同じ ``SIGMA_SA_REBAR_COMPRESSION``
    (材質によらず一定値)を、鉄筋引張側は :func:`core.section.checks.
    rebar_tension_allowable` を用いる。**H24 で削除済みの鉄筋材質
    (SD295 等)は選択できない**(``rebar_tension_allowable`` が拒む)。

    軸力が負(net で引張)でもモーメントが卓越していれば圧縮縁側に部分圧縮
    ゾーンが残ることがあり、``analyze_circular_section`` はそのケースを
    解ける(第51回、Kui_8 の地震時Nminケースで検証)。モーメントを伴わない
    純引張(M=0, N<0)は引き続き未対応で ``NotImplementedError`` となる。

    出典: フォーラムエイト UC-1 計算書サンプル Kui_4 の 6.3
    「仮想鉄筋コンクリート断面照査」(SD345、Do=1.4m、D35×24本@118、
    かぶり250mm)(第45回)。純軸圧縮(M=0, N=1870.5kN)で
    σc=1.005(計算例 0.99)、地震時(N=3163.5, M=897.5)で
    σc=4.78(計算例 4.72)・σs=54.6(同 53.99)、引張側が生じるケース
    (N=144.1, M=897.5)で σs(引張)=113.31(計算例 113.27、0.04%差)と
    確認した。**残差 1〜1.5% は既知の設計判断**(第31回付近に記録済み:
    換算断面積を Ac+n・As ではなく Ac+(n−1)・As(鉄筋が占めるコンクリートを
    控除する、より安全側の式)で計算しているため)であり、常に本ソフトの
    ほうが厳しい(応力度を大きく見る)側になる。確度C。
    """
    if fck not in SIGMA_CA_CONCRETE:
        raise ValueError(
            f"σck={fck} は許容曲げ圧縮応力度 σca の表(σck = "
            f"{sorted(SIGMA_CA_CONCRETE)})の範囲外です。"
            "適用する設計条件・発注者基準を別途確認してください"
        )
    if fck not in EC_CONCRETE:
        raise ValueError(f"σck={fck} は未対応です")
    increase = STRESS_INCREASE[case.value]
    layers = [rebar] if isinstance(rebar, RebarLayout) else rebar
    fibers = [
        fiber for layer in layers for fiber in layer.fibers(virtual_diameter)
    ]
    detail = analyze_circular_section(
        diameter=virtual_diameter,
        fibers=fibers,
        ec=EC_CONCRETE[fck],
        n_ratio=YOUNG_MODULUS_RATIO_RC,
        axial=axial,
        moment=moment,
    )
    sigma_ca = SIGMA_CA_CONCRETE[fck] * increase
    sigma_sa = rebar_tension_allowable(
        rebar_grade, case, underwater=False, increase=increase
    )
    checks = [
        StressCheck("仮想RC断面コンクリート圧縮応力度", detail.sigma_c, sigma_ca),
        StressCheck("仮想RC断面鉄筋引張応力度", detail.sigma_s_tension, sigma_sa),
        StressCheck(
            "仮想RC断面鉄筋圧縮応力度",
            detail.sigma_s_compression,
            SIGMA_SA_REBAR_COMPRESSION * increase,
        ),
    ]
    return VirtualRcSectionResult(detail=detail, checks=checks)
