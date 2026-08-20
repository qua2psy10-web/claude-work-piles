"""鉄筋コンクリート部材の最小鉄筋量・最大鉄筋量の照査(道示Ⅳ(H24) 7.3)。

**原典(スキャン)で照合済み**(docs/VERIFICATION.md 第23回・第25回)。

応力度やせん断力の照査が「発生値が許容値以下か」を見るのに対し、本モジュールは
**配筋そのものが足りているか・過密でないか**を見る。原典の趣旨は

    コンクリートのひび割れとともに耐力が減じて**急激に破壊することのないように**、
    軸方向引張鉄筋を配置しなければならない(7.3(1)1)i))

であり、**脆性破壊の防止**である。応力度照査を満たしていても配筋が薄すぎると
ひび割れた瞬間に耐力を失う、という状態を捕まえる。

最小鉄筋量
----------
規定は3つあり、それぞれ「満たすものとみなす」代替条件をもつ。

1) **曲げを受ける部材**: 部材の最大抵抗曲げモーメントがひび割れ曲げモーメント
   Mc 以上であること。ただし**部材に生じる曲げモーメントの 1.7 倍が Mc 以下**
   であればこの規定によらなくてよい。

       Mc = Zc(σbt + N/Ac),  σbt = 0.23・σck^(2/3)   …(解 7.3.1)

2) **軸方向力を受ける部材**: 軸方向鉄筋量を、軸方向力に対して計算上必要な
   コンクリート断面積 A′ の **0.8% 以上**とすればよい。

       A′ = max(A′1, A′2)                              …(解 7.3.2)
       A′1 = Na /(0.008・σsa + σca)
       A′2 = Nu /(0.008・σsy + 0.85・σck)

3) **ひび割れ制御**: 部材表面に沿った長さ 1m あたり 500mm² 以上の鉄筋を、
   中心間隔 300mm 以下で配置すればよい。1)・2) の軸方向鉄筋がこの目的を
   兼ねてよい。

最大鉄筋量
----------
* 軸方向引張鉄筋量 ≤ 部材の**有効断面積**(部材断面幅 × 有効高)の 2%
* 軸方向鉄筋量 ≤ 部材の**全断面積**の 6% 程度

.. note::
   1) の「最大抵抗曲げモーメント」(破壊抵抗曲げモーメント)は終局断面解析を
   要するため未実装である。本モジュールは Mc を算定して**ただし書きの
   1.7M ≤ Mc を判定**し、これを満たさない場合は別途の確認が必要である旨を
   注記する(黙って通さない)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models.pile import PileSpec, PileType
from core.section.rc import RebarLayout
from core.section.shear import (
    effective_depth,
    equivalent_square_width,
)
from core.standards import (
    CRACK_MOMENT_MARGIN,
    CRACK_TENSILE_STRENGTH_COEF,
    CRACK_TENSILE_STRENGTH_EXPONENT,
    MAX_TENSILE_REBAR_RATIO,
    MAX_TOTAL_REBAR_RATIO,
    MIN_REBAR_RATIO_AXIAL,
    REBAR_YIELD_POINT,
    SIGMA_CA_REBAR,
    SIGMA_CAG_CONCRETE,
    SURFACE_REBAR_MAX_SPACING,
    SURFACE_REBAR_MIN_AREA_PER_M,
    ULTIMATE_CONCRETE_COEF,
)


@dataclass(frozen=True)
class DetailingCheck:
    """構造細目の照査項目1つ。

    ``kind`` が ``"min"`` なら「``value`` が ``limit`` 以上であること」、
    ``"max"`` なら「``value`` が ``limit`` 以下であること」を要求する。
    どちらの場合も :attr:`ratio` は 1 以下が OK になるよう定義する。
    """

    name: str
    value: float
    limit: float
    unit: str
    kind: str  # "min" | "max"
    note: str = ""

    @property
    def ratio(self) -> float:
        if self.kind == "min":
            return self.limit / self.value if self.value else math.inf
        return self.value / self.limit if self.limit else math.inf

    @property
    def ok(self) -> bool:
        return self.ratio <= 1.0

    @property
    def judgement(self) -> str:
        return "OK" if self.ok else "NG"


def bending_tensile_strength(fck: int | float) -> float:
    """コンクリートの曲げ引張強度 σbt = 0.23・σck^(2/3) (N/mm2)。"""
    return CRACK_TENSILE_STRENGTH_COEF * float(fck) ** CRACK_TENSILE_STRENGTH_EXPONENT


def cracking_moment(diameter: float, fck: int | float, axial: float) -> float:
    """円形断面のひび割れ曲げモーメント Mc (kN·m)(式(解 7.3.1))。

        Mc = Zc(σbt + N/Ac)

    ``axial`` は軸方向力 (kN、圧縮正)。引張のときは Mc を小さくする方向に
    そのまま効く(N/Ac が負になる)。
    """
    zc = math.pi * (diameter * 1000.0) ** 3 / 32.0  # mm3
    ac = math.pi * (diameter * 1000.0) ** 2 / 4.0  # mm2
    sigma_n = axial * 1000.0 / ac  # N/mm2
    return zc * (bending_tensile_strength(fck) + sigma_n) / 1.0e6  # kN·m


def required_concrete_area(
    fck: int,
    rebar_grade: str,
    axial_allowable: float,
    axial_ultimate: float | None = None,
) -> tuple[float, float | None]:
    """軸方向力に対して計算上必要なコンクリート断面積 (mm2)(式(解 7.3.2))。

    Returns
    -------
    (A′1, A′2):
        A′2 は ``axial_ultimate``(レベル2の軸方向圧縮力)を与えた場合のみ。

    Parameters
    ----------
    axial_allowable:
        常時・暴風時及びレベル1地震時に対する照査時の軸方向圧縮力 Na (kN)。
    axial_ultimate:
        レベル2地震時に対する照査時の軸方向圧縮力 Nu (kN)。

    .. note::
       σsa(鉄筋の許容圧縮応力度)・σca(コンクリートの許容軸圧縮応力度)には
       **荷重の組合せによる割増しを乗じていない**。原典の Na は常時・暴風時・
       レベル1地震時を通した1つの値であり、どの割増しを充てるかが一意でない
       ためである。割増しを乗じないほうが分母が小さく A′ が大きくなるので、
       必要鉄筋量を大きめに見る**安全側**の扱いである。
    """
    if rebar_grade not in SIGMA_CA_REBAR:
        raise ValueError(
            f"鉄筋材質 {rebar_grade} の許容圧縮応力度が未定義です。"
            f"対応材質: {sorted(SIGMA_CA_REBAR)}"
        )
    if fck not in SIGMA_CAG_CONCRETE:
        raise ValueError(
            f"σck={fck} の許容軸圧縮応力度が未定義です。"
            f"対応値: {sorted(SIGMA_CAG_CONCRETE)}"
        )
    sigma_sa = SIGMA_CA_REBAR[rebar_grade]
    sigma_ca = SIGMA_CAG_CONCRETE[fck]
    a1 = axial_allowable * 1000.0 / (MIN_REBAR_RATIO_AXIAL * sigma_sa + sigma_ca)

    a2: float | None = None
    if axial_ultimate is not None:
        sigma_sy = REBAR_YIELD_POINT[rebar_grade]
        a2 = axial_ultimate * 1000.0 / (
            MIN_REBAR_RATIO_AXIAL * sigma_sy + ULTIMATE_CONCRETE_COEF * fck
        )
    return a1, a2


@dataclass(frozen=True)
class RebarDetailingResult:
    """軸方向鉄筋量の構造細目照査の結果。"""

    provided_area: float  # 配置された軸方向鉄筋量 (mm2)
    tensile_area: float  # 図心より引張側の軸方向鉄筋量 (mm2)
    gross_area: float  # 部材の全断面積 (mm2)
    effective_area: float  # 部材の有効断面積 b・d (mm2)
    a1: float  # A′1 (mm2)
    a2: float | None  # A′2 (mm2)
    required_min_area: float  # 0.008・A′ (mm2)
    cracking_moment: float  # Mc (kN·m)
    sigma_bt: float  # 曲げ引張強度 (N/mm2)
    design_moment: float  # 照査に用いた曲げモーメント M (kN·m)
    bar_spacing: float  # 鉄筋の中心間隔 (mm)
    surface_area_per_m: float  # 部材表面 1m あたりの鉄筋断面積 (mm2/m)
    checks: list[DetailingCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def required_concrete_area(self) -> float:
        """A′ = max(A′1, A′2) (mm2)。"""
        return self.a1 if self.a2 is None else max(self.a1, self.a2)

    @property
    def crack_moment_exempt(self) -> bool:
        """曲げの最小鉄筋量の規定によらなくてよいか(1.7M ≤ Mc)。"""
        return CRACK_MOMENT_MARGIN * abs(self.design_moment) <= self.cracking_moment

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def check_rebar_detailing(
    pile: PileSpec,
    rebar: RebarLayout,
    fck: int,
    rebar_grade: str,
    axial_allowable: float,
    moment: float,
    axial_ultimate: float | None = None,
) -> RebarDetailingResult:
    """軸方向鉄筋量の最小・最大の照査(道示Ⅳ 7.3)。

    Parameters
    ----------
    axial_allowable:
        常時・暴風時及びレベル1地震時を通した軸方向圧縮力の**最大値** Na (kN)。
    moment:
        部材に生じる曲げモーメントの最大値 (kN·m)。ひび割れ曲げモーメントの
        ただし書き(1.7M ≤ Mc)の判定に用いる。
    axial_ultimate:
        レベル2地震時の軸方向圧縮力 Nu (kN)。省略すると A′2 を算定しない。

    Raises
    ------
    ValueError
        場所打ち杭以外の場合(円形RC断面の規定を用いるため)。
    """
    if pile.pile_type != PileType.CAST_IN_PLACE:
        raise ValueError(
            f"{pile.pile_type.value}の鉄筋量の照査は未実装です"
            "(円形RC断面の規定を用いるため場所打ち杭のみ対応)"
        )

    d_mm = pile.diameter * 1000.0
    gross = math.pi * d_mm**2 / 4.0
    b = equivalent_square_width(pile.diameter)
    d = effective_depth(pile.diameter, rebar)
    effective = (b * 1000.0) * (d * 1000.0)

    # mm2 に換算(RebarLayout.bar_area は m2)
    bar_area_mm2 = rebar.bar_area * 1.0e6
    provided = rebar.count * bar_area_mm2
    tol = 1.0e-9 * max(rebar.radius(pile.diameter), 1.0)
    tensile = bar_area_mm2 * sum(
        1 for y in rebar.positions(pile.diameter) if y < -tol
    )

    a1, a2 = required_concrete_area(
        fck, rebar_grade, axial_allowable, axial_ultimate
    )
    required_concrete = a1 if a2 is None else max(a1, a2)
    min_area = MIN_REBAR_RATIO_AXIAL * required_concrete

    mc = cracking_moment(pile.diameter, fck, axial_allowable)
    sigma_bt = bending_tensile_strength(fck)

    # 鉄筋の中心間隔は鉄筋円の円周を本数で割る。表面 1m あたりの断面積は
    # 部材表面(πD)で割る(ひび割れ制御は表面での要求のため)。
    radius_mm = rebar.radius(pile.diameter) * 1000.0
    spacing = 2.0 * math.pi * radius_mm / rebar.count
    surface_per_m = provided / (math.pi * pile.diameter)  # mm2/m

    checks = [
        DetailingCheck(
            name="最小鉄筋量(軸方向力)",
            value=provided,
            limit=min_area,
            unit="mm²",
            kind="min",
            note=(
                f"計算上必要なコンクリート断面積 A′ = "
                f"{required_concrete:,.0f} mm² の "
                f"{MIN_REBAR_RATIO_AXIAL * 100:g}%(7.3(1)2)iii))"
            ),
        ),
        DetailingCheck(
            name="最大鉄筋量(引張鉄筋/有効断面積)",
            value=100.0 * tensile / effective,
            limit=MAX_TENSILE_REBAR_RATIO * 100.0,
            unit="%",
            kind="max",
            note="有効断面積 = 部材断面幅 b × 有効高 d(7.3(2))",
        ),
        DetailingCheck(
            name="最大鉄筋量(全鉄筋/全断面積)",
            value=100.0 * provided / gross,
            limit=MAX_TOTAL_REBAR_RATIO * 100.0,
            unit="%",
            kind="max",
            note="施工性の観点からの目安(7.3(2))",
        ),
        DetailingCheck(
            name="ひび割れ制御鉄筋(表面1mあたり)",
            value=surface_per_m,
            limit=SURFACE_REBAR_MIN_AREA_PER_M,
            unit="mm²/m",
            kind="min",
            note="7.3(1)3)ii)。軸方向鉄筋がこの目的を兼ねてよい",
        ),
        DetailingCheck(
            name="ひび割れ制御鉄筋(中心間隔)",
            value=spacing,
            limit=SURFACE_REBAR_MAX_SPACING,
            unit="mm",
            kind="max",
            note="7.3(1)3)ii)",
        ),
    ]

    notes: list[str] = []
    if axial_ultimate is None:
        notes.append(
            "レベル2地震時の軸方向圧縮力 Nu が未入力のため A′2 を算定して"
            "いない。**A′2 が A′1 を上回る場合、必要鉄筋量はここに示す値より"
            "大きくなる**(通常は A′1 が支配する)。"
        )

    result = RebarDetailingResult(
        provided_area=provided,
        tensile_area=tensile,
        gross_area=gross,
        effective_area=effective,
        a1=a1,
        a2=a2,
        required_min_area=min_area,
        cracking_moment=mc,
        sigma_bt=sigma_bt,
        design_moment=moment,
        bar_spacing=spacing,
        surface_area_per_m=surface_per_m,
        checks=checks,
        notes=notes,
    )
    if result.crack_moment_exempt:
        notes.append(
            f"曲げの最小鉄筋量(7.3(1)1))は、部材に生じる曲げモーメントの "
            f"{CRACK_MOMENT_MARGIN:g} 倍({CRACK_MOMENT_MARGIN * abs(moment):,.0f}"
            f" kN·m)がひび割れ曲げモーメント Mc = {mc:,.0f} kN·m 以下のため、"
            "ただし書きにより適用しない。"
        )
    else:
        notes.append(
            f"⚠ 曲げの最小鉄筋量(7.3(1)1))について、**最大抵抗曲げモーメントが"
            f"ひび割れ曲げモーメント Mc = {mc:,.0f} kN·m 以上であることの確認が"
            f"別途必要**である(部材に生じる曲げモーメントの {CRACK_MOMENT_MARGIN:g} 倍 "
            f"= {CRACK_MOMENT_MARGIN * abs(moment):,.0f} kN·m が Mc を超えるため、"
            "ただし書きは適用できない)。最大抵抗曲げモーメントの算定は未実装。"
        )
    return result
