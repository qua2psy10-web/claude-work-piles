"""杭体のせん断力に対する照査。

* **円形RC断面(場所打ち杭)** — 道示Ⅳ(H24) 5.1.3 / 5.2.3。以下に詳述。
* **鋼管断面(鋼管杭・鋼管ソイルセメント杭)** — :func:`check_steel_pipe_shear`。

円形RC断面(道示Ⅳ(H24) 5.1.3)

**原典(スキャン)で照合済み**(docs/VERIFICATION.md 第23回)。

照査式
------
    τm = Sh /(b・d)                                        …(5.1.1)
    Sh = S −(M/d)(tanβ + tanγ)                            …(5.1.2)

杭は等断面なので有効高は変化せず β = γ = 0、したがって **Sh = S** となる。

* コンクリートのみでせん断力を負担する場合  τm ≤ τa1
* 斜引張鉄筋と共同して負担する場合          τm ≤ τa2

τa1 を超える場合は、式(5.1.3) による断面積以上の斜引張鉄筋を配置する。

    Aw = 1.15・Sh' ・s /(σsa・d・(sinθ + cosθ))          …(5.1.3)
    ΣSh' = Sh − Sca,  Sca = τa1・b・d                     …(5.1.4)

円形断面の b と d(図-解4.2.2)
-------------------------------
原典の規定をそのまま実装している:

    円形断面では、円形断面を**面積の等しい正方形断面に置換え**、置換えられた
    正方形断面の圧縮縁から、**引張側の 1/4 部分の鉄筋の重心位置**までの距離を
    有効高とする。円形断面の幅は、面積の等しい正方形断面の幅とする。

したがって径 D の杭では

    b = √(π D² / 4) = (√π / 2)・D ≒ 0.8862 D
    d = b/2 −(引張側 1/4 部分にある鉄筋の y 座標の平均)

「引張側の 1/4 部分」は、断面を中心から見て引張縁を中心とする 90° の扇形
(引張縁方向から ±45°)とした。

τa1 の補正
-----------
τa1 には有効高 d・軸方向引張鉄筋比 pt・軸方向圧縮力の補正を乗じる
(表-4.2.2、表-4.2.3、式(4.2.1))。表の中間値は線形補間してよい。

.. note::
   地震の影響を考慮する場合は、τa1 に割増係数 1.50 を乗じる代わりに
   表-5.2.1 の τc を用いる(原典 4.2 の解説)。本実装もそれに従う。
"""  # noqa: D205
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models.loads import LoadCase
from core.models.pile import PileSpec, PileType
from core.section.rc import RebarLayout, StirrupLayout
from core.capacity.section import (
    CORROSION_ALLOWANCE_MM,
    corroded_tube,
    hollow_circle,
)
from core.standards import (
    PRECAST_CONCRETE_ALLOWABLE,
    REBAR_YIELD_POINT,
    SHEAR_CC_FOUNDATION,
    SHEAR_CE_BY_DEPTH,
    SHEAR_CN_MAX,
    SHEAR_CN_MIN,
    SHEAR_CPT_BY_RATIO,
    SHEAR_REBAR_YIELD_CAP,
    STRESS_INCREASE,
    TAU_A1_CONCRETE,
    TAU_MAX_CONCRETE,
    TAU_A2_CONCRETE,
    TAU_A_STEEL,
    TAU_C_CONCRETE,
    STEEL_ALLOWABLE_MAX_THICKNESS,
)

# 引張側 1/4 部分の境界(引張縁方向から ±45°)
_QUARTER_COS = math.cos(math.pi / 4.0)


def equivalent_square_width(diameter: float) -> float:
    """円形断面と面積の等しい正方形断面の幅 b (m)(図-解4.2.2)。"""
    return math.sqrt(math.pi * diameter**2 / 4.0)


def tension_quarter_positions(
    diameter: float, rebar: RebarLayout
) -> list[float]:
    """引張側 1/4 部分にある鉄筋の y 座標 (m)。

    y は断面中心を原点、圧縮縁側を正にとる(:mod:`core.section.rc` と同じ)。
    引張縁は y 負側なので、y ≤ −r・cos45° の鉄筋が該当する。
    """
    r = rebar.radius(diameter)
    threshold = -r * _QUARTER_COS
    # 浮動小数の丸めで境界上の鉄筋が落ちないよう、わずかに緩める
    tol = 1.0e-12 * max(r, 1.0)
    return [y for y in rebar.positions(diameter) if y <= threshold + tol]


def effective_depth(diameter: float, rebar: RebarLayout) -> float:
    """円形断面の有効高 d (m)(図-解4.2.2)。

    面積の等しい正方形断面の圧縮縁(y = +b/2)から、引張側 1/4 部分の鉄筋の
    重心位置までの距離。
    """
    quarter = tension_quarter_positions(diameter, rebar)
    if not quarter:
        raise ValueError(
            "引張側 1/4 部分に鉄筋がありません(本数・かぶりを確認してください)"
        )
    centroid = sum(quarter) / len(quarter)
    return equivalent_square_width(diameter) / 2.0 - centroid


def tensile_rebar_ratio(
    diameter: float, rebar: RebarLayout, b: float, d: float
) -> float:
    """軸方向引張鉄筋比 pt (%)。

    原典は中立軸より引張側の鉄筋量から求めることを規定しているが、
    「計算の簡略化のため**断面の図心位置から引張側**にある軸方向鉄筋の
    断面積の総和から求めてもよい」とされている。円形断面では図心が中心なので
    y < 0 の鉄筋が対象となる。本実装はこの簡略法を用いる。

    中立軸は一般に図心より圧縮側にあるため、簡略法は引張側の鉄筋を
    少なく数えることになり、pt が小さめ → cpt が小さめ → **安全側**である。

    .. note::
       本数が偶数のとき図心位置(y = 0)にちょうど乗る鉄筋が生じる。これは
       引張側ではないので数えない。浮動小数の丸めで符号が揺れるため、
       明示的に許容差を設けて判定を決定的にしている(数えないほうが pt が
       小さくなり安全側)。
    """
    tol = 1.0e-9 * max(rebar.radius(diameter), 1.0)
    area = sum(
        rebar.bar_area for y in rebar.positions(diameter) if y < -tol
    )
    return 100.0 * area / (b * d)


def _interpolate(
    table: tuple[tuple[float, float], ...],
    x: float,
    extrapolate_low: bool = False,
) -> float:
    """表の線形補間。範囲外は端の値で頭打ちにする。

    ``extrapolate_low`` を真にすると、表の**下限より小さい** x に対して
    最初の2点の勾配で線形外挿する(底版の cpt。
    :func:`core.section.footing.cpt_factor_footing` の説明を参照)。
    外挿値が負になる場合は 0 で下打ちする。
    """
    if x <= table[0][0]:
        if extrapolate_low and len(table) >= 2:
            (x0, y0), (x1, y1) = table[0], table[1]
            slope = (y1 - y0) / (x1 - x0)
            return max(0.0, y0 + slope * (x - x0))
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError("到達しない")  # pragma: no cover


def ce_factor(d: float) -> float:
    """部材断面の有効高 d に関する補正係数 ce(表-4.2.2)。d は m。"""
    return _interpolate(SHEAR_CE_BY_DEPTH, d * 1000.0)


def cpt_factor(pt: float) -> float:
    """軸方向引張鉄筋比 pt (%) に関する補正係数 cpt(表-4.2.3)。"""
    return _interpolate(SHEAR_CPT_BY_RATIO, pt)


def cn_factor(diameter: float, axial: float, moment: float) -> float:
    """軸方向圧縮力による補正係数 cN = 1 + M0/M(式(4.2.1))。

        M0 =(N / Ac)(Ic / y)

    円形断面では Ic/y = π D³/32(= 断面係数 Zc)である。1 ≤ cN ≤ 2。
    引張軸力・M = 0 のときは補正しない(cN = 1)。
    """
    if axial <= 0.0 or moment == 0.0:
        return SHEAR_CN_MIN
    area = math.pi * diameter**2 / 4.0
    section_modulus = math.pi * diameter**3 / 32.0
    m0 = axial / area * section_modulus  # kN·m
    return min(max(1.0 + m0 / abs(moment), SHEAR_CN_MIN), SHEAR_CN_MAX)


@dataclass(frozen=True)
class ShearCapacity:
    """レベル2地震時の断面のせん断耐力(道示Ⅳ(H24) 5.2.3、式(5.2.1))。

        Ps = Sc + Ss
        Sc = cc・ce・cpt・cN・τc・b・d
        Ss = Aw・σsy・d・(sinθ + cosθ) /(1.15 s)

    単位はいずれも kN。
    """

    sc: float  # コンクリートの負担するせん断耐力 (kN)
    ss: float  # 斜引張鉄筋の負担するせん断耐力 (kN)
    width: float  # b (m)
    effective_depth: float  # d (m)
    pt: float  # 軸方向引張鉄筋比 (%)
    cc: float
    ce: float
    cpt: float
    cn: float
    tau_c: float  # 表-5.2.1 の値 (N/mm2)
    sigma_sy: float | None  # 用いた斜引張鉄筋の降伏点 (N/mm2、345 で頭打ち)
    tau_max: float = 0.0  # 表-4.3.2 の平均せん断応力度の最大値 (N/mm2)

    @property
    def total(self) -> float:
        """斜引張破壊に対する耐力 Sus = Ps = Sc + Ss (kN)。"""
        return self.sc + self.ss

    @property
    def web_crushing_capacity(self) -> float:
        """ウェブコンクリートの圧壊に対する耐力 Suc = τmax・bw・d (kN)。

        RC部材なので PC鋼材の分力 Sp = 0 とする(道示Ⅲ 4.3.4(2))。
        """
        return (
            self.tau_max
            * (self.width * 1000.0)
            * (self.effective_depth * 1000.0)
            / 1000.0
        )

    @property
    def governing_capacity(self) -> float:
        """終局時に満たすべき耐力 min(Sus, Suc) (kN)。

        斜引張鉄筋を増やしても Suc は超えられないため、実質的な上限は
        この小さいほうである。
        """
        return min(self.total, self.web_crushing_capacity)

    @property
    def web_crushing_governs(self) -> bool:
        """斜め圧縮破壊が支配しているか(鉄筋を増やしても耐力が伸びない)。"""
        return self.web_crushing_capacity < self.total


def shear_capacity_level2(
    pile: PileSpec,
    rebar: RebarLayout,
    fck: int,
    axial: float,
    moment: float,
    stirrup: StirrupLayout | None = None,
    rebar_grade: str = "SD345",
) -> ShearCapacity:
    """レベル2地震時のせん断耐力 Ps(道示Ⅳ 5.2.3)。

    ``cc`` は荷重の正負交番作用の補正係数で、**橋台及び基礎については 1**
    としてよい(原典 5.2.3)。``cN`` は「杭のように軸力の作用が明確な部材に
    おいてのみ考慮する」とされているため、杭では考慮する。

    ``σsy`` は斜引張鉄筋の降伏点だが、**上限を 345 N/mm² とする**
    (SD390・SD490 を用いる場合の適用性が未検証のため)。

    ``stirrup`` を与えない場合は Ss = 0(コンクリートのみ)となる。
    """
    if pile.pile_type != PileType.CAST_IN_PLACE:
        raise ValueError(
            f"{pile.pile_type.value}のせん断耐力は未実装です"
            "(円形RC断面の規定を用いるため場所打ち杭のみ対応)"
        )
    if fck not in TAU_C_CONCRETE:
        raise ValueError(
            f"σck={fck} の τc が未定義です。対応値: {sorted(TAU_C_CONCRETE)}"
        )
    b = equivalent_square_width(pile.diameter)
    d = effective_depth(pile.diameter, rebar)
    pt = tensile_rebar_ratio(pile.diameter, rebar, b, d)
    ce = ce_factor(d)
    cpt = cpt_factor(pt)
    cn = cn_factor(pile.diameter, axial, moment)
    tau_c = TAU_C_CONCRETE[fck]

    # N → kN。b・d は mm に換算する
    sc = (
        SHEAR_CC_FOUNDATION * ce * cpt * cn * tau_c
        * (b * 1000.0) * (d * 1000.0) / 1000.0
    )

    ss = 0.0
    sigma_sy = None
    if stirrup is not None:
        stirrup.validated()
        if rebar_grade not in REBAR_YIELD_POINT:
            raise ValueError(
                f"鉄筋材質 {rebar_grade} の降伏点が未定義です。"
                f"対応材質: {sorted(REBAR_YIELD_POINT)}"
            )
        sigma_sy = min(REBAR_YIELD_POINT[rebar_grade], SHEAR_REBAR_YIELD_CAP)
        theta = math.radians(stirrup.angle_deg)
        ss = (
            stirrup.area * sigma_sy * (d * 1000.0)
            * (math.sin(theta) + math.cos(theta))
            / (1.15 * stirrup.spacing_mm)
            / 1000.0
        )
    if fck not in TAU_MAX_CONCRETE:
        raise ValueError(
            f"σck={fck} の τmax(表-4.3.2)が未定義です。"
            f"対応値: {sorted(TAU_MAX_CONCRETE)}"
        )
    return ShearCapacity(
        tau_max=TAU_MAX_CONCRETE[fck],
        sc=sc, ss=ss, width=b, effective_depth=d, pt=pt,
        cc=SHEAR_CC_FOUNDATION, ce=ce, cpt=cpt, cn=cn,
        tau_c=tau_c, sigma_sy=sigma_sy,
    )


@dataclass(frozen=True)
class ShearCheck:
    """1項目の照査結果(応力度の単位は N/mm2)。"""

    name: str
    stress: float
    allowable: float

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
class StirrupCheck:
    """斜引張鉄筋量の照査(必要量 vs 配置量)。単位は mm2/mm。"""

    required: float  # 必要量 Aw/s(式(5.1.3))
    provided: float  # 配置量 Aw/s
    sigma_sa: float  # 用いた許容引張応力度 (N/mm2)
    angle_deg: float

    @property
    def ratio(self) -> float:
        return self.required / self.provided if self.provided else math.inf

    @property
    def ok(self) -> bool:
        return self.ratio <= 1.0

    @property
    def judgement(self) -> str:
        return "OK" if self.ok else "NG"


@dataclass(frozen=True)
class ShearResult:
    """杭体1断面のせん断照査の結果。

    .. note::
       :attr:`checks` に入れているのは **τm ≤ τa2**(斜引張鉄筋と共同して
       負担する場合)だけである。杭には帯鉄筋が必ず配置されるため、これが
       合否を決める上限となる。τa1 との比較は :attr:`needs_stirrup` と
       :meth:`required_stirrup_ratio` で示す — τa1 を超えることは NG では
       なく「斜引張鉄筋がどれだけ必要か」を意味する。
       τa2 を超えた場合は鉄筋を増やしても解決せず、断面を大きくする等の
       対応が必要である(原典 5.1.3 の解説)。
    """

    depth: float  # 杭頭からの深さ (m)
    shear: float  # 作用せん断力 S (kN)
    moment: float  # 同断面の曲げモーメント (kN·m)
    axial: float  # 軸方向圧縮力 (kN)
    width: float  # 換算幅 b (m)
    effective_depth: float  # 有効高 d (m)
    tau_m: float  # 平均せん断応力度 (N/mm2)
    pt: float  # 軸方向引張鉄筋比 (%)
    ce: float
    cpt: float
    cn: float
    tau_a1: float  # 補正・割増後の許容せん断応力度 (N/mm2)
    tau_a2: float  # 割増後の許容せん断応力度 (N/mm2)
    seismic: bool  # 地震時(τc を用いたか)
    checks: list[ShearCheck] = field(default_factory=list)
    # 斜引張鉄筋の許容引張応力度 (N/mm2)。表-4.3.1 の「上記以外」の区分
    # (軸方向鉄筋とは異なる)。材質が不明な場合は None。
    stirrup_sigma_sa: float | None = None
    # 帯鉄筋を入力した場合の必要量 vs 配置量の照査
    stirrup: StirrupCheck | None = None

    @property
    def required_aw_per_spacing(self) -> float | None:
        """必要な斜引張鉄筋量 Aw/s (mm2/mm)。帯鉄筋(θ = 90°)として。"""
        if self.stirrup is not None:
            return self.stirrup.required
        if self.stirrup_sigma_sa is None:
            return None
        return self.required_stirrup_ratio(self.stirrup_sigma_sa)

    @property
    def needs_stirrup(self) -> bool:
        """コンクリートのみでは負担できず、斜引張鉄筋が必要か。"""
        return self.tau_m > self.tau_a1

    @property
    def all_ok(self) -> bool:
        if self.stirrup is not None and not self.stirrup.ok:
            return False
        return all(c.ok for c in self.checks)

    @property
    def concrete_shear_capacity(self) -> float:
        """コンクリートが負担するせん断力 Sca = τa1・b・d (kN)。"""
        return self.tau_a1 * (self.width * 1000.0) * (self.effective_depth * 1000.0) / 1000.0

    def required_stirrup_ratio(self, sigma_sa: float, angle_deg: float = 90.0) -> float:
        """必要な斜引張鉄筋量を間隔で割った値 Aw/s (mm2/mm)。

            Aw = 1.15・Sh' ・s /(σsa・d・(sinθ + cosθ))     …(5.1.3)
            Sh' = Sh − Sca

        間隔 s を決めれば必要断面積 Aw = (この値)× s で得られる。
        斜引張鉄筋が不要な場合は 0 を返す。

        Parameters
        ----------
        sigma_sa:
            斜引張鉄筋の許容引張応力度 (N/mm2)。軸方向鉄筋とは区分が異なる
            (表-4.3.1 の「上記以外」)ことに注意。
        angle_deg:
            斜引張鉄筋が部材軸方向となす角度 θ。帯鉄筋は 90°。
        """
        excess = abs(self.shear) - self.concrete_shear_capacity  # kN
        if excess <= 0.0:
            return 0.0
        theta = math.radians(angle_deg)
        d_mm = self.effective_depth * 1000.0
        return (
            1.15 * (excess * 1000.0)
            / (sigma_sa * d_mm * (math.sin(theta) + math.cos(theta)))
        )


def check_shear(
    pile: PileSpec,
    rebar: RebarLayout,
    fck: int,
    case: LoadCase,
    depth: float,
    shear: float,
    moment: float,
    axial: float,
    rebar_grade: str | None = None,
    stirrup: StirrupLayout | None = None,
) -> ShearResult:
    """場所打ち杭1断面のせん断照査(道示Ⅳ 5.1.3)。

    Parameters
    ----------
    shear:
        作用せん断力 S (kN)。杭は等断面なので Sh = S(式(5.1.2) の
        有効高の変化の項は 0)。
    moment:
        同じ断面の曲げモーメント (kN·m)。cN の算定に用いる。
    axial:
        軸方向圧縮力 (kN、圧縮正)。cN の算定に用いる。
    rebar_grade:
        鉄筋材質。与えると必要な斜引張鉄筋量まで算定する。斜引張鉄筋の
        許容引張応力度は**軸方向鉄筋とは区分が異なる**(表-4.3.1 の
        「上記以外」)ことに注意。

    Raises
    ------
    ValueError
        場所打ち杭以外、または σck が表の範囲外の場合。
    """
    if pile.pile_type != PileType.CAST_IN_PLACE:
        raise ValueError(
            f"{pile.pile_type.value}のせん断照査は未実装です"
            "(円形RC断面の規定を用いるため場所打ち杭のみ対応)"
        )
    if fck not in TAU_A1_CONCRETE:
        raise ValueError(
            f"σck={fck} の許容せん断応力度が未定義です。"
            f"対応値: {sorted(TAU_A1_CONCRETE)}"
        )

    b = equivalent_square_width(pile.diameter)
    d = effective_depth(pile.diameter, rebar)
    pt = tensile_rebar_ratio(pile.diameter, rebar, b, d)
    ce = ce_factor(d)
    cpt = cpt_factor(pt)
    cn = cn_factor(pile.diameter, axial, moment)

    # τm = Sh /(b・d)。kN と m を N と mm に換算する
    tau_m = abs(shear) * 1000.0 / (b * 1000.0 * d * 1000.0)

    increase = STRESS_INCREASE[case.value]
    if case.is_seismic:
        # 地震時は τa1 × 1.50 の代わりに τc を用いる(原典 4.2 の解説)
        base = TAU_C_CONCRETE[fck]
    else:
        base = TAU_A1_CONCRETE[fck] * increase
    tau_a1 = ce * cpt * cn * base
    tau_a2 = TAU_A2_CONCRETE[fck] * increase

    checks = [
        ShearCheck("平均せん断応力度(斜引張鉄筋と共同)", tau_m, tau_a2),
    ]
    stirrup_sigma_sa = None
    if rebar_grade is not None:
        # 循環インポートを避けるため遅延インポート
        from core.section.checks import rebar_tension_allowable

        stirrup_sigma_sa = rebar_tension_allowable(
            rebar_grade, case, underwater=True, increase=increase,
            axial_rebar=False,
        )

    stirrup_check = None
    if stirrup is not None:
        if stirrup_sigma_sa is None:
            raise ValueError(
                "帯鉄筋の照査には鉄筋材質(rebar_grade)の指定が必要です"
            )
        stirrup.validated()
        result = ShearResult(
            depth=depth, shear=shear, moment=moment, axial=axial,
            width=b, effective_depth=d, tau_m=tau_m, pt=pt,
            ce=ce, cpt=cpt, cn=cn, tau_a1=tau_a1, tau_a2=tau_a2,
            seismic=case.is_seismic, checks=checks,
            stirrup_sigma_sa=stirrup_sigma_sa,
        )
        stirrup_check = StirrupCheck(
            required=result.required_stirrup_ratio(
                stirrup_sigma_sa, stirrup.angle_deg
            ),
            provided=stirrup.aw_per_spacing,
            sigma_sa=stirrup_sigma_sa,
            angle_deg=stirrup.angle_deg,
        )
    return ShearResult(
        depth=depth,
        shear=shear,
        moment=moment,
        axial=axial,
        width=b,
        effective_depth=d,
        tau_m=tau_m,
        pt=pt,
        ce=ce,
        cpt=cpt,
        cn=cn,
        tau_a1=tau_a1,
        tau_a2=tau_a2,
        seismic=case.is_seismic,
        checks=checks,
        stirrup_sigma_sa=stirrup_sigma_sa,
        stirrup=stirrup_check,
    )


# ---------------------------------------------------------------------------
# 鋼管断面(鋼管杭・鋼管ソイルセメント杭)
# ---------------------------------------------------------------------------

# 薄肉円管の最大せん断応力度は τmax = 2V/A。
#
#   A = 2πrt、I = πr³t、中立軸の断面一次モーメント Q = 2r²t、
#   せん断流の幅 = 2t とすると
#       τmax = V・Q /(I・2t) = V /(πrt) = 2V / A
#
# これは**材料力学から厳密に導かれる**関係であり、推定ではない。
# 中空円形断面の厳密解(薄肉近似を用いない τ = VQ/(I・b))と比べると、
# 実用的な杭の板厚(D=600〜2000mm、t=9〜25mm)では差は 0.04% 以内で、
# かつ 2V/A のほうがわずかに**大きい**(安全側)。
STEEL_PIPE_SHEAR_FACTOR = 2.0


@dataclass(frozen=True)
class SteelPipeShearResult:
    """鋼管断面のせん断照査の結果。"""

    depth: float  # 杭頭からの深さ (m)
    shear: float  # 作用せん断力 (kN)
    area: float  # 腐食代控除後の断面積 (m2)
    thickness: float  # 腐食代控除後の板厚 (mm)
    tau_max: float  # 最大せん断応力度 τmax = 2V/A (N/mm2)
    tau_mean: float  # 平均せん断応力度 V/A (N/mm2、参考)
    allowable: float  # 許容せん断応力度(割増後) (N/mm2)
    checks: list[ShearCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def check_steel_pipe_shear(
    pile: PileSpec,
    steel_grade: str,
    case: LoadCase,
    depth: float,
    shear: float,
    corrosion_mm: float = CORROSION_ALLOWANCE_MM,
) -> SteelPipeShearResult:
    """鋼管杭・鋼管ソイルセメント杭のせん断照査。

    最大せん断応力度は薄肉円管の材料力学から

        τmax = 2V / A

    で求まる(:data:`STEEL_PIPE_SHEAR_FACTOR` の解説を参照)。許容せん断
    応力度は道示Ⅳ 表-4.4.1 の母材部の値(SKK400 = 80、SKK490 = 105)に
    荷重の組合せに応じた割増しを乗じる。

    .. warning::
       表-4.4.1 の**鋼管のせん断は「座屈を考慮しない場合」の値**である
       (原典の注記)。局部座屈が懸念される薄肉断面では別途の検討が要る。
       また同表は**板厚 40mm 以下**に適用するもので、これを超える場合は
       鋼橋編による。いずれも該当時に注記を出す。

    .. note::
       鋼管ソイルセメント杭はソイルセメント部を無視し、鋼管のみでせん断力を
       負担するものとして照査する(応力度照査と同じ扱い)。
    """
    if pile.pile_type not in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        raise ValueError(
            f"{pile.pile_type.value}に鋼管断面のせん断照査は適用できません"
        )
    if pile.wall_thickness is None:
        raise ValueError("鋼管杭は板厚 wall_thickness の入力が必要です")
    if steel_grade not in TAU_A_STEEL:
        raise ValueError(
            f"鋼材材質 {steel_grade} の許容せん断応力度が未定義です。"
            f"対応材質: {sorted(TAU_A_STEEL)}"
        )

    # 腐食しろは外面から控除する(外径が 2c 減り、内径は変わらない)
    outer, t_m = corroded_tube(pile.diameter, pile.wall_thickness, corrosion_mm)
    t = t_m * 1000.0
    area, _ = hollow_circle(outer, t_m)  # m2

    # kN, m2 → N/mm2 は 1/1000
    tau_mean = abs(shear) / area / 1000.0
    tau_max = STEEL_PIPE_SHEAR_FACTOR * tau_mean
    allowable = TAU_A_STEEL[steel_grade] * STRESS_INCREASE[case.value]

    notes: list[str] = []
    if pile.wall_thickness > STEEL_ALLOWABLE_MAX_THICKNESS:
        notes.append(
            f"板厚 {pile.wall_thickness:.1f} mm は表-4.4.1 の適用範囲"
            f"({STEEL_ALLOWABLE_MAX_THICKNESS:g} mm 以下)を超えている。"
            "鋼橋編の許容応力度を確認すること。"
        )
    notes.append(
        "許容せん断応力度は**座屈を考慮しない場合**の値である(表-4.4.1 の注記)。"
        "局部座屈が懸念される薄肉断面では別途の検討が必要。"
    )

    return SteelPipeShearResult(
        depth=depth,
        shear=shear,
        area=area,
        thickness=t,
        tau_max=tau_max,
        tau_mean=tau_mean,
        allowable=allowable,
        checks=[ShearCheck("最大せん断応力度 τmax = 2V/A", tau_max, allowable)],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 既製コンクリート杭(PHC杭)
# ---------------------------------------------------------------------------
#
# 出典: フォーラムエイト UC-1 計算書サンプル Kui_10 の 3.3「杭体応力度」
# (第2断面・PHC杭、φ600・t=90・B種、σce=8.0)の「せん断応力度の照査」
# (第55回)。
#
#     τ = S / Ae
#     CN = 1 + Mo/M(1.0 ≦ CN ≦ 2.0)
#     Mo = (σce + N/Ae)・Ie/y
#     τa = 0.85・CN(割増後は τa に STRESS_INCREASE を乗じる)
#
# Ae(換算断面積)・Ie(換算断面二次モーメント。Ze = Ie/y と整合)は、曲げ
# 応力度照査(:func:`core.section.checks.phc_effective_section`)と**同じ
# 値**を用いる。τa の基本値 0.85 N/mm² は
# ``PRECAST_CONCRETE_ALLOWABLE["PHC杭"].shear`` として既に定義済みだった
# (以前は未使用)。M=0(常時、曲げがない)のケースでは CN は 1+Mo/M が
# 発散し上限 2.0 に張り付くため、実装でもその場合は CN=2.0 とする
# (Kui_10 の常時ケースで CN=2.000 と明記されていることと整合)。


def phc_shear_correction_factor(
    sigma_ce: float, axial: float, area: float, inertia: float, y: float, moment: float
) -> float:
    """PHC杭のせん断照査における軸方向圧縮力の補正係数 CN。

        CN = 1 + Mo/M,  Mo = (σce + N/Ae)・Ie/y   (1.0 ≦ CN ≦ 2.0)

    ``area``(Ae)・``inertia``(Ie)は SI 単位(m2・m4)、``y`` は m。
    ``moment`` = 0(常時など曲げがないケース)では 1+Mo/M が発散するため、
    上限の 2.0 を返す(Kui_10 の常時ケースと整合)。
    """
    if area <= 0.0 or inertia <= 0.0 or y <= 0.0:
        raise ValueError("断面積・断面二次モーメント・y は正の値である必要があります")
    if moment == 0.0:
        return SHEAR_CN_MAX
    sigma_axial = axial / area / 1000.0  # N/mm2(圧縮正)
    mo = (sigma_ce + sigma_axial) * 1000.0 * inertia / y  # kN・m
    return min(max(1.0 + mo / abs(moment), SHEAR_CN_MIN), SHEAR_CN_MAX)


@dataclass(frozen=True)
class PhcShearResult:
    """PHC杭のせん断照査の結果。"""

    depth: float  # 杭頭からの深さ (m)
    shear: float  # 作用せん断力 S (kN)
    moment: float  # 同断面の曲げモーメント (kN·m)
    axial: float  # 軸方向圧縮力 (kN)
    area: float  # 換算断面積 Ae (m2)
    correction_factor: float  # CN
    tau: float  # τ = S/Ae (N/mm2)
    allowable: float  # τa(CN・割増後) (N/mm2)
    checks: list[ShearCheck] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def check_phc_shear(
    pile: PileSpec,
    material: "MaterialSpec",
    case: LoadCase,
    depth: float,
    shear: float,
    moment: float,
    axial: float,
) -> PhcShearResult:
    """PHC杭1断面のせん断照査。

    Ae・Ie は :func:`core.section.checks.phc_effective_section` と同じ値
    (製品カタログ値を指定しない場合はコンクリート部のみの幾何学的な近似)
    を用いる。
    """
    if pile.pile_type != PileType.PHC:
        raise ValueError(
            f"{pile.pile_type.value}にはPHC杭のせん断照査は適用できません"
        )
    if material.effective_prestress is None:
        raise ValueError(
            "PHC杭のせん断照査には有効プレストレス σce (N/mm2) の入力が"
            "必要です(MaterialSpec.effective_prestress)"
        )
    # 循環インポートを避けるため遅延インポート
    from core.section.checks import phc_effective_section

    area, section_modulus = phc_effective_section(pile, material)
    inertia = section_modulus * (pile.diameter / 2.0)
    y = pile.diameter / 2.0

    cn = phc_shear_correction_factor(
        material.effective_prestress, axial, area, inertia, y, moment
    )
    tau = abs(shear) / area / 1000.0
    increase = STRESS_INCREASE[case.value]
    allowable = PRECAST_CONCRETE_ALLOWABLE["PHC杭"].shear * cn * increase

    return PhcShearResult(
        depth=depth,
        shear=shear,
        moment=moment,
        axial=axial,
        area=area,
        correction_factor=cn,
        tau=tau,
        allowable=allowable,
        checks=[ShearCheck("せん断応力度 τ = S/Ae", tau, allowable)],
    )
