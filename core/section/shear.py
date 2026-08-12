"""円形RC断面(場所打ち杭)のせん断力に対する照査(道示Ⅳ(H24) 5.1.3)。

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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models.loads import LoadCase
from core.models.pile import PileSpec, PileType
from core.section.rc import RebarLayout
from core.standards import (
    SHEAR_CE_BY_DEPTH,
    SHEAR_CN_MAX,
    SHEAR_CN_MIN,
    SHEAR_CPT_BY_RATIO,
    STRESS_INCREASE,
    TAU_A1_CONCRETE,
    TAU_A2_CONCRETE,
    TAU_C_CONCRETE,
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


def _interpolate(table: tuple[tuple[float, float], ...], x: float) -> float:
    """表の線形補間。範囲外は端の値で頭打ちにする。"""
    if x <= table[0][0]:
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

    @property
    def required_aw_per_spacing(self) -> float | None:
        """必要な斜引張鉄筋量 Aw/s (mm2/mm)。帯鉄筋(θ = 90°)として。"""
        if self.stirrup_sigma_sa is None:
            return None
        return self.required_stirrup_ratio(self.stirrup_sigma_sa)

    @property
    def needs_stirrup(self) -> bool:
        """コンクリートのみでは負担できず、斜引張鉄筋が必要か。"""
        return self.tau_m > self.tau_a1

    @property
    def all_ok(self) -> bool:
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
    )
