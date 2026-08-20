"""底版(フーチング)本体の許容応力度法による照査(道示Ⅳ(H24) 8章)。

杭基礎の底版は、杭反力と底版自重・上載土重量の差によって曲げ・せん断を
受ける**単鉄筋長方形RC断面**として照査する。

照査項目
--------
1. **曲げ応力度** — :func:`check_footing_flexure`

       中立軸 x    : b・x²/2 = n・As・(d − x)
       σc = 2M /(b・x・(d − x/3))
       σs =   M /(As・(d − x/3))

   杭体(円形断面)と違い、底版は長方形断面なので閉じた式で解ける。
   コンクリートは引張を負担しない(ひび割れ断面)ものとし、圧縮鉄筋は
   考慮しない(**単鉄筋**。原典の計算例もそう明記している)。

2. **最小鉄筋量**(道示Ⅳ 7.3(1)) — :func:`check_footing_flexure` に内包

       (Mu ≧ Mc または 1.7・|M| ≦ Mc)かつ As/b ≧ 500 mm²/m

   Mc・1.7 倍・500mm²/m はいずれも既に :mod:`core.section.detailing` で
   杭体向けに実装済みの規定と同じもの。

3. **せん断応力度** — :func:`check_footing_shear`

       τm = Sh /(b・d)
       τa = ce・cpt・cdc・τa1

   杭体のせん断照査(:mod:`core.section.shear`)との違いは、軸方向圧縮力の
   補正 cN の代わりに**せん断スパン比による割増し cdc** を用いる点である。
   底版は柱前面から杭までの距離が部材高に対して短く、直接支持(アーチ)
   作用が効くため、せん断スパンが短いほど許容値を割り増す。

.. important::
   本モジュールは**照査断面の断面力 M・S を入力として受け取る**(杭体の
   断面照査が断面力を受け取るのと同じ設計)。杭反力・底版自重・上載土重量・
   浮力から各照査位置の M・S を求める計算は
   :mod:`core.section.footing_forces` にある(第59回)。
   ただし**照査位置の自動列挙**(杭中心・h/2・柱前面・柱間最大最小M)は
   柱の位置・寸法を入力モデルに持つ必要があり、依然として未実装である。

.. warning::
   照査式・係数は原典未照合であり、フォーラムエイト UC-1 の計算書サンプル
   との突合による(docs/VERIFICATION.md 第58回)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.models.loads import LoadCase
from core.section.detailing import bending_tensile_strength
from core.section.shear import ShearCheck, _interpolate, ce_factor
from core.standards import (
    CONCRETE_STRAIN_PEAK,
    CONCRETE_STRAIN_ULTIMATE,
    CRACK_MOMENT_MARGIN,
    REBAR_YIELD_POINT,
    SHEAR_CDC_BY_SPAN_RATIO,
    SHEAR_CDS_SPAN_RATIO_LIMIT,
    SHEAR_CPT_BY_RATIO,
    SIGMA_CA_CONCRETE,
    STRESS_INCREASE,
    SURFACE_REBAR_MIN_AREA_PER_M,
    TAU_A1_CONCRETE,
    TAU_A2_CONCRETE,
    TAU_C_CONCRETE,
    ULTIMATE_CONCRETE_COEF,
    YOUNG_MODULUS_RATIO_RC,
)


# ---------------------------------------------------------------------------
# 単鉄筋長方形断面の応力度
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RectangularRcStress:
    """単鉄筋長方形断面の応力度算定の結果。"""

    neutral_axis: float  # 圧縮縁からの中立軸位置 x (m)
    lever_arm: float  # 応力中心間距離 d − x/3 (m)
    sigma_c: float  # コンクリート圧縮縁の応力度 (N/mm2)
    sigma_s: float  # 引張鉄筋の応力度 (N/mm2)


def singly_reinforced_stress(
    width: float,
    effective_depth: float,
    rebar_area: float,
    moment: float,
    n_ratio: float = YOUNG_MODULUS_RATIO_RC,
) -> RectangularRcStress:
    """単鉄筋長方形断面の曲げ応力度。

    Parameters
    ----------
    width:
        部材断面幅 b (m)。底版では有効幅(曲げ)または全幅(せん断)。
    effective_depth:
        有効高 d (m)。圧縮縁から引張鉄筋重心までの距離。
    rebar_area:
        引張鉄筋量 As (m²)。
    moment:
        曲げモーメント (kN·m)。符号は問わない(絶対値で扱う)。
    n_ratio:
        ヤング係数比 n。道示Ⅲ の一定値 15 が既定。

    中立軸はコンクリート圧縮域の断面一次モーメントと換算鉄筋のそれが
    釣り合う位置として

        b・x²/2 = n・As・(d − x)

    から求める(二次方程式の正根)。
    """
    if width <= 0 or effective_depth <= 0:
        raise ValueError("部材幅・有効高は正の値である必要があります")
    if rebar_area <= 0:
        raise ValueError("鉄筋量は正の値である必要があります")

    na = n_ratio * rebar_area
    # b/2・x² + n·As·x − n·As·d = 0
    x = (-na + math.sqrt(na * na + 2.0 * width * na * effective_depth)) / width
    if x >= effective_depth:
        raise ValueError(
            "中立軸が引張鉄筋位置より下になりました(断面・鉄筋量を確認してください)"
        )
    lever = effective_depth - x / 3.0

    m = abs(moment)
    # kN/m² → N/mm² は 1/1000
    sigma_c = 2.0 * m / (width * x * lever) / 1000.0
    sigma_s = m / (rebar_area * lever) / 1000.0
    return RectangularRcStress(
        neutral_axis=x, lever_arm=lever, sigma_c=sigma_c, sigma_s=sigma_s
    )


def required_rebar_area(
    width: float,
    effective_depth: float,
    moment: float,
    sigma_sa: float,
    n_ratio: float = YOUNG_MODULUS_RATIO_RC,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> float:
    """引張鉄筋の応力度がちょうど許容値に達する鉄筋量 As (m²)。

    中立軸が As に依存するため反復で解く(不動点反復。単調で速く収束する)。
    ``moment`` は kN·m、``sigma_sa`` は N/mm²。
    """
    if sigma_sa <= 0:
        raise ValueError("許容引張応力度は正の値である必要があります")
    m = abs(moment)
    if m <= 0.0:
        return 0.0
    # 初期値: 応力中心間距離を 0.875d と見込む
    area = m / (sigma_sa * 1000.0 * 0.875 * effective_depth)
    for _ in range(max_iter):
        na = n_ratio * area
        x = (
            -na + math.sqrt(na * na + 2.0 * width * na * effective_depth)
        ) / width
        lever = effective_depth - x / 3.0
        updated = m / (sigma_sa * 1000.0 * lever)
        if abs(updated - area) <= tol * max(area, 1.0e-9):
            return updated
        area = updated
    return area


# ---------------------------------------------------------------------------
# ひび割れ/終局曲げモーメント(最小鉄筋量の照査に用いる)
# ---------------------------------------------------------------------------


def cracking_moment_rectangular(
    width: float, height: float, fck: int | float
) -> float:
    """長方形断面のひび割れ曲げモーメント Mc (kN·m)。

        Mc = σbt・Z,  Z = b・h²/6,  σbt = 0.23・σck^(2/3)

    軸方向力を持たない底版を対象とするため、円形断面版
    (:func:`core.section.detailing.cracking_moment`)の N/Ac 項は無い。

    出典: フォーラムエイト UC-1 計算書サンプル Kui_8 の 7.5「曲げ応力度
    照査」の最小鉄筋量照査(第58回)。σck=24 の2断面
    (b=12.0m → Mc=23920.96 kN·m、b=8.78m → Mc=17502.17 kN·m)から
    逆算した σbt がいずれも 1.913676 となり、``0.23・24^(2/3)`` =
    1.9136758 と**7桁で一致**した。
    """
    z = width * height * height / 6.0  # m3
    return bending_tensile_strength(fck) * 1000.0 * z  # kN·m


def _stress_block_factors() -> tuple[float, float]:
    """道示Ⅲ の応力度〜ひずみ曲線による等価応力ブロックの係数 (α, β)。

    圧縮合力 C = α・σcu・b・x、その作用位置は圧縮縁から β・x。
    放物線(ε ≦ ε0)＋一定(ε0 < ε ≦ εcu)の分布を積分して求める:

        r = ε0/εcu
        α = 1 − r/3
        β = 1 −(1/2 − r²/12)/α

    εcu = 0.0035、ε0 = 0.002 では α = 0.809524、β = 0.415971 となる。
    """
    r = CONCRETE_STRAIN_PEAK / CONCRETE_STRAIN_ULTIMATE
    alpha = 1.0 - r / 3.0
    beta = 1.0 - (0.5 - r * r / 12.0) / alpha
    return alpha, beta


def ultimate_moment_singly_reinforced(
    width: float,
    effective_depth: float,
    rebar_area: float,
    fck: int | float,
    rebar_grade: str = "SD345",
) -> float:
    """単鉄筋長方形断面の終局曲げモーメント Mu (kN·m)。

    引張鉄筋の降伏で決まるものとして、圧縮合力との釣合いから中立軸 xu を
    求め、応力中心間距離を掛ける:

        T = As・σsy
        xu = T /(α・σcu・b),  σcu = 0.85・σck
        Mu = T・(d − β・xu)

    α・β は :func:`_stress_block_factors`(道示Ⅲ の放物線＋矩形分布)。

    出典: Kui_8 7.5 の最小鉄筋量照査(第58回)。σck=24・SD345 の2断面で
    Mu = 37227.4(計算例 37228.61)、14926.9(同 14927.40)と
    **0.004% 以内で一致**した。単純な等価応力矩形ブロック(0.85σck × 0.8x)
    では 0.04% ずれるため、放物線を含む分布で積分しているものと判断した。
    """
    if rebar_grade not in REBAR_YIELD_POINT:
        raise ValueError(
            f"鉄筋材質 {rebar_grade} の降伏点が未定義です。"
            f"対応材質: {sorted(REBAR_YIELD_POINT)}"
        )
    alpha, beta = _stress_block_factors()
    sigma_sy = REBAR_YIELD_POINT[rebar_grade] * 1000.0  # kN/m2
    sigma_cu = ULTIMATE_CONCRETE_COEF * float(fck) * 1000.0  # kN/m2
    tension = rebar_area * sigma_sy  # kN
    x_u = tension / (alpha * sigma_cu * width)  # m
    return tension * (effective_depth - beta * x_u)  # kN·m


# ---------------------------------------------------------------------------
# 曲げ照査
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FootingFlexureResult:
    """底版1断面の曲げ照査の結果。"""

    label: str  # 照査位置の名称
    moment: float  # 設計曲げモーメント (kN·m)
    width: float  # 有効幅 b (m)
    height: float  # 部材高 h (m)
    effective_depth: float  # 有効高 d (m)
    rebar_area: float  # 配置鉄筋量 As (m²)
    required_area: float  # 必要鉄筋量 (m²)
    stress: RectangularRcStress
    checks: list[ShearCheck] = field(default_factory=list)
    # 最小鉄筋量(道示Ⅳ 7.3(1))
    ultimate_moment: float = 0.0  # Mu (kN·m)
    cracking_moment: float = 0.0  # Mc (kN·m)
    min_rebar_ok: bool = True
    min_rebar_note: str = ""

    @property
    def rebar_area_per_m(self) -> float:
        """幅1mあたりの鉄筋量 (mm²/m)。"""
        return self.rebar_area * 1.0e6 / self.width

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks) and self.min_rebar_ok


def check_footing_flexure(
    moment: float,
    width: float,
    height: float,
    effective_depth: float,
    rebar_area: float,
    fck: int,
    case: LoadCase,
    sigma_sa: float,
    rebar_grade: str = "SD345",
    label: str = "",
    n_ratio: float = YOUNG_MODULUS_RATIO_RC,
) -> FootingFlexureResult:
    """底版1断面の曲げ照査(単鉄筋長方形断面)。

    Parameters
    ----------
    moment:
        設計曲げモーメント (kN·m)。
    width, height, effective_depth:
        有効幅 b・部材高 h・有効高 d (m)。
    rebar_area:
        引張側に配置した主鉄筋量 As (m²)。
    sigma_sa:
        鉄筋の許容引張応力度 (N/mm²、**割増し後**)。底版は水中施工では
        ないため :func:`core.section.checks.rebar_tension_allowable` に
        ``underwater=False`` を与えて求めた値を想定する。呼び出し側で
        求めた値をそのまま用いるのは、上側・下側や方向ごとに鉄筋材質が
        変わり得るためである。

    最小鉄筋量(道示Ⅳ 7.3(1))も併せて判定する:
    ``(Mu ≧ Mc または 1.7・|M| ≦ Mc)かつ As/b ≧ 500 mm²/m``。
    """
    if fck not in SIGMA_CA_CONCRETE:
        raise ValueError(
            f"σck={fck} は許容曲げ圧縮応力度 σca の表(σck = "
            f"{sorted(SIGMA_CA_CONCRETE)})の範囲外です"
        )
    increase = STRESS_INCREASE[case.value]
    stress = singly_reinforced_stress(
        width, effective_depth, rebar_area, moment, n_ratio
    )
    sigma_ca = SIGMA_CA_CONCRETE[fck] * increase
    checks = [
        ShearCheck("底版コンクリート曲げ圧縮応力度", stress.sigma_c, sigma_ca),
        ShearCheck("底版鉄筋引張応力度", stress.sigma_s, sigma_sa),
    ]

    m_u = ultimate_moment_singly_reinforced(
        width, effective_depth, rebar_area, fck, rebar_grade
    )
    m_c = cracking_moment_rectangular(width, height, fck)
    per_m = rebar_area * 1.0e6 / width
    strength_ok = m_u >= m_c or CRACK_MOMENT_MARGIN * abs(moment) <= m_c
    area_ok = per_m >= SURFACE_REBAR_MIN_AREA_PER_M
    reasons = []
    if m_u >= m_c:
        reasons.append(f"Mu={m_u:,.0f} ≧ Mc={m_c:,.0f}")
    elif CRACK_MOMENT_MARGIN * abs(moment) <= m_c:
        reasons.append(
            f"{CRACK_MOMENT_MARGIN:g}M={CRACK_MOMENT_MARGIN * abs(moment):,.0f}"
            f" ≦ Mc={m_c:,.0f}"
        )
    else:
        reasons.append(
            f"Mu={m_u:,.0f} < Mc={m_c:,.0f} かつ "
            f"{CRACK_MOMENT_MARGIN:g}M="
            f"{CRACK_MOMENT_MARGIN * abs(moment):,.0f} > Mc"
        )
    reasons.append(
        f"As={per_m:,.0f} mm²/m "
        f"{'≧' if area_ok else '<'} {SURFACE_REBAR_MIN_AREA_PER_M:g} mm²/m"
    )

    return FootingFlexureResult(
        label=label,
        moment=moment,
        width=width,
        height=height,
        effective_depth=effective_depth,
        rebar_area=rebar_area,
        required_area=required_rebar_area(
            width, effective_depth, moment, sigma_sa, n_ratio
        ),
        stress=stress,
        checks=checks,
        ultimate_moment=m_u,
        cracking_moment=m_c,
        min_rebar_ok=strength_ok and area_ok,
        min_rebar_note="、".join(reasons),
    )


# ---------------------------------------------------------------------------
# せん断照査
# ---------------------------------------------------------------------------


def cpt_factor_footing(pt: float) -> float:
    """軸方向引張鉄筋比 pt (%) の補正係数 cpt(**底版用**)。

    表(:data:`core.standards.SHEAR_CPT_BY_RATIO`)は pt = 0.1% から
    始まるが、フォーラムエイトの計算例は pt < 0.1% でも表の最初の勾配で
    **線形外挿**している(Kui_8 7.6 の上側引張: pt = 0.0854% →
    cpt = 0.671 = 0.7 + 2.0×(0.0854 − 0.1))。外挿すると cpt が小さくなり
    許容せん断応力度も小さくなるので**安全側**である。

    .. note::
       杭体のせん断照査(:func:`core.section.shear.cpt_factor`)は従来
       どおり表の下限で頭打ちにしている。原典は表の範囲外の扱いを
       規定しておらず、杭側を変えると既存の照合結果に影響するため
       据え置いた(docs/VERIFICATION.md 第58回)。
    """
    return _interpolate(SHEAR_CPT_BY_RATIO, pt, extrapolate_low=True)


def cdc_factor(shear_span: float, column_face_depth: float) -> float:
    """せん断スパン比 a/d' による許容せん断応力度の割増し係数 cdc。

    ``shear_span`` は せん断スパン a (m)、``column_face_depth`` は柱前面
    位置での有効高 d' (m)。表(:data:`core.standards.
    SHEAR_CDC_BY_SPAN_RATIO`)を線形補間し、範囲外は端の値で頭打ちにする
    (a/d' ≧ 2.5 で割増しなし)。
    """
    if column_face_depth <= 0:
        raise ValueError("柱前面での有効高は正の値である必要があります")
    return _interpolate(SHEAR_CDC_BY_SPAN_RATIO, shear_span / column_face_depth)


def cds_factor(shear_span: float, column_face_depth: float) -> float:
    """斜引張鉄筋に対するせん断スパン比の低減係数 cds(診断値)。

        cds = min(a /(2.5・d'), 1.0)

    .. warning::
       **確度C。値は Kui_8 の2点と一致するが、これを用いる必要斜引張
       鉄筋量の式は検証できていない**(計算例は全ケースで斜引張鉄筋が
       不要だった)。本ソフトは cds を診断値として返すのみで、必要鉄筋量
       の算定には用いない。
    """
    if column_face_depth <= 0:
        raise ValueError("柱前面での有効高は正の値である必要があります")
    ratio = shear_span / (SHEAR_CDS_SPAN_RATIO_LIMIT * column_face_depth)
    return min(ratio, 1.0)


@dataclass(frozen=True)
class FootingShearResult:
    """底版1断面のせん断照査の結果。"""

    label: str
    shear: float  # 設計せん断力 S (kN)
    width: float  # 部材断面幅 b (m、フーチング全幅)
    effective_depth: float  # 有効高 d (m)
    rebar_area: float  # 引張側主鉄筋量 As (m²)
    pt: float  # 軸方向引張鉄筋比 (%)
    ce: float
    cpt: float
    cdc: float
    cds: float  # 診断値(必要斜引張鉄筋量の算定には用いない)
    shear_span: float  # a (m)
    column_face_depth: float  # d' (m)
    tau_m: float  # 平均せん断応力度 (N/mm2)
    tau_a: float  # ce・cpt・cdc・τa1 (N/mm2)
    tau_a2: float  # 斜引張鉄筋と共同 (N/mm2)
    concrete_shear: float  # Sca = τa・b・d (kN)
    seismic: bool
    checks: list[ShearCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def needs_stirrup(self) -> bool:
        """コンクリートのみでは負担できず、斜引張鉄筋が必要か。"""
        return abs(self.shear) > self.concrete_shear

    @property
    def stirrup_shear(self) -> float:
        """斜引張鉄筋が負担すべきせん断力 Sh' = Sh − Sca (kN)。"""
        return max(0.0, abs(self.shear) - self.concrete_shear)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


def check_footing_shear(
    shear: float,
    width: float,
    effective_depth: float,
    rebar_area: float,
    shear_span: float,
    column_face_depth: float,
    fck: int,
    case: LoadCase,
    label: str = "",
) -> FootingShearResult:
    """底版1断面のせん断照査。

        τm = Sh /(b・d)
        τa = ce・cpt・cdc・τa1   (地震時は τa1 に代えて τc を用いる)
        τa2 = 表の値 × 割増し

    Parameters
    ----------
    shear:
        設計せん断力 S (kN)。
    width:
        部材断面幅 b (m)。**曲げの有効幅ではなくフーチング全幅**を用いる
        (原典の計算例が「b：部材断面幅(mm)で、フーチング全幅とする」と
        明記している)。
    rebar_area:
        引張側の主鉄筋量 As (m²)。pt = As/(b・d) の算定に用いる。
    shear_span, column_face_depth:
        せん断スパン a (m) と柱前面位置での有効高 d' (m)。

    .. note::
       原典は「せん断スパン比により許容応力度の割増しを行う場合には、
       部材の有効高の変化の影響を考慮しない」としており、本実装は常に
       cdc による割増しを行うので Sh = S とする(底版は等厚なので
       いずれにせよ有効高は変化しない)。
    """
    if fck not in TAU_A1_CONCRETE:
        raise ValueError(
            f"σck={fck} の許容せん断応力度が未定義です。"
            f"対応値: {sorted(TAU_A1_CONCRETE)}"
        )
    increase = STRESS_INCREASE[case.value]
    pt = 100.0 * rebar_area / (width * effective_depth)
    ce = ce_factor(effective_depth)
    cpt = cpt_factor_footing(pt)
    cdc = cdc_factor(shear_span, column_face_depth)
    cds = cds_factor(shear_span, column_face_depth)

    # 杭体の照査と同じ扱い: 地震時は τa1 × 1.50 の代わりに τc を用いる
    base = TAU_C_CONCRETE[fck] if case.is_seismic else TAU_A1_CONCRETE[fck] * increase
    tau_a = ce * cpt * cdc * base
    tau_a2 = TAU_A2_CONCRETE[fck] * increase
    tau_m = abs(shear) / (width * effective_depth) / 1000.0
    concrete_shear = tau_a * width * effective_depth * 1000.0  # kN

    checks = [
        ShearCheck("底版平均せん断応力度(斜引張鉄筋と共同)", tau_m, tau_a2),
    ]
    notes: list[str] = []
    if tau_m > tau_a:
        notes.append(
            "コンクリートのみではせん断力を負担できないため斜引張鉄筋が"
            f"必要である(Sh' = {abs(shear) - concrete_shear:,.0f} kN)。"
            "**必要斜引張鉄筋量の算定式は原典・計算例とも未確認のため"
            "本ソフトは算定しない**(docs/VERIFICATION.md 第58回)。"
        )
    return FootingShearResult(
        label=label,
        shear=shear,
        width=width,
        effective_depth=effective_depth,
        rebar_area=rebar_area,
        pt=pt,
        ce=ce,
        cpt=cpt,
        cdc=cdc,
        cds=cds,
        shear_span=shear_span,
        column_face_depth=column_face_depth,
        tau_m=tau_m,
        tau_a=tau_a,
        tau_a2=tau_a2,
        concrete_shear=concrete_shear,
        seismic=case.is_seismic,
        checks=checks,
        notes=notes,
    )
