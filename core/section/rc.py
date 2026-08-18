"""円形断面の許容応力度法による応力度算定(道示Ⅲ 4章)。

仮定:
  * 平面保持
  * コンクリートは引張を負担しない(ひび割れ断面)
  * 応力度〜ひずみ関係は線形、ヤング係数比 n = Es/Ec

座標系は断面中心を原点とし、圧縮縁側を y 正にとる。中立軸位置を y_n と
すると、ひずみは (y − y_n) に比例する。

軸力 N と曲げモーメント M はいずれも曲率 k に比例するため、偏心量
e = M/N は y_n のみの関数となる。これを利用して y_n を二分法で求め、
その後 k を N から定める。

扱える断面
----------
コンクリート部は**中実円形または円環**(内径を与えれば中空)、鋼材は
:class:`SteelFiber` の集まりとして与える。これにより

* 場所打ち杭 — 中実円形 + 円形配置の軸方向鉄筋
* RC杭 — 円環 + 円環内に配置された軸方向鉄筋
* SC杭 — 円環コンクリート + その**外側**の鋼管(円環を細分した鋼材繊維)

を同じ解法で扱える。鋼材がコンクリート断面の内部にある場合(鉄筋)は
圧縮側で (n−1) 倍として重複計上を避けるが、外側にある場合(SC杭の鋼管)は
コンクリートと重ならないため n 倍のまま算入する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.standards import REBAR_NOMINAL_AREA


@dataclass(frozen=True)
class SteelFiber:
    """断面内の鋼材を表す1つの繊維(集中断面積として扱う)。

    ``embedded`` はその鋼材がコンクリート断面の内部にあるか。真なら圧縮側で
    (n−1) 倍として重複計上を避ける(軸方向鉄筋)。偽ならコンクリートと
    重ならないので n 倍のまま算入する(SC杭の外殻鋼管)。
    """

    area: float  # 断面積 (m2)
    y: float  # y 座標 (m、断面中心が原点、圧縮縁側が正)
    embedded: bool = True


def steel_tube_fibers(
    outer_diameter: float, thickness: float, divisions: int = 720
) -> list[SteelFiber]:
    """円環の鋼管を等角度に分割した鋼材繊維の列。

    各分割は面積を保存し、y 座標にはその区間の**図心**(∫y dA / ∫dA)を
    与える。したがって断面積と断面一次モーメントは分割数によらず厳密で、
    断面二次モーメントのみ O(1/分割数²) の誤差を持つ(既定の 720 分割で
    相対誤差 1e-4 程度。:mod:`tests.test_rc_section` で厳密解と突合)。

    SC杭の鋼管はコンクリートの外側にあるため ``embedded=False`` とする。
    """
    if outer_diameter <= 0 or thickness <= 0:
        raise ValueError("鋼管の外径・板厚は正の値である必要があります")
    if outer_diameter - 2.0 * thickness <= 0:
        raise ValueError("板厚が外径に対して大きすぎます(円環になりません)")
    if divisions < 8:
        raise ValueError("鋼管の分割数は 8 以上である必要があります")

    r = (outer_diameter - thickness) / 2.0  # 平均半径
    ring_area = math.pi * (
        outer_diameter**2 - (outer_diameter - 2.0 * thickness) ** 2
    ) / 4.0
    d_theta = 2.0 * math.pi / divisions
    fibers = []
    for i in range(divisions):
        theta1 = i * d_theta
        theta2 = theta1 + d_theta
        area = ring_area / divisions
        # ∫ r·cosθ · (r t dθ) / (r t Δθ) = r(sinθ2 − sinθ1)/Δθ
        y = r * (math.sin(theta2) - math.sin(theta1)) / d_theta
        fibers.append(SteelFiber(area=area, y=y, embedded=False))
    return fibers


@dataclass(frozen=True)
class RebarLayout:
    """円形配置の軸方向鉄筋。"""

    count: int  # 本数
    diameter_mm: float  # 呼び径 (mm)
    cover_mm: float  # かぶり(断面縁〜鉄筋中心) (mm)

    @property
    def bar_area(self) -> float:
        """鉄筋1本の**公称断面積** (m2)(JIS G 3112)。

        異形棒鋼はリブ・節があるため、公称断面積は呼び名の直径の円の面積とは
        一致しない(D25 では 506.7 mm² に対し π・25²/4 = 490.9 mm² で 3.2%
        小さい)。表にない呼び径は円の面積で代用する。
        """
        area_mm2 = REBAR_NOMINAL_AREA.get(self.diameter_mm)
        if area_mm2 is None:
            area_mm2 = math.pi * self.diameter_mm**2 / 4.0
        return area_mm2 / 1.0e6

    @property
    def total_area(self) -> float:
        return self.count * self.bar_area

    def radius(self, section_diameter: float) -> float:
        """鉄筋円の半径 (m)"""
        r = section_diameter / 2.0 - self.cover_mm / 1000.0
        if r <= 0:
            raise ValueError("かぶりが大きすぎて鉄筋を配置できません")
        return r

    def positions(self, section_diameter: float) -> list[float]:
        """各鉄筋の y 座標 (m)(断面中心が原点、圧縮縁側が正)。"""
        r = self.radius(section_diameter)
        return [
            r * math.cos(2.0 * math.pi * i / self.count) for i in range(self.count)
        ]

    def fibers(self, section_diameter: float) -> list[SteelFiber]:
        """鋼材繊維の列。鉄筋はコンクリート内部にあるので ``embedded=True``。"""
        area = self.bar_area
        return [
            SteelFiber(area=area, y=y, embedded=True)
            for y in self.positions(section_diameter)
        ]

    def check_fits_in_wall(
        self, section_diameter: float, inner_diameter: float
    ) -> None:
        """中空断面のコンクリート肉厚の中に鉄筋が収まるかを検査する。"""
        r = self.radius(section_diameter)
        if r <= inner_diameter / 2.0:
            raise ValueError(
                f"かぶり {self.cover_mm:g} mm では鉄筋円の半径が "
                f"{r * 1000:.0f} mm となり、中空部の半径 "
                f"{inner_diameter / 2.0 * 1000:.0f} mm 以内に入ってしまいます"
                "(コンクリートの肉厚の中に配置してください)"
            )


@dataclass(frozen=True)
class StirrupLayout:
    """斜引張鉄筋(帯鉄筋)の配置。

    円形断面の杭では、せん断ひび割れを横切る帯鉄筋は1断面あたり **2本**
    (円の両側)である。中間帯鉄筋を配置する場合はその分を ``legs`` に含める。
    """

    diameter_mm: float  # 呼び径 (mm)
    spacing_mm: float  # 部材軸方向の間隔 s (mm)
    legs: int = 2  # せん断ひび割れを横切る本数
    angle_deg: float = 90.0  # 部材軸方向となす角度 θ(帯鉄筋は 90°)

    def validated(self) -> "StirrupLayout":
        if self.diameter_mm <= 0 or self.spacing_mm <= 0:
            raise ValueError("帯鉄筋の径・間隔は正の値である必要があります")
        if self.legs < 1:
            raise ValueError("帯鉄筋の本数は 1 以上である必要があります")
        if not 0.0 < self.angle_deg <= 90.0:
            raise ValueError("帯鉄筋の角度 θ は 0〜90° の範囲である必要があります")
        return self

    @property
    def bar_area(self) -> float:
        """鉄筋1本の断面積 (mm2)"""
        return math.pi * self.diameter_mm**2 / 4.0

    @property
    def area(self) -> float:
        """間隔 s ごとに配置される斜引張鉄筋の断面積 Aw (mm2)"""
        return self.legs * self.bar_area

    @property
    def aw_per_spacing(self) -> float:
        """Aw / s (mm2/mm)。必要量との比較に用いる。"""
        return self.area / self.spacing_mm


@dataclass(frozen=True)
class RcStressResult:
    neutral_axis_y: float  # 中立軸の y 座標 (m)
    compression_depth: float  # 圧縮縁からの中立軸深さ x (m)
    curvature: float  # 曲率 k (1/m)
    sigma_c: float  # コンクリート圧縮応力度 (N/mm2、圧縮正)
    sigma_s_tension: float  # 鉄筋の最大引張応力度 (N/mm2、引張正)
    sigma_s_compression: float  # 鉄筋の最大圧縮応力度 (N/mm2、圧縮正)
    fully_compressed: bool  # 全断面圧縮か


def _width(radius: float, inner_radius: float, y: float) -> float:
    """深さ y におけるコンクリートの有効幅 (m)。円環なら中空部を差し引く。"""
    outer = 2.0 * math.sqrt(max(0.0, radius**2 - y**2))
    if inner_radius <= 0.0 or abs(y) >= inner_radius:
        return outer
    return outer - 2.0 * math.sqrt(inner_radius**2 - y**2)


def _concrete_integrals(
    radius: float, y_n: float, inner_radius: float = 0.0, divisions: int = 400
) -> tuple[float, float]:
    """圧縮側コンクリートの断面諸量を数値積分で求める。

    戻り値は (∫(y−y_n)·b dy, ∫(y−y_n)·y·b dy)。曲率 k と Ec を乗じると
    それぞれ軸力・断面中心まわりのモーメントになる。

    ``inner_radius`` を与えると円環断面として中空部を差し引く。中空部の
    境界 y = ±ri で幅が不連続に変化するため、[lower, −ri]・[−ri, ri]・
    [ri, radius] に**区間を分けて**積分する(不連続点をまたぐと
    シンプソン則の精度が落ちる)。
    """
    lower = max(y_n, -radius)
    if lower >= radius:
        return 0.0, 0.0

    breakpoints = [lower]
    for edge in (-inner_radius, inner_radius):
        if inner_radius > 0.0 and lower < edge < radius:
            breakpoints.append(edge)
    breakpoints.append(radius)

    s_area = 0.0
    s_moment = 0.0
    per_segment = max(2, (divisions // max(1, len(breakpoints) - 1)) // 2 * 2)
    for start, end in zip(breakpoints, breakpoints[1:]):
        if end <= start:
            continue
        h = (end - start) / per_segment
        for i in range(per_segment + 1):
            y = start + i * h
            w = 1.0 if i in (0, per_segment) else (4.0 if i % 2 == 1 else 2.0)
            base = (y - y_n) * _width(radius, inner_radius, y)
            s_area += w * base * h / 3.0
            s_moment += w * base * y * h / 3.0
    return s_area, s_moment


def _fiber_ratio(fiber: SteelFiber, n_ratio: float, y_n: float | None) -> float:
    """鋼材繊維に乗じるヤング係数比。

    コンクリート内部の鋼材が**有効なコンクリートと重なる**場合のみ (n−1) と
    する。有効なコンクリートは圧縮側(y > y_n)だけなので、引張側の鉄筋は
    n 倍で算入する。``y_n`` が None(全断面有効)なら断面全体が有効。
    """
    if not fiber.embedded:
        return n_ratio
    if y_n is None or fiber.y > y_n:
        return n_ratio - 1.0
    return n_ratio


def _section_sums(
    radius: float,
    inner_radius: float,
    fibers: "list[SteelFiber]",
    n_ratio: float,
    y_n: float,
) -> tuple[float, float]:
    """曲率1あたりの軸力係数・モーメント係数(Ec 倍で実値)。"""
    s_axial, s_moment = _concrete_integrals(radius, y_n, inner_radius)
    for fiber in fibers:
        contrib = _fiber_ratio(fiber, n_ratio, y_n) * fiber.area * (fiber.y - y_n)
        s_axial += contrib
        s_moment += contrib * fiber.y
    return s_axial, s_moment


def transformed_section(
    diameter: float,
    rebar: "RebarLayout | None" = None,
    n_ratio: float = 15.0,
    inner_diameter: float = 0.0,
    fibers: "list[SteelFiber] | None" = None,
) -> tuple[float, float]:
    """非ひび割れ(全断面有効)の換算断面積 At と断面二次モーメント It。

    ``rebar`` を渡す従来の呼び方と、``fibers`` に鋼材繊維を直接渡す呼び方の
    両方に対応する。コンクリート内部の鋼材は重複計上を避けるため (n−1) 倍、
    外側の鋼材(SC杭の鋼管)は n 倍で算入する。
    """
    items = list(fibers) if fibers is not None else []
    if rebar is not None:
        items += rebar.fibers(diameter)
    outer = diameter
    inner = inner_diameter
    area = math.pi * (outer**2 - inner**2) / 4.0
    inertia = math.pi * (outer**4 - inner**4) / 64.0
    for fiber in items:
        ratio = _fiber_ratio(fiber, n_ratio, None)
        area += ratio * fiber.area
        inertia += ratio * fiber.area * fiber.y**2
    return area, inertia


def analyze_circular_rc(
    diameter: float,
    rebar: RebarLayout,
    ec: float,
    n_ratio: float,
    axial: float,
    moment: float,
    tol: float = 1e-12,
    max_iter: int = 200,
    inner_diameter: float = 0.0,
) -> RcStressResult:
    """円形(中実・中空)RC断面の応力度を算定する。

    Parameters
    ----------
    diameter: コンクリート断面の外径 (m)
    inner_diameter: 中空部の内径 (m)。0 なら中実。
    ec: コンクリートのヤング係数 (kN/m2)
    n_ratio: ヤング係数比 n = Es/Ec
    axial: 軸力 (kN、圧縮正)
    moment: 曲げモーメント (kN·m、符号は問わない)
    """
    if inner_diameter > 0.0:
        rebar.check_fits_in_wall(diameter, inner_diameter)
    return analyze_circular_section(
        diameter=diameter,
        fibers=rebar.fibers(diameter),
        ec=ec,
        n_ratio=n_ratio,
        axial=axial,
        moment=moment,
        inner_diameter=inner_diameter,
        tol=tol,
        max_iter=max_iter,
    )


def analyze_circular_section(
    diameter: float,
    fibers: "list[SteelFiber]",
    ec: float,
    n_ratio: float,
    axial: float,
    moment: float,
    inner_diameter: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> RcStressResult:
    """円形(中実・中空)コンクリート断面 + 鋼材繊維の応力度を算定する。

    偏心量 e = M/N が換算断面の核 It/(At・R) 以下であれば全断面圧縮と
    なるため、換算断面の式 σ = N/At ± M・y/It で直接求める。
    それを超える場合はひび割れ断面として中立軸を二分法で求める。

    ``diameter`` はコンクリート断面の外径であり、鋼材繊維はその外側に
    あってもよい(SC杭の外殻鋼管)。返り値の ``sigma_c`` はコンクリートの
    圧縮縁(y = diameter/2)の応力度、``sigma_s_tension`` /
    ``sigma_s_compression`` は鋼材繊維の最大応力度である。

    Notes
    -----
    引張軸力(axial ≦ 0)には未対応。杭基礎で引抜きが生じる場合は
    別途照査が必要。
    """
    if axial <= 0:
        raise NotImplementedError(
            "引張軸力を受ける断面の応力度計算は未実装です"
            "(引抜き時は別途照査が必要)"
        )
    if inner_diameter < 0.0 or inner_diameter >= diameter:
        raise ValueError("中空部の内径は 0 以上、外径未満である必要があります")
    radius = diameter / 2.0
    inner_radius = inner_diameter / 2.0
    m_abs = abs(moment)
    target_e = m_abs / axial

    area_t, inertia_t = transformed_section(
        diameter, None, n_ratio, inner_diameter, fibers
    )
    kern = inertia_t / (area_t * radius)
    if target_e <= kern:
        return _uncracked_result(
            radius, fibers, n_ratio, ec, axial, m_abs, area_t, inertia_t
        )

    # ひび割れ断面: e(y_n) は y_n について単調増加
    lower, upper = -radius, radius - 1e-12
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        if _eccentricity(radius, inner_radius, fibers, n_ratio, mid) < target_e:
            lower = mid
        else:
            upper = mid
        if upper - lower < tol:
            break
    y_n = (lower + upper) / 2.0

    s_axial, _ = _section_sums(radius, inner_radius, fibers, n_ratio, y_n)
    if s_axial <= 0:
        raise ValueError("断面が軸力を負担できません(配筋・断面を見直してください)")
    curvature = axial / (ec * s_axial)

    # 応力度 (kN/m2 → N/mm2 は 1/1000)
    sigma_c = ec * curvature * (radius - y_n) / 1000.0
    tensions, compressions = [], []
    for fiber in fibers:
        sigma_s = n_ratio * ec * curvature * (fiber.y - y_n) / 1000.0
        if sigma_s >= 0:
            compressions.append(sigma_s)
        else:
            tensions.append(-sigma_s)
    return RcStressResult(
        neutral_axis_y=y_n,
        compression_depth=radius - y_n,
        curvature=curvature,
        sigma_c=sigma_c,
        sigma_s_tension=max(tensions, default=0.0),
        sigma_s_compression=max(compressions, default=0.0),
        fully_compressed=False,
    )


def _uncracked_result(
    radius: float,
    fibers: "list[SteelFiber]",
    n_ratio: float,
    ec: float,
    axial: float,
    m_abs: float,
    area_t: float,
    inertia_t: float,
) -> RcStressResult:
    """全断面圧縮(非ひび割れ)の場合の応力度。"""

    def concrete_stress(y: float) -> float:
        return axial / area_t + m_abs * y / inertia_t  # kN/m2

    sigma_c = concrete_stress(radius) / 1000.0
    compressions = [
        n_ratio * concrete_stress(fiber.y) / 1000.0 for fiber in fibers
    ]
    # 曲率 k = dε/dy = (M/It)/Ec、中立軸は応力度 0 の位置
    curvature = m_abs / (ec * inertia_t)
    y_n = -math.inf if m_abs == 0 else -axial * inertia_t / (m_abs * area_t)
    return RcStressResult(
        neutral_axis_y=y_n,
        compression_depth=math.inf if y_n == -math.inf else radius - y_n,
        curvature=curvature,
        sigma_c=sigma_c,
        sigma_s_tension=0.0,
        sigma_s_compression=max(compressions, default=0.0),
        fully_compressed=True,
    )


def _eccentricity(
    radius: float,
    inner_radius: float,
    fibers: "list[SteelFiber]",
    n_ratio: float,
    y_n: float,
) -> float:
    s_axial, s_moment = _section_sums(radius, inner_radius, fibers, n_ratio, y_n)
    if s_axial <= 0:
        return math.inf
    return s_moment / s_axial
