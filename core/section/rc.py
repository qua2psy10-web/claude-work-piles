"""円形RC断面の許容応力度法による応力度算定(道示Ⅲ 4章)。

仮定:
  * 平面保持
  * コンクリートは引張を負担しない(ひび割れ断面)
  * 応力度〜ひずみ関係は線形、ヤング係数比 n = Es/Ec

座標系は断面中心を原点とし、圧縮縁側を y 正にとる。中立軸位置を y_n と
すると、ひずみは (y − y_n) に比例する。

軸力 N と曲げモーメント M はいずれも曲率 k に比例するため、偏心量
e = M/N は y_n のみの関数となる。これを利用して y_n を二分法で求め、
その後 k を N から定める。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RebarLayout:
    """円形配置の軸方向鉄筋。"""

    count: int  # 本数
    diameter_mm: float  # 呼び径 (mm)
    cover_mm: float  # かぶり(断面縁〜鉄筋中心) (mm)

    @property
    def bar_area(self) -> float:
        """鉄筋1本の断面積 (m2)"""
        return math.pi * (self.diameter_mm / 1000.0) ** 2 / 4.0

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


def _concrete_integrals(
    radius: float, y_n: float, divisions: int = 400
) -> tuple[float, float]:
    """圧縮側コンクリートの断面諸量を数値積分で求める。

    戻り値は (∫(y−y_n)·b dy, ∫(y−y_n)·y·b dy)。曲率 k と Ec を乗じると
    それぞれ軸力・断面中心まわりのモーメントになる。
    """
    lower = max(y_n, -radius)
    if lower >= radius:
        return 0.0, 0.0
    # シンプソン則(区間数は偶数)
    n = divisions if divisions % 2 == 0 else divisions + 1
    h = (radius - lower) / n

    def integrand(y: float) -> tuple[float, float]:
        width = 2.0 * math.sqrt(max(0.0, radius**2 - y**2))
        base = (y - y_n) * width
        return base, base * y

    s_area = 0.0
    s_moment = 0.0
    for i in range(n + 1):
        y = lower + i * h
        w = 1.0 if i in (0, n) else (4.0 if i % 2 == 1 else 2.0)
        a, m = integrand(y)
        s_area += w * a
        s_moment += w * m
    factor = h / 3.0
    return s_area * factor, s_moment * factor


def _section_sums(
    diameter: float, rebar: RebarLayout, n_ratio: float, y_n: float
) -> tuple[float, float]:
    """曲率1あたりの軸力係数・モーメント係数(Ec 倍で実値)。

    圧縮側鉄筋はコンクリートとの重複を避けるため (n−1) 倍で算入する。
    """
    radius = diameter / 2.0
    s_axial, s_moment = _concrete_integrals(radius, y_n)
    bar_area = rebar.bar_area
    for y in rebar.positions(diameter):
        ratio = n_ratio if y <= y_n else n_ratio - 1.0
        contrib = ratio * bar_area * (y - y_n)
        s_axial += contrib
        s_moment += contrib * y
    return s_axial, s_moment


def transformed_section(
    diameter: float, rebar: RebarLayout, n_ratio: float
) -> tuple[float, float]:
    """非ひび割れ(全断面有効)の換算断面積 At と断面二次モーメント It。

    鉄筋はコンクリートとの重複を避けるため (n−1) 倍で算入する。
    """
    radius = diameter / 2.0
    area = math.pi * radius**2 + (n_ratio - 1.0) * rebar.total_area
    sum_y2 = sum(y**2 for y in rebar.positions(diameter)) * rebar.bar_area
    inertia = math.pi * diameter**4 / 64.0 + (n_ratio - 1.0) * sum_y2
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
) -> RcStressResult:
    """円形RC断面の応力度を算定する。

    偏心量 e = M/N が換算断面の核 It/(At・R) 以下であれば全断面圧縮と
    なるため、換算断面の式 σ = N/At ± M・y/It で直接求める。
    それを超える場合はひび割れ断面として中立軸を二分法で求める。

    Parameters
    ----------
    diameter: 断面直径 (m)
    ec: コンクリートのヤング係数 (kN/m2)
    n_ratio: ヤング係数比 n = Es/Ec
    axial: 軸力 (kN、圧縮正)
    moment: 曲げモーメント (kN·m、符号は問わない)

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
    radius = diameter / 2.0
    m_abs = abs(moment)
    target_e = m_abs / axial

    area_t, inertia_t = transformed_section(diameter, rebar, n_ratio)
    kern = inertia_t / (area_t * radius)
    if target_e <= kern:
        return _uncracked_result(
            diameter, rebar, n_ratio, ec, axial, m_abs, area_t, inertia_t
        )

    # ひび割れ断面: e(y_n) は y_n について単調増加
    lower, upper = -radius, radius - 1e-12
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        if _eccentricity(diameter, rebar, n_ratio, mid) < target_e:
            lower = mid
        else:
            upper = mid
        if upper - lower < tol:
            break
    y_n = (lower + upper) / 2.0

    s_axial, _ = _section_sums(diameter, rebar, n_ratio, y_n)
    if s_axial <= 0:
        raise ValueError("断面が軸力を負担できません(配筋・断面を見直してください)")
    curvature = axial / (ec * s_axial)

    # 応力度 (kN/m2 → N/mm2 は 1/1000)
    sigma_c = ec * curvature * (radius - y_n) / 1000.0
    tensions, compressions = [], []
    for y in rebar.positions(diameter):
        sigma_s = n_ratio * ec * curvature * (y - y_n) / 1000.0
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
    diameter: float,
    rebar: RebarLayout,
    n_ratio: float,
    ec: float,
    axial: float,
    m_abs: float,
    area_t: float,
    inertia_t: float,
) -> RcStressResult:
    """全断面圧縮(非ひび割れ)の場合の応力度。"""
    radius = diameter / 2.0

    def concrete_stress(y: float) -> float:
        return axial / area_t + m_abs * y / inertia_t  # kN/m2

    sigma_c = concrete_stress(radius) / 1000.0
    compressions = [
        n_ratio * concrete_stress(y) / 1000.0 for y in rebar.positions(diameter)
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
    diameter: float, rebar: RebarLayout, n_ratio: float, y_n: float
) -> float:
    s_axial, s_moment = _section_sums(diameter, rebar, n_ratio, y_n)
    if s_axial <= 0:
        return math.inf
    return s_moment / s_axial
