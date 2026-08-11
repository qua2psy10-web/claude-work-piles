"""杭体の深度方向断面力分布(Chang の式、道示Ⅳ(H24) 参考資料)。

弾性床上の梁の微分方程式
    EI・y'''' + kH・D・y = 0,   β = ⁴√(kH・D / (4EI))
の減衰解 y = e^(−βx)(A・cos βx + B・sin βx) を用いる。

杭頭(x = 0)の水平力 H0・モーメント M0 を境界条件として係数を定める:
    M = EI・y''  = 2EIβ²・e^(−βx)(A sin βx − B cos βx)
    S = EI・y''' = 2EIβ³・e^(−βx)[(A+B) cos βx + (B−A) sin βx]
    M(0) = −2EIβ²・B = M0        → B = −M0 / (2EIβ²)
    S(0) =  2EIβ³・(A+B) = H0    → A = H0 / (2EIβ³) − B

符号は :mod:`core.capacity.springs` の K1〜K4 と整合する。すなわち杭頭剛結
(回転角 0)では A = B = y0 となり、H0 = 4EIβ³y0(= K1・y0)、
M0 = −2EIβ²y0(= K3・y0)を満たす。

.. note::
   半無限長の杭(βL ≧ 3)を前提とした式である。βL < 3 の場合は適用範囲外。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionForce:
    """1深度の断面力。"""

    depth: float  # 杭頭からの深さ (m)
    displacement: float  # 水平変位 (m)
    moment: float  # 曲げモーメント (kN·m)
    shear: float  # せん断力 (kN)
    is_extremum: bool = False  # 曲げモーメントの極値点(S = 0)か


@dataclass(frozen=True)
class SectionForceDistribution:
    points: list[SectionForce]

    @property
    def max_moment(self) -> SectionForce:
        """曲げモーメントの絶対値が最大となる点。"""
        return max(self.points, key=lambda p: abs(p.moment))

    @property
    def max_underground_moment(self) -> SectionForce:
        """地中部の最大曲げモーメント点(せん断力が 0 となる極値点)。

        曲げモーメントの極値は dM/dx = S = 0 の位置に生じる。杭頭を除いた
        区間でせん断力が 0 となる点のうち、曲げモーメントの絶対値が最大の
        ものを返す。極値が無い場合は杭頭を除いた区間の最大値を返す。

        杭体の断面照査では杭頭断面とこの断面の両方を確認する必要がある。
        """
        extrema = [p for p in self.points if p.depth > 0 and p.is_extremum]
        if extrema:
            return max(extrema, key=lambda p: abs(p.moment))
        underground = [p for p in self.points if p.depth > 0]
        if not underground:
            return self.points[0]
        return max(underground, key=lambda p: abs(p.moment))

    @property
    def max_shear(self) -> SectionForce:
        return max(self.points, key=lambda p: abs(p.shear))


def chang_coefficients(ei: float, beta: float, h0: float, m0: float) -> tuple[float, float]:
    """Chang の式の係数 (A, B) を杭頭の境界条件から求める。"""
    b = -m0 / (2.0 * ei * beta**2)
    a = h0 / (2.0 * ei * beta**3) - b
    return a, b


def evaluate_at(
    ei: float,
    beta: float,
    h0: float,
    m0: float,
    depth: float,
    is_extremum: bool = False,
) -> SectionForce:
    """深さ depth (杭頭から) における変位・曲げモーメント・せん断力。"""
    a, b = chang_coefficients(ei, beta, h0, m0)
    t = beta * depth
    e = math.exp(-t)
    cos_t, sin_t = math.cos(t), math.sin(t)
    y = e * (a * cos_t + b * sin_t)
    moment = 2.0 * ei * beta**2 * e * (a * sin_t - b * cos_t)
    shear = 2.0 * ei * beta**3 * e * ((a + b) * cos_t + (b - a) * sin_t)
    return SectionForce(
        depth=depth,
        displacement=y,
        moment=moment,
        shear=shear,
        is_extremum=is_extremum,
    )


def extremum_depths(
    ei: float, beta: float, h0: float, m0: float, length: float
) -> list[float]:
    """曲げモーメントの極値(S = 0)が生じる深さを解析的に求める。

    S ∝ (A+B)cos βx + (B−A) sin βx = 0 より
        tan βx = (A+B)/(A−B)
    となる。周期 π ごとに解が現れるが、e^(−βx) で減衰するため浅い解ほど
    曲げモーメントが大きい。
    """
    a, b = chang_coefficients(ei, beta, h0, m0)
    t0 = math.atan2(a + b, a - b)
    while t0 < 0:
        t0 += math.pi
    depths = []
    t = t0
    while t / beta <= length:
        if t > 0:
            depths.append(t / beta)
        t += math.pi
    return depths


def distribution(
    ei: float,
    beta: float,
    h0: float,
    m0: float,
    length: float,
    pitch: float = 0.25,
) -> SectionForceDistribution:
    """杭頭から杭先端まで pitch (m) 刻みで断面力を算定する。

    曲げモーメントの極値点(S = 0)を解析的に求めて評価点に加えるため、
    刻み幅によらず最大曲げモーメントを取りこぼさない。
    """
    n = max(1, math.ceil(length / pitch))
    grid = {min(length, i * length / n) for i in range(n + 1)}
    extrema = set(extremum_depths(ei, beta, h0, m0, length))
    grid -= extrema  # 極値点は is_extremum つきで別途追加する
    points = [evaluate_at(ei, beta, h0, m0, d) for d in grid]
    points += [
        evaluate_at(ei, beta, h0, m0, d, is_extremum=True) for d in extrema
    ]
    points.sort(key=lambda p: p.depth)
    return SectionForceDistribution(points=points)
