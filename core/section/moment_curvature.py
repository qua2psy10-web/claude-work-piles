"""杭体の曲げモーメント〜曲率(M-φ)骨格曲線。

レベル2地震時の照査(地震時保有水平耐力法、道示Ⅴ)では、杭体が曲げ降伏
したあとの**曲げ剛性の低下**を追跡する必要がある。本モジュールはその骨格
曲線を、原点から始まる**折れ線**として表す。

    M
    │        ┌──── Mu (終局)
    │      ／
    │  ┌─ My (降伏)
    │ ／
    │Mc (ひび割れ)
    │/
    └──────────── φ

杭種による骨格曲線の型:

============================  ========================================
杭種                          骨格曲線
============================  ========================================
鋼管杭・鋼管ソイルセメント杭   バイリニア(降伏 My → 全塑性 Mp)
場所打ちRC杭・PHC杭・SC杭      トリリニア(ひび割れ Mc → 降伏 My → 終局 Mu)
============================  ========================================

.. important::
   **本モジュールは折れ点の値そのものを算定しない。** Mc・My・Mu と対応
   する曲率は利用者が与える(製品カタログ値、または別途の断面解析による)。
   鋼管杭の My・Mp のみ :mod:`core.analysis.level2` が算定できる
   (:func:`~core.analysis.level2.yield_moment_steel_pipe` /
   :func:`~core.analysis.level2.plastic_moment_steel_pipe`)。

   RC・PHC・SC杭のトリリニアの折れ点を断面諸元から導く式は、コンクリート
   の引張強度・終局ひずみなど原典未照合の定数を必要とするため、当てずっぽう
   を避けて実装していない(docs/VERIFICATION.md 参照)。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class MomentCurvature:
    """M-φ 骨格曲線(原点から始まる折れ線)。

    ``points`` は ``((φ1, M1), (φ2, M2), ...)`` の昇順の折れ点列。曲率・
    モーメントとも正の値で与え、負のモーメントに対しては原点対称に扱う。

    最終折れ点を超える曲率に対しては、モーメントを最終折れ点の値で頭打ちに
    する(完全塑性)。終局曲率を超えているかどうかは :meth:`exceeds_ultimate`
    で判定できる。
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 1:
            raise ValueError("M-φ 骨格曲線には少なくとも1つの折れ点が必要です")
        prev_phi = 0.0
        prev_m = 0.0
        for phi, moment in self.points:
            if phi <= prev_phi:
                raise ValueError(
                    f"曲率は 0 から昇順である必要があります(φ={phi:g})"
                )
            if moment <= prev_m:
                raise ValueError(
                    f"モーメントは 0 から昇順である必要があります(M={moment:g})"
                )
            prev_phi, prev_m = phi, moment

    # --- 構築 --------------------------------------------------------------

    @classmethod
    def bilinear(
        cls,
        yield_curvature: float,
        yield_moment: float,
        ultimate_curvature: float,
        ultimate_moment: float,
    ) -> "MomentCurvature":
        """バイリニア型(鋼管杭: 降伏 My → 全塑性 Mp)。"""
        return cls(
            (
                (yield_curvature, yield_moment),
                (ultimate_curvature, ultimate_moment),
            )
        )

    @classmethod
    def trilinear(
        cls,
        cracking_curvature: float,
        cracking_moment: float,
        yield_curvature: float,
        yield_moment: float,
        ultimate_curvature: float,
        ultimate_moment: float,
    ) -> "MomentCurvature":
        """トリリニア型(RC・PHC・SC杭: ひび割れ Mc → 降伏 My → 終局 Mu)。"""
        return cls(
            (
                (cracking_curvature, cracking_moment),
                (yield_curvature, yield_moment),
                (ultimate_curvature, ultimate_moment),
            )
        )

    # --- 折れ点 ------------------------------------------------------------

    @property
    def initial_ei(self) -> float:
        """初期(ひび割れ前)の曲げ剛性 EI = M1/φ1 (kN·m²)。"""
        phi, moment = self.points[0]
        return moment / phi

    @property
    def yield_moment(self) -> float:
        """降伏モーメント My (kN·m)。

        トリリニア(3点)なら2番目、バイリニア(2点)なら1番目の折れ点。
        """
        index = 1 if len(self.points) >= 3 else 0
        return self.points[index][1]

    @property
    def yield_curvature(self) -> float:
        index = 1 if len(self.points) >= 3 else 0
        return self.points[index][0]

    @property
    def ultimate_moment(self) -> float:
        return self.points[-1][1]

    @property
    def ultimate_curvature(self) -> float:
        return self.points[-1][0]

    # --- 骨格曲線 ----------------------------------------------------------

    def moment_at(self, curvature: float) -> float:
        """曲率に対するモーメント (kN·m)。符号は曲率に従う。"""
        phi = abs(curvature)
        sign = -1.0 if curvature < 0 else 1.0
        phis = [p[0] for p in self.points]
        if phi >= phis[-1]:
            return sign * self.points[-1][1]
        index = bisect.bisect_left(phis, phi)
        phi_hi, m_hi = self.points[index]
        phi_lo, m_lo = (0.0, 0.0) if index == 0 else self.points[index - 1]
        ratio = (phi - phi_lo) / (phi_hi - phi_lo)
        return sign * (m_lo + ratio * (m_hi - m_lo))

    def curvature_at(self, moment: float) -> float:
        """モーメントに対する曲率 (1/m)。符号はモーメントに従う。

        最終折れ点を超えるモーメントは骨格曲線上に存在しないため、終局
        曲率を返す(完全塑性として扱う)。
        """
        m_abs = abs(moment)
        sign = -1.0 if moment < 0 else 1.0
        moments = [p[1] for p in self.points]
        if m_abs >= moments[-1]:
            return sign * self.points[-1][0]
        index = bisect.bisect_left(moments, m_abs)
        phi_hi, m_hi = self.points[index]
        phi_lo, m_lo = (0.0, 0.0) if index == 0 else self.points[index - 1]
        ratio = (m_abs - m_lo) / (m_hi - m_lo)
        return sign * (phi_lo + ratio * (phi_hi - phi_lo))

    def secant_ei(self, moment: float) -> float:
        """モーメントに対する**割線**曲げ剛性 EI = M/φ(M) (kN·m²)。

        モーメントが 0 のときは初期剛性を返す。終局モーメントを超えた場合は
        終局点の割線剛性(Mu/φu、骨格曲線上で最も小さい割線剛性)で頭打ちに
        なる。
        """
        m_abs = abs(moment)
        if m_abs <= 0.0:
            return self.initial_ei
        return m_abs / abs(self.curvature_at(m_abs))

    def secant_ei_at(self, curvature: float) -> float:
        """**曲率**に対する割線曲げ剛性 EI = M(φ)/φ (kN·m²)。

        梁要素の剛性を低下させるのに用いる。曲率は変位場のみで決まり
        (剛性に依存しない)ため、こちらのほうが反復計算が安定する。
        曲率が 0 のときは初期剛性を返す。
        """
        phi = abs(curvature)
        if phi <= 0.0:
            return self.initial_ei
        return abs(self.moment_at(phi)) / phi

    def yielded(self, moment: float) -> bool:
        """降伏モーメントに達しているか。"""
        return abs(moment) >= self.yield_moment

    def exceeds_ultimate(self, moment: float) -> bool:
        """終局モーメントを超えているか(骨格曲線の外に出ている)。"""
        return abs(moment) > self.ultimate_moment
