"""分布バネモデル(BNWF: Beam on Nonlinear Winkler Foundation)。

杭を軸方向に分割した梁要素でモデル化し、各節点に弾塑性の水平地盤バネを
配置する。地盤バネは水平地盤反力度の上限値 pHU で頭打ちとなるため、
Chang の式(弾性・半無限長)では表せない**地盤の塑性化**を扱える。

座標と符号
----------
杭頭を原点、杭軸下向きを x、水平変位を y とする。梁要素の自由度は
節点ごとに (y, φ)、φ = dy/dx である。

一方、変位法(:mod:`core.analysis.displacement`)およびバネ定数
(:mod:`core.capacity.springs`)では杭頭の回転を **θ = −dy/dx** として

    ⎧H⎫   ⎡K1 K2⎤ ⎧u⎫
    ⎨M⎬ = ⎢K3 K4⎥ ⎨θ⎬
    ⎩ ⎭   ⎣     ⎦ ⎩ ⎭

と表している。両者は D = diag(1, −1) による相似変換で結ばれる:

    K_(u,θ) = D・K_(y,φ)・D

したがって半無限長・弾性の場合、本モデルの縮約剛性は

    K_(y,φ) → ⎡4EIβ³  2EIβ²⎤     (K1 = 4EIβ³, K2 = K3 = −2EIβ²,
              ⎣2EIβ²  2EIβ ⎦      K4 = 2EIβ に対応)

に収束する。この一致を :mod:`tests.test_bnwf` で固定しており、本モデルの
検証の基礎としている。

地盤バネ
--------
節点 i のバネ定数と上限値は、分担長 Δz_i を用いて

    k_i = kH_i・D・Δz_i  (kN/m)
    R_i = pHU_i・D・Δz_i (kN)

とする。載荷方向・反対方向とも同じ上限値を用いる(片側の受働抵抗のみを
考える細かいモデル化はしていない)。

水平方向地盤反力係数 kH は**節点ごと**に与えられる。Chang の式は地盤が
一様であることを前提とするため単一の kH しか持てないが、分布バネモデルには
その制約がない。:func:`core.capacity.springs.layered_kh` により当該深度の
地層の変形係数 E0 から求めた kH を節点ごとに与えるのが本ソフトの既定である
(換算載荷幅 BH は常時の条件で定めた共通値を用いる)。スカラーを渡せば
従来どおり杭長にわたって一定の kH として扱う。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.section.moment_curvature import MomentCurvature

# 杭頭自由度(y, φ)の数
_HEAD_DOF = 2

# 変位法の (u, θ) 系と梁の (y, φ) 系を結ぶ相似変換 D = diag(1, −1)
_SIGN = np.array([1.0, -1.0])


@dataclass(frozen=True)
class HeadResponse:
    """与えた杭頭変位に対する杭1本の応答。"""

    shear: float  # 杭頭水平力 H (kN)
    moment: float  # 杭頭モーメント M (kN·m)
    tangent: np.ndarray  # (u, θ) 系の接線剛性 2×2
    displacements: np.ndarray  # 各節点の水平変位 y (m)
    plastic_nodes: int  # 上限に達した地盤バネの数
    iterations: int
    # 降伏モーメントに達した梁要素の数(M-φ を与えた場合のみ。塑性ヒンジ)
    plastic_hinges: int = 0
    # 終局曲率 φu を超えた要素があるか(M-φ を与えた場合のみ)
    exceeds_ultimate_curvature: bool = False

    @property
    def yielded_ground(self) -> bool:
        return self.plastic_nodes > 0

    @property
    def yielded_body(self) -> bool:
        """杭体に塑性ヒンジが生じているか。"""
        return self.plastic_hinges > 0


def _element_stiffness(ei: float, le: float) -> np.ndarray:
    """Hermite 梁要素1つの剛性マトリクス 4×4。"""
    return (ei / le**3) * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le * le, -6.0 * le, 2.0 * le * le],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le * le, -6.0 * le, 4.0 * le * le],
        ]
    )


def _element_dofs(e: int) -> np.ndarray:
    return np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])


def _beam_stiffness(
    ei: "float | np.ndarray", element_length: float, n_elements: int
) -> np.ndarray:
    """Hermite 梁要素による全体剛性マトリクス。

    ``ei`` はスカラー(全要素で一定)または長さ ``n_elements`` の配列
    (要素ごとの曲げ剛性。M-φ による剛性低下を反映する場合に用いる)。
    """
    ndof = 2 * (n_elements + 1)
    k = np.zeros((ndof, ndof))
    le = element_length
    ei_array = np.atleast_1d(np.asarray(ei, dtype=float))
    if ei_array.size == 1:
        ei_array = np.full(n_elements, float(ei_array[0]))
    elif ei_array.shape != (n_elements,):
        raise ValueError(
            f"要素ごとの EI の要素数が分割数と一致しません "
            f"({ei_array.shape[0]} ≠ {n_elements})"
        )
    for e in range(n_elements):
        idx = _element_dofs(e)
        k[np.ix_(idx, idx)] += _element_stiffness(float(ei_array[e]), le)
    return k


def _node_kh(kh: float | np.ndarray, n_elements: int) -> np.ndarray:
    """節点ごとの kH 配列に整える。スカラーは全節点に展開する。"""
    array = np.atleast_1d(np.asarray(kh, dtype=float))
    if array.size == 1:
        if array[0] <= 0.0:
            raise ValueError("kH は正の値である必要があります")
        return np.full(n_elements + 1, float(array[0]))
    if array.shape != (n_elements + 1,):
        raise ValueError(
            f"kH の要素数が分割数と一致しません "
            f"({array.shape[0]} ≠ {n_elements + 1})"
        )
    if np.any(array < 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("節点ごとの kH は 0 以上の有限値である必要があります")
    if not np.any(array > 0.0):
        raise ValueError("節点ごとの kH がすべて 0 です(水平抵抗がありません)")
    return array.copy()


def tributary_lengths(length: float, n_elements: int) -> np.ndarray:
    """各節点が受け持つ長さ(台形則)。両端は要素長の 1/2。"""
    le = length / n_elements
    lengths = np.full(n_elements + 1, le)
    lengths[0] = le / 2.0
    lengths[-1] = le / 2.0
    return lengths


class PileLateralModel:
    """杭1本の水平方向モデル(弾性梁 + 弾塑性地盤バネ)。

    Parameters
    ----------
    ei:
        杭体の曲げ剛性 (kN·m²)。
    diameter:
        杭径 (m)。地盤バネの負担幅として用いる。
    length:
        杭長 (m)。
    kh:
        水平方向地盤反力係数 (kN/m³)。スカラーなら杭長にわたって一定、
        長さ 要素数 + 1 の配列なら**節点ごと**の値として扱う。配列の要素は
        0 以上であればよい(0 = その節点に水平抵抗がない)が、すべてが 0 では
        釣合いが解けないため少なくとも 1 つは正である必要がある。
    limits:
        節点ごとの地盤反力度の上限値 pHU (kN/m²)。``None`` なら弾性
        (上限なし)として扱う。要素数 + 1 個必要。
    reduction:
        節点ごとの土質定数の低減係数 DE(液状化。道示Ⅴ 8.2.4)。
        バネ定数 **および上限値** に乗じる。``None`` なら低減なし。

        .. note::
           提供資料で確認できているのは「側方地盤のバネ定数 kH に DE を
           乗じる」ことまでである。上限値 pHU にも乗じているのは、
           (a) 液状化した層は受働抵抗も失われると考えるのが自然であり、
           (b) 乗じないとバネの降伏変位 R/k が液状化層でかえって大きく
           なって挙動が不整合になり、(c) 抵抗を小さくする安全側の扱い
           だからである。**原典で要確認**。
    n_elements:
        分割数。
    moment_curvature:
        杭体の M-φ 骨格曲線(:class:`core.section.moment_curvature.
        MomentCurvature`)。与えると**杭体の曲げ剛性低下**を要素ごとに
        追跡する(レベル2の塑性ヒンジ)。``None`` なら杭体は弾性のまま。

        .. note::
           要素内でモーメントは線形に変化するが、本実装は要素ごとに
           **両端モーメントの大きいほう**で決まる割線剛性を要素全体に
           一様に適用する(集中化した近似)。分割数を増やすほど厳密解に
           近づく。
    """

    def __init__(
        self,
        ei: float,
        diameter: float,
        length: float,
        kh: float | np.ndarray,
        limits: np.ndarray | None = None,
        reduction: np.ndarray | None = None,
        n_elements: int = 50,
        moment_curvature: "MomentCurvature | None" = None,
    ) -> None:
        if n_elements < 2:
            raise ValueError("分割数は 2 以上である必要があります")
        if ei <= 0 or diameter <= 0 or length <= 0:
            raise ValueError("EI・杭径・杭長は正の値である必要があります")
        self.kh = _node_kh(kh, n_elements)
        self.moment_curvature = moment_curvature

        self.ei = ei
        self.diameter = diameter
        self.length = length
        self.n_elements = n_elements
        self.node_depths = np.linspace(0.0, length, n_elements + 1)

        tributary = tributary_lengths(length, n_elements)
        if reduction is None:
            self.reduction = np.ones(n_elements + 1)
        else:
            self.reduction = np.asarray(reduction, dtype=float)
            if self.reduction.shape != (n_elements + 1,):
                raise ValueError(
                    f"低減係数の要素数が分割数と一致しません "
                    f"({self.reduction.shape[0]} ≠ {n_elements + 1})"
                )
            if np.any(self.reduction < 0.0) or np.any(self.reduction > 1.0):
                raise ValueError("低減係数 DE は 0〜1 の範囲である必要があります")
        # 低減は**節点ごと**に行う。杭頭バネ K1〜K4 を用いる弾性解析では
        # 深度平均に頼らざるを得ないが、分布バネモデルでは層ごとに扱える。
        self.spring_k = self.kh * diameter * tributary * self.reduction  # kN/m
        if limits is None:
            self.spring_limit = np.full(n_elements + 1, np.inf)
        else:
            limits = np.asarray(limits, dtype=float)
            if limits.shape != (n_elements + 1,):
                raise ValueError(
                    f"上限値の要素数が分割数と一致しません "
                    f"({limits.shape[0]} ≠ {n_elements + 1})"
                )
            if np.any(limits <= 0):
                raise ValueError("地盤反力度の上限値は正の値である必要があります")
            self.spring_limit = limits * diameter * tributary * self.reduction  # kN

        self.element_length = length / n_elements
        # 弾性(初期)の梁剛性。head_stiffness() など弾性状態の参照に使う
        self.beam_elastic = _beam_stiffness(ei, self.element_length, n_elements)
        self.beam = self.beam_elastic.copy()
        self.element_ei = np.full(n_elements, float(ei))
        self._state = np.zeros(2 * (n_elements + 1))

    # --- 地盤バネ ----------------------------------------------------------

    def _spring_force(self, y: np.ndarray) -> np.ndarray:
        return np.clip(self.spring_k * y, -self.spring_limit, self.spring_limit)

    def _spring_tangent(self, y: np.ndarray) -> np.ndarray:
        elastic = np.abs(self.spring_k * y) < self.spring_limit
        return np.where(elastic, self.spring_k, 0.0)

    def _resistance(self, u: np.ndarray) -> np.ndarray:
        """全自由度に対する地盤バネの内力ベクトル(回転自由度は 0)。"""
        force = np.zeros_like(u)
        force[0::2] = self._spring_force(u[0::2])
        return force

    # --- 杭体の曲げ非線形(M-φ) --------------------------------------------

    def element_curvatures(self, u: np.ndarray | None = None) -> np.ndarray:
        """要素ごとの**両端曲率の絶対値の大きいほう** (1/m)。

        Hermite 要素の曲率 κ = d²v/dx² は変位場のみで決まり、曲げ剛性には
        依存しない。両端の曲率は

            κ1 = −6/le²·v1 − 4/le·φ1 + 6/le²·v2 − 2/le·φ2
            κ2 = +6/le²·v1 + 2/le·φ1 − 6/le²·v2 + 4/le·φ2

        (端力 ``ke @ ue = [F1, C1, F2, C2]`` に対し M1 = −C1 = EI·κ1、
        M2 = C2 = EI·κ2 と整合する)。
        """
        state = self._state if u is None else u
        le = self.element_length
        b1 = np.array([-6.0 / le**2, -4.0 / le, 6.0 / le**2, -2.0 / le])
        b2 = np.array([6.0 / le**2, 2.0 / le, -6.0 / le**2, 4.0 / le])
        curvatures = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            ue = state[_element_dofs(e)]
            curvatures[e] = max(abs(float(b1 @ ue)), abs(float(b2 @ ue)))
        return curvatures

    def element_moments(self, u: np.ndarray | None = None) -> np.ndarray:
        """要素ごとの曲げモーメント (kN·m、絶対値)。

        M-φ を与えていれば骨格曲線上の値、与えていなければ EI·κ。
        """
        curvatures = self.element_curvatures(u)
        if self.moment_curvature is None:
            return self.element_ei * curvatures
        return np.array(
            [abs(self.moment_curvature.moment_at(k)) for k in curvatures]
        )

    def _updated_element_ei(self, u: np.ndarray) -> np.ndarray:
        """現在の変位から、要素ごとの割線曲げ剛性を求める。"""
        if self.moment_curvature is None:
            return self.element_ei
        curvatures = self.element_curvatures(u)
        return np.array(
            [self.moment_curvature.secant_ei_at(k) for k in curvatures],
            dtype=float,
        )

    # --- 解 ----------------------------------------------------------------

    def solve(
        self,
        u_head: float,
        theta_head: float,
        max_iter: int = 40,
        tol: float = 1.0e-9,
    ) -> HeadResponse:
        """杭頭変位 (u, θ) を与えて杭頭反力と接線剛性を求める。

        杭頭の 2 自由度を拘束し、残りの自由度について Newton-Raphson で
        釣合いを解く。杭先端は自由(境界条件なし)。

        M-φ を与えている場合は、反復のたびに要素ごとの割線曲げ剛性を
        更新する(杭体の曲げ非線形)。梁の剛性低下ぶんは接線行列にも
        割線剛性として反映するため、収束は Newton 法より緩やかになる。
        その分 ``max_iter`` を大きめにとる。
        """
        u = self._state.copy()
        u[0] = u_head
        u[1] = -theta_head  # φ = −θ

        if self.moment_curvature is not None:
            max_iter = max(max_iter, 200)

        scale = max(abs(u_head), abs(theta_head), 1.0e-6)
        iterations = 0
        for iterations in range(1, max_iter + 1):
            if self.moment_curvature is not None:
                self.element_ei = self._updated_element_ei(u)
                self.beam = _beam_stiffness(
                    self.element_ei, self.element_length, self.n_elements
                )
            residual = self.beam @ u + self._resistance(u)
            free = residual[_HEAD_DOF:]
            if float(np.max(np.abs(free))) <= tol * self.ei * scale:
                break
            tangent = self._spring_tangent(u[0::2])
            jac = self.beam[_HEAD_DOF:, _HEAD_DOF:].copy()
            # 並進自由度の対角に地盤バネの接線剛性を加える
            diag_index = np.arange(0, jac.shape[0], 2)
            node_index = np.arange(1, self.n_elements + 1)
            jac[diag_index, diag_index] += tangent[node_index]
            u[_HEAD_DOF:] -= np.linalg.solve(jac, free)

        self._state = u
        residual = self.beam @ u + self._resistance(u)
        head = residual[:_HEAD_DOF]  # [F1, C1]

        hinges = 0
        exceeds_ultimate = False
        if self.moment_curvature is not None:
            moments = self.element_moments(u)
            hinges = int(
                sum(1 for m in moments if self.moment_curvature.yielded(m))
            )
            ultimate = self.moment_curvature.ultimate_curvature
            exceeds_ultimate = bool(
                np.any(self.element_curvatures(u) > ultimate)
            )

        return HeadResponse(
            shear=float(head[0]),
            moment=float(-head[1]),  # M = −C1
            tangent=self._condensed_tangent(u),
            displacements=u[0::2].copy(),
            plastic_nodes=int(np.count_nonzero(self._spring_tangent(u[0::2]) == 0.0)),
            iterations=iterations,
            plastic_hinges=hinges,
            exceeds_ultimate_curvature=exceeds_ultimate,
        )

    def _condensed_tangent(self, u: np.ndarray) -> np.ndarray:
        """杭頭 2 自由度に縮約した接線剛性((u, θ) 系)。"""
        tangent = self._spring_tangent(u[0::2])
        total = self.beam.copy()
        translational = np.arange(0, total.shape[0], 2)
        total[translational, translational] += tangent

        k_hh = total[:_HEAD_DOF, :_HEAD_DOF]
        k_hi = total[:_HEAD_DOF, _HEAD_DOF:]
        k_ih = total[_HEAD_DOF:, :_HEAD_DOF]
        k_ii = total[_HEAD_DOF:, _HEAD_DOF:]
        condensed = k_hh - k_hi @ np.linalg.solve(k_ii, k_ih)
        # (y, φ) 系 → (u, θ) 系
        return condensed * np.outer(_SIGN, _SIGN)

    def head_stiffness(self) -> np.ndarray:
        """弾性状態の杭頭剛性 ((u, θ) 系の 2×2)。

        地盤バネが上限に達しておらず、杭体も曲げ剛性が低下していない状態の
        値。Chang の式による [[K1, K2], [K3, K4]] と比較できる。
        """
        saved = self.beam
        self.beam = self.beam_elastic
        try:
            return self._condensed_tangent(np.zeros(2 * (self.n_elements + 1)))
        finally:
            self.beam = saved
