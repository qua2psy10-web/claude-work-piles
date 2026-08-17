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

    @property
    def yielded_ground(self) -> bool:
        return self.plastic_nodes > 0


def _beam_stiffness(ei: float, element_length: float, n_elements: int) -> np.ndarray:
    """Hermite 梁要素による全体剛性マトリクス。"""
    ndof = 2 * (n_elements + 1)
    k = np.zeros((ndof, ndof))
    le = element_length
    ke = (ei / le**3) * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le * le, -6.0 * le, 2.0 * le * le],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le * le, -6.0 * le, 4.0 * le * le],
        ]
    )
    for e in range(n_elements):
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        k[np.ix_(idx, idx)] += ke
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
    ) -> None:
        if n_elements < 2:
            raise ValueError("分割数は 2 以上である必要があります")
        if ei <= 0 or diameter <= 0 or length <= 0:
            raise ValueError("EI・杭径・杭長は正の値である必要があります")
        self.kh = _node_kh(kh, n_elements)

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

        self.beam = _beam_stiffness(ei, length / n_elements, n_elements)
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
        """
        u = self._state.copy()
        u[0] = u_head
        u[1] = -theta_head  # φ = −θ

        scale = max(abs(u_head), abs(theta_head), 1.0e-6)
        iterations = 0
        for iterations in range(1, max_iter + 1):
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

        return HeadResponse(
            shear=float(head[0]),
            moment=float(-head[1]),  # M = −C1
            tangent=self._condensed_tangent(u),
            displacements=u[0::2].copy(),
            plastic_nodes=int(np.count_nonzero(self._spring_tangent(u[0::2]) == 0.0)),
            iterations=iterations,
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

        地盤バネが上限に達していない状態の値。Chang の式による
        [[K1, K2], [K3, K4]] と比較できる。
        """
        return self._condensed_tangent(np.zeros(2 * (self.n_elements + 1)))
