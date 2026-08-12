"""レベル2地震時の照査(地震時保有水平耐力法、道示Ⅴ(H24))。

許容応力度設計法(フェーズ1〜3)と異なり、レベル2では基礎の**非線形**な
挙動を追跡し、次を照査する。

1. 基礎に生じる応答が**降伏**に達しないこと、または
2. 降伏する場合は、応答塑性率 μr が許容塑性率 μa 以下であり、
   かつ応答変位が許容変位以下であること

本モジュールは水平力を漸増させる**プッシュオーバー解析**により
水平力 H 〜 フーチング水平変位 δ の関係を求め、降伏点を判定する。

非線形性の取り込み範囲
----------------------
現時点で非線形として扱うのは**杭の軸方向バネのみ**である。
押込み側は押込み支持力の上限値 Pu、引抜き側は引抜き抵抗力の上限値 Pt で
頭打ちとなるバイリニアモデルとする。

水平方向の地盤反力(K1〜K4)は**弾性のまま**であり、水平地盤反力度の
上限値 pHU による塑性化は取り込んでいない。杭体の非線形(M-φ 関係)も
未実装で、杭体の降伏は利用者が与える降伏曲げモーメント My との比較で
判定する。したがって本解析は

    「軸方向バネの塑性化と杭体降伏の判定に基づく降伏点の推定」

であり、道示Ⅴ の完全な地震時保有水平耐力法ではない。制限は
:data:`LIMITATIONS` に列挙し、結果にも注記として付す。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.analysis.displacement import PileReaction, pile_x_coordinates
from core.capacity.bearing import BearingCapacity, compute_bearing_capacity
from core.capacity.section import CORROSION_ALLOWANCE_MM, pile_section
from core.capacity.springs import PileSection, axial_spring, lateral_springs
from core.models.loads import LoadCase
from core.models.pile import Footing, PileArrangement, PileSpec, PileType
from core.models.soil import SoilProfile
from core.standards import SIGMA_Y_STEEL, E0Method


# 本解析の制限(結果の注記として利用者に提示する)
LIMITATIONS: tuple[str, ...] = (
    "水平地盤反力は弾性(K1〜K4)のままであり、水平地盤反力度の上限値 pHU "
    "による塑性化を考慮していない。したがって降伏水平力を過大に、"
    "降伏変位を過小に評価するおそれがある。",
    "杭体の M-φ 関係(非線形)は未実装で、杭体の降伏は入力された降伏曲げ"
    "モーメント My との比較のみで判定する。塑性ヒンジ後の剛性低下は"
    "追跡していない。",
    "押込み支持力の上限値 Pu・引抜き抵抗力の上限値 Pt は、許容応力度設計法の"
    "式で安全率を 1 とした値として算定している(道示Ⅴ の規定との照合が未了)。",
    "許容塑性率 μa・許容変位 δa は道示Ⅴ の表が未照合のため、"
    "利用者の入力値をそのまま用いる。",
    "液状化に伴う地盤定数の低減、群杭効果、側方流動は考慮していない。",
    "水平方向地盤反力係数 kH はレベル1地震時の値(α = 2)を用いている。"
    "レベル2用の地盤反力係数の規定は未照合。",
)


@dataclass(frozen=True)
class AxialSpringModel:
    """杭1本の軸方向バネ(バイリニア)。

    変位 δ(沈下正)に対する軸力 N(押込み正)は

        N(δ) = kv・δ   ただし −Pt ≤ N ≤ Pu

    Pu・Pt は :meth:`from_bearing` で許容応力度設計法の式の安全率を 1 と
    して求める。
    """

    kv: float  # 弾性域の軸方向バネ定数 (kN/m)
    push_limit: float  # 押込み支持力の上限値 Pu (kN、正値)
    pull_limit: float  # 引抜き抵抗力の上限値 Pt (kN、正値)

    def __post_init__(self) -> None:
        if self.kv <= 0:
            raise ValueError("軸方向バネ定数 Kv は正の値である必要があります")
        if self.push_limit <= 0 or self.pull_limit < 0:
            raise ValueError("支持力の上限値が不正です")

    @classmethod
    def from_bearing(cls, kv: float, bearing: BearingCapacity) -> "AxialSpringModel":
        """支持力計算の結果から上限値を求める。

        許容押込み支持力 Ra =(Ru − Ws)/ n + Ws − W で n = 1 とすると
        Pu = Ru − W、許容引抜き力 Pa = Ruf / n + W で n = 1 とすると
        Pt = Ruf + W となる。

        .. warning::
           この「安全率を 1 とする」という導出は、既に照合済みの
           許容応力度設計法の式から一貫させたものであり、道示Ⅴ が
           レベル2用に別途定める上限値との照合は済んでいない。
        """
        return cls(
            kv=kv,
            push_limit=bearing.ru - bearing.w_pile,
            pull_limit=bearing.skin_resistance + bearing.w_pile,
        )

    def reaction(self, disp: float) -> float:
        """変位 δ (m) に対する軸力 (kN、押込み正)。"""
        return min(self.push_limit, max(-self.pull_limit, self.kv * disp))

    def tangent(self, disp: float) -> float:
        """接線剛性 (kN/m)。上限に達していれば 0。"""
        n = self.kv * disp
        return 0.0 if n >= self.push_limit or n <= -self.pull_limit else self.kv

    def is_plastic(self, disp: float) -> bool:
        return self.tangent(disp) == 0.0


@dataclass(frozen=True)
class PushoverStep:
    """プッシュオーバーの1ステップ。"""

    factor: float  # 設計水平力に対する倍率 λ
    h: float  # 水平力 (kN)
    m: float  # モーメント (kN·m)
    u: float  # フーチング水平変位 (m)
    v: float
    theta: float
    reactions: list[PileReaction]
    plastic_axial: int  # 軸方向バネが上限に達した杭の本数
    yielded_piles: int  # 杭体が降伏した杭の本数

    @property
    def max_axial(self) -> float:
        return max(r.axial for r in self.reactions)

    @property
    def min_axial(self) -> float:
        return min(r.axial for r in self.reactions)


@dataclass(frozen=True)
class YieldPoint:
    """基礎の降伏点。"""

    step: PushoverStep
    reason: str  # 降伏と判定した理由

    @property
    def displacement(self) -> float:
        return self.step.u

    @property
    def horizontal_force(self) -> float:
        return self.step.h


@dataclass(frozen=True)
class Level2Result:
    steps: list[PushoverStep]
    yield_point: YieldPoint | None
    response: PushoverStep | None  # λ = 1(設計レベル2荷重)の応答
    allowable_ductility: float | None
    allowable_displacement: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def yielded(self) -> bool:
        """設計レベル2荷重までに基礎が降伏するか。"""
        if self.yield_point is None or self.response is None:
            return False
        return self.yield_point.step.factor <= self.response.factor

    @property
    def response_ductility(self) -> float | None:
        """応答塑性率 μr = δr / δy。降伏しない場合は None。"""
        if not self.yielded or self.response is None or self.yield_point is None:
            return None
        dy = self.yield_point.displacement
        return self.response.u / dy if dy > 0 else math.inf

    @property
    def checks(self) -> list["Level2Check"]:
        results: list[Level2Check] = []
        if self.response is None:
            return results
        if not self.yielded:
            results.append(
                Level2Check(
                    name="基礎の降伏",
                    demand=self.response.h,
                    capacity=(
                        self.yield_point.horizontal_force
                        if self.yield_point
                        else math.inf
                    ),
                    unit="kN",
                    note="応答が降伏に達しないため、塑性率の照査は不要",
                )
            )
        else:
            mu = self.response_ductility
            if self.allowable_ductility is not None and mu is not None:
                results.append(
                    Level2Check(
                        name="応答塑性率",
                        demand=mu,
                        capacity=self.allowable_ductility,
                        unit="—",
                        note="μr = δr / δy",
                    )
                )
        if self.allowable_displacement is not None:
            results.append(
                Level2Check(
                    name="応答変位",
                    demand=self.response.u,
                    capacity=self.allowable_displacement,
                    unit="m",
                )
            )
        return results

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


@dataclass(frozen=True)
class Level2Check:
    name: str
    demand: float
    capacity: float
    unit: str
    note: str = ""

    @property
    def ratio(self) -> float:
        return abs(self.demand) / self.capacity if self.capacity else math.inf

    @property
    def ok(self) -> bool:
        return self.ratio <= 1.0

    @property
    def judgement(self) -> str:
        return "OK" if self.ok else "NG"


def _solve_step(
    xs: np.ndarray,
    axial: AxialSpringModel,
    k1: float,
    k2: float,
    k4: float,
    v_load: float,
    h_load: float,
    m_load: float,
    initial: np.ndarray,
    max_iter: int = 60,
    tol: float = 1.0e-8,
) -> np.ndarray:
    """1ステップの非線形釣合いを Newton-Raphson 法で解く。

    未知数は (u, v, θ)。水平方向は弾性、軸方向はバイリニア。
    """
    n = len(xs)
    x = initial.copy()
    # 荷重の代表スケール(収束判定を荷重の大きさに対する相対値で行う)
    scale = max(abs(v_load), abs(h_load), abs(m_load), 1.0)

    for _ in range(max_iter):
        u, v, theta = x
        disp = v + xs * theta
        forces = np.array([axial.reaction(float(d)) for d in disp])
        tangents = np.array([axial.tangent(float(d)) for d in disp])

        residual = np.array(
            [
                n * k1 * u + n * k2 * theta - h_load,
                float(forces.sum()) - v_load,
                n * k2 * u + n * k4 * theta + float((xs * forces).sum()) - m_load,
            ]
        )
        if float(np.max(np.abs(residual))) <= tol * scale:
            return x

        sum_kt = float(tangents.sum())
        sum_x_kt = float((xs * tangents).sum())
        sum_x2_kt = float((xs**2 * tangents).sum())
        jac = np.array(
            [
                [n * k1, 0.0, n * k2],
                [0.0, sum_kt, sum_x_kt],
                [n * k2, sum_x_kt, n * k4 + sum_x2_kt],
            ]
        )
        try:
            x = x - np.linalg.solve(jac, residual)
        except np.linalg.LinAlgError as exc:
            # 全杭が上限に達すると鉛直・回転方向の剛性が失われる。
            # これは変位が急増する状態そのものであり、解は存在しない。
            raise _Unstable(
                "軸方向バネがすべて上限に達し、鉛直方向の釣合いを保てません"
            ) from exc
    raise _Unstable("釣合い計算が収束しませんでした(変位が急増しています)")


class _Unstable(RuntimeError):
    """釣合いが解けない = 変位が急増する状態。"""


def pushover(
    arrangement: PileArrangement,
    axial: AxialSpringModel,
    k1: float,
    k2: float,
    k4: float,
    v_load: float,
    h_load: float,
    m_load: float,
    yield_moment: float | None = None,
    max_factor: float = 3.0,
    steps: int = 120,
) -> tuple[list[PushoverStep], YieldPoint | None]:
    """水平力を漸増させ、H〜δ 関係と基礎の降伏点を求める。

    鉛直力 ``v_load`` は一定に保ち、``h_load`` と ``m_load`` を倍率 λ で
    比例的に増加させる(λ = 1 が設計レベル2荷重)。

    Parameters
    ----------
    yield_moment:
        杭体の降伏曲げモーメント My (kN·m)。与えると「全杭の杭体が降伏」も
        降伏条件として判定する。省略時は軸方向支持力の上限到達のみで判定する。
    max_factor:
        水平力の最大倍率 λmax。
    steps:
        λ の分割数。

    Returns
    -------
    (各ステップ, 降伏点)。降伏点は λmax までに降伏しなければ None。

    Notes
    -----
    基礎の降伏は次のいずれかが最初に生じた時点とする。

    a) 押込み側の杭の軸力が押込み支持力の上限値 Pu に達する
    b) すべての杭の杭体が降伏する(``yield_moment`` を与えた場合)
    c) 釣合いが保てなくなる(変位の急増)

    この定義は道示Ⅳ 12.10 / Ⅴ の規定に基づくものだが、**原典との照合は
    済んでいない**。
    """
    if steps < 1:
        raise ValueError("steps は 1 以上である必要があります")
    if max_factor <= 0:
        raise ValueError("max_factor は正の値である必要があります")

    xs = np.array(pile_x_coordinates(arrangement), dtype=float)
    n_piles = len(xs)

    results: list[PushoverStep] = []
    yield_point: YieldPoint | None = None
    x = np.zeros(3)

    for factor in _load_factors(max_factor, steps):
        h = h_load * factor
        m = m_load * factor
        try:
            x = _solve_step(
                xs, axial, k1, k2, k4, v_load, h, m, initial=x
            )
        except _Unstable as exc:
            if yield_point is None and results:
                yield_point = YieldPoint(step=results[-1], reason=str(exc))
            break

        u, v, theta = (float(val) for val in x)
        disp = v + xs * theta
        reactions = [
            PileReaction(
                index=idx + 1,
                x=float(px),
                axial=axial.reaction(float(d)),
                shear=k1 * u + k2 * theta,
                moment=k2 * u + k4 * theta,
            )
            for idx, (px, d) in enumerate(zip(xs, disp))
        ]
        plastic = sum(1 for d in disp if axial.is_plastic(float(d)))
        if yield_moment is not None:
            yielded = sum(1 for r in reactions if abs(r.moment) >= yield_moment)
        else:
            yielded = 0

        step = PushoverStep(
            factor=factor,
            h=h,
            m=m,
            u=u,
            v=v,
            theta=theta,
            reactions=reactions,
            plastic_axial=plastic,
            yielded_piles=yielded,
        )
        results.append(step)

        if yield_point is None and factor > 0:
            reason = _yield_reason(step, axial, n_piles, yield_moment)
            if reason:
                yield_point = YieldPoint(step=step, reason=reason)

    return results, yield_point


def _load_factors(max_factor: float, steps: int) -> list[float]:
    """荷重倍率 λ の列。

    設計レベル2荷重 λ = 1 は応答値そのものなので、分割数によらず必ず
    サンプル点に含める(含めないと λ = 1 の応答が得られず、崩壊したものと
    誤判定される)。
    """
    factors = [max_factor * i / steps for i in range(steps + 1)]
    if max_factor >= 1.0 and not any(abs(f - 1.0) < 1.0e-12 for f in factors):
        factors.append(1.0)
        factors.sort()
    return factors


def _yield_reason(
    step: PushoverStep,
    axial: AxialSpringModel,
    n_piles: int,
    yield_moment: float | None,
) -> str | None:
    if step.max_axial >= axial.push_limit:
        return "押込み側の杭の軸力が押込み支持力の上限値に達した"
    if yield_moment is not None and step.yielded_piles >= n_piles:
        return "すべての杭の杭体が降伏した"
    return None


def analyze_level2(
    arrangement: PileArrangement,
    axial: AxialSpringModel,
    k1: float,
    k2: float,
    k4: float,
    v_load: float,
    h_load: float,
    m_load: float,
    yield_moment: float | None = None,
    allowable_ductility: float | None = None,
    allowable_displacement: float | None = None,
    max_factor: float = 3.0,
    steps: int = 120,
) -> Level2Result:
    """レベル2地震時の照査を行う。

    ``allowable_ductility``(許容塑性率 μa)と ``allowable_displacement``
    (許容変位 δa)は道示Ⅴ の表が未照合のため既定値を持たない。
    与えられた場合のみ該当する照査を行う。
    """
    steps_list, yield_point = pushover(
        arrangement,
        axial,
        k1=k1,
        k2=k2,
        k4=k4,
        v_load=v_load,
        h_load=h_load,
        m_load=m_load,
        yield_moment=yield_moment,
        max_factor=max_factor,
        steps=steps,
    )
    response = _response_step(steps_list)

    notes = list(LIMITATIONS)
    if response is None:
        notes.append(
            "設計レベル2荷重(λ = 1)に達する前に釣合いが保てなくなった。"
            "基礎が保有水平耐力に達していると考えられる。"
        )
    if yield_point is not None and allowable_ductility is None:
        notes.append(
            "基礎が降伏しているが許容塑性率 μa が未入力のため、"
            "応答塑性率の照査を行っていない。"
        )
    return Level2Result(
        steps=steps_list,
        yield_point=yield_point,
        response=response,
        allowable_ductility=allowable_ductility,
        allowable_displacement=allowable_displacement,
        notes=notes,
    )


def yield_moment_steel_pipe(
    pile: PileSpec,
    section: PileSection,
    axial: float,
    steel_grade: str = "SKK400",
    corrosion_mm: float = CORROSION_ALLOWANCE_MM,
) -> float:
    """鋼管杭の降伏曲げモーメント My (kN·m)。

    軸力 N が作用する状態で、断面の最外縁の応力度が降伏点 σy に達する
    モーメントとして

        N / A + My / Z = σy  →  My =(σy − N / A)・Z

    .. note::
       これは「最外縁が降伏する」という定義であり、断面全体が塑性化した
       全塑性モーメント Mp ではない。Mp を用いる規定であれば別途の実装が
       必要になる(道示Ⅴ の規定は未照合)。
    """
    if pile.pile_type not in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        raise ValueError(
            f"{pile.pile_type.value}には使用できません(鋼管杭のみ)"
        )
    if steel_grade not in SIGMA_Y_STEEL:
        raise ValueError(
            f"鋼材 {steel_grade} の降伏点が未定義です。"
            f"対応材質: {sorted(SIGMA_Y_STEEL)}"
        )
    sigma_y = SIGMA_Y_STEEL[steel_grade]
    section_modulus = section.inertia / (pile.diameter / 2.0)
    sigma_axial = axial / section.area / 1000.0  # N/mm2
    if sigma_axial >= sigma_y:
        raise ValueError(
            f"軸応力度 {sigma_axial:.1f} N/mm² が降伏点 {sigma_y} N/mm² に"
            "達しており、曲げに対する余裕がありません"
        )
    return (sigma_y - sigma_axial) * section_modulus * 1000.0


def run_level2(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    v_load: float,
    h_load: float,
    m_load: float,
    fck: int = 24,
    yield_moment: float | None = None,
    steel_grade: str = "SKK400",
    allowable_ductility: float | None = None,
    allowable_displacement: float | None = None,
    e0_method: E0Method = E0Method.N_VALUE,
    max_factor: float = 3.0,
    steps: int = 120,
) -> Level2Result:
    """杭・地盤の諸元からレベル2地震時の照査までを一括で行う。

    ``yield_moment`` を省略した場合、鋼管杭・鋼管ソイルセメント杭では
    :func:`yield_moment_steel_pipe` により自動算定する(軸力は設計レベル2
    荷重時の平均軸力 V / 杭本数を用いる)。それ以外の杭種では杭体降伏の
    判定を行わず、軸方向支持力の上限到達のみで降伏を判定する。
    """
    section = pile_section(pile, fck=fck)
    bearing = compute_bearing_capacity(pile, profile, footing.embedment)
    kv = axial_spring(pile, section)
    axial = AxialSpringModel.from_bearing(kv, bearing)
    springs = lateral_springs(
        pile, section, profile, footing.embedment, LoadCase.LEVEL1_EQ,
        e0_method=e0_method,
    )

    extra_notes: list[str] = []
    if yield_moment is None and pile.pile_type in (
        PileType.STEEL_PIPE,
        PileType.STEEL_PIPE_SOIL_CEMENT,
    ):
        mean_axial = v_load / (arrangement.nx * arrangement.ny)
        yield_moment = yield_moment_steel_pipe(
            pile, section, mean_axial, steel_grade=steel_grade
        )
        extra_notes.append(
            f"杭体の降伏曲げモーメントを My = {yield_moment:.0f} kN·m と"
            f"自動算定した(平均軸力 {mean_axial:.0f} kN、{steel_grade}、"
            "最外縁が降伏点に達する定義)。"
        )
    elif yield_moment is None:
        extra_notes.append(
            f"{pile.pile_type.value}の降伏曲げモーメント My が未入力のため、"
            "杭体降伏による降伏判定を行っていない"
            "(軸方向支持力の上限到達のみで判定)。"
        )

    result = analyze_level2(
        arrangement,
        axial,
        k1=springs.k1,
        k2=springs.k2,
        k4=springs.k4,
        v_load=v_load,
        h_load=h_load,
        m_load=m_load,
        yield_moment=yield_moment,
        allowable_ductility=allowable_ductility,
        allowable_displacement=allowable_displacement,
        max_factor=max_factor,
        steps=steps,
    )
    return Level2Result(
        steps=result.steps,
        yield_point=result.yield_point,
        response=result.response,
        allowable_ductility=result.allowable_ductility,
        allowable_displacement=result.allowable_displacement,
        notes=extra_notes + result.notes,
    )


def _response_step(steps: list[PushoverStep]) -> PushoverStep | None:
    """λ = 1(設計レベル2荷重)に最も近いステップ。"""
    reached = [s for s in steps if s.factor <= 1.0 + 1.0e-12]
    if not reached or reached[-1].factor < 1.0 - 1.0e-6:
        # λ = 1 まで解けていない(その前に不安定になった)
        return None
    return reached[-1]
