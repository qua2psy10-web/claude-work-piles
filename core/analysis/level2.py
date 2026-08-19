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

水平方向は、地層に地震時受働土圧係数 KEP が入力されていれば**分布バネ
モデル**(:mod:`core.analysis.bnwf`)で解き、水平地盤反力度の上限値 pHU に
よる地盤の塑性化を取り込む。KEP が無い場合は従来どおり杭頭バネ K1〜K4 に
よる弾性解析となり、:func:`check_soil_reaction_limit` により pHU を超える
区間があるかを**診断**して非安全側になっている深度を提示する。

杭体の曲げ剛性低下(M-φ 関係)は、``moment_curvature`` に骨格曲線を与えた
場合に**要素ごとに割線剛性を更新して追跡する**(分布バネモデルのとき)。
骨格曲線の折れ点の値そのものを算定できるのは鋼管杭のバイリニア型
(:func:`yield_moment_steel_pipe` / :func:`plastic_moment_steel_pipe`)のみで、
RC・PHC・SC杭のトリリニア型は利用者が与える。骨格曲線を与えない場合、杭体は
弾性のままで、降伏は ``yield_moment`` との比較のみで判定する。

したがって本解析は

    「軸方向バネ・水平地盤バネ・(骨格曲線を与えた場合は)杭体曲げの
      非線形を追跡するプッシュオーバーによる降伏点の推定」

である。道示Ⅴ の地震時保有水平耐力法との残る主な差は、除荷経路(履歴則)を
持たない単調載荷であることと、RC・PHC・SC杭の骨格曲線の折れ点を断面から
自動算定できないことである。制限は :data:`LIMITATIONS`
(および解析方法に応じて :data:`LIMITATION_ELASTIC_GROUND` /
:data:`LIMITATION_BNWF`)に列挙し、結果にも注記として付す。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.analysis.bnwf import PileLateralModel
from core.analysis.displacement import PileReaction, pile_x_coordinates
from core.analysis.section_forces import distribution
from core.capacity.bearing import BearingCapacity, compute_bearing_capacity
from core.capacity.lateral_limit import p_hu
from core.capacity.section import (
    CORROSION_ALLOWANCE_MM,
    corroded_tube,
    hollow_circle,
    pile_section,
)
from core.capacity.springs import (
    LateralSprings,
    PileSection,
    axial_spring,
    group_pile_factor,
    lateral_springs,
    layered_kh,
)
from core.models.loads import LoadCase
from core.models.pile import Footing, PileArrangement, PileSpec, PileType
from core.models.soil import SoilProfile
from core.section.moment_curvature import MomentCurvature
from core.section.rc import RebarLayout, StirrupLayout
from core.section.shear import ShearCapacity, shear_capacity_level2
from core.soil.liquefaction import SoilReduction
from core.standards import (
    ALLOWABLE_DUCTILITY_CIP_HIGH_GRADE,
    ALLOWABLE_DUCTILITY_PILE,
    ALLOWABLE_FOOTING_ROTATION,
    CONCRETE_AXIAL_CAPACITY_COEF,
    EC_CONCRETE,
    GROUP_PILE_SPACING_RATIO,
    HIGH_GRADE_REBAR_FOR_DUCTILITY,
    PRESTRESSING_STEEL_YIELD_POINT,
    REBAR_YIELD_POINT,
    SIGMA_Y_STEEL,
    E0Method,
    StructureType,
)


# 本解析の制限(結果の注記として利用者に提示する)
# 水平地盤バネを弾性のまま解いた場合にのみ付す制限
LIMITATION_ELASTIC_GROUND = (
    "水平地盤反力を弾性(杭頭バネ K1〜K4)のまま解いており、水平地盤反力度の"
    "上限値 pHU による塑性化を取り込んでいない。したがって降伏水平力を過大に、"
    "降伏変位を過小に評価するおそれがある。"
    "また水平方向地盤反力係数 kH は、Chang の式が地盤を一様と仮定するため、"
    "設計地盤面から 1/β の区間で平均した E0 による**単一の値**しか持てず、"
    "深度ごと・地層ごとの kH を反映できない。"
    "地層に KEP を入力すると分布バネモデル(BNWF)で解析され、"
    "いずれの制限も解消する。"
)

# 分布バネモデルで解いた場合の注意
LIMITATION_BNWF = (
    "分布バネモデルは杭を有限個の要素に分割した数値解であり、地盤バネが弾性の"
    "範囲では Chang の式(K1〜K4)に 2次で収束する。分割数が粗いと杭頭"
    "モーメントに数%の差が出る。"
)

LIMITATIONS: tuple[str, ...] = (
    "杭体の M-φ 骨格曲線を ``moment_curvature`` に与えると、分布バネモデル"
    "(BNWF)で**要素ごとに割線曲げ剛性を低下させる**(塑性ヒンジ)。"
    "ただし骨格曲線の**折れ点の値そのもの**を算定できるのは鋼管杭・鋼管"
    "ソイルセメント杭のバイリニア型(降伏 My → 全塑性 Mp)のみである。"
    "場所打ちRC杭・PHC杭・SC杭のトリリニア型(ひび割れ Mc・降伏 My・"
    "終局 Mu)の折れ点は、コンクリートの引張強度・終局ひずみなど原典未照合の"
    "定数を必要とするため算定していない。これらの杭種では折れ点を利用者が"
    "与える必要がある(与えなければ杭体は弾性のまま扱われる)。"
    "また要素内のモーメント変化は両端の大きいほうで代表させた近似であり、"
    "除荷経路(履歴則)は追跡していない(単調載荷のプッシュオーバーのみ)。",
    "押込み支持力の上限値 Pu・引抜き抵抗力の上限値 Pt は、**地盤から決まる値**"
    "(許容応力度設計法の式で安全率を 1 とした値)と、**杭体から決まる値**"
    "(Rpu = 0.85σck・Ac + σy・As、Ptu = σy・As)の小さいほうとしている。"
    "杭体側は**場所打ち杭**(軸方向鉄筋を入力した場合)、**PHC杭・SC杭**"
    "(コンクリートの負担項を持つ)、**鋼管杭・鋼管ソイルセメント杭**"
    "(Rpu = Ptu = σy・As。固化体・コンクリートの負担は見込まない)を算定する。"
    "他社製品の計算書サンプルとは一致するが**道示の原典は未照合**である。"
    "算定しない杭種(RC杭・H鋼杭)では地盤から決まる値のみとなり、"
    "**引抜き側を過大評価しうる**。"
    "押込み側は、計算例が杭の重量 W を控除していないのに対し本ソフトは"
    "控除している(本ソフトのほうが小さく安全側)。",
    "群杭効果のうち、杭中心間隔が 2.5D 未満のときの水平方向地盤反力係数の"
    "補正係数 μ は実装したが(分布バネモデルでは適用しない)、"
    "**仮想ケーソン基礎とみなした押込み支持力の上限**と支持力のブロック破壊、"
    "および側方流動は考慮していない。"
    "液状化に伴う土質定数の低減 DE は、液状化判定の結果を渡した場合にのみ"
    "考慮する。",
    "**水平方向地盤反力係数** kH は α = 2 で算定している"
    "(レベル1・レベル2とも α = 2 であることは第29回に照合済み)。"
    "換算載荷幅 BH は常時の条件で定めた共通値であり、地層ごとに求め直さない。"
    "(なお道示Ⅴ の**設計水平震度** kH(khc・khg)とは別量である。"
    "こちらは慣性力・土圧の算定に用いるもので、本ソフトはレベル2の荷重を"
    "利用者から与えられる前提のため算定していない。)",
)


@dataclass(frozen=True)
class PileBodyAxialLimits:
    """杭体そのものから決まる軸方向支持力の上限値 (kN)。

        押込み Rpu = 0.85・σck・Ac + σy・As
        引抜き Ptu = σy・As

    地盤がいくら支えても杭体が潰れる・鉄筋が降伏する以上の力は伝わらない。
    地盤から決まる上限値との**小さいほう**が実際の上限になる。
    """

    push: float  # Rpu
    pull: float  # Ptu
    concrete_area: float  # Ac (m2)
    rebar_area: float  # As (m2)


def pile_body_axial_limits(
    pile: PileSpec,
    rebar: "RebarLayout | None" = None,
    fck: int = 24,
    rebar_grade: str = "SD345",
    prestressing_steel_area: float | None = None,
    prestressing_steel_yield: float = PRESTRESSING_STEEL_YIELD_POINT,
    section: "PileSection | None" = None,
    steel_grade: str = "SKK400",
    corrosion_mm: float = CORROSION_ALLOWANCE_MM,
) -> PileBodyAxialLimits:
    """杭体から決まる軸方向支持力の上限値。

        Rpu = 0.85・σck・Ac + σy・As    (押込み)
        Ptu = σy・As                     (引抜き)

    ``Ac`` は杭体コンクリートの断面積、``As`` は鋼材の断面積、``σy`` は
    その降伏点。杭種により中身が変わる。

    ============ ==================== ==============================
    杭種         Ac                   As・σy
    ============ ==================== ==============================
    場所打ち杭   中実円 πD²/4         軸方向鉄筋 × 鉄筋の降伏点
    PHC杭        中空円環             PC鋼材量 × PC鋼材の降伏点
    SC杭         中空円環(鋼管内側)   鋼管の純断面積 × 鋼管の降伏点
    鋼管杭       0(見込まない)        鋼管の純断面積 × 鋼管の降伏点
    鋼管ソイル   0(固化体は見込まない) 鋼管の純断面積 × 鋼管の降伏点
    セメント杭
    ============ ==================== ==============================

    PHC杭では ``prestressing_steel_area`` (m²) の入力が必要である
    (本ソフトは PC鋼材量をモデルに持たないため)。鋼管杭・鋼管ソイル
    セメント杭では ``section``(:func:`core.capacity.section.pile_section`
    の戻り値。腐食代控除後の純断面積を持つ)の入力が必要である。SC杭は
    ``wall_thickness``・``concrete_thickness`` から自動算定する(応力度
    照査 :func:`core.section.checks._check_sc` と同じ幾何)。

    .. note::
       出典は他社製品の計算書サンプルであり**道示の原典は未照合**である。
       場所打ち杭 φ1200・σck=24・D25×24本 → 27267 kN(本式 27267.3)、
       PHC杭 φ800・t=110・σck=80・PC鋼材 25.120cm²・σy=1275 → 19417 kN
       (本式 19417.2)。引抜き側は同サンプルの 7.6.5 が
       ``Ppu = σy・As`` と明記している。

       鋼管ソイルセメント杭 φ1000(固化体)・鋼管径800・t=19.0・SKK490 →
       Rpu = Ppu = σy・As = 13894 kN(本式 13894.0。第42回、Kui_3の
       7.7.4/7.7.5)。**押込み側にも固化体・コンクリートの項が無く**、
       押込み・引抜きが同一の式になる点が場所打ち杭・PHC杭と異なる。

       SC杭 φ700・上杭SKK490(t=14mm)・σck=80・腐食代1mm → Kui_11 の
       7.7.4/7.7.5(第56回)に Rpu = 0.85・σck・Ac + σy・As = 19578kN、
       Ppu = σy・As = 8812kN と明記されており、**場所打ち杭・PHC杭と
       同型の式**(コンクリートの負担項を持つ)であることを確認した。
       鋼管ソイルセメント杭(コンクリート項なし)とは異なる扱いになる。

    .. important::
       **RC杭・H鋼杭のみ未対応**。式が確認できていないため、当てずっぽう
       を避けて対象外としている。
    """
    sigma_ck = fck * 1000.0  # N/mm2 → kN/m2
    if pile.pile_type == PileType.CAST_IN_PLACE:
        if rebar is None:
            raise ValueError("場所打ち杭には軸方向鉄筋の入力が必要です")
        if fck not in EC_CONCRETE:
            raise ValueError(f"σck={fck} は未対応です")
        if rebar_grade not in REBAR_YIELD_POINT:
            raise ValueError(
                f"鉄筋材質 {rebar_grade} の降伏点が未定義です。"
                f"対応材質: {sorted(REBAR_YIELD_POINT)}"
            )
        concrete_area = math.pi * pile.diameter**2 / 4.0
        steel_area = rebar.total_area
        sigma_y = REBAR_YIELD_POINT[rebar_grade] * 1000.0
    elif pile.pile_type == PileType.PHC:
        if pile.concrete_thickness is None:
            raise ValueError(
                "PHC杭はコンクリート部の肉厚 concrete_thickness (mm) の"
                "入力が必要です"
            )
        if prestressing_steel_area is None or prestressing_steel_area <= 0:
            raise ValueError(
                "PHC杭の杭体から決まる支持力の上限値には PC鋼材量 "
                "prestressing_steel_area (m²) の入力が必要です"
            )
        concrete_area, _ = hollow_circle(
            pile.diameter, pile.concrete_thickness / 1000.0
        )
        steel_area = prestressing_steel_area
        sigma_y = prestressing_steel_yield * 1000.0
        # PHC杭のコンクリートは σck = 80 が標準。ヤング係数の表引き
        # (EC_CONCRETE)は範囲外なので行わず、fck をそのまま使う
    elif pile.pile_type in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        if section is None:
            raise ValueError(
                "鋼管杭・鋼管ソイルセメント杭の杭体から決まる支持力の上限値"
                "には断面諸元 section の入力が必要です"
            )
        if steel_grade not in SIGMA_Y_STEEL:
            raise ValueError(
                f"鋼材種別 {steel_grade} の降伏点が未定義です。"
                f"対応材質: {sorted(SIGMA_Y_STEEL)}"
            )
        # 鋼管ソイルセメント杭も固化体・コンクリートの負担を見込まず、
        # 鋼管のみで決まる(Ac = 0)。押込み・引抜きが同一の式になる。
        concrete_area = 0.0
        steel_area = section.area
        sigma_y = SIGMA_Y_STEEL[steel_grade] * 1000.0
    elif pile.pile_type == PileType.SC:
        if pile.wall_thickness is None:
            raise ValueError("SC杭は鋼管の板厚 wall_thickness の入力が必要です")
        if pile.concrete_thickness is None:
            raise ValueError(
                "SC杭はコンクリート部の肉厚 concrete_thickness (mm) の"
                "入力が必要です"
            )
        if steel_grade not in SIGMA_Y_STEEL:
            raise ValueError(
                f"鋼材種別 {steel_grade} の降伏点が未定義です。"
                f"対応材質: {sorted(SIGMA_Y_STEEL)}"
            )
        # 応力度照査(core.section.checks._check_sc)と同じ幾何: 腐食は外面
        # から控除、鋼管の内径 = コンクリートの外径。
        steel_outer, t_steel = corroded_tube(
            pile.diameter, pile.wall_thickness, corrosion_mm
        )
        concrete_outer = steel_outer - 2.0 * t_steel
        concrete_area, _ = hollow_circle(
            concrete_outer, pile.concrete_thickness / 1000.0
        )
        steel_area, _ = hollow_circle(steel_outer, t_steel)
        sigma_y = SIGMA_Y_STEEL[steel_grade] * 1000.0
    else:
        raise ValueError(
            f"{pile.pile_type.value}の杭体から決まる支持力の上限値は未実装です"
            "(場所打ち杭・PHC杭・鋼管杭・鋼管ソイルセメント杭・SC杭のみ。"
            "式が確認できていません)"
        )
    steel = sigma_y * steel_area
    return PileBodyAxialLimits(
        push=CONCRETE_AXIAL_CAPACITY_COEF * sigma_ck * concrete_area + steel,
        pull=steel,
        concrete_area=concrete_area,
        rebar_area=steel_area,
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
    def from_bearing(
        cls,
        kv: float,
        bearing: BearingCapacity,
        body: "PileBodyAxialLimits | None" = None,
    ) -> "AxialSpringModel":
        """支持力計算の結果から上限値を求める。

        **地盤から決まる上限値**は、許容応力度設計法の式で安全率を 1 と
        して求める。許容押込み支持力 Ra =(Ru − Ws)/ n + Ws − W で n = 1
        とすると Pu = Ru − W、許容引抜き力 Pa = Ruf / n + W で n = 1 と
        すると Pt = Ruf + W となる。

        ``body`` を与えると、**杭体から決まる上限値**との小さいほうを採る
        (:func:`pile_body_axial_limits`)。杭体側を入れないと引抜きの
        上限値を大きく過大評価することがある(第35回の計算例では 1.47 倍)。

        .. warning::
           「安全率を 1 とする」という導出は、既に照合済みの許容応力度
           設計法の式から一貫させたものであり、道示Ⅴ が定める上限値との
           照合は済んでいない。第35回に計算例と突き合わせたところ、
           **押込み側は計算例が W を控除していない**(本ソフトのほうが
           小さく安全側)という違いが残っている。
        """
        push = bearing.ru - bearing.w_pile
        pull = bearing.skin_resistance + bearing.w_pile
        if body is not None:
            push = min(push, body.push)
            pull = min(pull, body.pull)
        return cls(kv=kv, push_limit=push, pull_limit=pull)

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
    plastic_ground_nodes: int = 0  # pHU に達した水平地盤バネの数(BNWF時)

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


def allowable_ductility_for(
    structure_type: StructureType,
    pile: PileSpec | None = None,
    rebar_grade: str | None = None,
) -> float | None:
    """杭基礎の許容塑性率 μa(道示Ⅴ 12.4)。

    直杭を前提とする(本ソフトは斜杭に未対応)。場所打ち杭に SD390・SD490 を
    用いる場合は許容塑性率が下がり、橋台では基礎の塑性化を考慮できない
    (None を返す)。
    """
    key = structure_type.value
    high_grade = (
        pile is not None
        and pile.pile_type == PileType.CAST_IN_PLACE
        and rebar_grade in HIGH_GRADE_REBAR_FOR_DUCTILITY
    )
    if high_grade:
        return ALLOWABLE_DUCTILITY_CIP_HIGH_GRADE[key]
    return ALLOWABLE_DUCTILITY_PILE[key]


@dataclass(frozen=True)
class Level2Result:
    steps: list[PushoverStep]
    yield_point: YieldPoint | None
    response: PushoverStep | None  # λ = 1(設計レベル2荷重)の応答
    allowable_ductility: float | None
    allowable_displacement: float | None
    allowable_rotation: float | None = ALLOWABLE_FOOTING_ROTATION
    soil_reaction: SoilReactionCheck | None = None  # pHU との突合(診断)
    notes: list[str] = field(default_factory=list)
    # レベル2のせん断耐力 Ps(道示Ⅳ 5.2.3)。場所打ち杭で軸方向鉄筋を
    # 入力した場合のみ算定する。
    shear_capacity: "ShearCapacity | None" = None
    response_shear: float | None = None  # 設計レベル2荷重時の杭頭せん断力 (kN)
    # kH に乗じた群杭の補正係数 μ。分布バネモデルでは 1.0(補正しない)
    group_factor: float = 1.0

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
        if self.allowable_rotation is not None:
            results.append(
                Level2Check(
                    name="フーチング底面の回転角",
                    demand=abs(self.response.theta),
                    capacity=self.allowable_rotation,
                    unit="rad",
                    note="過大な残留変位を生じさせないための規定(0.02 rad = 1/50)",
                )
            )
        if self.shear_capacity is not None and self.response_shear is not None:
            cap = self.shear_capacity
            results.append(
                Level2Check(
                    name="杭体のせん断耐力(斜引張破壊)",
                    demand=abs(self.response_shear),
                    capacity=cap.total,
                    unit="kN",
                    note=(
                        f"Sus = Sc + Ss = {cap.sc:.0f} + {cap.ss:.0f} kN"
                        "(道示Ⅳ 5.2.3)"
                    ),
                )
            )
            results.append(
                Level2Check(
                    name="コンクリートの斜め圧縮破壊",
                    demand=abs(self.response_shear),
                    capacity=cap.web_crushing_capacity,
                    unit="kN",
                    note=(
                        f"Suc = τmax・bw・d = {cap.tau_max:.1f}×"
                        f"{cap.width * 1000:.0f}×{cap.effective_depth * 1000:.0f}"
                        "(道示Ⅲ 4.3.4、表-4.3.2。斜引張鉄筋を増やしても"
                        "この上限は超えられない)"
                    ),
                )
            )
        if self.allowable_displacement is not None:
            results.append(
                Level2Check(
                    name="応答水平変位",
                    demand=self.response.u,
                    capacity=self.allowable_displacement,
                    unit="m",
                    note="道示の規定ではなく利用者が指定した制限値",
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
    lateral: LateralModel,
    v_load: float,
    h_load: float,
    m_load: float,
    initial: np.ndarray,
    max_iter: int = 60,
    tol: float = 1.0e-8,
) -> np.ndarray:
    """1ステップの非線形釣合いを Newton-Raphson 法で解く。

    未知数はフーチングの (u, v, θ)。軸方向はバイリニア、水平方向は
    ``lateral`` が返す接線剛性による(弾性バネまたは BNWF)。
    """
    x = initial.copy()
    # 荷重の代表スケール(収束判定を荷重の大きさに対する相対値で行う)
    scale = max(abs(v_load), abs(h_load), abs(m_load), 1.0)

    for _ in range(max_iter):
        u, v, theta = x
        disp = v + xs * theta
        forces = np.array([axial.reaction(float(d)) for d in disp])
        tangents = np.array([axial.tangent(float(d)) for d in disp])
        shear, moment, lateral_tangent = lateral.responses(float(u), float(theta))

        residual = np.array(
            [
                float(shear.sum()) - h_load,
                float(forces.sum()) - v_load,
                float(moment.sum()) + float((xs * forces).sum()) - m_load,
            ]
        )
        if float(np.max(np.abs(residual))) <= tol * scale:
            return x

        sum_kt = float(tangents.sum())
        sum_x_kt = float((xs * tangents).sum())
        sum_x2_kt = float((xs**2 * tangents).sum())
        jac = np.array(
            [
                [lateral_tangent[0, 0], 0.0, lateral_tangent[0, 1]],
                [0.0, sum_kt, sum_x_kt],
                [lateral_tangent[1, 0], sum_x_kt, lateral_tangent[1, 1] + sum_x2_kt],
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
    k1: float | None = None,
    k2: float | None = None,
    k4: float | None = None,
    *,
    lateral: LateralModel | None = None,
    v_load: float = 0.0,
    h_load: float = 0.0,
    m_load: float = 0.0,
    yield_moment: float | None = None,
    max_factor: float = 3.0,
    steps: int = 120,
) -> tuple[list[PushoverStep], YieldPoint | None]:
    """水平力を漸増させ、H〜δ 関係と基礎の降伏点を求める。

    鉛直力 ``v_load`` は一定に保ち、``h_load`` と ``m_load`` を倍率 λ で
    比例的に増加させる(λ = 1 が設計レベル2荷重)。

    Parameters
    ----------
    k1, k2, k4:
        弾性の杭頭バネ。``lateral`` を与える場合は不要。
    lateral:
        水平方向のモデル。省略時は ``k1``〜``k4`` による弾性バネを用いる。
        :class:`BnwfLateralModel` を与えると、地盤の塑性化(pHU による
        頭打ち)を解析に取り込む。
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
    if lateral is None:
        if k1 is None or k2 is None or k4 is None:
            raise ValueError("k1・k2・k4 または lateral のいずれかが必要です")
        lateral = LinearLateralModel(n_piles=n_piles, k1=k1, k2=k2, k4=k4)

    results: list[PushoverStep] = []
    yield_point: YieldPoint | None = None
    x = np.zeros(3)

    for factor in _load_factors(max_factor, steps):
        h = h_load * factor
        m = m_load * factor
        try:
            x = _solve_step(
                xs, axial, lateral, v_load, h, m, initial=x
            )
        except _Unstable as exc:
            if yield_point is None and results:
                yield_point = YieldPoint(step=results[-1], reason=str(exc))
            break

        u, v, theta = (float(val) for val in x)
        disp = v + xs * theta
        head_shear, head_moment, _ = lateral.responses(u, theta)
        reactions = [
            PileReaction(
                index=idx + 1,
                x=float(px),
                axial=axial.reaction(float(d)),
                shear=float(head_shear[idx]),
                moment=float(head_moment[idx]),
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
            plastic_ground_nodes=lateral.plastic_ground_nodes,
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
    k1: float | None = None,
    k2: float | None = None,
    k4: float | None = None,
    *,
    lateral: LateralModel | None = None,
    v_load: float = 0.0,
    h_load: float = 0.0,
    m_load: float = 0.0,
    yield_moment: float | None = None,
    allowable_ductility: float | None = None,
    allowable_displacement: float | None = None,
    allowable_rotation: float | None = ALLOWABLE_FOOTING_ROTATION,
    max_factor: float = 3.0,
    steps: int = 120,
) -> Level2Result:
    """レベル2地震時の照査を行う。

    Parameters
    ----------
    allowable_ductility:
        許容塑性率 μa。省略時は照査しない。杭種・下部構造から求める場合は
        :func:`allowable_ductility_for` を使う(:func:`run_level2` は自動)。
    allowable_rotation:
        フーチング底面位置の許容回転角 (rad)。道示Ⅴ の 0.02 rad が既定値。
    allowable_displacement:
        水平変位の制限値 (m)。道示Ⅴ の規定ではないため既定値を持たない。
    """
    steps_list, yield_point = pushover(
        arrangement,
        axial,
        k1=k1,
        k2=k2,
        k4=k4,
        lateral=lateral,
        v_load=v_load,
        h_load=h_load,
        m_load=m_load,
        yield_moment=yield_moment,
        max_factor=max_factor,
        steps=steps,
    )
    response = _response_step(steps_list)

    notes = [
        LIMITATION_BNWF if lateral is not None else LIMITATION_ELASTIC_GROUND,
        *LIMITATIONS,
    ]
    if response is None:
        notes.append(
            "設計レベル2荷重(λ = 1)に達する前に釣合いが保てなくなった。"
            "基礎が保有水平耐力に達していると考えられる。"
        )
    if yield_point is not None and allowable_ductility is None:
        notes.append(
            "基礎が降伏しているが許容塑性率 μa が未設定のため、"
            "応答塑性率の照査を行っていない。"
        )
    return Level2Result(
        steps=steps_list,
        yield_point=yield_point,
        response=response,
        allowable_ductility=allowable_ductility,
        allowable_displacement=allowable_displacement,
        allowable_rotation=allowable_rotation,
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
    # 断面係数は腐食しろ控除後の外径で求める(section も控除後の断面)
    outer, _ = corroded_tube(pile.diameter, pile.wall_thickness, corrosion_mm)
    section_modulus = section.inertia / (outer / 2.0)
    sigma_axial = axial / section.area / 1000.0  # N/mm2
    if sigma_axial >= sigma_y:
        raise ValueError(
            f"軸応力度 {sigma_axial:.1f} N/mm² が降伏点 {sigma_y} N/mm² に"
            "達しており、曲げに対する余裕がありません"
        )
    return (sigma_y - sigma_axial) * section_modulus * 1000.0


class LateralModel:
    """フーチング変位 (u, θ) から全杭の杭頭反力と接線剛性を返すモデル。

    フーチングを剛体としているため全杭の杭頭変位は等しいが、砂質地盤では
    最前列以外の杭の pHU が 1/2 になるため、杭ごとに応答が異なり得る。
    """

    def responses(
        self, u: float, theta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(杭ごとの H, 杭ごとの M, 全杭合計の接線剛性 2×2) を返す。"""
        raise NotImplementedError

    @property
    def plastic_ground_nodes(self) -> int:
        """直近の :meth:`responses` で塑性化した地盤バネの数。"""
        return 0

    @property
    def plastic_hinges(self) -> int:
        """直近の :meth:`responses` で降伏した杭体要素の数。

        杭体の曲げ非線形(M-φ)を扱わないモデルでは常に 0。
        """
        return 0

    @property
    def exceeds_ultimate_curvature(self) -> bool:
        """終局曲率 φu を超えた杭体要素があるか。"""
        return False


@dataclass
class LinearLateralModel(LateralModel):
    """従来どおりの弾性杭頭バネ K1〜K4 によるモデル。"""

    n_piles: int
    k1: float
    k2: float
    k4: float

    def responses(
        self, u: float, theta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shear = np.full(self.n_piles, self.k1 * u + self.k2 * theta)
        moment = np.full(self.n_piles, self.k2 * u + self.k4 * theta)
        tangent = self.n_piles * np.array(
            [[self.k1, self.k2], [self.k2, self.k4]]
        )
        return shear, moment, tangent


@dataclass
class BnwfLateralModel(LateralModel):
    """分布バネモデル(BNWF)による杭頭応答。

    最前列とそれ以外で pHU が異なるため、2 つの杭モデルを持ち、杭の
    x 座標から所属を決める。
    """

    front_mask: np.ndarray  # 各杭が最前列か
    front: PileLateralModel
    back: PileLateralModel
    _plastic: int = 0
    _hinges: int = 0
    _exceeds_ultimate: bool = False

    def responses(
        self, u: float, theta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        front = self.front.solve(u, theta)
        n_front = int(np.count_nonzero(self.front_mask))
        n_back = len(self.front_mask) - n_front

        shear = np.where(self.front_mask, front.shear, 0.0)
        moment = np.where(self.front_mask, front.moment, 0.0)
        tangent = n_front * front.tangent
        self._plastic = n_front * front.plastic_nodes
        self._hinges = n_front * front.plastic_hinges
        self._exceeds_ultimate = front.exceeds_ultimate_curvature

        if n_back:
            back = self.back.solve(u, theta)
            shear = np.where(self.front_mask, shear, back.shear)
            moment = np.where(self.front_mask, moment, back.moment)
            tangent = tangent + n_back * back.tangent
            self._plastic += n_back * back.plastic_nodes
            self._hinges += n_back * back.plastic_hinges
            self._exceeds_ultimate = (
                self._exceeds_ultimate or back.exceeds_ultimate_curvature
            )
        return shear, moment, tangent

    @property
    def plastic_ground_nodes(self) -> int:
        return self._plastic

    @property
    def plastic_hinges(self) -> int:
        """直近の :meth:`responses` で降伏した杭体要素の数(全杭合計)。"""
        return self._hinges

    @property
    def exceeds_ultimate_curvature(self) -> bool:
        """終局曲率 φu を超えた杭体要素があるか。"""
        return self._exceeds_ultimate

    @property
    def kh_range(self) -> tuple[float, float]:
        """節点ごとの kH の (最小, 最大) (kN/m3)。一定なら同じ値になる。"""
        return float(self.front.kh.min()), float(self.front.kh.max())


def build_bnwf_model(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    section: PileSection,
    springs: LateralSprings,
    reduction: SoilReduction | None = None,
    n_elements: int = 50,
    layered: bool = True,
    moment_curvature: "MomentCurvature | None" = None,
) -> BnwfLateralModel:
    """杭・地盤の諸元から BNWF モデルを組み立てる。

    節点ごとの pHU を :func:`core.capacity.lateral_limit.p_hu` で求める。
    杭先端が地盤モデルの下端より深い場合は、最下層の値を延長して用いる。

    水平方向地盤反力係数 kH も**節点ごと**に、当該深度の地層の変形係数 E0 から
    :func:`core.capacity.springs.layered_kh` で求める。換算載荷幅 BH は
    ``springs`` が常時の条件で定めた共通値を用いる。``layered`` を偽にすると
    従来どおり ``springs.kh``(1/β 区間で平均した E0 による単一値)を杭長に
    わたって一定として扱う。

    ``reduction`` を与えると、液状化の低減係数 DE を**節点ごとに**バネ定数と
    上限値に乗じる。杭頭バネ K1〜K4 による弾性解析では深度平均に頼るしか
    ないが、分布バネモデルでは層ごとに扱えるためこちらのほうが原典に忠実。
    """
    depths = np.linspace(0.0, pile.length, n_elements + 1) + footing.embedment
    xs = np.array(pile_x_coordinates(arrangement), dtype=float)
    front_mask = xs >= xs.max() - 1.0e-9
    # 群杭の補正係数 μ は分布バネモデルには適用しないため、springs 側で
    # 1.0 になっている。ここでも乗じない。
    kh: float | np.ndarray = (
        np.array(
            layered_kh(profile, [float(d) for d in depths], springs.bh, springs.alpha)
        )
        if layered
        else springs.kh
    )

    def limits(front_row: bool) -> np.ndarray:
        return np.array(
            [
                p_hu(
                    profile,
                    min(float(d), profile.total_depth),
                    pile.diameter,
                    arrangement.spacing_y,
                    front_row=front_row,
                )
                for d in depths
            ]
        )

    de = (
        np.array([reduction.factor_at(float(d)) for d in depths])
        if reduction is not None
        else None
    )

    def make(front_row: bool) -> PileLateralModel:
        return PileLateralModel(
            ei=section.ei,
            diameter=pile.diameter,
            length=pile.length,
            kh=kh,
            limits=limits(front_row),
            reduction=de,
            n_elements=n_elements,
            moment_curvature=moment_curvature,
        )

    return BnwfLateralModel(
        front_mask=front_mask, front=make(True), back=make(False)
    )


@dataclass(frozen=True)
class SoilReactionPoint:
    """1深度における地盤反力度と、その上限値。"""

    depth: float  # 地表面からの深さ (m)
    reaction: float  # 弾性モデルの地盤反力度 kH・y (kN/m²)
    limit: float  # 上限値 pHU (kN/m²)

    @property
    def ratio(self) -> float:
        return abs(self.reaction) / self.limit if self.limit else math.inf

    @property
    def exceeded(self) -> bool:
        return self.ratio > 1.0


@dataclass(frozen=True)
class SoilReactionCheck:
    """弾性の地盤バネが pHU を超えていないかの診断。

    本ソフトの水平地盤バネは弾性のままなので、pHU を超える区間があれば
    **その区間の地盤抵抗を過大に見積もっている**(非安全側)。塑性化を
    取り込んだ解析(分布バネモデル)が必要であることを示す診断として用いる。
    """

    points: list[SoilReactionPoint]
    front_row: bool  # 判定に用いた杭が最前列か

    @property
    def exceedances(self) -> list[SoilReactionPoint]:
        return [p for p in self.points if p.exceeded]

    @property
    def ok(self) -> bool:
        return not self.exceedances

    @property
    def max_ratio(self) -> float:
        return max((p.ratio for p in self.points), default=0.0)

    @property
    def exceeded_depth_range(self) -> tuple[float, float] | None:
        found = self.exceedances
        if not found:
            return None
        return (min(p.depth for p in found), max(p.depth for p in found))


def check_soil_reaction_limit(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    section: PileSection,
    springs: LateralSprings,
    head_shear: float,
    head_moment: float,
) -> SoilReactionCheck:
    """弾性モデルの地盤反力度を pHU と比較する。

    杭体の水平変位分布 y(x) を Chang の式から求め、地盤反力度
    p = kH・y を深度ごとの pHU と突き合わせる。

    フーチングを剛体としているため全杭の y(x) は等しい。一方 pHU は砂質
    地盤で最前列以外が 1/2 になるため、**最前列以外の杭が支配する**。
    杭が2列以上ある場合はそちらで判定する。
    """
    forces = distribution(
        ei=section.ei,
        beta=springs.beta,
        h0=head_shear,
        m0=head_moment,
        length=pile.length,
    )
    front_row = arrangement.nx < 2
    points: list[SoilReactionPoint] = []
    for point in forces.points:
        depth = footing.embedment + point.depth
        if depth > profile.total_depth:
            break
        points.append(
            SoilReactionPoint(
                depth=depth,
                reaction=springs.kh * point.displacement,
                limit=p_hu(
                    profile,
                    depth,
                    pile.diameter,
                    arrangement.spacing_y,
                    front_row=front_row,
                ),
            )
        )
    return SoilReactionCheck(points=points, front_row=front_row)


def plastic_moment_steel_pipe(
    pile: PileSpec,
    axial: float,
    steel_grade: str = "SKK400",
    corrosion_mm: float = CORROSION_ALLOWANCE_MM,
) -> float:
    """鋼管杭の全塑性モーメント Mp (kN·m)。軸力の影響を含む。

    鋼管杭・鋼管ソイルセメント杭の M-φ 関係はバイリニア型で、Mp を上限と
    する(道示Ⅳ の断面計算式による)。

    薄肉円環断面(平均半径 r、板厚 t)の完全塑性状態を解くと、塑性中立軸の
    角度を θ0 として

        N = −4・σy・t・r・θ0
        M =  4・σy・t・r²・cos θ0

    したがって軸力 0 のとき Mp0 = 4・σy・t・r²、squash 軸力
    Np = σy・A に対して

        Mp(N)= Mp0・cos(π・N /(2・Np))

    となる。腐食代を控除した板厚を用いる。

    .. note::
       上式は**薄肉近似の厳密解**である。鋼管杭の D/t では中実解との差は
       0.1% 程度(:func:`plastic_section_modulus_hollow` と比較するテストで
       固定している)。道示Ⅳ が示す断面計算式そのものとの照合は未了。
    """
    if pile.pile_type not in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        raise ValueError(f"{pile.pile_type.value}には使用できません(鋼管杭のみ)")
    if steel_grade not in SIGMA_Y_STEEL:
        raise ValueError(
            f"鋼材 {steel_grade} の降伏点が未定義です。"
            f"対応材質: {sorted(SIGMA_Y_STEEL)}"
        )
    if pile.wall_thickness is None:
        raise ValueError("鋼管杭は板厚 wall_thickness の入力が必要です")

    # 腐食しろは外面から控除する(外径が 2c 減り、内径は変わらない)
    outer, t = corroded_tube(pile.diameter, pile.wall_thickness, corrosion_mm)
    r = (outer - t) / 2.0  # 平均半径
    sigma_y = SIGMA_Y_STEEL[steel_grade]

    area = 2.0 * math.pi * r * t
    squash = sigma_y * area * 1000.0  # kN
    if abs(axial) >= squash:
        raise ValueError(
            f"軸力 {axial:.0f} kN が全塑性軸力 {squash:.0f} kN 以上であり、"
            "曲げ耐力が残っていません"
        )
    mp0 = 4.0 * sigma_y * t * r**2 * 1000.0  # kN·m
    return mp0 * math.cos(math.pi * axial / (2.0 * squash))


def plastic_section_modulus_hollow(outer: float, thickness: float) -> float:
    """中空円形断面の塑性断面係数 Zp = (D³ − d³)/ 6 (m³)。

    :func:`plastic_moment_steel_pipe` の薄肉近似を検証するための厳密値。
    """
    inner = outer - 2.0 * thickness
    if inner <= 0:
        raise ValueError("肉厚が外径に対して大きすぎます")
    return (outer**3 - inner**3) / 6.0


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
    corrosion_mm: float = CORROSION_ALLOWANCE_MM,
    structure_type: StructureType = StructureType.PIER,
    rebar_grade: str | None = None,
    rebar: "RebarLayout | None" = None,
    stirrup: "StirrupLayout | None" = None,
    allowable_ductility: float | None = None,
    allowable_displacement: float | None = None,
    allowable_rotation: float | None = ALLOWABLE_FOOTING_ROTATION,
    e0_method: E0Method = E0Method.N_VALUE,
    reduction: SoilReduction | None = None,
    use_bnwf: bool = True,
    bnwf_elements: int = 100,
    bnwf_layered_kh: bool = True,
    max_factor: float = 3.0,
    steps: int = 120,
    moment_curvature: "MomentCurvature | None" = None,
) -> Level2Result:
    """杭・地盤の諸元からレベル2地震時の照査までを一括で行う。

    ``yield_moment`` を省略した場合、鋼管杭・鋼管ソイルセメント杭では
    :func:`yield_moment_steel_pipe` により自動算定する(軸力は設計レベル2
    荷重時の平均軸力 V / 杭本数を用いる)。それ以外の杭種では杭体降伏の
    判定を行わず、軸方向支持力の上限到達のみで降伏を判定する。

    ``allowable_ductility`` を省略した場合は ``structure_type`` と
    ``rebar_grade`` から :func:`allowable_ductility_for` により決定する。

    ``use_bnwf`` が真で、かつ全ての地層に KEP が入力されていれば、水平方向を
    分布バネモデル(BNWF)で解き、pHU による地盤の塑性化を解析に反映する。
    KEP が無い場合は従来どおり杭頭バネ K1〜K4 による弾性解析となる。

    ``bnwf_elements`` は杭の分割数。既定の 100 分割では、弾性状態で
    Chang の解析解に対し杭頭モーメントで 1% 程度の差になる(2次収束するので
    分割を倍にすると誤差は約 1/4)。

    ``bnwf_layered_kh`` が真(既定)なら、分布バネモデルの kH を**節点ごとに**
    当該深度の地層の E0 から算定する。偽にすると杭長にわたって単一の kH
    (1/β 区間で平均した E0 による値)を用いる。**偽は Chang の解析解との
    突合(一様地盤の仮定を共有させる)のための経路であり、設計では真を使う。**
    """
    section = pile_section(pile, fck=fck)
    # 軸方向バネの上限 Pu・Pt は周面摩擦力に依存するので、DE は分布バネ
    # モデルの有無にかかわらず支持力側に反映する
    bearing = compute_bearing_capacity(
        pile, profile, footing.embedment, reduction=reduction
    )
    kv = axial_spring(pile, section)
    body_limits = None
    if pile.pile_type == PileType.CAST_IN_PLACE and rebar is not None:
        body_limits = pile_body_axial_limits(
            pile, rebar, fck, rebar_grade or "SD345"
        )
    elif pile.pile_type in (PileType.STEEL_PIPE, PileType.STEEL_PIPE_SOIL_CEMENT):
        body_limits = pile_body_axial_limits(
            pile, section=section, steel_grade=steel_grade
        )
    elif (
        pile.pile_type == PileType.SC
        and pile.wall_thickness is not None
        and pile.concrete_thickness is not None
    ):
        body_limits = pile_body_axial_limits(
            pile, fck=fck, steel_grade=steel_grade, corrosion_mm=corrosion_mm
        )
    axial = AxialSpringModel.from_bearing(kv, bearing, body=body_limits)
    has_k_ep = all(layer.k_ep is not None for layer in profile.layers)
    use_bnwf_actual = use_bnwf and has_k_ep
    # 分布バネモデルでは DE を**節点ごとに**乗じるので、ここでは低減前の
    # kH を求める。弾性解析に落ちる場合のみ、平均した DE を kH に織り込む。
    # 群杭の補正係数 μ は**線形の地盤反力係数に対するもの**であり、基礎地盤の
    # 非線形性を考慮する場合には適用しないとされている。分布バネモデルは
    # まさにその非線形モデルなので、BNWF を使う経路では μ を乗じない。
    group_factor = group_pile_factor(
        min(arrangement.spacing_x, arrangement.spacing_y), pile.diameter
    )
    springs = lateral_springs(
        pile, section, profile, footing.embedment, LoadCase.LEVEL1_EQ,
        e0_method=e0_method,
        reduction=None if use_bnwf_actual else reduction,
        group_factor=1.0 if use_bnwf_actual else group_factor,
    )

    extra_notes: list[str] = []
    if group_factor < 1.0:
        extra_notes.append(
            f"杭中心間隔が {GROUP_PILE_SPACING_RATIO:g}D 未満のため群杭の"
            f"補正係数 μ = {group_factor:.3f} が求まるが、"
            + (
                "**分布バネモデル(基礎地盤の非線形性を考慮する場合)では "
                "μ による補正を考慮しない**とされているため、乗じていない。"
                if use_bnwf_actual
                else "杭頭バネによる弾性解析のため、kH に乗じている。"
            )
        )
    if reduction is not None and reduction.has_reduction:
        span = reduction.reduced_depth_range()
        detail = (
            "分布バネモデルでは節点ごとに乗じている"
            if use_bnwf and all(layer.k_ep is not None for layer in profile.layers)
            else "杭頭バネを用いる弾性解析のため、kH の平均区間で層厚加重平均"
            "した値を乗じている"
        )
        extra_notes.append(
            f"液状化による土質定数の低減 DE を考慮している"
            f"(低減区間: 深さ {span[0]:.1f}〜{span[1]:.1f} m、"
            f"{reduction.motion_type.value})。水平方向は{detail}。"
        )
        if bearing.has_reduced_skin:
            lost = bearing.skin_resistance_unreduced - bearing.skin_resistance
            extra_notes.append(
                f"周面摩擦力度 f にも DE を乗じており、極限支持力 Ru が "
                f"{lost:.0f} kN 減少している。これは軸方向バネの上限"
                f"(押込み Pu・引抜き Pt)を直接下げるため、浮上りの発生"
                "しやすさに影響する。**f への DE の適用は原典未確認**"
                "(安全側の判断。docs/VERIFICATION.md 参照)。"
            )
        if bearing.tip_zone_liquefies:
            extra_notes.append(
                f"⚠ 杭先端付近が液状化すると判定されている"
                f"(DE={bearing.tip_de:.2f})。先端支持力度 qd は低減して"
                "いないため、支持層の設定を確認すること。"
            )
    if allowable_ductility is None:
        allowable_ductility = allowable_ductility_for(
            structure_type, pile, rebar_grade
        )
        if allowable_ductility is None:
            extra_notes.append(
                f"{structure_type.value}の場所打ち杭に {rebar_grade} を用いる"
                "場合、基礎の塑性化を考慮できない(許容塑性率の規定がない)。"
                "基礎が降伏しない設計とする必要がある。"
            )
        else:
            extra_notes.append(
                f"許容塑性率 μa = {allowable_ductility:g} を"
                f"{structure_type.value}の杭基礎(直杭)として自動設定した。"
            )

    if yield_moment is None and pile.pile_type in (
        PileType.STEEL_PIPE,
        PileType.STEEL_PIPE_SOIL_CEMENT,
    ):
        mean_axial = v_load / (arrangement.nx * arrangement.ny)
        yield_moment = plastic_moment_steel_pipe(
            pile, mean_axial, steel_grade=steel_grade, corrosion_mm=corrosion_mm
        )
        first_yield = yield_moment_steel_pipe(
            pile, section, mean_axial, steel_grade=steel_grade,
            corrosion_mm=corrosion_mm,
        )
        extra_notes.append(
            f"鋼管杭の M-φ 関係はバイリニア型で、全塑性モーメント "
            f"Mp = {yield_moment:.0f} kN·m を上限とする"
            f"(死荷重時の平均軸力 {mean_axial:.0f} kN、{steel_grade}、"
            f"腐食代 {corrosion_mm:g} mm 控除)。参考: 最外縁が降伏点に達する"
            f"モーメントは {first_yield:.0f} kN·m。"
        )
    elif yield_moment is None:
        extra_notes.append(
            f"{pile.pile_type.value}の降伏曲げモーメント My が未入力のため、"
            "杭体降伏による降伏判定を行っていない"
            "(軸方向支持力の上限到達のみで判定)。"
        )

    soil_reaction, soil_notes = _soil_reaction_diagnosis(
        pile, arrangement, footing, profile, section, springs, v_load, h_load, m_load,
        kv, axial, bnwf=use_bnwf_actual,
    )
    extra_notes.extend(soil_notes)

    lateral: LateralModel | None = None
    if use_bnwf_actual and soil_reaction is not None:
        lateral = build_bnwf_model(
            pile, arrangement, footing, profile, section, springs,
            reduction=reduction, n_elements=bnwf_elements,
            layered=bnwf_layered_kh, moment_curvature=moment_curvature,
        )
        extra_notes.append(
            f"水平方向は分布バネモデル(BNWF、{bnwf_elements} 分割)で解析し、"
            "地盤反力度が pHU に達した節点は頭打ちとして扱っている"
            "(杭頭バネ K1〜K4 による弾性解析ではない)。"
        )
        if moment_curvature is not None:
            mc = moment_curvature
            extra_notes.append(
                f"杭体の曲げ非線形を M-φ 骨格曲線(折れ点 {len(mc.points)} 点、"
                f"My = {mc.yield_moment:,.0f} kN·m、"
                f"Mu = {mc.ultimate_moment:,.0f} kN·m、"
                f"φu = {mc.ultimate_curvature:.6g} 1/m)として考慮し、"
                "要素ごとに割線曲げ剛性を低下させている(塑性ヒンジ)。"
                "要素内のモーメント変化は両端の大きいほうで代表させた近似である。"
            )
        kh_min, kh_max = lateral.kh_range
        extra_notes.append(
            f"水平方向地盤反力係数 kH は**節点ごとに**当該深度の地層の E0 から"
            f"算定している(kH = {kh_min:,.0f}〜{kh_max:,.0f} kN/m³。"
            f"換算載荷幅 BH = {springs.bh:.3f} m は常時の条件で定めた共通値)。"
            + (
                f"参考: 1/β 区間で平均した E0 による単一値は "
                f"{springs.kh:,.0f} kN/m³。"
                if kh_max > kh_min
                else "杭長にわたって同一の地層のため単一値と一致する。"
            )
        )

    result = analyze_level2(
        arrangement,
        axial,
        k1=springs.k1,
        k2=springs.k2,
        k4=springs.k4,
        lateral=lateral,
        v_load=v_load,
        h_load=h_load,
        m_load=m_load,
        yield_moment=yield_moment,
        allowable_ductility=allowable_ductility,
        allowable_displacement=allowable_displacement,
        allowable_rotation=allowable_rotation,
        max_factor=max_factor,
        steps=steps,
    )
    capacity, response_shear, shear_notes = _level2_shear(
        pile, rebar, stirrup, fck, rebar_grade or "SD345", result
    )
    return Level2Result(
        steps=result.steps,
        yield_point=result.yield_point,
        response=result.response,
        allowable_ductility=result.allowable_ductility,
        allowable_displacement=result.allowable_displacement,
        allowable_rotation=result.allowable_rotation,
        soil_reaction=soil_reaction,
        notes=extra_notes + shear_notes + result.notes,
        shear_capacity=capacity,
        response_shear=response_shear,
        group_factor=springs.group_factor,
    )


def _level2_shear(
    pile: PileSpec,
    rebar: "RebarLayout | None",
    stirrup: "StirrupLayout | None",
    fck: int,
    rebar_grade: str,
    result: Level2Result,
) -> tuple["ShearCapacity | None", float | None, list[str]]:
    """レベル2のせん断耐力 Ps と、設計レベル2荷重時の杭頭せん断力。

    場所打ち杭で軸方向鉄筋が入力されている場合のみ算定する。せん断力は
    最も厳しい杭(押込み軸力が最大の杭)の杭頭せん断力とする。
    """
    if pile.pile_type != PileType.CAST_IN_PLACE:
        return None, None, []
    if rebar is None:
        return None, None, [
            "場所打ち杭の軸方向鉄筋が未入力のため、レベル2のせん断耐力"
            "(道示Ⅳ 5.2.3)を照査していない。"
        ]
    if result.response is None:
        return None, None, []
    critical = max(result.response.reactions, key=lambda r: r.axial)
    capacity = shear_capacity_level2(
        pile, rebar, fck,
        axial=critical.axial, moment=critical.moment,
        stirrup=stirrup, rebar_grade=rebar_grade,
    )
    notes: list[str] = []
    if stirrup is None:
        notes.append(
            "帯鉄筋が未入力のため、レベル2のせん断耐力はコンクリートの負担分 Sc "
            "のみで評価している(Ss = 0)。実際の配筋を入力すると Ss を"
            "算入できる。"
        )
    return capacity, critical.shear, notes


def _soil_reaction_diagnosis(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    section: PileSection,
    springs: LateralSprings,
    v_load: float,
    h_load: float,
    m_load: float,
    kv: float,
    axial: AxialSpringModel,
    bnwf: bool,
) -> tuple[SoilReactionCheck | None, list[str]]:
    """設計レベル2荷重時の地盤反力度を pHU と突き合わせる。

    KEP が未入力の層があれば診断を行わず、その旨を注記として返す。
    """
    from core.analysis.displacement import solve_stability

    if any(layer.k_ep is None for layer in profile.layers):
        return None, [
            "地層の地震時受働土圧係数 KEP が未入力のため、水平地盤反力度が"
            "上限値 pHU を超えていないかの診断を行っていない。"
            "KEP を入力すると、弾性の地盤バネが非安全側になる区間を"
            "確認できる。"
        ]

    # 設計レベル2荷重(λ = 1)における杭頭反力を用いる
    elastic = solve_stability(
        arrangement,
        kv=kv,
        k1=springs.k1,
        k2=springs.k2,
        k4=springs.k4,
        v_load=v_load,
        h_load=h_load,
        m_load=m_load,
    )
    critical = max(elastic.reactions, key=lambda r: r.axial)
    try:
        check = check_soil_reaction_limit(
            pile, arrangement, footing, profile, section, springs,
            head_shear=critical.shear, head_moment=critical.moment,
        )
    except ValueError as exc:
        return None, [f"pHU の診断を行えませんでした: {exc}"]

    notes: list[str] = []
    if check.ok:
        notes.append(
            f"設計レベル2荷重時の地盤反力度は上限値 pHU 以下"
            f"(最大で pHU の {check.max_ratio * 100:.0f}%)。"
            "弾性の地盤バネのままでも大きな乖離はないと考えられる。"
        )
    elif bnwf:
        top, bottom = check.exceeded_depth_range  # type: ignore[misc]
        notes.append(
            f"弾性解析であれば深さ {top:.1f}〜{bottom:.1f} m で地盤反力度が"
            f"上限値 pHU を超えていた(最大で pHU の {check.max_ratio * 100:.0f}%)。"
            "本解析は分布バネモデルでこの塑性化を考慮している。"
        )
    else:
        top, bottom = check.exceeded_depth_range  # type: ignore[misc]
        notes.append(
            f"**深さ {top:.1f}〜{bottom:.1f} m で地盤反力度が上限値 pHU を"
            f"超えている(最大で pHU の {check.max_ratio * 100:.0f}%)。**"
            "本解析は水平地盤バネを弾性としているため、この区間の地盤抵抗を"
            "過大に評価しており、結果は非安全側である。"
        )
    return check, notes


def _response_step(steps: list[PushoverStep]) -> PushoverStep | None:
    """λ = 1(設計レベル2荷重)に最も近いステップ。"""
    reached = [s for s in steps if s.factor <= 1.0 + 1.0e-12]
    if not reached or reached[-1].factor < 1.0 - 1.0e-6:
        # λ = 1 まで解けていない(その前に不安定になった)
        return None
    return reached[-1]
