"""液状化の判定(道示Ⅴ(H24) 8.2)。

H24年道示では、液状化の判定はレベル2地震動(タイプI・タイプII)に対して行う。

判定フロー:
  1. 判定対象層のスクリーニング(8.2.2)
  2. FL = R / L の算定(8.2.3)
       R = cw・RL          (動的せん断強度比)
       L = rd・khg・σv/σ'v (地震時せん断応力比)
  3. FL ≦ 1.0 の層について土質定数の低減係数 DE を決定(8.2.4)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.soil import SoilLayer, SoilProfile, SoilType
from core.standards import (
    DE_TABLE,
    KHG0_LIQUEFACTION,
    LIQUEFACTION_MAX_D10,
    LIQUEFACTION_MAX_D50,
    LIQUEFACTION_MAX_DEPTH,
    LIQUEFACTION_MAX_FC,
    LIQUEFACTION_MAX_GWL,
    LIQUEFACTION_MAX_IP,
    GroundMotionType,
    GroundType,
)


def n1_value(n: float, sigma_v_eff: float) -> float:
    """有効上載圧 100kN/m2 相当に換算したN値 N1(道示Ⅴ 8.2.3)

    N1 = 170・N / (σ'v + 70)
    """
    return 170.0 * n / (sigma_v_eff + 70.0)


def na_sand(n1: float, fc: float) -> float:
    """砂質土の粒度の影響を考慮した補正N値 Na(道示Ⅴ 8.2.3)"""
    if fc < 10.0:
        c1, c2 = 1.0, 0.0
    elif fc < 60.0:
        c1, c2 = (fc + 40.0) / 50.0, (fc - 10.0) / 18.0
    else:
        c1, c2 = fc / 20.0 - 1.0, (fc - 10.0) / 18.0
    return c1 * n1 + c2


def na_gravel(n1: float, d50: float) -> float:
    """礫質土の補正N値 Na(道示Ⅴ 8.2.3)

    Na = {1 − 0.36・log10(D50/2)}・N1
    """
    return (1.0 - 0.36 * math.log10(d50 / 2.0)) * n1


def rl_value(na: float) -> float:
    """繰返し三軸強度比 RL(道示Ⅴ 8.2.3)"""
    rl = 0.0882 * math.sqrt(na / 1.7)
    if na >= 14.0:
        rl += 1.6e-6 * (na - 14.0) ** 4.5
    return rl


def cw_value(rl: float, motion: GroundMotionType) -> float:
    """地震動特性による補正係数 cw(道示Ⅴ 8.2.3)"""
    if motion == GroundMotionType.LEVEL2_TYPE1:
        return 1.0
    # タイプII
    if rl <= 0.1:
        return 1.0
    if rl <= 0.4:
        return 3.3 * rl + 0.67
    return 2.0


def rd_value(depth: float) -> float:
    """地震時せん断応力比の深さ方向低減係数 rd = 1.0 − 0.015x"""
    return 1.0 - 0.015 * depth


def reduction_factor_de(fl: float, depth: float, r: float) -> float:
    """土質定数の低減係数 DE(道示Ⅴ(H24) 表-8.2.4)

    FL > 1.0(液状化しない)の場合は低減しない(DE = 1.0)。

    DE は液状化対象層の地盤定数(水平方向地盤反力係数 kH、最大周面摩擦力度、
    最大地盤反力度)に乗じて用いる。

    .. warning::
       **本ソフトは DE を算定するのみで、安定計算には反映していない。**
       液状化層を含む地盤での地震時照査には、kH・周面摩擦力度への
       DE の適用が必要(未実装)。
    """
    if fl > 1.0:
        return 1.0
    fl_idx = 0 if fl <= 1.0 / 3.0 else (1 if fl <= 2.0 / 3.0 else 2)
    depth_idx = 0 if depth <= 10.0 else 1
    r_idx = 0 if r <= 0.3 else 1
    return DE_TABLE[(fl_idx, depth_idx, r_idx)]


def _screening_reason(
    layer: SoilLayer, depth: float, gwl: float
) -> str | None:
    """判定対象層のスクリーニング(道示Ⅴ 8.2.2)。対象なら None、対象外なら理由。"""
    if gwl > LIQUEFACTION_MAX_GWL:
        return f"地下水位が地表面から{LIQUEFACTION_MAX_GWL:.0f}mより深い"
    if depth > LIQUEFACTION_MAX_DEPTH:
        return f"深度{LIQUEFACTION_MAX_DEPTH:.0f}m超"
    if depth < gwl:
        return "地下水位以浅(非飽和)"
    if layer.soil_type == SoilType.CLAY:
        return "粘性土(砂質土・礫質土でない)"
    if not layer.is_alluvial:
        return "沖積層でない"
    if layer.fc is None:
        return "細粒分含有率FCが未入力"
    if layer.fc > LIQUEFACTION_MAX_FC and (
        layer.ip is None or layer.ip > LIQUEFACTION_MAX_IP
    ):
        return f"FC>{LIQUEFACTION_MAX_FC:.0f}%かつIP>{LIQUEFACTION_MAX_IP:.0f}"
    if layer.soil_type == SoilType.GRAVEL and layer.d50 is None:
        return "礫質土でD50が未入力"
    if layer.d50 is not None and layer.d50 > LIQUEFACTION_MAX_D50:
        return f"D50>{LIQUEFACTION_MAX_D50:.0f}mm"
    if layer.d10 is not None and layer.d10 > LIQUEFACTION_MAX_D10:
        return f"D10>{LIQUEFACTION_MAX_D10:.0f}mm"
    return None


@dataclass(frozen=True)
class SliceResult:
    """判定単位(スライス)ごとの結果。対象外の場合は数値フィールドが None。"""

    depth_top: float
    depth_bottom: float
    depth_mid: float
    layer_name: str
    soil_type: SoilType
    is_target: bool
    excluded_reason: str | None
    sigma_v: float | None = None
    sigma_v_eff: float | None = None
    n1: float | None = None
    na: float | None = None
    rl: float | None = None
    # タイプI
    l_type1: float | None = None
    r_type1: float | None = None
    fl_type1: float | None = None
    de_type1: float = 1.0
    # タイプII
    l_type2: float | None = None
    r_type2: float | None = None
    fl_type2: float | None = None
    de_type2: float = 1.0


@dataclass(frozen=True)
class LiquefactionAssessment:
    slices: list[SliceResult]
    liquefiable_type1: bool  # タイプIで FL≦1 の層があるか
    liquefiable_type2: bool


def evaluate_at(
    profile: SoilProfile,
    depth: float,
    ground_type: GroundType,
    cz_type1: float = 1.0,
    cz_type2: float = 1.0,
    depth_top: float | None = None,
    depth_bottom: float | None = None,
) -> SliceResult:
    """1深度についてFL・DEを算定する。"""
    layer = profile.layer_at(depth)
    top = depth if depth_top is None else depth_top
    bottom = depth if depth_bottom is None else depth_bottom
    reason = _screening_reason(layer, depth, profile.gwl)
    if reason is not None:
        return SliceResult(
            depth_top=top,
            depth_bottom=bottom,
            depth_mid=depth,
            layer_name=layer.name,
            soil_type=layer.soil_type,
            is_target=False,
            excluded_reason=reason,
        )

    sigma_v, sigma_v_eff = profile.stresses_at(depth)
    n1 = n1_value(layer.n_value, sigma_v_eff)
    if layer.soil_type == SoilType.GRAVEL:
        na = na_gravel(n1, layer.d50)  # type: ignore[arg-type]  # スクリーニング済み
    else:
        na = na_sand(n1, layer.fc)  # type: ignore[arg-type]  # スクリーニング済み
    rl = rl_value(na)
    rd = rd_value(depth)
    stress_ratio = sigma_v / sigma_v_eff

    per_motion: dict[GroundMotionType, tuple[float, float, float, float]] = {}
    for motion, cz in (
        (GroundMotionType.LEVEL2_TYPE1, cz_type1),
        (GroundMotionType.LEVEL2_TYPE2, cz_type2),
    ):
        khg = cz * KHG0_LIQUEFACTION[motion][ground_type]
        l_val = rd * khg * stress_ratio
        r_val = cw_value(rl, motion) * rl
        fl = r_val / l_val
        de = reduction_factor_de(fl, depth, r_val)
        per_motion[motion] = (l_val, r_val, fl, de)

    l1, r1, fl1, de1 = per_motion[GroundMotionType.LEVEL2_TYPE1]
    l2, r2, fl2, de2 = per_motion[GroundMotionType.LEVEL2_TYPE2]
    return SliceResult(
        depth_top=top,
        depth_bottom=bottom,
        depth_mid=depth,
        layer_name=layer.name,
        soil_type=layer.soil_type,
        is_target=True,
        excluded_reason=None,
        sigma_v=sigma_v,
        sigma_v_eff=sigma_v_eff,
        n1=n1,
        na=na,
        rl=rl,
        l_type1=l1,
        r_type1=r1,
        fl_type1=fl1,
        de_type1=de1,
        l_type2=l2,
        r_type2=r2,
        fl_type2=fl2,
        de_type2=de2,
    )


def assess_liquefaction(
    profile: SoilProfile,
    ground_type: GroundType,
    cz_type1: float = 1.0,
    cz_type2: float = 1.0,
    pitch: float = 1.0,
) -> LiquefactionAssessment:
    """地盤モデル全体の液状化判定。

    各層を pitch (m) 以下の等分スライスに分割し、スライス中央深度で評価する。
    """
    slices: list[SliceResult] = []
    for top, bottom, _layer in profile.layer_boundaries():
        n_div = max(1, math.ceil((bottom - top) / pitch - 1e-9))
        dz = (bottom - top) / n_div
        for i in range(n_div):
            s_top = top + i * dz
            s_bottom = s_top + dz
            mid = (s_top + s_bottom) / 2.0
            slices.append(
                evaluate_at(
                    profile,
                    mid,
                    ground_type,
                    cz_type1,
                    cz_type2,
                    depth_top=s_top,
                    depth_bottom=s_bottom,
                )
            )
    liq1 = any(s.fl_type1 is not None and s.fl_type1 <= 1.0 for s in slices)
    liq2 = any(s.fl_type2 is not None and s.fl_type2 <= 1.0 for s in slices)
    return LiquefactionAssessment(
        slices=slices, liquefiable_type1=liq1, liquefiable_type2=liq2
    )


# ---------------------------------------------------------------------------
# 土質定数の低減(道示Ⅴ(H24) 8.2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoilReduction:
    """深度ごとの土質定数の低減係数 DE。

    液状化が生じると判定された土層では、**側方地盤のバネ定数 kH に DE を
    乗じる**(提供解説資料により確認済み。docs/VERIFICATION.md 第7回)。

    .. important::
       DE は**レベル2地震動に対する液状化判定**から得られる。本ソフトは
       レベル1地震動に対する液状化判定(khg0 が異なる)を実装していないため、
       常時・レベル1地震時の照査にこの DE を用いるかは利用者の判断とする
       (用いる場合は安全側の代用となる)。

    .. note::
       **周面摩擦力度 f** への低減は
       :func:`core.capacity.bearing.compute_bearing_capacity` で適用している。
       ただし原典未確認であり、低減しないほうが明確に非安全側であることを
       根拠とした安全側の判断である(docs/VERIFICATION.md 第21回)。

       **受働土圧強度**(pHU の算定に用いる pEP)への低減は未適用。
       レベル2の分布バネモデルでは pHU そのものに DE を乗じているため、
       杭前面地盤の抵抗としては低減が効いている。
    """

    # (上端深度, 下端深度, DE) の並び。深度は地表面から。
    segments: tuple[tuple[float, float, float], ...]
    motion_type: GroundMotionType

    @classmethod
    def from_assessment(
        cls, assessment: LiquefactionAssessment, motion_type: GroundMotionType
    ) -> "SoilReduction":
        """液状化判定の結果から低減係数を取り出す。"""
        segments = tuple(
            (
                s.depth_top,
                s.depth_bottom,
                s.de_type1
                if motion_type == GroundMotionType.LEVEL2_TYPE1
                else s.de_type2,
            )
            for s in assessment.slices
        )
        return cls(segments=segments, motion_type=motion_type)

    @property
    def has_reduction(self) -> bool:
        """低減される区間があるか。"""
        return any(de < 1.0 for _, _, de in self.segments)

    def factor_at(self, depth: float) -> float:
        """深度 ``depth`` における DE。範囲外は 1.0(低減なし)。"""
        for top, bottom, de in self.segments:
            if top <= depth <= bottom:
                return de
        return 1.0

    def mean_factor(self, top: float, bottom: float) -> float:
        """区間 [top, bottom] における DE の層厚加重平均。

        本ソフトの kH は杭頭直下の一定区間を平均した1つの値なので、DE も
        同じ区間で平均して乗じる。

        .. warning::
           層ごとに kH を変える(分布バネモデル)ほうが原典に忠実である。
           この平均化は、杭頭バネ K1〜K4 を用いる弾性解析での近似である。
        """
        if bottom <= top:
            return self.factor_at(top)
        total = 0.0
        weighted = 0.0
        for seg_top, seg_bottom, de in self.segments:
            overlap = min(seg_bottom, bottom) - max(seg_top, top)
            if overlap <= 0:
                continue
            weighted += de * overlap
            total += overlap
        if total <= 0:
            return 1.0
        # 判定範囲(通常は地表面から 20 m)より深い区間は低減しない
        uncovered = (bottom - top) - total
        return (weighted + uncovered) / (bottom - top)

    def reduced_depth_range(self) -> tuple[float, float] | None:
        """低減される区間の深度範囲。無ければ None。"""
        reduced = [(t, b) for t, b, de in self.segments if de < 1.0]
        if not reduced:
            return None
        return (min(t for t, _ in reduced), max(b for _, b in reduced))
