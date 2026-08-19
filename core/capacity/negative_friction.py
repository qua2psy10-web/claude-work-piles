"""負の周面摩擦力(NF)の検討(道示Ⅳ(H24) 12.4.3)。

圧密沈下が生じる軟弱層では、地盤が杭より大きく沈下することで杭に
下向きの周面摩擦力(負の周面摩擦力)が作用する。

    中立点  : 地盤と杭の沈下量が等しくなる深さ。安全側に軟弱層下端とする。
    NF      : 中立点以浅の周面摩擦力の総和
    Nmax    = 死荷重による軸力 + NF
    照査     : Nmax ≦ Ru / n_NF   (n_NF は通常の安全率より小さい値)

.. note::
   中立点を軟弱層下端に固定するのは安全側の簡易法である。厳密には
   沈下量の釣合いから求める必要がある。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.models.pile import PileSpec
from core.models.soil import SoilLayer, SoilProfile, SoilType
from core.standards import NF_SAFETY_FACTOR


@dataclass(frozen=True)
class NegativeFrictionSegment:
    layer_name: str
    depth_top: float
    depth_bottom: float
    fn: float  # 負の周面摩擦力度 (kN/m2)
    force: float  # U·L·fn (kN)


@dataclass(frozen=True)
class NegativeFrictionResult:
    neutral_depth: float  # 中立点の深さ(地表面から) (m)
    segments: list[NegativeFrictionSegment]
    nf: float  # 負の周面摩擦力の総和 (kN)
    dead_load: float  # 死荷重による杭頭軸力 (kN)
    n_max: float  # 最大軸力 (kN)
    ru: float  # 極限支持力 (kN)

    @property
    def allowable(self) -> float:
        """NF 考慮時の許容軸力 (kN)"""
        return self.ru / NF_SAFETY_FACTOR

    @property
    def ratio(self) -> float:
        return self.n_max / self.allowable if self.allowable else math.inf

    @property
    def ok(self) -> bool:
        return self.ratio <= 1.0

    @property
    def judgement(self) -> str:
        return "OK" if self.ok else "NG"


def negative_friction_intensity(layer: SoilLayer, sigma_v_eff: float) -> float:
    """負の周面摩擦力度 fn (kN/m2)(道示Ⅳ 12.4.3)。

    粘性土: fn = c(粘着力。未入力時は c = 10N で推定)
    砂質土: fn = 0.3・σ'v(有効上載圧に比例)

    .. note::
       粘性土の c=10N 推定式は、フォーラムエイト UC-1 計算書サンプル
       Kui_12 の 2.4「負の周面摩擦力に対する検討」(第57回)で、粘着力
       未入力の粘性土2層(N=2.0→fn=20.0、N=3.8→fn=38.0)がいずれも
       fn=10・N と厳密に一致することを確認した。従来は二次資料のみに
       基づく確度Cだったが、この一致により確度Bに格上げした。
    """
    if layer.soil_type == SoilType.CLAY:
        return layer.cohesion if layer.cohesion is not None else 10.0 * layer.n_value
    return 0.3 * sigma_v_eff


def compute_negative_friction(
    pile: PileSpec,
    profile: SoilProfile,
    embedment: float,
    dead_load: float,
    ru: float,
    consolidating_layers: set[str] | None = None,
    pitch: float = 0.5,
) -> NegativeFrictionResult:
    """負の周面摩擦力を算定し、最大軸力を照査する。

    Parameters
    ----------
    consolidating_layers:
        圧密沈下層とみなす層名の集合。省略時は杭頭以深で最初に現れる
        連続した粘性土層(N値10以下)を圧密層とみなす。
    dead_load:
        死荷重による杭1本あたりの杭頭軸力 (kN)。
    ru:
        杭の極限支持力 (kN)。
    """
    tip_depth = embedment + pile.length
    boundaries = profile.layer_boundaries()

    if consolidating_layers is None:
        names = set()
        for top, bottom, layer in boundaries:
            if bottom <= embedment:
                continue
            if layer.soil_type == SoilType.CLAY and layer.n_value <= 10.0:
                names.add(layer.name)
            elif names:
                break  # 圧密層が途切れたら終了
        consolidating_layers = names

    # 中立点 = 圧密層の下端(杭頭以深に圧密層がなければ NF なし)
    neutral = embedment
    for top, bottom, layer in boundaries:
        if layer.name in consolidating_layers and bottom > embedment:
            neutral = max(neutral, min(bottom, tip_depth))

    perimeter = math.pi * pile.diameter
    segments: list[NegativeFrictionSegment] = []
    nf = 0.0
    for top, bottom, layer in boundaries:
        seg_top = max(top, embedment)
        seg_bottom = min(bottom, neutral)
        if seg_bottom <= seg_top:
            continue
        # 有効上載圧が深さで変わるため区間を細分して積分する
        n_div = max(1, math.ceil((seg_bottom - seg_top) / pitch))
        dz = (seg_bottom - seg_top) / n_div
        force = 0.0
        fn_sum = 0.0
        for i in range(n_div):
            mid = seg_top + (i + 0.5) * dz
            _, sigma_v_eff = profile.stresses_at(mid)
            fn = negative_friction_intensity(layer, sigma_v_eff)
            fn_sum += fn
            force += perimeter * dz * fn
        nf += force
        segments.append(
            NegativeFrictionSegment(
                layer_name=layer.name,
                depth_top=seg_top,
                depth_bottom=seg_bottom,
                fn=fn_sum / n_div,
                force=force,
            )
        )

    return NegativeFrictionResult(
        neutral_depth=neutral,
        segments=segments,
        nf=nf,
        dead_load=dead_load,
        n_max=dead_load + nf,
        ru=ru,
    )
