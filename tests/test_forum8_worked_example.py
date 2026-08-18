"""フォーラムエイト UC-1「基礎の設計」計算書サンプル Kui_1 との突合(第34回)。

**他社製品の出力(二次資料)であり道示の原典ではない。** ただし H24版の
許容応力度設計法による**通しの計算例**であり、本ソフトが依存する連鎖

    層ごとの kH → 杭頭バネ K1〜K4 → 変位法 → 杭反力・断面力 → 応力度

を外部の独立実装と数値で突き合わせられる唯一の資料である。定数の照合では
なく**手順の照合**にあたる。

対象: 場所打ち杭 φ1200・L=25m・12本(4列×3行、間隔 3.0m)、σck=24、
SD345、n=15、杭頭剛結およびヒンジ、杭先端ヒンジ。
"""
import math

import numpy as np
import pytest

from core.analysis.bnwf import PileLateralModel
from core.analysis.displacement import solve_stability
from core.capacity.springs import PileSection, axial_spring, kh_from_e0
from core.models import (
    ConstructionMethod,
    PileArrangement,
    PileSpec,
    PileType,
)
from core.section.rc import RebarLayout, analyze_circular_rc

DIAMETER = 1.2
LENGTH = 25.0
EC = 2.5e4 * 1000.0  # 2.50×10⁴ N/mm² → kN/m²
INERTIA = math.pi * DIAMETER**4 / 64.0
AREA = math.pi * DIAMETER**2 / 4.0
EI = EC * INERTIA
SECTION = PileSection(area=AREA, inertia=INERTIA, young=EC)
PILE = PileSpec(
    pile_type=PileType.CAST_IN_PLACE,
    method=ConstructionMethod.CAST_IN_PLACE,
    diameter=DIAMETER,
    length=LENGTH,
)
ARRANGEMENT = PileArrangement(nx=4, ny=3, spacing_x=3.0, spacing_y=3.0)

# 1.5 地層データ: (層厚, α·E0 常時, α·E0 地震時)
LAYERS = [
    (5.0, 5600.0, 11200.0),
    (12.0, 10640.0, 21280.0),
    (6.0, 56000.0, 112000.0),
    (2.0, 140000.0, 280000.0),
]
# 1.6 水平方向地盤反力係数 kH (kN/m³)
EXPECTED_KH = [(3466, 6932), (6586, 13171), (34661, 69323), (86654, 173307)]
KV = 560774.0  # 1.6 杭軸方向バネ定数 (kN/m)


def _node_kh(seismic: bool, n_elements: int) -> np.ndarray:
    depths = np.linspace(0.0, LENGTH, n_elements + 1)
    tops, values = [], []
    top = 0.0
    for thickness, normal, quake in LAYERS:
        top += thickness
        tops.append(top)
        values.append(quake if seismic else normal)
    out = []
    for d in depths:
        for boundary, alpha_e0 in zip(tops, values):
            if d <= boundary + 1e-9:
                out.append(alpha_e0)
                break
    # BH は本ソフトの手順どおり常時条件で1つ定める(計算例も全層で共通)
    return np.array([kh_from_e0(v, _BH, 1.0) for v in out], dtype=float)


def _bh_from_expected() -> float:
    """計算例の kH と α·E0 から換算載荷幅 B' を逆算する。

    kH = (α·E0/0.3)(B'/0.3)^(−3/4) を B' について解く。計算例は B' を
    表に出していないが、全層・全ケースで同じ値になっているはずである。
    """
    bhs = []
    for (_, alpha_e0, alpha_e0_eq), (kh, kh_eq) in zip(LAYERS, EXPECTED_KH):
        for a, k in ((alpha_e0, kh), (alpha_e0_eq, kh_eq)):
            bhs.append(0.3 * (a / 0.3 / k) ** (4.0 / 3.0))
    assert max(bhs) - min(bhs) < 1e-3, "全層で B' が共通でない"
    return sum(bhs) / len(bhs)


_BH = _bh_from_expected()


def test_layered_kh_matches_every_layer_and_case():
    """層ごと・ケースごとの kH が 8 通りすべて一致すること。

    共通の換算載荷幅 B' から、当該層の α·E0 だけを変えて求める。
    第30回に実装した「BH は共通・kH は層ごと」がそのまま裏付けられる。
    """
    assert _BH == pytest.approx(2.83, abs=0.01)
    for (_, alpha_e0, alpha_e0_eq), (kh, kh_eq) in zip(LAYERS, EXPECTED_KH):
        assert kh_from_e0(alpha_e0, _BH, 1.0) == pytest.approx(kh, rel=2e-4)
        assert kh_from_e0(alpha_e0_eq, _BH, 1.0) == pytest.approx(kh_eq, rel=2e-4)
    # 地震時は常時のちょうど 2 倍(α の比)
    for (_, a, a_eq), (k, k_eq) in zip(LAYERS, EXPECTED_KH):
        assert a_eq == pytest.approx(2.0 * a)
        assert k_eq == pytest.approx(2.0 * k, rel=1e-3)


def test_axial_spring_matches():
    """杭軸方向バネ定数 Kv = a·Ap·Ep/L が一致すること。"""
    assert axial_spring(PILE, SECTION) == pytest.approx(KV, rel=1e-5)


def test_head_springs_need_the_layered_finite_length_model():
    """計算例の K1〜K4 は Chang の半無限長式では再現できないこと。

    半無限長の Chang では K1 = 4EIβ³・K2 = 2EIβ²・K4 = 2EIβ なので、
    3 つから逆算した β は一致しなければならない。計算例では一致せず、
    さらに恒等式 K1·K4 = 2·K2² も成り立たない。つまり計算例は
    **層ごとの kH をもつ有限長の数値解**である。
    """
    for k1, k2, k4 in ((33490, 120476, 807526), (54140, 164549, 943120)):
        beta_1 = (k1 / (4.0 * EI)) ** (1.0 / 3.0)
        beta_2 = (k2 / (2.0 * EI)) ** 0.5
        beta_4 = k4 / (2.0 * EI)
        assert beta_2 != pytest.approx(beta_1, rel=0.01)
        assert beta_4 != pytest.approx(beta_1, rel=0.01)
        assert k1 * k4 != pytest.approx(2.0 * k2**2, rel=0.02)


@pytest.mark.parametrize(
    "seismic, expected",
    [(False, (33490, 120476, 807526)), (True, (54140, 164549, 943120))],
)
def test_bnwf_reproduces_the_head_springs(seismic, expected):
    """分布バネモデルが計算例の K1〜K4 を 0.3% 以内で再現すること。"""
    n_elements = 400
    model = PileLateralModel(
        EI, DIAMETER, LENGTH, _node_kh(seismic, n_elements), n_elements=n_elements
    )
    k = model.head_stiffness()
    k1, k2, k4 = expected
    assert float(k[0, 0]) == pytest.approx(k1, rel=3e-3)
    assert float(-k[0, 1]) == pytest.approx(k2, rel=3e-3)
    assert float(k[1, 1]) == pytest.approx(k4, rel=3e-3)


def test_displacement_method_reproduces_the_pile_reactions():
    """変位法が計算例の変位・杭反力・杭頭モーメントを再現すること。

    杭頭バネは計算例の値をそのまま与え、**変位法ソルバーだけ**を検証する。
    K2 は本ソフトの符号規約では負であることに注意(計算例は絶対値で表示)。
    """
    result = solve_stability(
        ARRANGEMENT, kv=KV, k1=54140.0, k2=-164549.0, k4=943120.0,
        v_load=16627.4, h_load=2410.5, m_load=19623.1,
    )
    assert result.u * 1000.0 == pytest.approx(4.72, abs=0.01)
    assert result.v * 1000.0 == pytest.approx(2.47, abs=0.01)
    assert result.theta == pytest.approx(0.00033262, rel=1e-3)

    by_x = {round(r.x, 1): r for r in result.reactions}
    assert by_x[-4.5].axial == pytest.approx(546.25, abs=0.05)
    assert by_x[4.5].axial == pytest.approx(2224.99, abs=0.05)
    assert by_x[4.5].shear == pytest.approx(200.87, abs=0.05)
    assert by_x[4.5].moment == pytest.approx(-463.17, abs=0.1)


def test_permanent_case_axial_reaction():
    """常時(モーメント・水平力なし)は全杭が V/12 を等分すること。"""
    result = solve_stability(
        ARRANGEMENT, kv=KV, k1=33490.0, k2=-120476.0, k4=807526.0,
        v_load=17019.8, h_load=0.0, m_load=0.0,
    )
    for r in result.reactions:
        assert r.axial == pytest.approx(1418.32, abs=0.01)
    assert result.v * 1000.0 == pytest.approx(2.53, abs=0.01)


REBAR = RebarLayout(count=24, diameter_mm=25.0, cover_mm=150.0)


def test_rebar_nominal_area_matches():
    """D25 × 24 本 = 121.608 cm²(公称断面積 506.7 mm²)。"""
    assert REBAR.total_area * 1.0e4 == pytest.approx(121.608, abs=1e-3)


@pytest.mark.parametrize(
    "axial, moment, sigma_c, sigma_s",
    [
        (546.25, 463.17, 4.25, 76.30),   # ケース3 地震時
        (352.01, 463.17, 4.34, 93.24),   # ケース6 地震時(浮)
    ],
)
def test_section_stresses_match(axial, moment, sigma_c, sigma_s):
    """杭体応力度が計算例と一致すること(本ソフトが 1% 以内で安全側)。

    計算例の「第1断面」の σc・σs は、**押込み軸力が最小の杭**(引張側が
    最も厳しい杭)の杭頭断面から取られている。最大軸力の杭ではない。

    残る 1% 以内の差は換算断面の扱いによる
    (:func:`test_transformed_section_deducts_the_concrete_the_rebar_displaces`)。
    """
    result = analyze_circular_rc(
        DIAMETER, REBAR, EC, 15.0, axial=axial, moment=moment
    )
    assert result.sigma_c == pytest.approx(sigma_c, rel=0.01)
    assert result.sigma_s_tension == pytest.approx(sigma_s, rel=0.01)
    # 本ソフトのほうが大きい = 安全側
    assert result.sigma_c >= sigma_c
    assert result.sigma_s_tension >= sigma_s


def test_permanent_case_section_stress():
    """常時は全断面圧縮で、鉄筋も圧縮になること(計算例 σc=1.08, σs=−16.20)。"""
    result = analyze_circular_rc(
        DIAMETER, REBAR, EC, 15.0, axial=1418.32, moment=0.0
    )
    assert result.fully_compressed
    assert result.sigma_c == pytest.approx(1.09, abs=0.01)   # 計算例 1.08
    assert result.sigma_s_compression == pytest.approx(16.35, abs=0.02)  # 16.20
    assert result.sigma_s_tension == 0.0


def test_transformed_section_deducts_the_concrete_the_rebar_displaces():
    """換算断面積は Ac +(n−1)·As。計算例は Ac + n·As で 0.9% 大きい。

    鉄筋が占める分のコンクリートを差し引くかどうかの違いである。差し引く
    (本ソフト)ほうが換算断面積が小さくなり、応力度は**大きく = 安全側**に
    出る。純軸圧縮で検算すると

        本ソフト: At = 1.301225 m² → σc = 1.090、σs = 15×1.090 = 16.35
        計算例  : At = 1.313385 m² → σc = 1.080、σs = 15×1.080 = 16.20

    となり、計算例の表示値 1.08 / 16.20 は**控除なし**で説明できる。
    安全側なので追随しない。
    """
    from core.section.rc import transformed_section

    n = 15.0
    area_t, _ = transformed_section(DIAMETER, REBAR, n)
    concrete = math.pi * DIAMETER**2 / 4.0
    steel = REBAR.total_area
    assert area_t == pytest.approx(concrete + (n - 1.0) * steel)
    assert area_t < concrete + n * steel

    sigma_c_ours = 1418.32 / area_t / 1000.0
    sigma_c_theirs = 1418.32 / (concrete + n * steel) / 1000.0
    assert sigma_c_ours == pytest.approx(1.090, abs=0.001)
    assert sigma_c_theirs == pytest.approx(1.080, abs=0.001)
    assert sigma_c_ours > sigma_c_theirs


def test_underground_maximum_moment_comes_from_the_hinged_head_case():
    """地中部最大曲げは**杭頭ヒンジ**の解析から出ていること。

    計算例は杭頭を剛結とヒンジの両方で解き、断面計算にはヒンジ時の断面力を
    採用している(3章の脚注「(*)は、ヒンジ時の断面力を採用する」)。
    剛結のまま地中部の極値を取ると 193 kN·m 程度にしかならず、計算例の
    409.04 kN·m には合わない。

    本ソフトは**杭頭剛結のみ**に対応しており、ヒンジは未対応である。
    ここでは分布バネモデルの杭頭剛性を回転自由に縮約して再現する。
    """
    n_elements = 400
    model = PileLateralModel(
        EI, DIAMETER, LENGTH, _node_kh(True, n_elements), n_elements=n_elements
    )
    k = model.head_stiffness()
    k1, k2, k4 = float(k[0, 0]), float(k[0, 1]), float(k[1, 1])
    k1_hinged = k1 - k2 * k2 / k4

    result = solve_stability(
        ARRANGEMENT, kv=KV, k1=k1_hinged, k2=0.0, k4=0.0,
        v_load=16627.4, h_load=2410.5, m_load=19623.1,
    )
    # 杭頭が回転自由なので、杭体側の回転は杭頭モーメントが 0 になる値をとる
    response = model.solve(result.u, -k2 / k4 * result.u)
    y = response.displacements
    le = LENGTH / n_elements
    moments = -EI * (y[2:] - 2.0 * y[1:-1] + y[:-2]) / le**2
    assert abs(moments[0]) < 20.0  # 杭頭はほぼモーメントフリー
    assert abs(moments).max() == pytest.approx(409.04, rel=5e-3)


# --- 5章・7章(第35回に追加された章)--------------------------------------

# 1.5 地層データの最大周面摩擦力度 f (kN/m²)
EXPECTED_F = [0.0, 38.0, 100.0, 200.0]


def _bearing_profile():
    """計算例と同じ f になる地盤モデル。

    粘性土は c を与えるとその値が f になる。計算例の 38.0 は 10N
    (= 10 × 3.8)だが、本ソフトは N < 5 の粘性土で N 値推定を拒むため、
    ここでは c = 38 として同じ f を与える(:func:`test_soft_clay_needs_cohesion`)。
    """
    from core.models import SoilLayer, SoilProfile, SoilType

    return SoilProfile(
        layers=[
            SoilLayer(
                name="1", soil_type=SoilType.CLAY, thickness=5.0, n_value=2.0,
                gamma_t=16.0, gamma_sat=16.0, cohesion=0.0,
            ),
            SoilLayer(
                name="2", soil_type=SoilType.CLAY, thickness=12.0, n_value=3.8,
                gamma_t=16.0, gamma_sat=16.0, cohesion=38.0,
            ),
            SoilLayer(
                name="3", soil_type=SoilType.SAND, thickness=6.0, n_value=20.0,
                gamma_t=18.0, gamma_sat=18.0,
            ),
            SoilLayer(
                name="4", soil_type=SoilType.SAND, thickness=2.0, n_value=50.0,
                gamma_t=20.0, gamma_sat=20.0,
            ),
        ],
        gwl=0.0,
    )


def test_sand_skin_friction_matches():
    """場所打ち杭の砂質土の f = 5N(上限 200)が一致すること。"""
    from core.capacity.bearing import skin_friction_intensity
    from core.models import SoilLayer, SoilType

    for n_value, expected in ((20.0, 100.0), (50.0, 200.0)):
        layer = SoilLayer(
            name="s", soil_type=SoilType.SAND, thickness=1.0, n_value=n_value,
            gamma_t=18.0, gamma_sat=18.0,
        )
        got = skin_friction_intensity(ConstructionMethod.CAST_IN_PLACE, layer)
        assert got == pytest.approx(expected)


def test_soft_clay_needs_cohesion():
    """N < 5 の粘性土では c を要求すること(計算例は 10N を使っている)。

    計算例は N = 3.8 に対して f = 38.0 = 10N をそのまま適用している。
    本ソフトは道示の注記に従い、この範囲では N 値による推定を行わず
    エラーとする。**推定しない側**の判断なので追随しない。
    """
    from core.capacity.bearing import skin_friction_intensity
    from core.models import SoilLayer, SoilType

    layer = SoilLayer(
        name="c", soil_type=SoilType.CLAY, thickness=1.0, n_value=3.8,
        gamma_t=16.0, gamma_sat=16.0,
    )
    with pytest.raises(ValueError, match="軟弱粘性土"):
        skin_friction_intensity(ConstructionMethod.CAST_IN_PLACE, layer)


def test_ultimate_bearing_capacity_matches_without_the_tip_zone_exclusion():
    """極限支持力 Ru が計算例の設計極限押込力 8882.00 kN と一致すること。

    **ただし先端 1D 区間の周面摩擦を除外しない場合**である。本ソフトは
    既定で除外しており(道示Ⅳ 12.4.1 の規定として実装)、その場合は
    7977 kN と 10.2% 小さくなる。計算例は除外していない。

    除外するほうが支持力を小さく見るので**本ソフトが安全側**である。
    どちらが H24 の規定かは原典未照合(docs/VERIFICATION.md 第35回)。
    """
    from core.capacity.bearing import compute_bearing_capacity

    profile = _bearing_profile()
    without = compute_bearing_capacity(
        PILE, profile, 0.0, exclude_tip_zone=False
    )
    assert without.ru == pytest.approx(8882.00, abs=0.2)
    assert without.tip_resistance == pytest.approx(3392.92, abs=0.1)
    assert without.skin_resistance == pytest.approx(5488.99, abs=0.1)

    with_exclusion = compute_bearing_capacity(
        PILE, profile, 0.0, exclude_tip_zone=True
    )
    assert with_exclusion.ru == pytest.approx(7977.13, abs=0.2)
    assert with_exclusion.ru < without.ru  # 除外するほうが安全側


def test_group_pile_pressure_coefficients_match():
    """水平地盤反力度の上限値の補正係数が計算例と一致すること。

    計算例 7.1 の「単杭および群杭に関する補正係数」:
        単杭 αp — 砂質土 3.000、粘性土 1.500(2<N)/ 1.000(N≦2)
        群杭 ηp·αp — 砂質土 2.500、粘性土 ηp = 1.000
    砂質土の 2.500 は ηp·αp = min(s/D, αp) = min(3.0/1.2, 3.0) から出る。
    """
    from core.capacity.lateral_limit import alpha_p, eta_p_alpha_p
    from core.models import SoilLayer, SoilType

    def layer(soil_type, n_value):
        return SoilLayer(
            name="x", soil_type=soil_type, thickness=1.0, n_value=n_value,
            gamma_t=18.0, gamma_sat=18.0, k_ep=3.0,
        )

    assert alpha_p(layer(SoilType.SAND, 20.0)) == pytest.approx(3.000)
    assert alpha_p(layer(SoilType.CLAY, 3.8)) == pytest.approx(1.500)
    assert alpha_p(layer(SoilType.CLAY, 2.0)) == pytest.approx(1.000)

    spacing, diameter = 3.0, 1.2
    assert eta_p_alpha_p(
        layer(SoilType.SAND, 20.0), diameter, spacing
    ) == pytest.approx(2.500)
    # 粘性土は ηp = 1.0 なので αp がそのまま出る
    assert eta_p_alpha_p(
        layer(SoilType.CLAY, 3.8), diameter, spacing
    ) == pytest.approx(1.500)
    assert eta_p_alpha_p(
        layer(SoilType.CLAY, 2.0), diameter, spacing
    ) == pytest.approx(1.000)


def test_level2_kh_equals_the_seismic_kh_because_the_corrections_cancel():
    """レベル2の kHE が地震時の kH に等しいこと。

    計算例 7.1 は 単杭 αk = 1.500、群杭 ηk = 0.66667 を掲げ、
    7.1 の地盤反力係数 kHE(6932.334 / 13171.434 / 69323.339 / 173308.351)は
    1.6 の**地震時 kH と同じ値**である。αk · ηk = 1.5 × 2/3 = 1.0 で
    相殺するためで、本ソフトが kH(α = 2)をそのままレベル2に使うのと
    数値的に一致する。

    .. note::
       相殺するのは**群杭**の場合である。単杭なら αk のみが効いて
       1.5 倍になるはずだが、本ソフトは群杭のみを扱うため影響しない。
    """
    alpha_k, eta_k = 1.5, 2.0 / 3.0
    assert alpha_k * eta_k == pytest.approx(1.0)
    expected_khe = [6932.334, 13171.434, 69323.339, 173308.351]
    for (_, _, alpha_e0_eq), khe in zip(LAYERS, expected_khe):
        assert kh_from_e0(alpha_e0_eq, _BH, 1.0) * alpha_k * eta_k == pytest.approx(
            khe, rel=2e-4
        )


def test_rebar_nominal_area_confirmed_again_by_the_anchorage_calculation():
    """杭頭補強鉄筋の定着長からも D25 = 506.7 mm² が確認できること。

    計算例 6.4: σsa = 200.00、τoa = 1.600、Ast = 506.7、u = 80、
    Lo = 792 mm。Lo = σsa·Ast /(τoa·u)= 200×506.7 /(1.6×80)= 791.7。
    """
    assert REBAR.bar_area * 1.0e6 == pytest.approx(506.7)
    lo = 200.0 * 506.7 / (1.6 * 80.0)
    assert lo == pytest.approx(792.0, abs=0.5)
