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


# --- 7.6 予備計算(第36回に追加された章)-----------------------------------


def test_characteristic_value_and_loading_width_match_7_6_2():
    """7.6.2 の β・1/β・平均 α·E0・BH・kH0 が一致すること。

    計算例は「**※地震時BH算出時のα・Eoの取扱い:常時**」と明記しており、
    第29回に修正した「BH は常時の条件で定める」がそのまま裏付けられる。
    """
    from core.capacity.springs import lateral_springs, mean_e0
    from core.models import LoadCase, SoilLayer, SoilProfile, SoilType

    profile = SoilProfile(
        layers=[
            SoilLayer(
                name="1", soil_type=SoilType.CLAY, thickness=5.0, n_value=2.0,
                e0=5600.0, gamma_t=16.0, gamma_sat=16.0, cohesion=30.0,
            ),
            SoilLayer(
                name="2", soil_type=SoilType.CLAY, thickness=12.0, n_value=3.8,
                e0=10640.0, gamma_t=16.0, gamma_sat=16.0, cohesion=30.0,
            ),
            SoilLayer(
                name="3", soil_type=SoilType.SAND, thickness=6.0, n_value=20.0,
                e0=56000.0, gamma_t=18.0, gamma_sat=18.0, phi=30.0,
            ),
            SoilLayer(
                name="4", soil_type=SoilType.SAND, thickness=2.0, n_value=50.0,
                e0=140000.0, gamma_t=20.0, gamma_sat=20.0, phi=40.0,
            ),
        ],
        gwl=0.0,
    )
    springs = lateral_springs(PILE, SECTION, profile, 0.0, LoadCase.PERMANENT)
    assert INERTIA == pytest.approx(0.101787619, rel=1e-6)
    assert springs.beta == pytest.approx(0.149629, rel=1e-5)
    assert 1.0 / springs.beta == pytest.approx(6.6832, abs=0.001)
    assert mean_e0(profile, 0.0, 1.0 / springs.beta) == pytest.approx(6869.3, abs=0.2)
    assert springs.bh == pytest.approx(2.8319, abs=0.0002)
    assert springs.e0 / 0.3 == pytest.approx(22897.7, abs=0.5)


@pytest.mark.parametrize(
    "k_ep, sigma_v_eff, cohesion, expected",
    [
        (1.000, 0.00, 30.0, 60.00),      # 層1 上端
        (1.000, 34.95, 30.0, 94.95),     # 層1 下端
        (1.000, 118.83, 30.0, 178.83),   # 層2 下端
        (3.505, 118.83, 0.0, 416.52),    # 層3 上端(φ=30)
        (3.505, 172.77, 0.0, 605.59),    # 層3 下端
        (5.996, 172.77, 0.0, 1035.94),   # 層4 上端(φ=40)
        (5.996, 194.75, 0.0, 1167.73),   # 層4 下端
    ],
)
def test_passive_pressure_formula_matches_7_6_3(
    k_ep, sigma_v_eff, cohesion, expected
):
    """受働土圧強度 pEp = KEp・σ'v + 2c・√KEp が一致すること。

    計算例は壁面摩擦角を **δE = −φ/6** と定めており、そこから KEp を
    求めている(φ=30 → 3.505、φ=40 → 5.996)。本ソフトは KEp を
    利用者入力に委ねているので、KEp を与えたうえで式だけを照合する。
    """
    got = k_ep * sigma_v_eff + 2.0 * cohesion * math.sqrt(k_ep)
    assert got == pytest.approx(expected, abs=0.05)


def test_non_front_row_halving_applies_to_sand_only():
    """砂質土のみ最前列以外を 1/2 とすること(7.6.3 の表)。

    計算例の pHu:
        層1・2(粘性土): 1列目 = 2列目以降(60.00 / 94.95 / 142.43 / 268.25)
        層3・4(砂質土): 2列目以降がちょうど 1/2
                        (1041.30 → 520.65、2589.85 → 1294.93)
    """
    from core.standards import NON_FRONT_ROW_FACTOR_SAND

    assert NON_FRONT_ROW_FACTOR_SAND == pytest.approx(0.5)
    for front, back in ((1041.30, 520.65), (1513.98, 756.99),
                        (2589.85, 1294.93), (2919.32, 1459.66)):
        assert back == pytest.approx(front * NON_FRONT_ROW_FACTOR_SAND, abs=0.01)


def test_pile_body_axial_limits_match_7_6_4():
    """杭体から決まる支持力の上限値が計算例と一致すること。

    7.6.4: Rpu = 0.85・σck・Ac + σy・As = 27267 kN
    (φ1200、σck = 24 N/mm²、D25 × 24本、σy = 345 N/mm²)

    引抜き側は 7.6.5 の本文を入手できていないが、同サンプルの
    **設計極限引抜力 PTu = 4195 kN が σy・As = 4195.5 kN と一致**する。
    """
    from core.analysis.level2 import pile_body_axial_limits

    limits = pile_body_axial_limits(PILE, REBAR, fck=24, rebar_grade="SD345")
    assert limits.concrete_area == pytest.approx(1.131, abs=0.001)
    assert limits.rebar_area * 1.0e4 == pytest.approx(121.608, abs=1e-3)
    assert limits.push == pytest.approx(27267.0, abs=1.0)
    assert limits.pull == pytest.approx(4195.0, abs=1.0)


def test_pile_body_limit_governs_the_uplift_and_is_the_safe_side():
    """引抜きは杭体から決まる上限値が支配し、入れないと 1.41 倍の過大評価になる。

    7.6.5 は地盤から決まる極限引抜き力を Pu + W = U·Σ(Li·fi) + W とし、
    W を**有効重量**(水中部 16.61 kN/m × 25 m = 415.4 kN)としている。
    本ソフトも浮力を控除した W = 415.0 kN を使うので同じ 5904 kN になる。
    """
    from core.analysis.level2 import AxialSpringModel, pile_body_axial_limits
    from core.capacity.bearing import compute_bearing_capacity

    bearing = compute_bearing_capacity(
        PILE, _bearing_profile(), 0.0, exclude_tip_zone=False
    )
    body = pile_body_axial_limits(PILE, REBAR, fck=24, rebar_grade="SD345")

    without = AxialSpringModel.from_bearing(KV, bearing)
    with_body = AxialSpringModel.from_bearing(KV, bearing, body=body)

    assert with_body.pull_limit == pytest.approx(4195.0, abs=1.0)
    assert without.pull_limit == pytest.approx(5904.0, abs=1.0)  # 計算例 5904
    assert without.pull_limit / with_body.pull_limit == pytest.approx(1.41, abs=0.01)
    assert with_body.pull_limit < without.pull_limit  # 入れるほうが安全側
    # 押込みは地盤側が支配する(杭体 27267 ≫ 地盤 8189)
    assert with_body.push_limit == pytest.approx(without.push_limit)


def test_pile_body_limits_are_only_for_verified_pile_types():
    """場所打ち杭・PHC杭以外は式が確認できていないので拒むこと。"""
    from core.analysis.level2 import pile_body_axial_limits

    steel = PileSpec(
        pile_type=PileType.STEEL_PIPE,
        method=ConstructionMethod.DRIVEN,
        diameter=1.2,
        length=25.0,
        wall_thickness=12.0,
    )
    with pytest.raises(ValueError, match="場所打ち杭・PHC杭のみ"):
        pile_body_axial_limits(steel, REBAR, fck=24)


def test_uplift_limit_formula_matches_7_6_5():
    """7.6.5 の式・値がそのまま一致すること。

        1) 地盤から決まる  Pu + W = U·Σ(Li·fi) + W = 5489.0 + 415.4 = 5904
           W は**有効重量**(水中部単位長重量 16.61 kN/m × 25.000 m)
        2) 杭体から決まる  Ppu = σy·As = 4195
        3) PTu = min(Pu + W, Ppu) = 4195

    第36回に σy·As = 4195.5 から推定した内容が、本文で確認できた。
    """
    from core.analysis.level2 import pile_body_axial_limits
    from core.capacity.bearing import compute_bearing_capacity

    bearing = compute_bearing_capacity(
        PILE, _bearing_profile(), 0.0, exclude_tip_zone=False
    )
    assert bearing.skin_resistance == pytest.approx(5489.0, abs=0.1)
    # 有効重量(浮力控除後)。計算例 415.4 kN
    assert bearing.w_pile == pytest.approx(415.4, abs=0.5)
    ground = bearing.skin_resistance + bearing.w_pile
    assert ground == pytest.approx(5904.0, abs=1.0)

    body = pile_body_axial_limits(PILE, REBAR, fck=24, rebar_grade="SD345")
    assert body.pull == pytest.approx(4195.0, abs=1.0)
    assert min(ground, body.pull) == pytest.approx(4195.0, abs=1.0)


# --- Kui_2(PHC杭・中掘り杭工法)の 5.2〜5.3(第37回)-----------------------
#
# 別サンプル Kui_2 は PHC杭 φ800・中掘り杭(セメントミルク噴出攪拌)・突出杭。
# 5.3「許容支持力・引抜力の計算」が載っており、**許容値の式と安全率**を
# 直接突き合わせられる(Kui_1 には無かった章)。


def test_allowable_capacity_formulas_and_safety_factors_match():
    """許容支持力・引抜力の式と安全率が計算例と一致すること。

    Kui_2 の 5.3(杭タイプ1)は式と安全率を明記している。

        Ra =(Ru − Ws)/ n + Ws − W   n = 3.0(常時)/ 2.0(地震時)
        Pa = Pu / n + W              n = 6.0(常時)/ 3.0(地震時)

    Ru = 5520、Ws = 93.1(杭で置き換えられる部分の土の有効重量)、
    W = 97.3(杭の有効重量)、Pu = 1750 に対する結果は
    1805 / 2709(支持力)、389 / 681(引抜力)。
    """
    from core.capacity.bearing import BearingCapacity
    from core.models import LoadCase

    bearing = BearingCapacity(
        qd=7500.0, tip_area=0.503, tip_resistance=3772.5, ru=5520.0,
        w_soil=93.1, w_pile=97.3, skin_resistance=1750.0,
    )
    assert bearing.allowable_push(LoadCase.PERMANENT) == pytest.approx(1805, abs=1)
    assert bearing.allowable_push(LoadCase.LEVEL1_EQ) == pytest.approx(2709, abs=1)
    assert bearing.allowable_pull(LoadCase.PERMANENT) == pytest.approx(389, abs=1)
    assert bearing.allowable_pull(LoadCase.LEVEL1_EQ) == pytest.approx(681, abs=1)


def test_safety_factors_match_the_worked_example():
    """支持杭の安全率 3.0 / 2.0(押込み)、6.0 / 3.0(引抜き)。"""
    from core.standards import SAFETY_FACTORS_PULL, SAFETY_FACTORS_PUSH

    assert SAFETY_FACTORS_PUSH["常時"]["支持杭"] == pytest.approx(3.0)
    assert SAFETY_FACTORS_PUSH["レベル1地震時"]["支持杭"] == pytest.approx(2.0)
    assert SAFETY_FACTORS_PULL["常時"] == pytest.approx(6.0)
    assert SAFETY_FACTORS_PULL["レベル1地震時"] == pytest.approx(3.0)


def test_inner_digging_axial_spring_coefficient_matches():
    """中掘り杭工法の a = 0.010・(L/D)+ 0.36 と Kv が一致すること。

    Kui_2 の 5.2(杭タイプ1): L = 23.300 m、D = 0.8000 m、
    Ap = 0.24850 m²、Ep = 4.00×10⁷ kN/m² → a = 0.6513、Kv = 277829 kN/m。
    """
    from core.capacity.springs import PileSection, axial_spring
    from core.standards import KV_A_COEF

    slope, intercept = KV_A_COEF["中掘り"]
    assert (slope, intercept) == (0.010, 0.36)
    assert slope * (23.300 / 0.8) + intercept == pytest.approx(0.6513, abs=1e-4)

    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.INNER_DIGGING,
        diameter=0.8,
        length=23.300,
        concrete_thickness=115.0,
    )
    section = PileSection(area=0.24850, inertia=1.0, young=4.0e7)
    assert axial_spring(phc, section) == pytest.approx(277829, rel=1e-4)


def test_skin_friction_intensity_is_a_user_input_in_the_worked_examples():
    """計算例の f は**利用者入力**であり、f の表の照合には使えないこと。

    決定的な証拠は Kui_1 の 1.5 地層データで、N = 2 の粘性土に **f = 0.0**
    が入っていること。10N = 20 でも c = 30 でもなく、どの推定式からも
    出てこない値である。Kui_2 でも N = 2 の粘性土が f = 0.0 になっている。

    したがって計算例の f と本ソフトの推定値が食い違っても、それは
    **本ソフトの誤りを意味しない**。実際 Kui_2(中掘り杭)では
    砂質土 N=20 → 計算例 20.0 に対し本ソフトは 3N = 60.0 である。
    """
    from core.capacity.bearing import skin_friction_intensity
    from core.models import SoilLayer, SoilType
    from core.standards import F_SPECS

    # 中掘り杭の砂質土は 3N(第4回以降の値)
    assert F_SPECS["中掘り"]["砂質土"] == (3.0, "N")
    layer = SoilLayer(
        name="s", soil_type=SoilType.SAND, thickness=1.0, n_value=20.0,
        gamma_t=18.0, gamma_sat=18.0,
    )
    assert skin_friction_intensity(
        ConstructionMethod.INNER_DIGGING, layer
    ) == pytest.approx(60.0)


def test_phc_pile_body_axial_limits_match_kui2():
    """PHC杭でも同じ式で、PC鋼材の降伏点を使うこと(Kui_2 の 7.5.4)。

        Rpu = 0.85・σck・Ac + σy・As
            σck = 80.00 ×10³ kN/m²、Ac = 0.238 m²(φ800・t=110)
            σy  = 1275.00 ×10³ kN/m²(**PC鋼材**の降伏点)
            As  = 25.120 ×10⁻⁴ m²(**PC鋼材量**)
        → 19417 kN
    """
    from core.analysis.level2 import pile_body_axial_limits
    from core.standards import PRESTRESSING_STEEL_YIELD_POINT

    assert PRESTRESSING_STEEL_YIELD_POINT == 1275.0
    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.INNER_DIGGING,
        diameter=0.8,
        length=26.0,
        concrete_thickness=110.0,
    )
    limits = pile_body_axial_limits(
        phc, fck=80, prestressing_steel_area=25.120e-4
    )
    assert limits.concrete_area == pytest.approx(0.238, abs=0.001)
    assert limits.push == pytest.approx(19417.0, abs=1.0)
    # 7.5.5: Ppu = σy·As = 3203 kN
    assert limits.pull == pytest.approx(3203.0, abs=1.0)


def test_min_is_exercised_on_both_sides_across_the_two_samples():
    """min(地盤, 杭体)の支配側が2つのサンプルで入れ替わること。

    Kui_1(場所打ち杭): 引抜きは杭体が支配(地盤 5904 > 杭体 4195)
    Kui_2(PHC杭 (1)杭): 引抜きは地盤が支配(地盤 1851 < 杭体 3203)
    どちらか一方だけを見ていたら min の必要性に気づけなかった。
    """
    assert min(5904.0, 4195.0) == 4195.0   # Kui_1 → 杭体
    assert min(1851.0, 3203.0) == 1851.0   # Kui_2 → 地盤


def test_phc_pile_body_limits_need_the_prestressing_steel_area():
    """PHC杭は PC鋼材量の入力を要求すること(モデルに持っていないため)。"""
    from core.analysis.level2 import pile_body_axial_limits

    phc = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.INNER_DIGGING,
        diameter=0.8,
        length=26.0,
        concrete_thickness=110.0,
    )
    with pytest.raises(ValueError, match="PC鋼材量"):
        pile_body_axial_limits(phc, fck=80)


# --- Kui_3(鋼管ソイルセメント杭・液状化考慮)(第39回)------------------------
#
# 別サンプル Kui_3 は鋼管ソイルセメント杭 φ1000(固化体径)/鋼管径800mm、
# 継杭(上杭 t=19.0mm/中杭 t=14.0mm/下杭 t=11.0mm、いずれも SKK490)、
# 液状化考慮(地震時(液有))のケースを含む。


def test_bilinear_m_phi_matches_the_worked_example_for_each_segment():
    """全塑性モーメント Mp・降伏モーメント My が3区間すべて一致すること。

    軸力 N = 959.9 kN(死荷重時軸力、浮力無視)に対し、上杭・中杭・下杭で
    板厚が異なる(19.0 / 14.0 / 11.0 mm)。鋼管ソイルセメント杭は鋼管部で
    照査するので、鋼管杭と同じ式(:func:`plastic_moment_steel_pipe` /
    :func:`yield_moment_steel_pipe`)がそのまま使える。
    """
    from core.analysis.level2 import plastic_moment_steel_pipe, yield_moment_steel_pipe
    from core.capacity.section import pile_section

    axial = 959.9
    segments = [
        (19.0, 3429.9, 2466.6),  # 上杭
        (14.0, 2495.6, 1764.8),  # 中杭
        (11.0, 1919.6, 1330.4),  # 下杭
    ]
    for thickness, expected_mp, expected_my in segments:
        pile = PileSpec(
            pile_type=PileType.STEEL_PIPE_SOIL_CEMENT,
            method=ConstructionMethod.STEEL_PIPE_SOIL_CEMENT,
            diameter=0.8, length=10.0, wall_thickness=thickness,
            soil_cement_diameter=1.0,
        )
        section = pile_section(pile)
        mp = plastic_moment_steel_pipe(
            pile, axial, steel_grade="SKK490", corrosion_mm=1.0
        )
        my = yield_moment_steel_pipe(
            pile, section, axial, steel_grade="SKK490", corrosion_mm=1.0
        )
        assert mp == pytest.approx(expected_mp, rel=3e-4), thickness
        assert my == pytest.approx(expected_my, rel=3e-4), thickness


def test_allowable_capacity_with_liquefaction_matches():
    """液状化考慮(地震時(液有))を含む許容支持力・引抜力が一致すること。

    地盤から決まる極限支持力・極限引抜力は、DE を周面摩擦力度に乗じた
    Σ(Li・fi・DEi) から求まる(常時/地震時(液無)と地震時(液有)で
    Σ(Li・fi) が変わる)。安全率は Kui_2 で確認したものと同じ
    (押込み 3.0/2.0、引抜き 6.0/3.0)。6項目すべて計算例と一致する。
    """
    from core.capacity.bearing import BearingCapacity
    from core.models import LoadCase

    qd_ap = 7500.0 * 0.785
    skin_normal = 3.142 * 3410.0   # 常時・地震時(液無)
    skin_liq = 3.142 * 2993.7      # 地震時(液有)
    ws = 203.5
    w = 278.1

    normal = BearingCapacity(
        qd=7500.0, tip_area=0.785, tip_resistance=qd_ap,
        ru=qd_ap + skin_normal, w_soil=ws, w_pile=w,
        skin_resistance=skin_normal,
    )
    liquefied = BearingCapacity(
        qd=7500.0, tip_area=0.785, tip_resistance=qd_ap,
        ru=qd_ap + skin_liq, w_soil=ws, w_pile=w,
        skin_resistance=skin_liq,
    )
    assert normal.ru == pytest.approx(16603, abs=2)
    assert liquefied.ru == pytest.approx(15296, abs=3)

    assert normal.allowable_push(LoadCase.PERMANENT) == pytest.approx(5392, abs=1)
    assert normal.allowable_push(LoadCase.LEVEL1_EQ) == pytest.approx(8125, abs=1)
    assert liquefied.allowable_push(LoadCase.LEVEL1_EQ) == pytest.approx(7471, abs=1)

    assert normal.allowable_pull(LoadCase.PERMANENT) == pytest.approx(2064, abs=1)
    assert normal.allowable_pull(LoadCase.LEVEL1_EQ) == pytest.approx(3849, abs=1)
    assert liquefied.allowable_pull(LoadCase.LEVEL1_EQ) == pytest.approx(3413, abs=1)
