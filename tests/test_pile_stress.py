"""杭体応力度照査・杭頭結合部・負の周面摩擦力のテスト。"""
import dataclasses
import math

import pytest

from core.capacity.negative_friction import (
    compute_negative_friction,
    negative_friction_intensity,
)
from core.models import (
    ConstructionMethod,
    Footing,
    LoadCase,
    PileArrangement,
    PileSpec,
    PileType,
    SoilLayer,
    SoilProfile,
    SoilType,
)
from core.section.checks import MaterialSpec, check_section
from core.section.pile_head import (
    check_pile_head,
    edge_distances,
    horizontal_bearing_stress,
    horizontal_edge_punching_shear,
    punching_shear_area,
)
from core.section.rc import RebarLayout
from core.standards import SIGMA_A_STEEL, SIGMA_SA_REBAR_STATIC, STRESS_INCREASE

CIP = PileSpec(
    pile_type=PileType.CAST_IN_PLACE,
    method=ConstructionMethod.CAST_IN_PLACE,
    diameter=1.0,
    length=20.0,
)
STEEL = PileSpec(
    pile_type=PileType.STEEL_PIPE,
    method=ConstructionMethod.DRIVEN,
    diameter=1.0,
    length=20.0,
    wall_thickness=12.0,
)
MATERIAL = MaterialSpec(
    fck=24, rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)
)


def test_steel_pipe_stress_hand_calculation():
    """σ = N/A ± M/Z。腐食しろ 1mm を**外面から**控除する。

    外径 1.000 → 0.998 m、板厚 12 → 11 mm、内径は 0.976 m のまま。
    """
    d_out = 1.0 - 2 * 0.001
    t = 0.011
    d_in = d_out - 2 * t
    area = math.pi * (d_out**2 - d_in**2) / 4
    inertia = math.pi * (d_out**4 - d_in**4) / 64
    z = inertia / (d_out / 2)

    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=2000.0, moment=300.0
    )
    compression = next(c for c in result.checks if "圧縮" in c.name)
    expected = (2000.0 / area + 300.0 / z) / 1000.0
    assert compression.stress == pytest.approx(expected, rel=1e-9)
    assert compression.allowable == pytest.approx(SIGMA_A_STEEL["SKK400"])


def test_steel_pipe_tension_side():
    """曲げが軸圧縮を上回ると引張側が生じる。"""
    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=100.0, moment=2000.0
    )
    tension = next(c for c in result.checks if "引張" in c.name)
    assert tension.stress > 0


def test_no_tension_when_axial_dominates():
    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=5000.0, moment=50.0
    )
    tension = next(c for c in result.checks if "引張" in c.name)
    assert tension.stress == 0.0


def test_storm_case_uses_125_not_150():
    """暴風時の割増は 1.25(地震時 1.5 とは異なる)。"""
    normal = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=2000.0, moment=300.0
    )
    storm = check_section(
        STEEL, MATERIAL, LoadCase.STORM, depth=0.0, axial=2000.0, moment=300.0
    )
    seismic = check_section(
        STEEL, MATERIAL, LoadCase.LEVEL1_EQ, depth=0.0, axial=2000.0, moment=300.0
    )
    base = normal.checks[0].allowable
    assert storm.checks[0].allowable == pytest.approx(base * 1.25)
    assert seismic.checks[0].allowable == pytest.approx(base * 1.50)
    assert storm.checks[0].allowable < seismic.checks[0].allowable


def test_seismic_case_increases_allowable():
    normal = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=2000.0, moment=300.0
    )
    seismic = check_section(
        STEEL, MATERIAL, LoadCase.LEVEL1_EQ, depth=0.0, axial=2000.0, moment=300.0
    )
    assert seismic.checks[0].allowable == pytest.approx(
        normal.checks[0].allowable * STRESS_INCREASE["レベル1地震時"]
    )
    # 発生応力度は同じ
    assert seismic.checks[0].stress == pytest.approx(normal.checks[0].stress)


def test_cast_in_place_checks_concrete_and_rebar():
    result = check_section(
        CIP, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=1500.0, moment=800.0
    )
    names = [c.name for c in result.checks]
    assert "コンクリート圧縮応力度" in names
    assert "鉄筋引張応力度" in names
    assert result.rc_detail is not None
    # 水中施工の許容曲げ圧縮は表-4.2.5(σck=24 → 8.0)。**0.8 倍の低減はない**
    concrete = next(c for c in result.checks if "コンクリート" in c.name)
    assert concrete.allowable == pytest.approx(8.0)
    rebar = next(c for c in result.checks if "鉄筋" in c.name)
    # 場所打ち杭は水中施工なので「水中又は地下水位以下に設ける部材」の 160
    assert rebar.allowable == pytest.approx(
        SIGMA_SA_REBAR_STATIC["水中又は地下水位以下に設ける部材"]["SD345"]
    )


def test_cast_in_place_uses_the_fixed_young_modulus_ratio_15():
    """RC の応力度計算は n = 15 を用いる(n = Es/Ec ではない)。

    かつて Es/Ec(σck=24 で 8.0)を用いていたが、道示Ⅲ 3.3 は σck に
    よらない一定値 15 を規定する。n = 8 は鉄筋引張応力度を約 18% 過小に
    評価する**非安全側**の誤りであったため、その再発をここで止める。
    """
    result = check_section(
        CIP, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=1500.0, moment=800.0
    )
    detail = result.rc_detail
    assert detail is not None
    # n = 15 での値。n = 8 なら鉄筋が約 18% 小さく出る。
    # 値は第34回に D25 の公称断面積(506.7 mm²)へ改めた分だけ動いている
    assert detail.sigma_s_tension == pytest.approx(141.953, abs=0.01)
    assert detail.sigma_c == pytest.approx(10.906, abs=0.01)

    # 独立に、同じ断面を n = 15 で解いた結果と一致すること
    from core.section.rc import analyze_circular_rc
    from core.standards import EC_CONCRETE, YOUNG_MODULUS_RATIO_RC

    assert YOUNG_MODULUS_RATIO_RC == 15.0
    expected = analyze_circular_rc(
        diameter=CIP.diameter,
        rebar=MATERIAL.rebar,
        ec=EC_CONCRETE[MATERIAL.fck],
        n_ratio=YOUNG_MODULUS_RATIO_RC,
        axial=1500.0,
        moment=800.0,
    )
    assert detail.sigma_s_tension == pytest.approx(expected.sigma_s_tension)


def test_cast_in_place_rejects_a_grade_without_an_allowable_stress():
    """水中コンクリートの表(24〜30)にない σck は弾く。

    Ec の表は 21〜60 を持つが、水中で施工する場所打ち杭の許容応力度は
    道示Ⅳ 表-4.2.5 が σck = 24/27/30 のみを規定する。
    """
    for fck in (21, 50):
        material = dataclasses.replace(MATERIAL, fck=fck)
        with pytest.raises(ValueError, match="表-4.2.5"):
            check_section(
                CIP, material, LoadCase.PERMANENT,
                depth=0.0, axial=1500.0, moment=800.0,
            )


def test_cast_in_place_rebar_allowable_by_load_case():
    """場所打ち杭の鉄筋は「水中又は地下水位以下に設ける部材」の区分。

    SD345: 常時 160、暴風時 160×1.25 = 200、地震時は区分が変わり
    軸方向鉄筋の基本値 200 に割増 1.50 を乗じて 300。
    """
    expected = {
        LoadCase.PERMANENT: 160.0,
        LoadCase.STORM: 200.0,
        LoadCase.LEVEL1_EQ: 300.0,
    }
    for case, allowable in expected.items():
        result = check_section(
            CIP, MATERIAL, case, depth=0.0, axial=1500.0, moment=800.0
        )
        rebar = next(c for c in result.checks if "鉄筋" in c.name)
        assert rebar.allowable == pytest.approx(allowable), case


def test_sd390_static_allowable_is_180_not_200():
    """SD390 の常時の基本値は SD345 と同じ 180(かつては 200 で非安全側)。"""
    from core.section.checks import rebar_tension_allowable

    for grade in ("SD345", "SD390", "SD490"):
        assert rebar_tension_allowable(
            grade, LoadCase.PERMANENT, underwater=False, increase=1.0
        ) == 180.0
        assert rebar_tension_allowable(
            grade, LoadCase.PERMANENT, underwater=True, increase=1.0
        ) == 160.0
    # 地震時だけ材質で差がつく
    seismic = {
        g: rebar_tension_allowable(g, LoadCase.LEVEL1_EQ, underwater=True, increase=1.0)
        for g in ("SD345", "SD390", "SD490")
    }
    assert seismic == {"SD345": 200.0, "SD390": 230.0, "SD490": 290.0}


def test_removed_rebar_grade_is_rejected_with_a_reason():
    from core.section.checks import rebar_tension_allowable

    with pytest.raises(ValueError, match="削除"):
        rebar_tension_allowable(
            "SD295", LoadCase.PERMANENT, underwater=True, increase=1.0
        )


def test_cast_in_place_requires_rebar():
    with pytest.raises(ValueError, match="鉄筋"):
        check_section(
            CIP, MaterialSpec(fck=24), LoadCase.PERMANENT, 0.0, 1500.0, 800.0
        )


def test_unimplemented_pile_type_raises():
    """H鋼杭は資料の警告により応力度照査を実装していない。"""
    from core.models.pile import HSection

    h_pile = PileSpec(
        pile_type=PileType.H_STEEL,
        method=ConstructionMethod.DRIVEN,
        diameter=0.4,
        length=20.0,
        h_section=HSection(
            height=400.0, width=400.0, web_thickness=13.0, flange_thickness=21.0
        ),
    )
    with pytest.raises(NotImplementedError, match="H鋼杭"):
        check_section(h_pile, MATERIAL, LoadCase.PERMANENT, 0.0, 1000.0, 100.0)


# --- PHC杭(全断面有効) ---------------------------------------------------

PHC = PileSpec(
    pile_type=PileType.PHC,
    method=ConstructionMethod.DRIVEN,
    diameter=0.6,
    length=20.0,
    concrete_thickness=90.0,
)


def _phc_section():
    """テストの期待値算定に用いる中空断面の A・Z。"""
    outer, inner = 0.6, 0.6 - 2 * 0.09
    area = math.pi * (outer**2 - inner**2) / 4.0
    inertia = math.pi * (outer**4 - inner**4) / 64.0
    return area, inertia / (outer / 2.0)


def test_phc_stress_matches_hand_calculation():
    area, z = _phc_section()
    axial, moment = 2000.0, 100.0
    result = check_section(PHC, MATERIAL, LoadCase.PERMANENT, 0.0, axial, moment)

    sigma_n = axial / area / 1000.0
    sigma_b = moment / z / 1000.0
    by_name = {c.name: c for c in result.checks}
    assert by_name["軸圧縮応力度"].stress == pytest.approx(sigma_n)
    assert by_name["曲げ圧縮応力度"].stress == pytest.approx(sigma_n + sigma_b)
    # 圧縮側が卓越しているため引張の照査は現れない
    assert "曲げ引張応力度" not in by_name
    assert by_name["軸圧縮応力度"].allowable == pytest.approx(23.0)
    assert by_name["曲げ圧縮応力度"].allowable == pytest.approx(27.0)
    assert result.all_ok


def test_phc_allowable_compression_is_increased_by_load_case():
    result = check_section(PHC, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 2000.0, 100.0)
    by_name = {c.name: c for c in result.checks}
    increase = STRESS_INCREASE[LoadCase.LEVEL1_EQ.value]
    assert by_name["曲げ圧縮応力度"].allowable == pytest.approx(27.0 * increase)
    assert by_name["軸圧縮応力度"].allowable == pytest.approx(23.0 * increase)


def test_phc_permanent_case_allows_no_tension():
    # 曲げが卓越して引張が生じるケース
    result = check_section(PHC, MATERIAL, LoadCase.PERMANENT, 0.0, 100.0, 400.0)
    tension = next(c for c in result.checks if c.name == "曲げ引張応力度")
    assert tension.stress > 0.0
    assert tension.allowable == 0.0
    assert tension.judgement == "NG"
    assert not result.all_ok


def test_phc_seismic_tension_allowable_depends_on_prestress():
    high = MaterialSpec(effective_prestress=8.0)
    mid = MaterialSpec(effective_prestress=5.0)
    low = MaterialSpec(effective_prestress=2.0)
    for material, expected in ((high, 5.0), (mid, 3.0), (low, 0.0)):
        result = check_section(
            PHC, material, LoadCase.LEVEL1_EQ, 0.0, 100.0, 400.0
        )
        tension = next(c for c in result.checks if c.name == "曲げ引張応力度")
        # 地震時の許容曲げ引張応力度には割増しを重ねない
        assert tension.allowable == pytest.approx(expected)


def test_phc_seismic_tension_requires_prestress_input():
    with pytest.raises(ValueError, match="有効プレストレス"):
        check_section(PHC, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 100.0, 400.0)
    # 引張が生じなければ σce の入力は不要
    check_section(PHC, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 2000.0, 100.0)


def test_phc_requires_concrete_thickness():
    thin = PileSpec(
        pile_type=PileType.PHC,
        method=ConstructionMethod.DRIVEN,
        diameter=0.6,
        length=20.0,
    )
    with pytest.raises(ValueError, match="肉厚"):
        check_section(thin, MATERIAL, LoadCase.PERMANENT, 0.0, 1000.0, 100.0)


def test_stress_check_judgement():
    # 過大な断面力で NG になること
    result = check_section(
        STEEL, MATERIAL, LoadCase.PERMANENT, depth=0.0, axial=50000.0, moment=5000.0
    )
    assert not result.all_ok


# --- 杭頭結合部 -------------------------------------------------------------


def test_punching_shear_area():
    # h = 1.5 − 0.1 = 1.4 → A = π(1.0+1.4)×1.4
    assert punching_shear_area(1.0, 1.5) == pytest.approx(math.pi * 2.4 * 1.4)


def test_punching_area_requires_thickness():
    with pytest.raises(ValueError, match="フーチング厚"):
        punching_shear_area(1.0, 0.05)


def test_pile_head_checks():
    """押込み力に対する押抜きせん断と支圧を照査する(道示Ⅳ 12.9.3)。"""
    result = check_pile_head(
        pile_diameter=1.0,
        footing_height=1.5,
        fck=24,
        case=LoadCase.PERMANENT,
        axial=2000.0,
        shear=200.0,
        moment=150.0,
    )
    names = [c.name for c in result.checks]
    assert "杭頭押抜きせん断応力度" in names
    assert "杭頭支圧応力度" in names
    tau = next(c for c in result.checks if "押抜き" in c.name)
    assert tau.stress == pytest.approx(2000.0 / result.punching_area / 1000.0)
    assert tau.allowable == pytest.approx(0.90)
    # 支圧は軸力を杭断面積で除した値。許容値は SIGMA_CVA_PILE_HEAD_BEARING
    # (第43回。曲げ圧縮の σca=8.0 とは別表で 7.2)
    bearing = next(c for c in result.checks if c.name == "杭頭支圧応力度")
    assert bearing.stress == pytest.approx(2000.0 / (math.pi / 4) / 1000.0)
    assert bearing.allowable == pytest.approx(7.2)


def test_punching_shear_allowable_is_not_increased():
    """杭頭結合部の τa3 には荷重組合せによる割増しを行わない(道示Ⅳ 4.2)。"""
    allowables = {}
    for case in (LoadCase.PERMANENT, LoadCase.STORM, LoadCase.LEVEL1_EQ):
        result = check_pile_head(1.0, 1.5, 24, case, 2000.0, 200.0, 150.0)
        tau = next(c for c in result.checks if "押抜き" in c.name)
        allowables[case] = tau.allowable
    # 地震時・暴風時も常時と同じ 0.90
    for case, allowable in allowables.items():
        assert allowable == pytest.approx(0.90), case

    # 一方、支圧は通常どおり割増しされる(原典未確認のため現状の扱い)
    normal = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 0.0, 0.0)
    seismic = check_pile_head(1.0, 1.5, 24, LoadCase.LEVEL1_EQ, 2000.0, 0.0, 0.0)
    n_b = next(c for c in normal.checks if "支圧" in c.name)
    s_b = next(c for c in seismic.checks if "支圧" in c.name)
    assert s_b.allowable == pytest.approx(n_b.allowable * 1.5)


def test_punching_shear_rejects_grade_outside_table():
    """τa3 の表は σck = 21〜30 のみ。範囲外は明示的にエラー。"""
    with pytest.raises(ValueError, match="範囲外"):
        check_pile_head(1.0, 1.5, 40, LoadCase.PERMANENT, 2000.0, 0.0, 0.0)


def test_removed_rebar_grade_is_rejected():
    """SD295 は H24 の道示Ⅳで削除されており選択できない。"""
    material = MaterialSpec(
        fck=24,
        rebar_grade="SD295",
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
    )
    with pytest.raises(ValueError, match="削除"):
        check_section(CIP, material, LoadCase.PERMANENT, 0.0, 1500.0, 800.0)


def test_unknown_rebar_grade_is_rejected():
    material = MaterialSpec(
        fck=24,
        rebar_grade="SD500",  # 存在しない材質
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
    )
    with pytest.raises(ValueError, match="未対応"):
        check_section(CIP, material, LoadCase.PERMANENT, 0.0, 1500.0, 800.0)


def test_sd490_is_now_supported():
    """H24 で新たに規定された SD490 を扱えること。"""
    material = MaterialSpec(
        fck=24,
        rebar_grade="SD490",
        rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0),
    )
    result = check_section(CIP, material, LoadCase.LEVEL1_EQ, 0.0, 1500.0, 800.0)
    rebar = next(c for c in result.checks if "鉄筋" in c.name)
    assert rebar.allowable == pytest.approx(290.0 * 1.5)


def test_pile_head_bearing_ignores_uplift():
    """支圧は押込み時のみ。引抜き時は 0 とする。"""
    result = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, -2000.0, 0.0, 0.0)
    bearing = next(c for c in result.checks if "支圧" in c.name)
    assert bearing.stress == 0.0


def test_edge_distance_standard():
    """縁端距離が 1.0D 以上なら標準、未満なら水平押抜きせん断の照査が必要。"""
    arrangement = PileArrangement(nx=2, ny=3, spacing_x=2.5, spacing_y=2.5)
    # 橋軸方向: 幅8.0 → 8/2 − 1.25 = 2.75m、直角方向: 8/2 − 2.5 = 1.5m
    wide = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)
    edge = edge_distances(wide, arrangement, diameter=1.0)
    assert edge.edge_x == pytest.approx(2.75)
    assert edge.edge_y == pytest.approx(1.5)
    assert edge.minimum == pytest.approx(1.5)
    assert edge.required == pytest.approx(1.0)
    assert edge.is_standard
    assert not edge.needs_horizontal_punching_check

    # 幅を詰めると縁端距離が 1.0D を下回る
    narrow = Footing(width_x=8.0, width_y=5.5, height=1.5, embedment=2.0)
    edge = edge_distances(narrow, arrangement, diameter=1.0)
    assert edge.edge_y == pytest.approx(0.25)
    assert not edge.is_standard
    assert edge.needs_horizontal_punching_check


def test_edge_distance_scales_with_diameter():
    """必要縁端距離は杭径に比例する。"""
    arrangement = PileArrangement(nx=2, ny=2, spacing_x=3.0, spacing_y=3.0)
    footing = Footing(width_x=6.0, width_y=6.0, height=2.0, embedment=2.0)
    # 縁端距離 = 3.0 − 1.5 = 1.5m
    assert edge_distances(footing, arrangement, 1.0).is_standard
    assert not edge_distances(footing, arrangement, 2.0).is_standard


def test_check_pile_head_includes_edge_distance_when_given():
    arrangement = PileArrangement(nx=2, ny=3, spacing_x=2.5, spacing_y=2.5)
    footing = Footing(width_x=8.0, width_y=8.0, height=1.5, embedment=2.0)
    with_edge = check_pile_head(
        1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 200.0, 150.0,
        footing=footing, arrangement=arrangement,
    )
    assert with_edge.edge_distance is not None
    without = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 200.0, 150.0)
    assert without.edge_distance is None


def test_pile_head_uplift_uses_the_pull_out_resistance_thickness():
    """引抜き時の押抜きせん断は専用の抵抗厚さ ht を使い、押込み側の面積とは
    異なること(Kui_5 6.2、第43回)。

    従来は押込み側の面積(フーチング厚基準、ここでは h=1.4m)を引抜き時にも
    流用していたが、正しくは ht(標準 100mm、押込み側よりずっと薄い)を
    使う専用の仮想破壊面であり、応力度は押込み側よりはるかに大きくなる。
    """
    from core.standards import PULL_OUT_RESISTANCE_THICKNESS

    push = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, 2000.0, 0.0, 0.0)
    pull = check_pile_head(1.0, 1.5, 24, LoadCase.PERMANENT, -2000.0, 0.0, 0.0)
    push_tau = push.checks[0]
    pull_tau = pull.checks[0]
    expected_pull_area = punching_shear_area(
        1.0, PULL_OUT_RESISTANCE_THICKNESS, embedment=0.0
    )
    assert pull_tau.stress == pytest.approx(2000.0 / expected_pull_area / 1000.0)
    assert pull_tau.stress > push_tau.stress


# --- 水平支圧応力度・水平方向押抜きせん断(Kui_5 6.2、第43回)---------------


def test_horizontal_bearing_stress_matches_method_a_with_moment():
    """方法A(既設鋼管杭、L=D=0.6m)はモーメント項を含む式を使うこと。"""
    sigma_ch = horizontal_bearing_stress(
        shear=100.3, diameter=0.6, embedment=0.6, moment=90.0
    )
    assert sigma_ch == pytest.approx(2.78, abs=0.01)


def test_horizontal_bearing_stress_matches_method_b_without_moment():
    """方法B(増し杭、L=0.1m)はモーメント項を含まない式を使うこと。

    モーメント抵抗は仮想RC断面が負担するため、水平支圧応力度の算定には
    水平力 PH のみを用いる(モーメントを与えても既定では無視される)。
    """
    sigma_ch = horizontal_bearing_stress(shear=167.1, diameter=1.0, embedment=0.1)
    assert sigma_ch == pytest.approx(1.67, abs=0.01)


def test_check_pile_head_bearing_can_opt_into_moment():
    """``include_moment_in_bearing=True`` で方法Aのモーメント項を有効化できる。"""
    without_moment = check_pile_head(
        pile_diameter=0.6, footing_height=1.2, fck=24, case=LoadCase.LEVEL1_EQ,
        axial=1040.9, shear=100.3, moment=90.0, embedment=0.6,
    )
    with_moment = check_pile_head(
        pile_diameter=0.6, footing_height=1.2, fck=24, case=LoadCase.LEVEL1_EQ,
        axial=1040.9, shear=100.3, moment=90.0, embedment=0.6,
        include_moment_in_bearing=True,
    )
    ch_without = next(c for c in without_moment.checks if "水平支圧" in c.name)
    ch_with = next(c for c in with_moment.checks if "水平支圧" in c.name)
    assert ch_without.stress < ch_with.stress
    assert ch_with.stress == pytest.approx(2.78, abs=0.01)


def test_horizontal_edge_punching_shear_matches_both_connection_methods():
    """既設鋼管杭(方法A)・増し杭(方法B)の両方で τh が一致すること。"""
    tau_a = horizontal_edge_punching_shear(
        shear=100.3, diameter=0.6, embedment=0.6, effective_thickness=2.45
    )
    assert tau_a == pytest.approx(0.006, abs=0.0005)

    tau_b = horizontal_edge_punching_shear(
        shear=167.1, diameter=1.0, embedment=0.1, effective_thickness=2.45
    )
    assert tau_b == pytest.approx(0.011, abs=0.0005)


# --- 負の周面摩擦力 ---------------------------------------------------------


def nf_profile() -> SoilProfile:
    return SoilProfile(
        layers=[
            SoilLayer(
                name="Ac", soil_type=SoilType.CLAY, thickness=10.0, n_value=3.0,
                gamma_t=15.0, gamma_sat=15.5, cohesion=30.0,
            ),
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=20.0, n_value=40.0,
                gamma_t=19.0, gamma_sat=20.0,
            ),
        ],
        gwl=1.0,
    )


def test_negative_friction_intensity_clay_uses_cohesion():
    clay = nf_profile().layers[0]
    assert negative_friction_intensity(clay, 100.0) == 30.0


def test_negative_friction_intensity_sand_uses_effective_stress():
    sand = nf_profile().layers[1]
    assert negative_friction_intensity(sand, 100.0) == pytest.approx(30.0)


def test_negative_friction_neutral_point_at_soft_layer_bottom():
    result = compute_negative_friction(
        CIP, nf_profile(), embedment=1.0, dead_load=1000.0, ru=9000.0
    )
    # 圧密層(Ac, N=3)の下端 10m が中立点
    assert result.neutral_depth == pytest.approx(10.0)
    # NF = π×1.0×(10−1)×30 = 848.2 kN
    assert result.nf == pytest.approx(math.pi * 9.0 * 30.0, rel=1e-6)
    assert result.n_max == pytest.approx(1000.0 + result.nf)
    assert result.allowable == pytest.approx(9000.0 / 1.2)
    assert result.ok


def test_negative_friction_absent_without_soft_layer():
    profile = SoilProfile(
        layers=[
            SoilLayer(
                name="Ds", soil_type=SoilType.SAND, thickness=30.0, n_value=40.0,
                gamma_t=19.0, gamma_sat=20.0,
            )
        ],
        gwl=1.0,
    )
    result = compute_negative_friction(
        CIP, profile, embedment=1.0, dead_load=1000.0, ru=9000.0
    )
    assert result.nf == 0.0
    assert result.n_max == 1000.0


def test_negative_friction_ng_when_excessive():
    result = compute_negative_friction(
        CIP, nf_profile(), embedment=1.0, dead_load=8000.0, ru=5000.0
    )
    assert not result.ok
    assert result.judgement == "NG"


# --- RC杭(中空のひび割れ断面) ---------------------------------------------

RC = PileSpec(
    pile_type=PileType.RC,
    method=ConstructionMethod.PREBORING,
    diameter=0.6,
    length=20.0,
    concrete_thickness=90.0,
)
RC_MATERIAL = MaterialSpec(
    rebar=RebarLayout(count=12, diameter_mm=16.0, cover_mm=40.0)
)


def test_rc_pile_uses_the_precast_table_not_the_input_fck():
    """許容応力度は RC杭の表の値で、MaterialSpec.fck に依存しないこと。"""
    from core.standards import PRECAST_CONCRETE_ALLOWABLE

    allow = PRECAST_CONCRETE_ALLOWABLE["RC杭"]
    for fck in (24, 30, 40):
        material = dataclasses.replace(RC_MATERIAL, fck=fck)
        result = check_section(RC, material, LoadCase.PERMANENT, 0.0, 800.0, 60.0)
        by_name = {c.name: c for c in result.checks}
        assert by_name["コンクリート圧縮応力度"].allowable == pytest.approx(
            allow.bending_compression
        )
        assert by_name["軸圧縮応力度"].allowable == pytest.approx(
            allow.axial_compression
        )


def test_rc_pile_rebar_allowable_takes_the_underwater_value():
    """鉄筋の許容引張応力度は常時に水中の値(小さい側)を用いること。"""
    result = check_section(RC, RC_MATERIAL, LoadCase.PERMANENT, 0.0, 800.0, 300.0)
    by_name = {c.name: c for c in result.checks}
    assert by_name["鉄筋引張応力度"].allowable == pytest.approx(
        SIGMA_SA_REBAR_STATIC["水中又は地下水位以下に設ける部材"]["SD345"]
    )
    # 一般の部材の 180 ではない(安全側を採っている)
    assert by_name["鉄筋引張応力度"].allowable < 180.0
    assert any("水中" in n for n in result.notes)


def test_rc_pile_cracks_under_large_moment():
    small = check_section(RC, RC_MATERIAL, LoadCase.PERMANENT, 0.0, 800.0, 20.0)
    large = check_section(RC, RC_MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 800.0, 300.0)
    assert small.rc_detail.fully_compressed
    assert small.rc_detail.sigma_s_tension == 0.0
    assert not large.rc_detail.fully_compressed
    assert large.rc_detail.sigma_s_tension > 0.0


def test_rc_pile_requires_thickness_and_rebar():
    no_thickness = RC.model_copy(update={"concrete_thickness": None})
    with pytest.raises(ValueError, match="肉厚"):
        check_section(no_thickness, RC_MATERIAL, LoadCase.PERMANENT, 0.0, 800.0, 60.0)
    with pytest.raises(ValueError, match="軸方向鉄筋"):
        check_section(RC, MaterialSpec(), LoadCase.PERMANENT, 0.0, 800.0, 60.0)


def test_rc_pile_seismic_increase_applies_to_every_allowable():
    permanent = check_section(RC, RC_MATERIAL, LoadCase.PERMANENT, 0.0, 800.0, 60.0)
    seismic = check_section(RC, RC_MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 800.0, 60.0)
    increase = STRESS_INCREASE[LoadCase.LEVEL1_EQ.value]
    p = {c.name: c.allowable for c in permanent.checks}
    s = {c.name: c.allowable for c in seismic.checks}
    for name in ("軸圧縮応力度", "コンクリート圧縮応力度"):
        assert s[name] == pytest.approx(p[name] * increase)


# --- SC杭(鋼管 + コンクリートの合成断面) ---------------------------------

SC = PileSpec(
    pile_type=PileType.SC,
    method=ConstructionMethod.PREBORING,
    diameter=0.6,
    length=20.0,
    wall_thickness=9.0,
    concrete_thickness=80.0,
)


def test_sc_pile_checks_both_materials():
    result = check_section(SC, MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0)
    names = [c.name for c in result.checks]
    assert names == [
        "軸圧縮応力度",
        "コンクリート圧縮応力度",
        "鋼管圧縮応力度",
        "鋼管引張応力度",
    ]
    by_name = {c.name: c for c in result.checks}
    assert by_name["鋼管圧縮応力度"].allowable == pytest.approx(
        SIGMA_A_STEEL["SKK400"]
    )


def test_sc_pile_steel_allowable_is_flagged_as_unverified():
    """既定では表-4.4.1 を適用するが、適用根拠が未照合である旨を注記する。"""
    from core.section.checks import SC_STEEL_ALLOWABLE_NOTE

    result = check_section(SC, MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0)
    assert SC_STEEL_ALLOWABLE_NOTE in result.notes
    assert "原典未照合" in SC_STEEL_ALLOWABLE_NOTE


def test_sc_pile_steel_allowable_can_be_given_directly():
    material = MaterialSpec(sc_steel_allowable=120.0)
    result = check_section(SC, material, LoadCase.LEVEL1_EQ, 0.0, 1200.0, 80.0)
    by_name = {c.name: c for c in result.checks}
    increase = STRESS_INCREASE[LoadCase.LEVEL1_EQ.value]
    assert by_name["鋼管圧縮応力度"].allowable == pytest.approx(120.0 * increase)
    assert any("利用者指定" in n for n in result.notes)
    with pytest.raises(ValueError, match="鋼管の許容応力度"):
        check_section(
            SC, MaterialSpec(sc_steel_allowable=0.0),
            LoadCase.PERMANENT, 0.0, 1200.0, 80.0,
        )


def test_sc_pile_steel_carries_tension_that_concrete_does_not():
    """曲げが大きいと鋼管に引張が生じ、コンクリートは引張を負担しないこと。"""
    result = check_section(SC, MaterialSpec(), LoadCase.LEVEL1_EQ, 0.0, 1400.0, 350.0)
    assert not result.rc_detail.fully_compressed
    assert result.rc_detail.sigma_s_tension > 0.0
    # コンクリートの照査項目に引張は現れない(ひび割れ断面として扱うため)
    assert not any("引張" in c.name and "鋼管" not in c.name for c in result.checks)


def test_sc_pile_deducts_the_corrosion_allowance():
    """腐食代を控除した板厚で解いているので、控除量を増やすと応力度が上がる。"""
    thin = check_section(
        SC, MaterialSpec(corrosion_mm=3.0), LoadCase.PERMANENT, 0.0, 1200.0, 200.0
    )
    thick = check_section(
        SC, MaterialSpec(corrosion_mm=0.0), LoadCase.PERMANENT, 0.0, 1200.0, 200.0
    )
    assert thin.rc_detail.sigma_c > thick.rc_detail.sigma_c
    assert any("腐食代 3 mm" in n for n in thin.notes)


def test_sc_pile_requires_both_thicknesses():
    with pytest.raises(ValueError, match="板厚"):
        check_section(
            SC.model_copy(update={"wall_thickness": None}),
            MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0,
        )
    with pytest.raises(ValueError, match="肉厚"):
        check_section(
            SC.model_copy(update={"concrete_thickness": None}),
            MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0,
        )
    # 肉厚が鋼管内径に対して大きすぎる
    with pytest.raises(ValueError, match="中空断面"):
        check_section(
            SC.model_copy(update={"concrete_thickness": 300.0}),
            MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0,
        )


def test_sc_pile_uses_the_h24_concrete_young_modulus():
    """Ec は SC杭に定められた 3.5×10⁴ N/mm²(H24版)であること。"""
    from core.standards import EC_SC_PILE_CONCRETE

    result = check_section(SC, MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0)
    assert any(f"{EC_SC_PILE_CONCRETE / 1000.0:,.0f} N/mm²" in n for n in result.notes)
    # concrete_young を与えるとそちらが優先される
    override = check_section(
        SC.model_copy(update={"concrete_young": 4.0e7}),
        MaterialSpec(), LoadCase.PERMANENT, 0.0, 1200.0, 80.0,
    )
    assert any("40,000 N/mm²" in n for n in override.notes)


# --- Forum8 のサンプル出力との突合(第33回) ---------------------------------
#
# 出典: フォーラムエイト UC-1「基礎の設計」設計計算書サンプル kiso-Kui_1
# (場所打ち杭 φ1200・L=25m・12本、σck=24、SD345、n=15)の 1章。
# **他社製品の出力(二次資料)**であり道示の原典ではないが、独立した実装の
# 許容応力度表なので、本ソフトの値と突き合わせる価値がある。


def test_allowable_stress_table_matches_the_forum8_sample():
    """サンプルの許容応力度表(σck=24、SD345)と一致すること。

    | 割増 | σca | τa1 | τa2 | σsa(引張) |
    |---|---|---|---|---|
    | 1.00 | 8.00 | 0.230 | 1.700 | 160.00 |
    | 1.50 | 12.00 | 0.350 | 2.550 | 300.00 |

    とくに σsa は、常時 160(水中又は地下水位以下)→ 地震時 300 であり、
    **160 × 1.5 = 240 ではない**。地震時に基本値が 200(軸方向鉄筋)へ
    切り替わってから割増を乗じる扱いが独立に裏付けられた。
    """
    from core.standards import (
        STRESS_INCREASE,
        UNDERWATER_CONCRETE_ALLOWABLE,
        YOUNG_MODULUS_RATIO_RC,
    )
    from core.section.checks import rebar_tension_allowable

    allow = UNDERWATER_CONCRETE_ALLOWABLE[24]
    normal = STRESS_INCREASE["常時"]
    seismic = STRESS_INCREASE["レベル1地震時"]

    assert YOUNG_MODULUS_RATIO_RC == 15.0
    assert allow.bending_compression * normal == pytest.approx(8.00)
    assert allow.bending_compression * seismic == pytest.approx(12.00)
    assert allow.tau_a1 * normal == pytest.approx(0.230)
    assert allow.tau_a2 * normal == pytest.approx(1.700)
    assert allow.tau_a2 * seismic == pytest.approx(2.550)
    # τa1 の割増後は 0.23×1.5 = 0.345。サンプルの表示は 0.350(丸め)で、
    # 本ソフトのほうが小さい = 安全側なので、この差は追随しない
    assert allow.tau_a1 * seismic == pytest.approx(0.345)
    assert allow.tau_a1 * seismic < 0.350

    assert rebar_tension_allowable(
        "SD345", LoadCase.PERMANENT, underwater=True, increase=normal
    ) == pytest.approx(160.00)
    assert rebar_tension_allowable(
        "SD345", LoadCase.LEVEL1_EQ, underwater=True, increase=seismic
    ) == pytest.approx(300.00)


def test_allowable_horizontal_displacement_matches_the_forum8_sample():
    """杭径 1200mm では許容水平変位が常時・地震時とも 15mm であること。"""
    from core.standards import (
        ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD,
        ALLOWABLE_DISPLACEMENT_MM,
        ALLOWABLE_DISPLACEMENT_RATIO,
    )

    diameter = 1.2
    assert diameter < ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD
    assert ALLOWABLE_DISPLACEMENT_MM == pytest.approx(15.0)
    # 1.5m 以上なら杭径の 1%(φ1200 では 12mm となり 15mm より小さい)
    assert diameter * ALLOWABLE_DISPLACEMENT_RATIO * 1000 == pytest.approx(12.0)


def test_rebar_compression_is_checked_against_the_forum8_value():
    """鉄筋の許容曲げ圧縮応力度 σsa' = 200(割増1.50 で 300)を照査すること。

    サンプルの表に σsa' があるのに本ソフトが持っていなかったため、第33回に
    追加した。**値は他社製品の出力から読み取ったもので原典未照合**であり、
    その旨を注記に出す。
    """
    from core.section.checks import REBAR_COMPRESSION_NOTE
    from core.standards import SIGMA_SA_REBAR_COMPRESSION

    assert SIGMA_SA_REBAR_COMPRESSION == 200.0
    normal = check_section(CIP, MATERIAL, LoadCase.PERMANENT, 0.0, 1500.0, 800.0)
    seismic = check_section(CIP, MATERIAL, LoadCase.LEVEL1_EQ, 0.0, 1500.0, 800.0)
    by_name = {c.name: c for c in normal.checks}
    assert by_name["鉄筋圧縮応力度"].allowable == pytest.approx(200.0)
    assert by_name["鉄筋圧縮応力度"].stress == pytest.approx(
        normal.rc_detail.sigma_s_compression
    )
    assert {c.name: c.allowable for c in seismic.checks}["鉄筋圧縮応力度"] == (
        pytest.approx(300.0)
    )
    assert REBAR_COMPRESSION_NOTE in normal.notes
    assert "原典は未照合" in REBAR_COMPRESSION_NOTE

    # 「鉄筋」で最初に引っかかるのは引張側のままであること(既存テストの前提)
    assert next(c for c in normal.checks if "鉄筋" in c.name).name == "鉄筋引張応力度"


def test_rebar_compression_does_not_govern_for_cast_in_place_piles():
    """場所打ち杭ではコンクリート圧縮が必ず先に支配すること。

    平面保持より鉄筋位置の応力度は σs = n・σc(鉄筋位置)で、鉄筋はかぶりの
    内側にあるから σs < n・σc(圧縮縁)。したがって

        σs < n・σca = 15 × 8.0 = 120 < 200   (σck = 24)

    となる。割増係数は σca・σsa' の双方に同じく効くので荷重の組合せによらない。
    **RC杭では n・σca = 15 × 13.5 = 202.5 で 200 に接する**ので、この包含は
    成り立たない(だからこそ照査項目として持つ必要がある)。
    """
    from core.standards import UNDERWATER_CONCRETE_ALLOWABLE, YOUNG_MODULUS_RATIO_RC
    from core.standards import PRECAST_CONCRETE_ALLOWABLE, SIGMA_SA_REBAR_COMPRESSION

    n = YOUNG_MODULUS_RATIO_RC
    assert n * UNDERWATER_CONCRETE_ALLOWABLE[24].bending_compression == 120.0
    assert 120.0 < SIGMA_SA_REBAR_COMPRESSION
    # RC杭は接する
    assert n * PRECAST_CONCRETE_ALLOWABLE["RC杭"].bending_compression == 202.5
    assert 202.5 > SIGMA_SA_REBAR_COMPRESSION

    pile = PileSpec(
        pile_type=PileType.CAST_IN_PLACE,
        method=ConstructionMethod.CAST_IN_PLACE,
        diameter=1.2,
        length=25.0,
    )
    material = MaterialSpec(
        fck=24, rebar=RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)
    )
    for axial, moment in ((8600.0, 200.0), (5000.0, 1500.0), (2000.0, 2500.0)):
        result = check_section(
            pile, material, LoadCase.PERMANENT, 0.0, axial, moment
        )
        detail = result.rc_detail
        assert detail.sigma_s_compression < n * detail.sigma_c
        by_name = {c.name: c for c in result.checks}
        assert by_name["鉄筋圧縮応力度"].ratio < by_name["コンクリート圧縮応力度"].ratio
