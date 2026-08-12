"""基準定数のピン止めテスト。

`core.standards` の定数は道示H24に基づく設計判断そのものであり、
うっかり変更すると全計算結果が静かに変わる。本テストは現在の値を
明示的に固定し、変更が「意図的な照合結果の反映」であることを
コードレビューで確認できるようにするためのもの。

**定数を修正する場合は、本テストの期待値も同時に更新し、
`docs/VERIFICATION.md` の照合欄に出典を記録すること。**
"""
import pytest

from core import standards as st


def test_safety_factors_push():
    """押込みの安全率(道示Ⅳ 表-12.4.1)。支持杭・摩擦杭で異なる。"""
    assert st.SAFETY_FACTORS_PUSH == {
        "常時": {"支持杭": 3.0, "摩擦杭": 4.0},
        "暴風時": {"支持杭": 2.0, "摩擦杭": 3.0},
        "レベル1地震時": {"支持杭": 2.0, "摩擦杭": 3.0},
    }


def test_safety_factors_pull():
    """引抜きの安全率(道示Ⅳ 表-12.4.3)。押込み(支持杭)の2倍。"""
    assert st.SAFETY_FACTORS_PULL == {
        "常時": 6.0,
        "暴風時": 3.0,
        "レベル1地震時": 3.0,
    }
    # 引抜きは押込みより厳しい安全率を用いる
    for case, n_pull in st.SAFETY_FACTORS_PULL.items():
        for n_push in st.SAFETY_FACTORS_PUSH[case].values():
            assert n_pull >= n_push


def test_friction_pile_is_stricter_than_end_bearing():
    for case, factors in st.SAFETY_FACTORS_PUSH.items():
        assert factors["摩擦杭"] > factors["支持杭"], case


def test_kv_coefficients():
    """Kv の係数 a = slope・(L/D) + intercept(工法別)。

    公開資料で裏付けが取れているもの(docs/VERIFICATION.md 参照):
      場所打ち        a = 0.031(L/D) − 0.15
      打込み(打撃)    a = 0.014(L/D) + 0.720
      バイブロハンマ  a = 0.017(L/D) − 0.014
    """
    assert st.KV_A_COEF["場所打ち"] == (0.031, -0.15)
    assert st.KV_A_COEF["打込み(打撃)"] == (0.014, 0.72)
    assert st.KV_A_COEF["バイブロハンマ"] == (0.017, -0.014)
    # 未照合の工法
    assert st.KV_A_COEF["中掘り"] == (0.010, 0.36)
    assert st.KV_A_COEF["プレボーリング"] == (0.013, 0.53)
    assert st.KV_A_COEF["鋼管ソイルセメント"] == (0.040, 0.15)
    assert st.KV_A_COEF["回転"] == (0.013, 0.54)


def test_all_methods_have_capacity_specs():
    """全工法が qd・f の定義を持つこと(定義漏れの検出)。"""
    methods = set(st.KV_A_COEF)
    assert set(st.QD_SPECS) == methods
    assert set(st.F_SPECS) == methods
    assert set(st.F_MAX) == methods
    for method in methods:
        # 周面摩擦力度は3土質すべてに定義が必要
        assert set(st.F_SPECS[method]) == {"砂質土", "礫質土", "粘性土"}
        assert set(st.F_MAX[method]) == {"砂質土", "礫質土", "粘性土"}
        # 先端支持力度は支持層になり得る土質のみ
        assert st.QD_SPECS[method]
        for spec in st.QD_SPECS[method].values():
            assert spec.kind in {"N", "qu", "steps"}
            if spec.kind == "steps":
                # 閾値は降順に並んでいること(高い区分から判定するため)
                thresholds = [t for t, _ in spec.steps]
                assert thresholds == sorted(thresholds, reverse=True)
            else:
                assert spec.coef > 0


def test_qd_specs_driven_and_cast_in_place():
    assert st.QD_SPECS["打込み(打撃)"]["砂質土"] == st.QdSpec(
        "N", coef=130.0, cap=6500.0
    )
    # 場所打ち杭: 砂層 N≧30 で 3,000、良質な砂れき層 N≧50 で 5,000
    assert st.QD_SPECS["場所打ち"]["砂質土"].steps == ((30.0, 3000.0),)
    assert st.QD_SPECS["場所打ち"]["礫質土"].steps == (
        (50.0, 5000.0),
        (30.0, 3000.0),
    )
    assert st.QD_SPECS["場所打ち"]["粘性土"] == st.QdSpec("qu", coef=3.0, cap=3000.0)
    # 場所打ち杭は砂質土・礫質土・粘性土を支持層にできる
    assert set(st.QD_SPECS["場所打ち"]) == {"砂質土", "礫質土", "粘性土"}
    # 打込み杭は粘性土を支持層としない
    assert "粘性土" not in st.QD_SPECS["打込み(打撃)"]


def test_qd_specs_embedded_methods():
    """中掘り(セメントミルク)・プレボーリング・鋼管ソイルセメントは同値。

    砂層 150N(≦7,500)、砂れき層 200N(≦10,000)。
    """
    for method in ("中掘り", "プレボーリング", "鋼管ソイルセメント"):
        assert st.QD_SPECS[method]["砂質土"] == st.QdSpec(
            "N", coef=150.0, cap=7500.0
        ), method
        assert st.QD_SPECS[method]["礫質土"] == st.QdSpec(
            "N", coef=200.0, cap=10000.0
        ), method


def test_qd_specs_rotary_by_wing_ratio():
    """回転杭は羽根外径比(1.5 / 2.0)により qd が異なる。"""
    assert st.QD_SPECS_ROTARY == {
        1.5: {
            "砂質土": st.QdSpec("N", coef=120.0, cap=6000.0),
            "礫質土": st.QdSpec("N", coef=130.0, cap=6500.0),
        },
        2.0: {
            "砂質土": st.QdSpec("N", coef=100.0, cap=5000.0),
            "礫質土": st.QdSpec("N", coef=115.0, cap=5750.0),
        },
    }
    assert st.DEFAULT_WING_RATIO == 1.5
    assert st.DEFAULT_WING_RATIO in st.QD_SPECS_ROTARY
    # 羽根が大きいほど qd は小さい(単位面積あたりの支持力度)
    for soil in ("砂質土", "礫質土"):
        assert (
            st.QD_SPECS_ROTARY[2.0][soil].coef < st.QD_SPECS_ROTARY[1.5][soil].coef
        )
    # QD_SPECS["回転"] は既定(1.5倍)と一致していること
    assert st.QD_SPECS["回転"] == st.QD_SPECS_ROTARY[st.DEFAULT_WING_RATIO]


def test_tip_treatment_sources():
    """先端処理方式ごとに qd の算定に用いる工法。"""
    assert st.TIP_TREATMENT_QD_SOURCE == {
        st.TipTreatment.FINAL_DRIVING: "打込み(打撃)",
        st.TipTreatment.CEMENT_MILK: "中掘り",
        st.TipTreatment.CONCRETE: "場所打ち",
    }
    assert set(st.TIP_TREATMENT_QD_SOURCE) == set(st.TipTreatment)
    # 参照先はすべて QD_SPECS に定義されていること
    for source in st.TIP_TREATMENT_QD_SOURCE.values():
        assert source in st.QD_SPECS


def test_f_specs_pinned():
    """打込み・場所打ちの式形は複数ソースで確認済み。

    中掘り・プレボーリングは 2026-08-11 に 10N/0.8c → 3N/1.0c へ変更
    (独立3ソースが一致、旧値は支持力を過大評価していた)。
    """
    assert st.F_SPECS["打込み(打撃)"]["砂質土"] == (2.0, "N")
    assert st.F_MAX["打込み(打撃)"]["砂質土"] == 100.0
    assert st.F_SPECS["打込み(打撃)"]["粘性土"] == (1.0, "c")
    assert st.F_MAX["打込み(打撃)"]["粘性土"] == 150.0

    # 場所打ち杭は提供解説資料と完全一致(砂 5N≦200、粘 c または 10N≦150)
    assert st.F_SPECS["場所打ち"]["砂質土"] == (5.0, "N")
    assert st.F_MAX["場所打ち"]["砂質土"] == 200.0
    assert st.F_SPECS["場所打ち"]["粘性土"] == (1.0, "c")
    assert st.F_MAX["場所打ち"]["粘性土"] == 150.0
    # 上限に達する N 値(砂質土: 5N=200 → N=40)
    assert st.F_MAX["場所打ち"]["砂質土"] / st.F_SPECS["場所打ち"]["砂質土"][0] == 40.0
    assert st.MIN_N_FOR_CLAY_FRICTION_FROM_N == 5.0

    for method in ("中掘り", "プレボーリング"):
        assert st.F_SPECS[method]["砂質土"] == (3.0, "N")
        assert st.F_MAX[method]["砂質土"] == 150.0
        assert st.F_SPECS[method]["粘性土"] == (1.0, "c")
        assert st.F_MAX[method]["粘性土"] == 100.0

    # 鋼管ソイルセメントは未照合のため旧値を据え置き
    assert st.F_SPECS["鋼管ソイルセメント"]["砂質土"] == (10.0, "N")
    assert st.F_MAX["鋼管ソイルセメント"]["砂質土"] == 200.0


def test_cement_methods_are_not_shared_with_soil_cement():
    """中掘り系と鋼管ソイルセメントは別テーブル(照合状況が異なるため)。"""
    assert st.F_SPECS["中掘り"] is not st.F_SPECS["鋼管ソイルセメント"]
    assert st.F_SPECS["中掘り"] is st.F_SPECS["プレボーリング"]


def test_liquefaction_constants():
    assert st.KHG0_LIQUEFACTION[st.GroundMotionType.LEVEL2_TYPE1] == {
        st.GroundType.TYPE_I: 0.50,
        st.GroundType.TYPE_II: 0.45,
        st.GroundType.TYPE_III: 0.40,
    }
    assert st.KHG0_LIQUEFACTION[st.GroundMotionType.LEVEL2_TYPE2] == {
        st.GroundType.TYPE_I: 0.80,
        st.GroundType.TYPE_II: 0.70,
        st.GroundType.TYPE_III: 0.60,
    }
    assert st.LIQUEFACTION_MAX_DEPTH == 20.0
    assert st.LIQUEFACTION_MAX_GWL == 10.0
    assert st.LIQUEFACTION_MAX_FC == 35.0
    assert st.LIQUEFACTION_MAX_IP == 15.0


def test_de_table_is_complete_and_monotonic():
    """DE 表は 12 通り(FL3区分×深度2区分×R2区分)すべてが定義されること。"""
    assert len(st.DE_TABLE) == 12
    for fl_idx in range(3):
        for depth_idx in range(2):
            for r_idx in range(2):
                assert (fl_idx, depth_idx, r_idx) in st.DE_TABLE
    # FL が大きいほど(液状化しにくいほど)低減が緩む
    for depth_idx in range(2):
        for r_idx in range(2):
            values = [st.DE_TABLE[(i, depth_idx, r_idx)] for i in range(3)]
            assert values == sorted(values)
    # 深いほど、R が大きいほど低減が緩む
    for fl_idx in range(3):
        for r_idx in range(2):
            assert (
                st.DE_TABLE[(fl_idx, 0, r_idx)] <= st.DE_TABLE[(fl_idx, 1, r_idx)]
            )
        for depth_idx in range(2):
            assert (
                st.DE_TABLE[(fl_idx, depth_idx, 0)]
                <= st.DE_TABLE[(fl_idx, depth_idx, 1)]
            )
    # すべて 0〜1 の範囲
    assert all(0.0 <= v <= 1.0 for v in st.DE_TABLE.values())


def test_de_table_values_match_reference():
    """DE の取り得る値と配置が提供解説資料(道示Ⅴ 表-8.2.4)と一致すること。

    資料は R の区分を持たないため、最も液状化しやすい区分を
    「0(または 1/6)」と併記している。下記はその R 区分を展開した形。
    """
    assert set(st.DE_TABLE.values()) == {0.0, 1 / 6, 1 / 3, 2 / 3, 1.0}
    # FL 最小・浅い: 0(R≦0.3)/ 1/6(R>0.3)
    assert st.DE_TABLE[(0, 0, 0)] == 0.0
    assert st.DE_TABLE[(0, 0, 1)] == pytest.approx(1 / 6)
    # FL 最小・深い: 1/3
    assert st.DE_TABLE[(0, 1, 0)] == pytest.approx(1 / 3)
    # 中間区分・浅い: 1/3、深い: 2/3
    assert st.DE_TABLE[(1, 0, 0)] == pytest.approx(1 / 3)
    assert st.DE_TABLE[(1, 1, 0)] == pytest.approx(2 / 3)
    # FL 最大区分・浅い: 2/3、深い: 1.0
    assert st.DE_TABLE[(2, 0, 0)] == pytest.approx(2 / 3)
    assert st.DE_TABLE[(2, 1, 0)] == 1.0


def test_kh_constants():
    assert st.E0_FROM_N == 2800.0
    assert st.ALLOWABLE_DISPLACEMENT_MM == 15.0
    assert st.ALLOWABLE_DISPLACEMENT_RATIO == 0.01
    assert st.ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD == 1.5


def test_alpha_kh_by_e0_method():
    """α は E0 の推定方法により決まる(道示Ⅳ 9.5.2)。

    N値・平板載荷試験: 常時1 / 地震時2
    孔内水平載荷試験・室内試験: 常時4 / 地震時8
    """
    assert st.ALPHA_KH == {
        st.E0Method.N_VALUE: (1.0, 2.0),
        st.E0Method.PLATE_LOADING: (1.0, 2.0),
        st.E0Method.BOREHOLE_LATERAL: (4.0, 8.0),
        st.E0Method.LAB_COMPRESSION: (4.0, 8.0),
    }
    # すべての推定方法が定義されていること
    assert set(st.ALPHA_KH) == set(st.E0Method)
    # 地震時は常時のちょうど2倍
    for normal, seismic in st.ALPHA_KH.values():
        assert seismic == pytest.approx(2.0 * normal)


def test_stress_increase_factors():
    """割増係数(道示Ⅰ 荷重の組合せ)。

    2026-08-11 に暴風時を 1.5 → 1.25 へ修正
    (風荷重 1.25 / 地震の影響 1.50 で2ソースが一致)。
    """
    assert st.STRESS_INCREASE == {"常時": 1.0, "暴風時": 1.25, "レベル1地震時": 1.50}


def test_concrete_allowable_stresses():
    """道示Ⅲ 表-3.2.1。σck=21〜30 は提供解説資料と完全一致。"""
    assert st.SIGMA_CA_CONCRETE == {21: 7.0, 24: 8.0, 27: 9.0, 30: 10.0, 40: 13.0}
    assert st.SIGMA_CAG_CONCRETE == {21: 5.5, 24: 6.5, 27: 7.0, 30: 8.0, 40: 10.0}
    assert st.TAU_A1_CONCRETE == {21: 0.35, 24: 0.38, 27: 0.40, 30: 0.42, 40: 0.50}
    assert st.TAU_A2_CONCRETE == {21: 1.6, 24: 1.7, 27: 1.8, 30: 1.9, 40: 2.2}
    # 軸圧縮は曲げ圧縮より小さい
    for grade in st.SIGMA_CA_CONCRETE:
        assert st.SIGMA_CAG_CONCRETE[grade] < st.SIGMA_CA_CONCRETE[grade]


def test_rebar_allowable_stresses():
    """道示Ⅳ 表4.3.1。SD345=180 は2ソースで確認済み。

    SD295・SR235 は H24 改定で下部構造編の表から削除されたため、
    許容引張応力度が規定されておらず本表には含めない。
    """
    assert st.SIGMA_SA_REBAR == {"SD345": 180.0, "SD390": 200.0}
    assert st.SIGMA_SA_REBAR_SEVERE == {"SD345": 160.0, "SD390": 180.0}
    # 腐食性環境の許容値は一般の部材以下
    for grade in st.SIGMA_SA_REBAR:
        assert st.SIGMA_SA_REBAR_SEVERE[grade] <= st.SIGMA_SA_REBAR[grade]
    # 削除された材質は表に含まれない
    assert "SD295" in st.REMOVED_REBAR_GRADES
    assert "SR235" in st.REMOVED_REBAR_GRADES
    assert not (st.REMOVED_REBAR_GRADES & set(st.SIGMA_SA_REBAR))


def test_other_allowable_stresses():
    assert st.CIP_CONCRETE_REDUCTION == 0.8
    assert st.SIGMA_A_STEEL["SKK400"] == 140.0
    assert st.SIGMA_A_STEEL["SKK490"] == 185.0
    assert st.TAU_A_PUNCHING[24] == 0.90
    assert st.NF_SAFETY_FACTOR == 1.2


def test_material_constants():
    assert st.E_STEEL == 2.0e8
    assert st.E_REBAR == 2.0e8
    assert st.EC_CONCRETE[24] == 2.5e7
    assert st.GAMMA_W == 9.8


def test_concrete_tables_share_grades():
    """コンクリート関連の表は同じ σck を網羅していること。"""
    grades = set(st.EC_CONCRETE)
    for table in (
        st.SIGMA_CA_CONCRETE,
        st.SIGMA_CAG_CONCRETE,
        st.TAU_A1_CONCRETE,
        st.TAU_A2_CONCRETE,
    ):
        assert set(table) == grades


def test_punching_shear_table_covers_21_to_30_only():
    """τa3(道示Ⅳ 表4.2.1)は σck = 21〜30 のみを規定する。"""
    assert st.TAU_A_PUNCHING == {21: 0.85, 24: 0.90, 27: 0.95, 30: 1.00}
    # 40 は表の範囲外(別途、設計条件・発注者基準の確認が必要)
    assert 40 not in st.TAU_A_PUNCHING
    assert set(st.TAU_A_PUNCHING) < set(st.EC_CONCRETE)


def test_rebar_tables_share_grades():
    assert set(st.SIGMA_SA_REBAR) == set(st.SIGMA_SA_REBAR_SEVERE)


def test_stress_increase_matches_safety_factor_cases():
    """許容応力度の割増と支持力の安全率は同じ荷重ケースを扱うこと。"""
    assert set(st.STRESS_INCREASE) == set(st.SAFETY_FACTORS_PUSH)
    assert set(st.STRESS_INCREASE) == set(st.SAFETY_FACTORS_PULL)


def test_precast_concrete_allowable_pinned():
    """既製コンクリート杭の許容応力度(提供解説資料により照合済み)。"""
    expected = {
        # 杭種: (σck, 曲げ圧縮, 軸圧縮, せん断, 曲げ引張)
        "RC杭": (40.0, 13.5, 11.5, 0.36, None),
        "PHC杭": (80.0, 27.0, 23.0, 0.85, 0.0),
        "SC杭": (80.0, 27.0, 23.0, 0.85, None),
    }
    assert set(st.PRECAST_CONCRETE_ALLOWABLE) == set(expected)
    for name, (fck, bend, axial, shear, tension) in expected.items():
        a = st.PRECAST_CONCRETE_ALLOWABLE[name]
        assert (a.fck, a.bending_compression, a.axial_compression, a.shear) == (
            fck, bend, axial, shear
        )
        assert a.bending_tension == tension
    # 軸圧縮は曲げ圧縮より小さい
    for a in st.PRECAST_CONCRETE_ALLOWABLE.values():
        assert a.axial_compression < a.bending_compression


def test_phc_bending_tension_table_pinned():
    """PHC杭の地震時の許容曲げ引張応力度は σce の降順に並ぶこと。"""
    assert st.PHC_BENDING_TENSION_BY_PRESTRESS == ((7.8, 5.0), (3.9, 3.0))
    thresholds = [t for t, _ in st.PHC_BENDING_TENSION_BY_PRESTRESS]
    assert thresholds == sorted(thresholds, reverse=True)


def test_h_steel_reference_allowable_is_not_used_for_checks():
    """H形鋼杭の参考値は照査に用いていないこと(資料の警告による)。"""
    from core.section.checks import UNIMPLEMENTED_STRESS_CHECK
    from core.models.pile import PileType

    assert st.H_STEEL_REFERENCE_ALLOWABLE == {
        "SS400相当": (140.0, 210.0),
        "SM490相当": (185.0, 277.0),
    }
    assert PileType.H_STEEL in UNIMPLEMENTED_STRESS_CHECK
