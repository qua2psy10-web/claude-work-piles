"""基準定数のピン止めテスト。

`core.standards` の定数は道示H24に基づく設計判断そのものであり、
うっかり変更すると全計算結果が静かに変わる。本テストは現在の値を
明示的に固定し、変更が「意図的な照合結果の反映」であることを
コードレビューで確認できるようにするためのもの。

**定数を修正する場合は、本テストの期待値も同時に更新し、
`docs/VERIFICATION.md` の照合欄に出典を記録すること。**
"""
import inspect
from pathlib import Path

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
    """道示Ⅳ(H24) 表-4.2.1。**原典(スキャン)で照合済み**(第23回)。

    表は σck = 21〜30 のみを規定する。σck = 40 の行は原典に存在しないため
    削除した。τa1 は 0.35/0.38/0.40/0.42(約1.5倍、非安全側)を
    0.22/0.23/0.24/0.25 に、σcag は 27/30 を 7.0/8.0 → 7.5/8.5 に修正した。
    """
    assert st.SIGMA_CA_CONCRETE == {21: 7.0, 24: 8.0, 27: 9.0, 30: 10.0}
    assert st.SIGMA_CAG_CONCRETE == {21: 5.5, 24: 6.5, 27: 7.5, 30: 8.5}
    assert st.TAU_A1_CONCRETE == {21: 0.22, 24: 0.23, 27: 0.24, 30: 0.25}
    assert st.TAU_A2_CONCRETE == {21: 1.6, 24: 1.7, 27: 1.8, 30: 1.9}
    assert st.TAU_C_CONCRETE == {21: 0.33, 24: 0.35, 27: 0.36, 30: 0.37}
    # 軸圧縮は曲げ圧縮より小さい
    for grade in st.SIGMA_CA_CONCRETE:
        assert st.SIGMA_CAG_CONCRETE[grade] < st.SIGMA_CA_CONCRETE[grade]
    # 40 は表外(解説に「21〜30 の範囲について規定している」と明記)
    assert 40 not in st.SIGMA_CA_CONCRETE


def test_tau_a1_equals_tau_c_divided_by_the_safety_factor_1_5():
    """τa1 = τc / 1.5(原典 4.2 の解説)。2つの表の整合を固定する。"""
    for grade, tau_c in st.TAU_C_CONCRETE.items():
        assert st.TAU_A1_CONCRETE[grade] == pytest.approx(tau_c / 1.5, abs=0.005)


def test_underwater_concrete_is_not_a_0_8_reduction():
    """水中施工は表-4.2.5 で扱う。0.8 倍の低減は道示Ⅳ に存在しない。

    表-4.2.5 の許容応力度は、同じ**設計基準強度**に対する表-4.2.1 の値と
    一致する。水中施工は「呼び強度 → 水中コンクリートの設計基準強度」の
    読替え(30→24、36→27、40→30)で考慮されている。低減されるのは
    付着応力度のみである。
    """
    assert not hasattr(st, "CIP_CONCRETE_REDUCTION")
    assert sorted(st.UNDERWATER_CONCRETE_ALLOWABLE) == [24, 27, 30]
    for fck, uw in st.UNDERWATER_CONCRETE_ALLOWABLE.items():
        assert uw.bending_compression == st.SIGMA_CA_CONCRETE[fck]
        assert uw.axial_compression == st.SIGMA_CAG_CONCRETE[fck]
        assert uw.tau_a1 == st.TAU_A1_CONCRETE[fck]
        assert uw.tau_a2 == st.TAU_A2_CONCRETE[fck]
        # 付着応力度だけは大気中より小さい
        assert uw.bond < st.BOND_ALLOWABLE_CONCRETE[fck]
    # 呼び強度との対応
    assert [uw.nominal_strength for uw in st.UNDERWATER_CONCRETE_ALLOWABLE.values()] == [
        30, 36, 40
    ]


def test_shear_correction_factor_tables():
    """道示Ⅳ 表-4.2.2(ce)・表-4.2.3(cpt)、式(4.2.1) の cN。"""
    assert st.SHEAR_CE_BY_DEPTH == (
        (300.0, 1.4), (1000.0, 1.0), (3000.0, 0.7), (5000.0, 0.6), (10000.0, 0.5)
    )
    assert st.SHEAR_CPT_BY_RATIO == (
        (0.1, 0.7), (0.2, 0.9), (0.3, 1.0), (0.5, 1.2), (1.0, 1.5)
    )
    assert (st.SHEAR_CN_MIN, st.SHEAR_CN_MAX) == (1.0, 2.0)
    assert st.SHEAR_REBAR_YIELD_CAP == 345.0
    # ce は有効高が大きいほど小さく、cpt は鉄筋比が大きいほど大きい
    assert [c for _, c in st.SHEAR_CE_BY_DEPTH] == sorted(
        (c for _, c in st.SHEAR_CE_BY_DEPTH), reverse=True
    )
    assert [c for _, c in st.SHEAR_CPT_BY_RATIO] == sorted(
        c for _, c in st.SHEAR_CPT_BY_RATIO
    )


def test_rebar_allowable_stresses():
    """道示Ⅳ(H24) 表-4.3.1。**原典(スキャン)で照合済み**(第23回)。

    表は荷重の組合せの区分ごとに基本値を与える。従来は区分を持たず
    SD345=180 / SD390=200 の1本だけで、SD390 の常時(正しくは 180)と
    水中部材の 160 を扱えていなかった(いずれも非安全側)。
    """
    assert st.SIGMA_SA_REBAR_STATIC == {
        "一般の部材": {"SD345": 180.0, "SD390": 180.0, "SD490": 180.0},
        "水中又は地下水位以下に設ける部材": {
            "SD345": 160.0, "SD390": 160.0, "SD490": 160.0
        },
    }
    assert st.SIGMA_SA_REBAR_SEISMIC == {
        "軸方向鉄筋": {"SD345": 200.0, "SD390": 230.0, "SD490": 290.0},
        "上記以外": {"SD345": 200.0, "SD390": 200.0, "SD490": 200.0},
    }
    assert st.SIGMA_CA_REBAR == {"SD345": 200.0, "SD390": 230.0, "SD490": 290.0}
    # 水中部材は一般の部材以下
    for grade in st.REBAR_GRADES:
        assert (
            st.SIGMA_SA_REBAR_STATIC["水中又は地下水位以下に設ける部材"][grade]
            <= st.SIGMA_SA_REBAR_STATIC["一般の部材"][grade]
        )
    # 削除された材質は表に含まれない
    assert "SD295" in st.REMOVED_REBAR_GRADES
    assert "SR235" in st.REMOVED_REBAR_GRADES
    assert not (st.REMOVED_REBAR_GRADES & set(st.REBAR_GRADES))
    assert st.HIGH_GRADE_REBAR_MIN_FCK == 30


def test_other_allowable_stresses():
    assert st.SIGMA_A_STEEL["SKK400"] == 140.0
    assert st.SIGMA_A_STEEL["SKK490"] == 185.0
    assert st.TAU_A_STEEL == {"SKK400": 80.0, "SKK490": 105.0}
    assert st.STEEL_ALLOWABLE_MAX_THICKNESS == 40.0
    assert st.TAU_A_PUNCHING[24] == 0.90
    assert st.NF_SAFETY_FACTOR == 1.2


def test_material_constants():
    assert st.E_STEEL == 2.0e8
    assert st.E_REBAR == 2.0e8
    assert st.EC_CONCRETE[24] == 2.5e7
    assert st.GAMMA_W == 9.8


def test_allowable_stress_tables_share_grades_and_have_a_young_modulus():
    """許容応力度の表どうしは同じ σck を網羅し、いずれも Ec を持つこと。

    Ec の表(道示Ⅲ 表-3.3.3)は σck = 21〜60 を規定するが、下部構造の
    許容応力度(道示Ⅳ)は 21〜40 までである。したがって両者は一致せず、
    許容応力度の表が Ec の表の**部分集合**であることを要件とする。
    """
    tables = (
        st.SIGMA_CA_CONCRETE,
        st.SIGMA_CAG_CONCRETE,
        st.TAU_A1_CONCRETE,
        st.TAU_A2_CONCRETE,
    )
    grades = set(tables[0])
    for table in tables:
        assert set(table) == grades
    assert grades < set(st.EC_CONCRETE)


def test_young_modulus_table_matches_doushi_iii_table_3_3_3():
    """道示Ⅲ 表-3.3.3 の全範囲。σck = 80 は表になく、持たないこと。"""
    assert st.EC_CONCRETE == {
        21: 2.35e7,
        24: 2.50e7,
        27: 2.65e7,
        30: 2.80e7,
        40: 3.10e7,
        50: 3.30e7,
        60: 3.50e7,
    }
    # 既製杭(PHC・SC)の標準強度は表の範囲外。外挿せず入力に委ねている
    assert 80 not in st.EC_CONCRETE


def test_sc_pile_concrete_young_modulus():
    """SC杭のコンクリートは H24版で Ec = 3.5×10⁴ N/mm²(H29版は 4.0×10⁴)。

    フォーラムエイト UC-1 計算書サンプル Kui_10 の 1.2「杭の条件」に
    「杭体のヤング係数 SC杭：3.50×10⁴ N/mm²」と明記されており、この値が
    独立に裏付けられた(第52回)。この Ec は断面剛性(EI・Kv)に用いる値で、
    応力度照査専用のヤング係数比 n(YOUNG_MODULUS_RATIO_SC=6.00)とは別物。
    """
    assert st.EC_SC_PILE_CONCRETE == 3.5e7
    # σck = 80 は表の範囲外なので、表引きではなく専用の定数である
    assert 80 not in st.EC_CONCRETE
    # 表-3.3.3 の上限(σck = 60)と同値。外挿せず頭打ちにしていると解釈できる
    assert st.EC_SC_PILE_CONCRETE == st.EC_CONCRETE[60]


def test_sc_pile_stress_check_young_modulus_ratio_is_the_fixed_value_6():
    """SC杭の応力度照査専用のヤング係数比は Es/Ec ではなく固定値 6.00
    (第52回、Kui_10の3.3「杭体応力度」に n=6.00 と明記)。EC_SC_PILE_CONCRETE
    (=3.5×10⁴、Es/Ec=5.71相当)とは意図的に一致しない。"""
    assert st.YOUNG_MODULUS_RATIO_SC == 6.0
    assert st.E_STEEL / st.EC_SC_PILE_CONCRETE != pytest.approx(
        st.YOUNG_MODULUS_RATIO_SC, abs=0.1
    )


def test_young_modulus_ratio_is_the_fixed_value_15():
    """RC の応力度計算のヤング係数比は Es/Ec ではなく一定値 15(道示Ⅲ 3.3)。"""
    assert st.YOUNG_MODULUS_RATIO_RC == 15.0
    # Es/Ec から算出した値とは一致しない(σck=24 なら 8.0)
    assert st.E_REBAR / st.EC_CONCRETE[24] != st.YOUNG_MODULUS_RATIO_RC


def test_punching_shear_table_covers_21_to_30_only():
    """τa3(道示Ⅳ 表4.2.1)は σck = 21〜30 のみを規定する。"""
    assert st.TAU_A_PUNCHING == {21: 0.85, 24: 0.90, 27: 0.95, 30: 1.00}
    # 40 は表の範囲外(別途、設計条件・発注者基準の確認が必要)
    assert 40 not in st.TAU_A_PUNCHING
    assert set(st.TAU_A_PUNCHING) < set(st.EC_CONCRETE)


def test_rebar_tables_share_grades():
    for table in (*st.SIGMA_SA_REBAR_STATIC.values(),
                  *st.SIGMA_SA_REBAR_SEISMIC.values(),
                  st.SIGMA_CA_REBAR):
        assert set(table) == set(st.REBAR_GRADES)


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
    # 第31回に RC杭・SC杭を実装したので、未実装は H鋼杭のみ
    assert set(UNIMPLEMENTED_STRESS_CHECK) == {PileType.H_STEEL}


def test_sc_steel_allowable_reuses_the_verified_table_but_says_so():
    """SC杭の鋼管には表-4.4.1 の値を流用しており、それを注記していること。"""
    from core.section.checks import SC_STEEL_ALLOWABLE_NOTE

    for keyword in ("表-4.4.1", "原典未照合", "sc_steel_allowable"):
        assert keyword in SC_STEEL_ALLOWABLE_NOTE
    # 流用元は第23回に原典照合済みの表そのもの(値を別に持っていない)
    assert st.SIGMA_A_STEEL == {"SKK400": 140.0, "SKK490": 185.0}


def test_steel_yield_points_are_consistent_with_allowables():
    """鋼管杭の降伏点と許容応力度の比が安全率 1.7 と整合すること。"""
    assert st.SIGMA_Y_STEEL == {"SKK400": 235.0, "SKK490": 315.0}
    assert set(st.SIGMA_Y_STEEL) == set(st.SIGMA_A_STEEL)
    for grade, sigma_y in st.SIGMA_Y_STEEL.items():
        ratio = sigma_y / st.SIGMA_A_STEEL[grade]
        assert 1.65 <= ratio <= 1.75, f"{grade}: σy/σa = {ratio:.2f}"


def test_level2_allowable_ductility_pinned():
    """杭基礎の許容塑性率(直杭)。二次資料2件で一致。"""
    assert st.ALLOWABLE_DUCTILITY_PILE == {"橋脚": 4.0, "橋台": 3.0}
    # 橋台のほうが小さい(塑性化に対する余裕が小さい)
    assert st.ALLOWABLE_DUCTILITY_PILE["橋台"] < st.ALLOWABLE_DUCTILITY_PILE["橋脚"]
    assert set(st.ALLOWABLE_DUCTILITY_PILE) == {t.value for t in st.StructureType}


def test_cast_in_place_high_grade_rebar_reduces_ductility():
    """SD390・SD490 を用いる場所打ち杭は許容塑性率が下がる(確度C・安全側)。"""
    assert st.ALLOWABLE_DUCTILITY_CIP_HIGH_GRADE == {"橋脚": 2.0, "橋台": None}
    assert st.HIGH_GRADE_REBAR_FOR_DUCTILITY == frozenset({"SD390", "SD490"})
    for key, value in st.ALLOWABLE_DUCTILITY_CIP_HIGH_GRADE.items():
        # 通常値より必ず小さい(None = 塑性化不可)
        assert value is None or value < st.ALLOWABLE_DUCTILITY_PILE[key]
    # 実装している鉄筋材質のうち高強度側と整合すること
    assert "SD390" in st.REBAR_GRADES


def test_allowable_footing_rotation_pinned():
    """許容変位はフーチング底面の回転角 0.02 rad。

    二次資料には「0.02rad(約1/60rad)」とあるが、0.02 rad は 1/50 rad で
    あり併記が一致しない。数値 0.02 のほうを採用している。
    """
    assert st.ALLOWABLE_FOOTING_ROTATION == 0.02
    assert 1.0 / st.ALLOWABLE_FOOTING_ROTATION == pytest.approx(50.0)


def test_p_hu_factors_pinned():
    """杭前面地盤の pHU の係数(道示Ⅳ 12.10)。"""
    assert st.ALPHA_P_PILE == {"砂質土": 3.0, "礫質土": 3.0, "粘性土": 1.5}
    assert st.ALPHA_P_SOFT_CLAY == 1.0
    assert st.SOFT_CLAY_N_THRESHOLD == 2.0
    assert st.NON_FRONT_ROW_FACTOR_SAND == 0.5
    # 軟弱粘性土の特例は通常の粘性土より小さいこと
    assert st.ALPHA_P_SOFT_CLAY < st.ALPHA_P_PILE["粘性土"]
    # 砂質土・礫質土は同じ扱い
    assert st.ALPHA_P_PILE["砂質土"] == st.ALPHA_P_PILE["礫質土"]
    # 土質の網羅
    from core.models.soil import SoilType
    assert set(st.ALPHA_P_PILE) == {t.value for t in SoilType}


def test_rebar_detailing_constants():
    """道示Ⅳ 7.3 最小鉄筋量・最大鉄筋量。**原典で照合済み**(第23回)。"""
    assert st.MIN_REBAR_RATIO_AXIAL == 0.008
    assert st.MAX_TENSILE_REBAR_RATIO == 0.02
    assert st.MAX_TOTAL_REBAR_RATIO == 0.06
    assert st.ULTIMATE_CONCRETE_COEF == 0.85
    assert st.CRACK_TENSILE_STRENGTH_COEF == 0.23
    assert st.CRACK_TENSILE_STRENGTH_EXPONENT == pytest.approx(2.0 / 3.0)
    assert st.CRACK_MOMENT_MARGIN == 1.7
    assert st.SURFACE_REBAR_MIN_AREA_PER_M == 500.0
    assert st.SURFACE_REBAR_MAX_SPACING == 300.0
    # 引張鉄筋の上限は全鉄筋の上限より厳しい
    assert st.MAX_TENSILE_REBAR_RATIO < st.MAX_TOTAL_REBAR_RATIO
    # 最小は最大より小さい(当然だが、値の取り違えを検出する)
    assert st.MIN_REBAR_RATIO_AXIAL < st.MAX_TENSILE_REBAR_RATIO


def test_rebar_yield_points():
    """材質記号がそのまま降伏点(SD345 → 345)。せん断耐力では 345 で頭打ち。"""
    assert st.REBAR_YIELD_POINT == {"SD345": 345.0, "SD390": 390.0, "SD490": 490.0}
    assert set(st.REBAR_YIELD_POINT) == set(st.REBAR_GRADES)
    for grade, yield_point in st.REBAR_YIELD_POINT.items():
        assert float(grade.removeprefix("SD")) == yield_point
    assert st.SHEAR_REBAR_YIELD_CAP == min(st.REBAR_YIELD_POINT.values())
    assert st.SHEAR_CC_FOUNDATION == 1.0


def test_tau_max_is_a_different_table_from_tau_c():
    """表-4.3.2(斜め圧縮破壊の上限)と表-5.2.1(負担できる値)は別物。"""
    assert st.TAU_MAX_CONCRETE == {
        21: 2.8, 24: 3.2, 27: 3.6, 30: 4.0, 40: 5.3, 50: 6.0, 60: 6.0,
    }
    # τc(0.33〜0.37)とはひと桁違う
    for fck, tau_c in st.TAU_C_CONCRETE.items():
        assert st.TAU_MAX_CONCRETE[fck] > tau_c * 5
    # 高強度側は 6.0 で頭打ち
    assert st.TAU_MAX_CONCRETE[50] == st.TAU_MAX_CONCRETE[60] == 6.0


def test_group_pile_spacing_is_a_threshold_not_a_minimum():
    """2.5D は群杭影響を考慮する境界。最小寸法の規定ではない(第28回で訂正)。"""
    assert st.GROUP_PILE_SPACING_RATIO == 2.5
    # 「最小値」という誤った説明に戻らないよう、注意書きを固定する
    source = inspect.getsource(st)
    marker = source[: source.index("GROUP_PILE_SPACING_RATIO = ")]
    comment = marker.rsplit("\n\n", 1)[-1]
    assert "これは「最小値」ではない" in comment
    assert "令和7年改訂版" in comment  # H24 の条文そのものではない旨


def test_group_pile_kh_correction():
    """μ = 1 − 0.2(2.5 − L/D)。境界で連続し、L=D でも 0.7 を下回らない。"""
    from core.capacity.springs import group_pile_factor

    assert st.GROUP_PILE_KH_COEF == 0.2
    # 2.5D 以上では補正しない
    assert group_pile_factor(2.5, 1.0) == 1.0
    assert group_pile_factor(4.0, 1.0) == 1.0
    # 境界で連続(不連続なジャンプが無いこと)
    assert group_pile_factor(2.499, 1.0) == pytest.approx(1.0, abs=1e-3)
    # 提供資料の計算例: D=1.00 m、L=2.20 m → μ = 0.94
    assert group_pile_factor(2.20, 1.00) == pytest.approx(0.94)
    # 単調増加、かつ下限は L=D の 0.7
    values = [group_pile_factor(r, 1.0) for r in (1.0, 1.5, 2.0, 2.5)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.7)
    # 相似則: μ は L/D のみの関数
    assert group_pile_factor(2.2, 1.0) == pytest.approx(group_pile_factor(4.4, 2.0))


def test_region_cz_covers_every_zone_and_allows_1_20():
    """地域別補正係数。cIz は A1・B1 で 1.20(1.0 が上限ではない)。"""
    assert st.REGION_CZ == {
        "A1": (1.20, 1.00),
        "A2": (1.00, 1.00),
        "B1": (1.20, 0.85),
        "B2": (1.00, 0.85),
        "C": (0.80, 0.70),
    }
    assert st.CZ_MAX == 1.20
    assert st.CZ_MIN == 0.70
    # タイプI はタイプII 以上(どの地域でも)
    for zone, (cz1, cz2) in st.REGION_CZ.items():
        assert cz1 >= cz2, zone
    # 入力モデルが 1.20 を受け付けること(第28回まで 1.0 で頭打ちだった)
    from core.models.project import SeismicConditions

    assert SeismicConditions(cz_type1=1.20, cz_type2=1.00).cz_type1 == 1.20


def test_khg0_matches_the_supplied_table():
    """khg0(地盤面)は提供資料の表4と一致。橋の慣性力用 khc0 とは別物。"""
    assert st.KHG0_LIQUEFACTION == {
        st.GroundMotionType.LEVEL2_TYPE1: {
            st.GroundType.TYPE_I: 0.50,
            st.GroundType.TYPE_II: 0.45,
            st.GroundType.TYPE_III: 0.40,
        },
        st.GroundMotionType.LEVEL2_TYPE2: {
            st.GroundType.TYPE_I: 0.80,
            st.GroundType.TYPE_II: 0.70,
            st.GroundType.TYPE_III: 0.60,
        },
    }
    # タイプII のほうが大きく、良い地盤ほど大きい
    for ground in st.GroundType:
        t1 = st.KHG0_LIQUEFACTION[st.GroundMotionType.LEVEL2_TYPE1][ground]
        t2 = st.KHG0_LIQUEFACTION[st.GroundMotionType.LEVEL2_TYPE2][ground]
        assert t2 > t1
