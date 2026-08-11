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
    assert set(st.QD_MAX) == methods
    assert set(st.F_MAX) == methods
    for method in methods:
        # 周面摩擦力度は3土質すべてに定義が必要
        assert set(st.F_SPECS[method]) == {"砂質土", "礫質土", "粘性土"}
        assert set(st.F_MAX[method]) == {"砂質土", "礫質土", "粘性土"}
        # 先端支持力度は支持層になり得る土質のみ
        assert st.QD_SPECS[method]
        assert set(st.QD_MAX[method]) <= set(st.QD_SPECS[method])


def test_qd_specs():
    assert st.QD_SPECS["打込み(打撃)"]["砂質土"] == (130.0, "N")
    assert st.QD_SPECS["場所打ち"]["砂質土"] == (3000.0, "const")
    assert st.QD_SPECS["場所打ち"]["粘性土"] == (3.0, "qu")
    assert st.QD_SPECS["中掘り"]["砂質土"] == (200.0, "N")
    # 場所打ち杭は砂質土・礫質土・粘性土を支持層にできる
    assert set(st.QD_SPECS["場所打ち"]) == {"砂質土", "礫質土", "粘性土"}
    # 打込み杭は粘性土を支持層としない
    assert "粘性土" not in st.QD_SPECS["打込み(打撃)"]


def test_qd_max_pinned():
    """要確認: 中掘り・プレボーリングは 10,000 の可能性(VERIFICATION.md 参照)。"""
    assert st.QD_MAX["打込み(打撃)"]["砂質土"] == 6500.0
    assert st.QD_MAX["場所打ち"]["粘性土"] == 3000.0
    assert st.QD_MAX["中掘り"]["砂質土"] == 12000.0
    assert st.QD_MAX["プレボーリング"]["砂質土"] == 12000.0


def test_f_specs_pinned():
    """打込み・場所打ちの式形は複数ソースで確認済み。

    中掘り・プレボーリングは 2026-08-11 に 10N/0.8c → 3N/1.0c へ変更
    (独立3ソースが一致、旧値は支持力を過大評価していた)。
    """
    assert st.F_SPECS["打込み(打撃)"]["砂質土"] == (2.0, "N")
    assert st.F_MAX["打込み(打撃)"]["砂質土"] == 100.0
    assert st.F_SPECS["打込み(打撃)"]["粘性土"] == (1.0, "c")
    assert st.F_MAX["打込み(打撃)"]["粘性土"] == 150.0

    assert st.F_SPECS["場所打ち"]["砂質土"] == (5.0, "N")
    assert st.F_MAX["場所打ち"]["砂質土"] == 200.0  # 要確認: 150 の可能性
    assert st.F_SPECS["場所打ち"]["粘性土"] == (1.0, "c")
    assert st.F_MAX["場所打ち"]["粘性土"] == 150.0

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
    assert st.ALPHA_KH_NORMAL == 1.0
    assert st.ALPHA_KH_SEISMIC == 2.0
    assert st.ALLOWABLE_DISPLACEMENT_MM == 15.0
    assert st.ALLOWABLE_DISPLACEMENT_RATIO == 0.01
    assert st.ALLOWABLE_DISPLACEMENT_DIA_THRESHOLD == 1.5


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
    """道示Ⅲ 表-3.2.3。SD345=180 は2ソースで確認済み。

    SD295 は資料間で 140 / 160 / 180 と食い違うため、最も安全側の 140 を据え置き。
    """
    assert st.SIGMA_SA_REBAR == {"SD295": 140.0, "SD345": 180.0, "SD390": 200.0}
    assert st.SIGMA_SA_REBAR_SEVERE == {"SD295": 140.0, "SD345": 160.0, "SD390": 180.0}
    # 腐食性環境の許容値は一般の部材以下
    for grade in st.SIGMA_SA_REBAR:
        assert st.SIGMA_SA_REBAR_SEVERE[grade] <= st.SIGMA_SA_REBAR[grade]


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
        st.TAU_A_PUNCHING,
    ):
        assert set(table) == grades


def test_rebar_tables_share_grades():
    assert set(st.SIGMA_SA_REBAR) == set(st.SIGMA_SA_REBAR_SEVERE)


def test_stress_increase_matches_safety_factor_cases():
    """許容応力度の割増と支持力の安全率は同じ荷重ケースを扱うこと。"""
    assert set(st.STRESS_INCREASE) == set(st.SAFETY_FACTORS_PUSH)
    assert set(st.STRESS_INCREASE) == set(st.SAFETY_FACTORS_PULL)
