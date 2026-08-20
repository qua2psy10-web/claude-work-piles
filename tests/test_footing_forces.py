"""底版照査断面の断面力算定の検証。

出典はフォーラムエイト UC-1「基礎の設計」計算書サンプル。

* Kui_5 7.2.3「断面力算出」— 柱前面のみ2箇所、杭頭水平反力・杭頭モーメントを
  含まない最も単純な形。
* Kui_4 7.5.4「断面力算出」— 8照査位置、Mp1/Mp2/Mp3 の3成分すべてを含む。
* Kui_8 8.5.4「断面力算出」— 上側引張のせん断スパン。

docs/VERIFICATION.md 第59回。
"""
from __future__ import annotations

import pytest

from core.section.footing_forces import (
    FootingSection,
    PileReaction,
    dead_load_forces,
    design_moment,
    design_shear,
    effective_width,
    section_forces,
    shear_span,
)


# ---------------------------------------------------------------------------
# Kui_5 — 単純な片持ち(柱前面のみ、H・Mt なし)
# ---------------------------------------------------------------------------


KUI5_LONGITUDINAL = FootingSection(depth=5.500, thickness=1.500, gamma_concrete=24.50)
KUI5_TRANSVERSE = FootingSection(depth=4.500, thickness=1.500, gamma_concrete=24.50)


def test_kui5_longitudinal_dead_load():
    """Kui_5 橋軸方向 L=1.400: W1=282.97, Σ(W・x)=198.08。"""
    total, moment = dead_load_forces(1.400, KUI5_LONGITUDINAL)
    assert total == pytest.approx(282.97, abs=0.01)
    assert moment == pytest.approx(198.08, abs=0.01)


def test_kui5_longitudinal_section_forces():
    """Kui_5 橋軸方向 L=1.400: Sp=1851.00, Mp=1203.15, S=1568.03, M=1005.07。"""
    piles = [PileReaction(position=0.750, vertical=1851.00)]
    forces = section_forces(1.400, piles, KUI5_LONGITUDINAL)
    assert forces.pile_shear == pytest.approx(1851.00)
    assert forces.pile_moment == pytest.approx(1203.15, abs=0.01)
    assert forces.shear == pytest.approx(1568.03, abs=0.01)
    assert forces.moment == pytest.approx(1005.07, abs=0.01)


def test_kui5_transverse_section_forces():
    """Kui_5 橋軸直角方向 L=1.600: Sp=1851.00, Mp=1573.35, S=1586.40, M=1361.67。"""
    piles = [PileReaction(position=0.750, vertical=1851.00)]
    forces = section_forces(1.600, piles, KUI5_TRANSVERSE)
    assert forces.dead_load == pytest.approx(264.60, abs=0.01)
    assert forces.dead_load_moment == pytest.approx(211.68, abs=0.01)
    assert forces.pile_moment == pytest.approx(1573.35, abs=0.01)
    assert forces.shear == pytest.approx(1586.40, abs=0.01)
    assert forces.moment == pytest.approx(1361.67, abs=0.01)


def test_kui5_no_horizontal_or_head_moment_terms():
    """H・Mt を与えなければ Mp2・Mp3 は 0 で Mp = Mp1。"""
    piles = [PileReaction(position=0.750, vertical=1851.00)]
    forces = section_forces(1.400, piles, KUI5_LONGITUDINAL)
    assert forces.pile_moment_horizontal == 0.0
    assert forces.pile_moment_head == 0.0
    assert forces.pile_moment == forces.pile_moment_vertical


# ---------------------------------------------------------------------------
# Kui_4 — Mp1/Mp2/Mp3 の3成分すべて
# ---------------------------------------------------------------------------


KUI4 = FootingSection(
    depth=8.500,
    thickness=2.500,
    dry_soil=2.000,
    gamma_concrete=24.50,
    gamma_moist=19.00,
)
KUI4_TOTAL_WIDTH = 11.500
KUI4_HG = 1.250
KUI4_COLUMN_FACE = 4.650

# 左側2杭列。Mp2 = Σ(Hi)・hg = 133.42 より Hi = 133.42/1.250。
KUI4_PILES = [
    PileReaction(
        position=1.200, vertical=3909.03, horizontal=133.42 / KUI4_HG, head_moment=2515.97
    ),
    PileReaction(
        position=4.233, vertical=4610.55, horizontal=133.42 / KUI4_HG, head_moment=2515.97
    ),
]


@pytest.mark.parametrize(
    "position,total,moment",
    [
        (1.200, 1012.35, 607.41),
        (3.400, 2868.33, 4876.15),
        (4.650, 3922.86, 9120.64),
    ],
)
def test_kui4_dead_load(position, total, moment):
    """Kui_4 橋軸方向の ΣW・Σ(W・x)(上載土・浮力なしのケース)。"""
    got_total, got_moment = dead_load_forces(position, KUI4)
    assert got_total == pytest.approx(total, abs=0.02)
    assert got_moment == pytest.approx(moment, abs=0.02)


def test_kui4_position1_pile_moment_components():
    """Kui_4 L=1.200: Mp1=0.00, Mp2=133.42, Mp3=2515.97, Mp=2649.39。"""
    forces = section_forces(1.200, KUI4_PILES, KUI4, centroid_height=KUI4_HG)
    assert forces.pile_shear == pytest.approx(3909.03)
    assert forces.pile_moment_vertical == pytest.approx(0.0, abs=1e-9)
    assert forces.pile_moment_horizontal == pytest.approx(133.42, abs=0.01)
    assert forces.pile_moment_head == pytest.approx(2515.97)
    assert forces.pile_moment == pytest.approx(2649.39, abs=0.01)


def test_kui4_position4_column_face():
    """Kui_4 L=4.650(柱前面): Mp1=15408.77, Mp=20707.54。"""
    forces = section_forces(KUI4_COLUMN_FACE, KUI4_PILES, KUI4, centroid_height=KUI4_HG)
    assert forces.pile_shear == pytest.approx(8519.58)
    assert forces.pile_moment_vertical == pytest.approx(15408.77, abs=0.03)
    assert forces.pile_moment_horizontal == pytest.approx(266.84, abs=0.01)
    assert forces.pile_moment_head == pytest.approx(5031.94)
    assert forces.pile_moment == pytest.approx(20707.54, abs=0.03)


def test_kui4_position2_only_outer_piles_contribute():
    """L=3.400 では杭列2(4.233)は照査断面より内側なので寄与しない。

    計算例の Vi は表示が6桁(3909.03)で丸められており、真値は 3909.036 程度。
    そのため Mp1 は 0.02 程度ずれる(照査位置4で 15408.77 に一致することから
    逆算できる)。
    """
    forces = section_forces(3.400, KUI4_PILES, KUI4, centroid_height=KUI4_HG)
    assert forces.pile_shear == pytest.approx(3909.03)
    assert forces.pile_moment_vertical == pytest.approx(8599.88, abs=0.02)
    assert forces.pile_moment == pytest.approx(11249.26, abs=0.02)


@pytest.mark.parametrize(
    "position,expected_mo",
    [(1.200, 240.23), (4.650, 1363.16)],
)
def test_kui4_design_moment_per_metre(position, expected_mo):
    """Mo = (Mp − Σ(W・x)) / B。b = B = 8.500(下側引張)、α = 1.000。"""
    forces = section_forces(position, KUI4_PILES, KUI4, centroid_height=KUI4_HG)
    width = effective_width(8.500, 5.000, 2.390, upper_tension=False)
    assert width == pytest.approx(8.500)
    assert design_moment(forces.moment, width) == pytest.approx(expected_mo, abs=0.02)


def test_kui4_design_shear_per_metre():
    """Kui_4 L=1.200: So = (3909.03 − 1012.35)/8.500 = 340.79 kN/m。"""
    forces = section_forces(1.200, KUI4_PILES, KUI4, centroid_height=KUI4_HG)
    assert design_shear(forces.shear, 8.500) == pytest.approx(340.79, abs=0.01)


def test_kui4_shear_span_lower_tension():
    """Kui_4 L=1.200(下側引張): a = |M'/S'| = 3.450。"""
    got = shear_span(1.200, KUI4_PILES, KUI4_COLUMN_FACE, upper_tension=False)
    assert got == pytest.approx(3.450, abs=0.001)


def test_kui4_mirrored_reproduces_far_side():
    """反対側から見た L=6.850 は、鏡像化した L=4.650 と一致する。

    Kui_4 の照査位置5(L=6.850)は Mp2=−266.84, Mp3=−5031.94 と符号が反転し、
    ΣW・Σ(W・x) は照査位置4(L=4.650)と同値になる。
    """
    far_piles = [
        PileReaction(position=7.267, vertical=5312.30, horizontal=133.42 / KUI4_HG,
                     head_moment=2515.97),
        PileReaction(position=10.300, vertical=6013.81, horizontal=133.42 / KUI4_HG,
                     head_moment=2515.97),
    ]
    mirrored = [p.mirrored(KUI4_TOTAL_WIDTH) for p in far_piles]
    forces = section_forces(
        KUI4_TOTAL_WIDTH - 6.850, mirrored, KUI4, centroid_height=KUI4_HG
    )
    assert forces.pile_shear == pytest.approx(11326.11)
    assert forces.pile_moment_vertical == pytest.approx(22962.88, abs=0.03)
    assert forces.pile_moment_horizontal == pytest.approx(-266.84, abs=0.01)
    assert forces.pile_moment_head == pytest.approx(-5031.94)
    assert forces.pile_moment == pytest.approx(17664.10, abs=0.03)
    assert forces.dead_load == pytest.approx(3922.86, abs=0.02)


def test_mirrored_keeps_vertical_sign():
    """鏡像化しても鉛直反力の符号は変わらない。"""
    pile = PileReaction(position=1.200, vertical=100.0, horizontal=20.0, head_moment=30.0)
    got = pile.mirrored(11.500)
    assert got.position == pytest.approx(10.300)
    assert got.vertical == pytest.approx(100.0)
    assert got.horizontal == pytest.approx(-20.0)
    assert got.head_moment == pytest.approx(-30.0)


# ---------------------------------------------------------------------------
# Kui_8 — 上側引張のせん断スパン・有効幅
# ---------------------------------------------------------------------------


def test_kui8_shear_span_upper_tension():
    """Kui_8 L=1.200(上側引張): a = L + L' = 1.650 + min(2.700/2, 2.390) = 3.000。"""
    got = shear_span(
        1.200,
        [],
        2.850,
        upper_tension=True,
        column_width=2.700,
        face_effective_depth=2.390,
    )
    assert got == pytest.approx(3.000, abs=0.001)


def test_upper_tension_shear_span_uses_effective_depth_when_smaller():
    """L' は柱幅の1/2と有効高の小さい方。有効高が小さければそちらを採る。"""
    got = shear_span(
        1.200, [], 2.850, upper_tension=True, column_width=6.000, face_effective_depth=1.000
    )
    assert got == pytest.approx(1.650 + 1.000, abs=1e-9)


def test_effective_width_upper_tension_capped_by_total_width():
    """上側引張の有効幅 tc + 1.5d は底版全幅 B で頭打ちになる。"""
    assert effective_width(8.500, 5.000, 2.390, upper_tension=True) == pytest.approx(8.500)
    assert effective_width(20.0, 5.000, 2.390, upper_tension=True) == pytest.approx(8.585)


# ---------------------------------------------------------------------------
# 浮力・上載土
# ---------------------------------------------------------------------------


def test_buoyant_head_capped_by_footing_and_submerged_soil():
    """浮力の作用高さ hw' は (h1 + h2) と hw の小さい方。"""
    section = FootingSection(depth=5.0, thickness=2.0, submerged_soil=1.0, water_head=10.0)
    assert section.buoyant_head == pytest.approx(3.0)
    section = FootingSection(depth=5.0, thickness=2.0, submerged_soil=1.0, water_head=1.5)
    assert section.buoyant_head == pytest.approx(1.5)


def test_buoyancy_reduces_dead_load():
    """浮力は自重を軽減する(断面力を減らす方向に働く)。"""
    dry = FootingSection(depth=5.0, thickness=2.0)
    wet = FootingSection(depth=5.0, thickness=2.0, water_head=2.0, gamma_water=10.0)
    assert dead_load_forces(3.0, wet)[0] < dead_load_forces(3.0, dry)[0]


def test_submerged_and_dry_soil_both_counted():
    """水位上下の上載土がそれぞれの単位重量で加算される。"""
    section = FootingSection(
        depth=2.0, thickness=1.0, submerged_soil=1.0, dry_soil=2.0, gamma_water=0.0
    )
    total, _ = dead_load_forces(1.0, section)
    assert total == pytest.approx(2.0 * (1.0 * 24.5 + 1.0 * 20.0 + 2.0 * 19.0))


# ---------------------------------------------------------------------------
# 異常系
# ---------------------------------------------------------------------------


def test_negative_position_rejected():
    with pytest.raises(ValueError, match="照査位置"):
        dead_load_forces(-1.0, KUI4)


@pytest.mark.parametrize("kwargs", [{"depth": 0.0}, {"thickness": 0.0}])
def test_invalid_geometry_rejected(kwargs):
    base = {"depth": 5.0, "thickness": 2.0}
    with pytest.raises(ValueError):
        FootingSection(**{**base, **kwargs})


def test_negative_soil_height_rejected():
    with pytest.raises(ValueError, match="0以上"):
        FootingSection(depth=5.0, thickness=2.0, dry_soil=-1.0)


def test_zero_width_rejected():
    with pytest.raises(ValueError, match="有効幅"):
        design_moment(100.0, 0.0)


def test_upper_tension_shear_span_requires_column_data():
    with pytest.raises(ValueError, match="柱幅と柱前面の有効高"):
        shear_span(1.0, [], 2.0, upper_tension=True)


def test_lower_tension_shear_span_requires_nonzero_shear():
    with pytest.raises(ValueError, match="せん断力が0"):
        shear_span(1.0, [], 2.0, upper_tension=False)
