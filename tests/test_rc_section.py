"""円形RC断面の応力度算定のテスト。"""
import math

import pytest

from core.section.rc import RebarLayout, _concrete_integrals, analyze_circular_rc

EC = 2.5e7  # kN/m2(σck=24)
N_RATIO = 8.0  # E_REBAR/EC = 2.0e8/2.5e7
D = 1.0
REBAR = RebarLayout(count=24, diameter_mm=25.0, cover_mm=125.0)


def test_rebar_geometry():
    # D25 の**公称断面積**は 506.7 mm²(JIS G 3112)。呼び名の直径の円の
    # 面積 π・25²/4 = 490.9 mm² より 3.2% 大きい(異形棒鋼のリブ・節による)
    assert REBAR.bar_area == pytest.approx(506.7e-6)
    assert REBAR.bar_area > math.pi * 0.025**2 / 4
    assert REBAR.total_area == pytest.approx(24 * 506.7e-6)
    assert REBAR.radius(D) == pytest.approx(0.375)
    ys = REBAR.positions(D)
    assert len(ys) == 24
    assert max(ys) == pytest.approx(0.375)
    assert sum(ys) == pytest.approx(0.0, abs=1e-12)


def test_concrete_integrals_full_section():
    """中立軸を断面外に置くと、全断面積・断面二次モーメントに一致する。"""
    radius = 0.5
    y_n = -radius
    s_area, s_moment = _concrete_integrals(radius, y_n, divisions=2000)
    # ∫(y+R)·b dy = R·A (Aは円の面積、∫y·b dy = 0 のため)
    area = math.pi * radius**2
    assert s_area == pytest.approx(radius * area, rel=1e-4)
    # ∫(y+R)·y·b dy = I(∫y²b dy)
    inertia = math.pi * radius**4 / 4.0
    assert s_moment == pytest.approx(inertia, rel=1e-4)


def test_uncracked_section_matches_hand_calculation():
    """全断面圧縮のケースを換算断面の手計算と突合する。

    N=1500 kN, M=100 kN·m, D=1.0m, 24-D25 (かぶり125mm), n=8
      A  = πD²/4 + (n−1)·As = 0.785398 + 7×0.0117810 = 0.867865 m²
      I  = πD⁴/64 + (n−1)·ΣAs·y² = 0.0490874 + 7×0.00082849 = 0.0548868 m⁴
      σ  = N/A ± M·y/I
    """
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=100.0)
    assert result.fully_compressed

    area = math.pi * D**2 / 4 + (N_RATIO - 1) * REBAR.total_area
    sum_y2 = sum(y**2 for y in REBAR.positions(D)) * REBAR.bar_area
    inertia = math.pi * D**4 / 64 + (N_RATIO - 1) * sum_y2
    sigma_c = (1500.0 / area + 100.0 * 0.5 / inertia) / 1000.0
    assert result.sigma_c == pytest.approx(sigma_c, rel=2e-3)

    # 最外縁鉄筋(y=0.375)の圧縮応力度 = n × その位置のコンクリート応力度
    sigma_c_at_bar = (1500.0 / area + 100.0 * 0.375 / inertia) / 1000.0
    assert result.sigma_s_compression == pytest.approx(
        N_RATIO * sigma_c_at_bar, rel=2e-3
    )
    assert result.sigma_s_tension == 0.0


def test_cracked_section_equilibrium():
    """ひび割れ断面で、求めた応力状態が N・M の釣合いを満たすこと。"""
    axial, moment = 1500.0, 800.0
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=axial, moment=moment)
    assert not result.fully_compressed

    # 応力分布から軸力・モーメントを積分し直して照合する
    radius = D / 2
    y_n = result.neutral_axis_y
    k = result.curvature
    s_area, s_moment = _concrete_integrals(radius, y_n, divisions=4000)
    n_axial = EC * k * s_area
    n_moment = EC * k * s_moment
    for y in result_positions():
        ratio = N_RATIO if y <= y_n else N_RATIO - 1.0
        force = ratio * REBAR.bar_area * EC * k * (y - y_n)
        n_axial += force
        n_moment += force * y
    assert n_axial == pytest.approx(axial, rel=1e-3)
    assert n_moment == pytest.approx(moment, rel=1e-3)


def result_positions():
    return REBAR.positions(D)


def test_larger_moment_reduces_compression_depth():
    depths = []
    for moment in (200.0, 600.0, 1200.0):
        r = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=moment)
        depths.append(r.compression_depth)
    assert depths[0] > depths[1] > depths[2]


def test_moment_sign_does_not_matter():
    plus = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=800.0)
    minus = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=-800.0)
    assert plus.sigma_c == pytest.approx(minus.sigma_c)
    assert plus.sigma_s_tension == pytest.approx(minus.sigma_s_tension)


def test_pure_axial_gives_uniform_stress():
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=1500.0, moment=0.0)
    area = math.pi * D**2 / 4 + (N_RATIO - 1) * REBAR.total_area
    assert result.sigma_c == pytest.approx(1500.0 / area / 1000.0, rel=2e-3)
    assert result.sigma_s_tension == 0.0


def test_pure_tension_axial_not_supported():
    """モーメントを伴わない純引張(net軸力<0, M=0)は未対応(部分圧縮ゾーンが
    存在しえず、コンクリート無引張の単純化モデルでは解けない)。"""
    with pytest.raises(NotImplementedError):
        analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=-100.0, moment=0.0)


def test_net_tension_with_large_moment_still_has_partial_compression_zone():
    """net軸力が引張でもモーメントが卓越していれば、圧縮縁側に部分圧縮
    ゾーンが残ることがある(道示Ⅴ耐震設計で杭頭のNminが地震時に軸力反転
    してもモーメントが大きい場合など)。フォーラムエイト UC-1 Kui_8の
    計算例(6.3仮想RC断面照査、地震時Nmin: N=-43kN, M=147kN・m)で
    実際に発生することを確認した(第51回)。ここでは別断面(D=1.0,
    REBAR)で、独立した数値積分による軸力・モーメントの釣合いチェックで
    解の正しさを検証する。"""
    result = analyze_circular_rc(D, REBAR, EC, N_RATIO, axial=-100.0, moment=500.0)
    assert not result.fully_compressed
    assert result.sigma_c > 0.0
    assert result.sigma_s_tension > 0.0

    radius = D / 2.0
    y_n = result.neutral_axis_y
    k = result.curvature

    def width(y: float) -> float:
        return 2.0 * math.sqrt(max(0.0, radius**2 - y**2))

    n = 4000
    ys = [y_n + (radius - y_n) * i / n for i in range(n + 1)]
    n_force = 0.0
    m_force = 0.0
    for i in range(n):
        y_mid = (ys[i] + ys[i + 1]) / 2.0
        dy = ys[i + 1] - ys[i]
        stress = EC * k * (y_mid - y_n)  # kN/m2, 圧縮正
        n_force += stress * width(y_mid) * dy
        m_force += stress * width(y_mid) * y_mid * dy
    for fiber in REBAR.fibers(D):
        ratio = (N_RATIO - 1.0) if fiber.y > y_n else N_RATIO
        sigma = ratio * EC * k * (fiber.y - y_n)
        n_force += sigma * fiber.area
        m_force += sigma * fiber.area * fiber.y

    assert n_force == pytest.approx(-100.0, abs=0.1)
    assert m_force == pytest.approx(500.0, abs=0.1)


def test_excessive_cover_rejected():
    bad = RebarLayout(count=12, diameter_mm=25.0, cover_mm=600.0)
    with pytest.raises(ValueError, match="かぶり"):
        bad.radius(D)


# --- 中空断面・鋼管合成断面 --------------------------------------------------


def _annulus_properties(outer: float, inner: float) -> tuple[float, float]:
    """円環の (断面積, 断面二次モーメント)。厳密解。"""
    return (
        math.pi * (outer**2 - inner**2) / 4.0,
        math.pi * (outer**4 - inner**4) / 64.0,
    )


def test_concrete_integrals_of_an_annulus_match_the_closed_form():
    """円環でも全断面圧縮の積分が厳密解に一致すること。"""
    radius, inner = 0.5, 0.3
    s_area, s_moment = _concrete_integrals(radius, -radius, inner, divisions=2000)
    area, inertia = _annulus_properties(2.0 * radius, 2.0 * inner)
    assert s_area == pytest.approx(radius * area, rel=1e-4)
    assert s_moment == pytest.approx(inertia, rel=1e-4)


def test_steel_tube_fibers_preserve_area_and_first_moment_exactly():
    """鋼管の繊維分割は断面積と断面一次モーメントを厳密に保つこと。"""
    from core.section.rc import steel_tube_fibers

    d, t = 0.6, 0.012
    fibers = steel_tube_fibers(d, t)
    area, inertia = _annulus_properties(d, d - 2.0 * t)
    assert sum(f.area for f in fibers) == pytest.approx(area, rel=1e-12)
    assert sum(f.area * f.y for f in fibers) == pytest.approx(0.0, abs=1e-12)
    # 断面二次モーメントのみ分割による誤差を持つ(集中質量化のため)
    lumped = sum(f.area * f.y**2 for f in fibers)
    assert lumped == pytest.approx(inertia, rel=1e-3)
    assert lumped < inertia  # 各区間の自身まわりの慣性を落としている分
    # 鋼管はコンクリートの外側にあるので (n−1) 控除の対象にしない
    assert all(not f.embedded for f in fibers)


def test_steel_tube_fibers_reject_invalid_geometry():
    from core.section.rc import steel_tube_fibers

    with pytest.raises(ValueError):
        steel_tube_fibers(0.6, 0.0)
    with pytest.raises(ValueError, match="円環"):
        steel_tube_fibers(0.6, 0.4)
    with pytest.raises(ValueError, match="分割数"):
        steel_tube_fibers(0.6, 0.012, divisions=4)


HOLLOW_REBAR = RebarLayout(count=12, diameter_mm=19.0, cover_mm=40.0)


def test_hollow_section_uncracked_matches_the_transformed_section_formula():
    """中空断面の全断面圧縮を σ = N/At ± M·y/It と突合する。"""
    from core.section.rc import transformed_section

    inner = 0.42  # D=0.6、肉厚 90mm
    area_t, inertia_t = transformed_section(D, HOLLOW_REBAR, N_RATIO, inner)
    # 換算断面積・断面二次モーメントを手計算で組み立てる
    area_c, inertia_c = _annulus_properties(D, inner)
    ys = HOLLOW_REBAR.positions(D)
    bar = HOLLOW_REBAR.bar_area
    assert area_t == pytest.approx(
        area_c + (N_RATIO - 1.0) * len(ys) * bar
    )
    assert inertia_t == pytest.approx(
        inertia_c + (N_RATIO - 1.0) * bar * sum(y**2 for y in ys)
    )

    n_load = 1500.0
    m_load = 20.0  # 核の内側に収まる小さな偏心
    result = analyze_circular_rc(
        D, HOLLOW_REBAR, EC, N_RATIO, n_load, m_load, inner_diameter=inner
    )
    assert result.fully_compressed
    assert result.sigma_c == pytest.approx(
        (n_load / area_t + m_load * (D / 2.0) / inertia_t) / 1000.0
    )


def test_hollowing_the_section_raises_the_concrete_stress():
    """同じ外径・同じ断面力なら、中空にしたほうがコンクリート応力度が大きい。

    一方**鉄筋の引張応力度は下がる**。中空にすると圧縮域の面積が減るので
    中立軸が引張側へ深く入り(圧縮域を広げて軸力を負担するため)、引張鉄筋が
    中立軸に近づいてひずみが小さくなるからである。直感に反するが、釣合いは
    :func:`test_hollow_cracked_section_satisfies_equilibrium` で独立に
    確認している。
    """
    solid = analyze_circular_rc(D, HOLLOW_REBAR, EC, N_RATIO, 1500.0, 300.0)
    hollow = analyze_circular_rc(
        D, HOLLOW_REBAR, EC, N_RATIO, 1500.0, 300.0, inner_diameter=0.42
    )
    assert hollow.sigma_c > solid.sigma_c
    assert hollow.compression_depth > solid.compression_depth
    assert hollow.sigma_s_tension < solid.sigma_s_tension


def test_rebar_must_sit_inside_the_concrete_wall():
    # D=1.0、内径 0.42 → 肉厚 290mm。かぶり 300mm では中空部に落ちる
    outside = RebarLayout(count=12, diameter_mm=19.0, cover_mm=300.0)
    with pytest.raises(ValueError, match="中空部"):
        analyze_circular_rc(
            D, outside, EC, N_RATIO, 1500.0, 300.0, inner_diameter=0.42
        )
    # 肉厚の中に収まっていれば通る
    inside = RebarLayout(count=12, diameter_mm=19.0, cover_mm=120.0)
    assert inside.radius(D) > 0.42 / 2.0
    analyze_circular_rc(
        D, inside, EC, N_RATIO, 1500.0, 300.0, inner_diameter=0.42
    )


def test_inner_diameter_must_be_smaller_than_the_outer():
    from core.section.rc import analyze_circular_section

    with pytest.raises(ValueError, match="内径"):
        analyze_circular_section(
            D, HOLLOW_REBAR.fibers(D), EC, N_RATIO, 1500.0, 300.0,
            inner_diameter=D,
        )


def _equilibrium(
    outer: float,
    inner: float,
    fibers: list,
    ec: float,
    n_ratio: float,
    result,
    strips: int = 20000,
) -> tuple[float, float]:
    """求まった (中立軸, 曲率) から軸力とモーメントを積み直す。

    断面の幅を独立に組み立てて数値積分するので、解法そのものとは別の経路で
    釣合いを確かめられる。返り値は (N, M) (kN, kN·m)。
    """
    r_out, r_in = outer / 2.0, inner / 2.0
    y_n, k = result.neutral_axis_y, result.curvature
    if result.fully_compressed:
        y_n = -math.inf

    n_sum = 0.0
    m_sum = 0.0
    h = 2.0 * r_out / strips
    for i in range(strips):
        y = -r_out + (i + 0.5) * h
        if y <= y_n:
            continue
        width = 2.0 * math.sqrt(max(0.0, r_out**2 - y**2))
        if r_in > 0.0 and abs(y) < r_in:
            width -= 2.0 * math.sqrt(r_in**2 - y**2)
        sigma = ec * k * (y - y_n)  # kN/m2
        force = sigma * width * h
        n_sum += force
        m_sum += force * y
    for fiber in fibers:
        ratio = n_ratio
        if fiber.embedded and fiber.y > y_n:
            ratio = n_ratio - 1.0
        sigma = ec * k * (fiber.y - y_n)
        force = ratio * sigma * fiber.area
        n_sum += force
        m_sum += force * fiber.y
    return n_sum, m_sum


def test_hollow_cracked_section_satisfies_equilibrium():
    """中空のひび割れ断面が、独立な数値積分で釣合っていること。"""
    inner = 0.42
    n_load, m_load = 1500.0, 350.0
    result = analyze_circular_rc(
        D, HOLLOW_REBAR, EC, N_RATIO, n_load, m_load, inner_diameter=inner
    )
    assert not result.fully_compressed
    n_calc, m_calc = _equilibrium(
        D, inner, HOLLOW_REBAR.fibers(D), EC, N_RATIO, result
    )
    assert n_calc == pytest.approx(n_load, rel=1e-3)
    assert m_calc == pytest.approx(m_load, rel=1e-3)


def test_composite_steel_tube_section_satisfies_equilibrium():
    """SC杭型(外側鋼管 + 中空コンクリート)でも釣合っていること。"""
    from core.section.rc import analyze_circular_section, steel_tube_fibers

    d_out, t = 0.6, 0.008
    concrete_outer = d_out - 2.0 * t
    concrete_inner = concrete_outer - 2.0 * 0.080
    ec = 3.5e7
    n_ratio = 2.0e8 / ec
    fibers = steel_tube_fibers(d_out, t)

    n_load, m_load = 1400.0, 350.0
    result = analyze_circular_section(
        diameter=concrete_outer,
        fibers=fibers,
        ec=ec,
        n_ratio=n_ratio,
        axial=n_load,
        moment=m_load,
        inner_diameter=concrete_inner,
    )
    assert not result.fully_compressed
    # 鋼管は引張・圧縮の両方を負担する
    assert result.sigma_s_tension > 0.0
    assert result.sigma_s_compression > 0.0
    n_calc, m_calc = _equilibrium(
        concrete_outer, concrete_inner, fibers, ec, n_ratio, result
    )
    assert n_calc == pytest.approx(n_load, rel=1e-3)
    assert m_calc == pytest.approx(m_load, rel=1e-3)
