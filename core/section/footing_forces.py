"""底版(フーチング)照査断面の断面力算定(道示Ⅳ(H24) 8章)。

底版は、下向きに作用する**底版自重・上載土重量**(浮力は控除)と、上向きに
作用する**杭頭反力**の差によって曲げ・せん断を受ける。本モジュールは、
照査位置における設計断面力 M・S を、片持ち梁として算定する。

    S  = Sp − ΣW
    M  = Mp − Σ(W・x)

Sp・Mp は照査断面より**外側**(自由端側)の杭頭反力による断面力、
ΣW・Σ(W・x) は同じく外側の底版自重・上載土重量・浮力による断面力である。

杭頭反力による曲げモーメントは3成分の和をとる。

    Mp1 = Σ(Vi・xi)     鉛直反力による
    Mp2 = Σ(Hi)・hg     水平反力による(hg は底版図心までの高さ)
    Mp3 = Σ(Mti)        杭頭モーメントによる
    Mp  = Mp1 + Mp2 + Mp3

曲げ照査は単位幅(1m)あたりで行うため、有効幅 b で除して換算する。

    Mo = (Mp − Σ(W・x)) / b
    M  = α・Mo

有効幅 b は、下側引張では底版全幅 B、上側引張では柱幅 tc に有効高の 1.5 倍を
加えた値(ただし B 以下)をとる。

.. note::
   単位重量 γc・γsat・γt・γw は**標準定数ではなく設計条件**である
   (原典の計算例も設計条件として明示している)。既定値は計算例の値だが、
   利用者が変更できるようにしている。特に γw は、本ソフトが地盤の有効応力に
   用いる :data:`core.standards.GAMMA_W` (=9.8) とは別に指定する。

.. warning::
   算定式は原典未照合であり、フォーラムエイト UC-1 の計算書サンプル
   (Kui_4・Kui_5・Kui_8)との突合による(docs/VERIFICATION.md 第59回)。
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 入力
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FootingSection:
    """断面力算定に用いる底版の諸元。

    高さはいずれも底版下面を基準とし、``depth`` は計算方向に直交する
    奥行き長(原典の「奥行き長 b」)である。
    """

    depth: float
    """奥行き長 b(m)。計算方向に直交する方向の底版幅。"""

    thickness: float
    """フーチング厚 h1(m)。"""

    submerged_soil: float = 0.0
    """水位より下の上載土厚 h2(m)。"""

    dry_soil: float = 0.0
    """水位より上の上載土厚 h3(m)。"""

    water_head: float = 0.0
    """水位 hw(m)。底版下面からの高さ。"""

    gamma_concrete: float = 24.5
    """フーチング単位重量 γc(kN/m³)。"""

    gamma_saturated: float = 20.0
    """上載土の飽和重量 γsat(kN/m³)。"""

    gamma_moist: float = 19.0
    """上載土の湿潤重量 γt(kN/m³)。"""

    gamma_water: float = 10.0
    """水の単位重量 γw(kN/m³)。"""

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("奥行き長は正の値が必要")
        if self.thickness <= 0:
            raise ValueError("フーチング厚は正の値が必要")
        for name, value in (
            ("水位より下の上載土厚", self.submerged_soil),
            ("水位より上の上載土厚", self.dry_soil),
            ("水位", self.water_head),
        ):
            if value < 0:
                raise ValueError(f"{name}は0以上が必要")

    @property
    def buoyant_head(self) -> float:
        """浮力の作用高さ hw'(m)。(h1 + h2) と hw の小さい方。"""
        return min(self.thickness + self.submerged_soil, self.water_head)


@dataclass(frozen=True)
class PileReaction:
    """1杭列ぶんの杭頭反力。

    ``position`` は、断面力を算定する側の底版端からの距離(m)である。
    """

    position: float
    """底版端からの距離(m)。"""

    vertical: float
    """鉛直反力 Vi(kN)。押込みを正とする。杭列の合計値。"""

    horizontal: float = 0.0
    """水平反力 Hi(kN)。杭列の合計値。"""

    head_moment: float = 0.0
    """杭頭モーメント Mti(kN·m)。杭列の合計値。"""

    def mirrored(self, total_width: float) -> "PileReaction":
        """反対側の端を基準とした反力に変換する。

        断面力を反対側から片持ち梁として算定する場合、位置は反転し、
        水平反力・杭頭モーメントは作用方向が逆になるため符号が反転する
        (鉛直反力は反転しない)。
        """
        return PileReaction(
            position=total_width - self.position,
            vertical=self.vertical,
            horizontal=-self.horizontal,
            head_moment=-self.head_moment,
        )


# ---------------------------------------------------------------------------
# 断面力
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FootingSectionForces:
    """照査位置における断面力の内訳。"""

    position: float
    """照査位置 L(m)。底版端からの距離。"""

    dead_load: float
    """底版自重・上載土重量・浮力の合計 ΣW(kN)。"""

    dead_load_moment: float
    """同じく Σ(W・x)(kN·m)。"""

    pile_shear: float
    """杭鉛直反力によるせん断力 Sp(kN)。"""

    pile_moment_vertical: float
    """Mp1 = Σ(Vi・xi)(kN·m)。"""

    pile_moment_horizontal: float
    """Mp2 = Σ(Hi)・hg(kN·m)。"""

    pile_moment_head: float
    """Mp3 = Σ(Mti)(kN·m)。"""

    @property
    def pile_moment(self) -> float:
        """杭頭反力による曲げモーメント Mp = Mp1 + Mp2 + Mp3(kN·m)。"""
        return (
            self.pile_moment_vertical
            + self.pile_moment_horizontal
            + self.pile_moment_head
        )

    @property
    def shear(self) -> float:
        """設計せん断力 S = Sp − ΣW(kN)。"""
        return self.pile_shear - self.dead_load

    @property
    def moment(self) -> float:
        """設計曲げモーメント M = Mp − Σ(W・x)(kN·m)。"""
        return self.pile_moment - self.dead_load_moment


def dead_load_forces(position: float, section: FootingSection) -> tuple[float, float]:
    """照査位置より外側の自重・上載土重量・浮力による ΣW と Σ(W・x) を返す。

        W1 = L・h1・b・γc      (フーチング)
        W2 = L・h2・b・γsat    (水位より下の上載土)
        W3 = L・h3・b・γt      (水位より上の上載土)
        W4 = −L・hw'・b・γw    (浮力)
        x1 = x2 = x3 = x4 = L/2

    いずれも等分布荷重なので重心位置は L/2 で共通である。
    """
    if position < 0:
        raise ValueError("照査位置は0以上が必要")
    area = position * section.depth
    total = area * (
        section.thickness * section.gamma_concrete
        + section.submerged_soil * section.gamma_saturated
        + section.dry_soil * section.gamma_moist
        - section.buoyant_head * section.gamma_water
    )
    return total, total * position / 2.0


def section_forces(
    position: float,
    piles: list[PileReaction],
    section: FootingSection,
    centroid_height: float | None = None,
) -> FootingSectionForces:
    """照査位置における断面力を算定する。

    ``piles`` のうち、照査位置以内(``position`` 以下)にあるもの、すなわち
    照査断面より**外側**の杭列のみが断面力に寄与する。

    ``centroid_height`` は hg(底版下面から図心までの高さ)で、既定では
    フーチング厚の 1/2 とする。テーパ付きの場合は図心位置を明示的に与える。
    """
    outer = [p for p in piles if p.position <= position + 1e-9]
    hg = section.thickness / 2.0 if centroid_height is None else centroid_height
    total_w, total_wx = dead_load_forces(position, section)
    return FootingSectionForces(
        position=position,
        dead_load=total_w,
        dead_load_moment=total_wx,
        pile_shear=sum(p.vertical for p in outer),
        pile_moment_vertical=sum(p.vertical * (position - p.position) for p in outer),
        pile_moment_horizontal=sum(p.horizontal for p in outer) * hg,
        pile_moment_head=sum(p.head_moment for p in outer),
    )


# ---------------------------------------------------------------------------
# 単位幅あたりへの換算
# ---------------------------------------------------------------------------


def effective_width(
    total_width: float,
    column_width: float,
    effective_depth: float,
    *,
    upper_tension: bool,
) -> float:
    """曲げ照査に用いる有効幅 b(m)。

        下側引張  b = B
        上側引張  b = tc + 1.5・d ≦ B
    """
    if total_width <= 0:
        raise ValueError("底版全幅は正の値が必要")
    if not upper_tension:
        return total_width
    if column_width <= 0:
        raise ValueError("柱幅は正の値が必要")
    if effective_depth <= 0:
        raise ValueError("有効高は正の値が必要")
    return min(column_width + 1.5 * effective_depth, total_width)


def design_moment(moment: float, width: float, alpha: float = 1.0) -> float:
    """単位幅(1m)あたりの設計曲げモーメント M = α・Mo(kN·m/m)。

    ``Mo = M / b`` であり、``alpha`` は有効幅の換算係数である。
    """
    if width <= 0:
        raise ValueError("有効幅は正の値が必要")
    return alpha * moment / width


def design_shear(shear: float, width: float) -> float:
    """単位幅(1m)あたりの設計せん断力 So = S / b(kN/m)。"""
    if width <= 0:
        raise ValueError("有効幅は正の値が必要")
    return shear / width


# ---------------------------------------------------------------------------
# せん断スパン
# ---------------------------------------------------------------------------


def shear_span(
    position: float,
    piles: list[PileReaction],
    column_face: float,
    *,
    upper_tension: bool,
    column_width: float | None = None,
    face_effective_depth: float | None = None,
) -> float:
    """せん断スパン a(m)。

        下側引張  a = |M'／S'|
        上側引張  a = L + L'

    M'・S' は、照査断面とそれより外側の杭鉛直反力によって**柱前面**に生じる
    曲げモーメント・せん断力である。L は照査断面から柱前面までの距離、
    L' は計算方向の柱幅の 1/2 と柱前面における有効高のうち小さい方の値。
    """
    if not upper_tension:
        outer = [p for p in piles if p.position <= position + 1e-9]
        shear_at_face = sum(p.vertical for p in outer)
        if shear_at_face == 0:
            raise ValueError("柱前面のせん断力が0のためせん断スパンを算定できない")
        moment_at_face = sum(p.vertical * (column_face - p.position) for p in outer)
        return abs(moment_at_face / shear_at_face)
    if column_width is None or face_effective_depth is None:
        raise ValueError("上側引張では柱幅と柱前面の有効高が必要")
    if column_width <= 0 or face_effective_depth <= 0:
        raise ValueError("柱幅・有効高は正の値が必要")
    return abs(column_face - position) + min(column_width / 2.0, face_effective_depth)
