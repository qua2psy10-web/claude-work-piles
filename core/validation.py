"""入力の妥当性チェック。

計算の前に、**物理的に成立しない入力**と**前提を外れている疑いのある入力**を
洗い出す。趣旨は次のとおり。

    間違った答えが「OK」と表示されることは、照査が無いことより危険である。

たとえば杭径 1.0 m の杭を 0.5 m 間隔で配置すると杭どうしが重なるが、
支持力も変位も何事もなく計算でき、総合判定は「OK」になってしまう。
利用者はまず気づかない。

重大度
------
* :attr:`Severity.ERROR` — **物理的に成立しない**。計算しても意味がないので
  :func:`core.analysis.stability.analyze` は例外を送出する。
* :attr:`Severity.WARNING` — 計算はできるが、規定違反や前提逸脱の疑いがある。
  計算は続け、結果に注記として付す。

.. note::
   ここで扱うのは**入力どうしの整合**である。個々の値域(正値・範囲)は
   pydantic のモデル側で検証している。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from core.models.loads import FootingLoads
from core.models.pile import Footing, PileArrangement, PileSpec, PileType
from core.models.soil import SoilProfile
from core.section.checks import MaterialSpec
from core.standards import MIN_PILE_SPACING_RATIO


class Severity(str, Enum):
    ERROR = "エラー"
    WARNING = "警告"


@dataclass(frozen=True)
class ValidationIssue:
    """入力の問題1件。"""

    severity: Severity
    field: str  # どの入力に関するものか
    message: str  # 何が問題か
    remedy: str = ""  # どうすればよいか

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def __str__(self) -> str:
        text = f"[{self.severity.value}] {self.field}: {self.message}"
        return f"{text} — {self.remedy}" if self.remedy else text


class InvalidInputError(ValueError):
    """物理的に成立しない入力。"""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        super().__init__(
            "入力が物理的に成立しません:\n"
            + "\n".join(
                f"  ・{i.field}: {i.message}"
                + (f" → {i.remedy}" if i.remedy else "")
                for i in issues
            )
        )


def _check_arrangement(
    pile: PileSpec, arrangement: PileArrangement, footing: Footing
) -> list[ValidationIssue]:
    """杭配置とフーチングの幾何。"""
    issues: list[ValidationIssue] = []
    d = pile.diameter

    for label, spacing, count in (
        ("橋軸方向", arrangement.spacing_x, arrangement.nx),
        ("直角方向", arrangement.spacing_y, arrangement.ny),
    ):
        if count <= 1:
            continue  # 1列なら間隔は結果に影響しない
        if spacing <= d:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"杭中心間隔({label})",
                    f"中心間隔 {spacing:.2f} m が杭径 {d:.2f} m 以下で、"
                    "杭どうしが重なります",
                    f"中心間隔を杭径より大きくしてください",
                )
            )
        elif spacing < MIN_PILE_SPACING_RATIO * d:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"杭中心間隔({label})",
                    f"中心間隔 {spacing:.2f} m が杭径の "
                    f"{spacing / d:.2f} 倍しかありません",
                    f"道示Ⅳ は杭中心間隔の最小値を規定しています"
                    f"(一般に {MIN_PILE_SPACING_RATIO:g}D 程度とされますが"
                    "**本ソフトでは原典未照合**)。適用する規定を確認してください",
                )
            )

    # 最外周杭がフーチングに収まっているか
    edge_x = footing.width_x / 2.0 - (arrangement.nx - 1) / 2.0 * arrangement.spacing_x
    edge_y = footing.width_y / 2.0 - (arrangement.ny - 1) / 2.0 * arrangement.spacing_y
    for label, edge, width in (
        ("橋軸方向", edge_x, footing.width_x),
        ("直角方向", edge_y, footing.width_y),
    ):
        if edge < d / 2.0:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"フーチング幅({label})",
                    f"最外周杭の中心からフーチング縁端までが {edge:.2f} m しかなく、"
                    f"杭(半径 {d / 2:.2f} m)がフーチングからはみ出します",
                    f"フーチング幅({label} {width:.2f} m)を広げるか、"
                    "杭間隔・本数を見直してください",
                )
            )
    return issues


def _check_section(pile: PileSpec, material: MaterialSpec) -> list[ValidationIssue]:
    """杭断面と配筋の整合。"""
    issues: list[ValidationIssue] = []
    rebar = material.rebar
    if rebar is not None and pile.pile_type == PileType.CAST_IN_PLACE:
        radius = pile.diameter / 2.0 - rebar.cover_mm / 1000.0
        if radius <= 0:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "かぶり",
                    f"かぶり {rebar.cover_mm:.0f} mm が杭の半径 "
                    f"{pile.diameter / 2 * 1000:.0f} mm 以上で、鉄筋を配置できません",
                    "かぶりを小さくするか、杭径を大きくしてください",
                )
            )
        else:
            # 隣接する鉄筋の中心間隔(鉄筋円の円周 ÷ 本数)
            spacing = 2.0 * math.pi * radius * 1000.0 / rebar.count
            if spacing <= rebar.diameter_mm:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        "軸方向鉄筋",
                        f"鉄筋の中心間隔 {spacing:.0f} mm が鉄筋径 "
                        f"{rebar.diameter_mm:.0f} mm 以下で、鉄筋どうしが重なります",
                        "本数を減らすか、径を細くしてください",
                    )
                )
            elif spacing < 2.0 * rebar.diameter_mm:
                issues.append(
                    ValidationIssue(
                        Severity.WARNING,
                        "軸方向鉄筋",
                        f"鉄筋のあき {spacing - rebar.diameter_mm:.0f} mm が"
                        f"鉄筋径 {rebar.diameter_mm:.0f} mm 未満です",
                        "コンクリートの充填性(粗骨材寸法との関係)を確認してください",
                    )
                )

    stirrup = material.stirrup
    if stirrup is not None and rebar is not None:
        # せん断ひび割れ(概ね 45°)を横切るには、間隔が有効高より十分小さい
        # 必要がある。有効高は概ね 0.75D 程度なので、その半分を目安とする。
        rough_depth = 0.75 * pile.diameter * 1000.0
        if stirrup.spacing_mm > rough_depth:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "帯鉄筋の間隔",
                    f"間隔 {stirrup.spacing_mm:.0f} mm が有効高の目安"
                    f"({rough_depth:.0f} mm)を超えています",
                    "せん断ひび割れを横切らないおそれがあります。"
                    "構造細目(道示Ⅳ 7.10)の最大間隔を確認してください",
                )
            )
    return issues


def _check_ground(
    pile: PileSpec, footing: Footing, profile: SoilProfile
) -> list[ValidationIssue]:
    """地盤モデルと杭・フーチングの深さの整合。"""
    issues: list[ValidationIssue] = []
    total = profile.total_depth
    tip = footing.embedment + pile.length

    if footing.embedment >= total:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                "根入れ深さ",
                f"フーチング下面 {footing.embedment:.2f} m が地盤モデルの深さ "
                f"{total:.2f} m 以上です",
                "地層を追加するか、根入れ深さを見直してください",
            )
        )
    elif tip > total:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                "杭長",
                f"杭先端 {tip:.2f} m が地盤モデルの深さ {total:.2f} m を"
                "超えています",
                "地層を追加するか、杭長を短くしてください",
            )
        )

    if profile.gwl > total:
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                "地下水位",
                f"地下水位 {profile.gwl:.2f} m が地盤モデルの深さ "
                f"{total:.2f} m より深く、実質的に「地下水なし」として扱われます",
                "意図した設定か確認してください",
            )
        )
    return issues


def _check_loads(loads: list[FootingLoads]) -> list[ValidationIssue]:
    """荷重の整合。"""
    issues: list[ValidationIssue] = []
    if not loads:
        issues.append(
            ValidationIssue(
                Severity.ERROR, "荷重", "荷重ケースが1つもありません",
                "少なくとも1ケース入力してください",
            )
        )
        return issues

    seen: dict[str, int] = {}
    for load in loads:
        seen[load.case.value] = seen.get(load.case.value, 0) + 1
    for case, count in seen.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "荷重ケース",
                    f"「{case}」が {count} 回入力されています",
                    "計算書には同じ名前の節が複数現れます。意図したものか"
                    "確認してください",
                )
            )
    for load in loads:
        if load.v <= 0:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"鉛直力({load.case.value})",
                    f"V = {load.v:.1f} kN で、押込み側の荷重がありません",
                    "符号(下向き正)を確認してください",
                )
            )
    return issues


def validate_inputs(
    pile: PileSpec,
    arrangement: PileArrangement,
    footing: Footing,
    profile: SoilProfile,
    loads: list[FootingLoads],
    material: MaterialSpec | None = None,
) -> list[ValidationIssue]:
    """入力の妥当性を検査し、問題の一覧を返す。

    エラーを含む場合でも例外は送出しない(呼び出し側が判断する)。
    :func:`raise_on_error` を併用すると、エラーがあるときだけ送出できる。
    """
    issues: list[ValidationIssue] = []
    issues += _check_arrangement(pile, arrangement, footing)
    issues += _check_ground(pile, footing, profile)
    issues += _check_loads(loads)
    if material is not None:
        issues += _check_section(pile, material)
    return issues


def raise_on_error(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """エラーがあれば送出し、無ければ警告のみを返す。"""
    errors = [i for i in issues if i.is_error]
    if errors:
        raise InvalidInputError(errors)
    return [i for i in issues if not i.is_error]
