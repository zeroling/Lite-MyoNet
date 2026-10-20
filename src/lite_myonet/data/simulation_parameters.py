from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


SIMULATION_GRID_STEP = Decimal("0.001")
SIMULATION_GRID_TOLERANCE = Decimal("0.000005")


@dataclass(frozen=True)
class CanonicalSimulationParameter:
    """文件名仿真参数在千分位理论网格上的无歧义表示。"""

    raw: str
    milli: int
    canonical: str
    residual: str


def _format_decimal(value):
    return format(value, "f")


def canonicalize_simulation_parameter(raw_value):
    """把文件名中的浮点尾差恢复到最近的 0.001 仿真网格点。

    只有与最近网格点的距离不超过 5e-6 时才允许恢复。这个阈值覆盖
    当前数据中观测到的最大 4e-6 尾差，同时远小于相邻网格点 1e-3
    的间隔，因而不会把两个真实相邻参数合并。
    """

    raw = str(raw_value).strip()
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"仿真参数不是有效十进制数：{raw!r}。") from error
    if not value.is_finite():
        raise ValueError(f"仿真参数必须是有限十进制数：{raw!r}。")

    milli = int((value / SIMULATION_GRID_STEP).to_integral_value(rounding=ROUND_HALF_UP))
    canonical_value = Decimal(milli) * SIMULATION_GRID_STEP
    residual = value - canonical_value
    if abs(residual) > SIMULATION_GRID_TOLERANCE:
        raise ValueError(
            f"仿真参数 {raw!r} 距离最近的千分位仿真网格点 "
            f"{canonical_value:.3f} 为 {abs(residual)}，超过允许的 "
            f"{SIMULATION_GRID_TOLERANCE}。"
        )

    return CanonicalSimulationParameter(
        raw=raw,
        milli=milli,
        canonical=f"{canonical_value:.3f}",
        residual=_format_decimal(residual),
    )
