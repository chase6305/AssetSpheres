"""Read saved sphere results without loading geometry or export backends."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .model import FitResult


def read_result(path: str | Path) -> FitResult:
    """Reject nonfinite JSON numbers before validating the saved result schema."""

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Result JSON contains a non-finite number: {value}")

    def parse_finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            reject_nonfinite(value)
        return number

    return FitResult.from_dict(
        json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite,
            parse_float=parse_finite_float,
        )
    )
