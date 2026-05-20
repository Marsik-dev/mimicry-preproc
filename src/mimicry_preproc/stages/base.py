"""Stage protocol and result wrapper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class StageResult(Generic[T]):
    data: T
    dropped_count: int = 0
    warnings: list[str] = field(default_factory=list)
