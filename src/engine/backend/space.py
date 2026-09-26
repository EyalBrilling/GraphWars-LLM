from dataclasses import dataclass
from typing import Tuple

@dataclass
class Point2D:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Point2D") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class CoordinateBounds:
    x_min: float = -20.0
    x_max: float = 20.0
    y_min: float = -15.0
    y_max: float = 15.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def contains(self, p: Point2D) -> bool:
        return self.x_min <= p.x <= self.x_max and self.y_min <= p.y <= self.y_max


@dataclass
class Shooter:
    name: str
    position: Point2D
    radius: float = 0.6
    color: str = "#3B82F6"  # Blue


@dataclass
class Target:
    id: str
    position: Point2D
    radius: float = 0.8
    is_alive: bool = True
    color: str = "#EF4444"  # Red
