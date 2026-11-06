from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.engine.backend.space import Point2D


class BaseObstacle(ABC):
    """Abstract base class for obstacles in the 2D Cartesian plane."""

    def __init__(self, name: str = "obstacle", color: str = "#6B7280"):
        self.name = name
        self.color = color

    @abstractmethod
    def contains_point(self, p: Point2D) -> bool:
        """Check if a given point lies inside the obstacle."""
        pass

    @abstractmethod
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) bounding box."""
        pass


@dataclass
class CircleObstacle(BaseObstacle):
    center: Point2D
    radius: float
    name: str = "circle_obstacle"
    color: str = "#4B5563"

    def contains_point(self, p: Point2D) -> bool:
        return self.center.distance_to(p) <= self.radius

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return (
            self.center.x - self.radius,
            self.center.x + self.radius,
            self.center.y - self.radius,
            self.center.y + self.radius,
        )


@dataclass
class BoxObstacle(BaseObstacle):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    name: str = "box_obstacle"
    color: str = "#4B5563"

    def contains_point(self, p: Point2D) -> bool:
        return (self.x_min <= p.x <= self.x_max) and (self.y_min <= p.y <= self.y_max)

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


class PolygonObstacle(BaseObstacle):
    def __init__(self, vertices: List[Point2D], name: str = "polygon_obstacle", color: str = "#4B5563"):
        super().__init__(name=name, color=color)
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        self.vertices = vertices

    def contains_point(self, p: Point2D) -> bool:
        """Ray-casting algorithm to determine point-in-polygon."""
        inside = False
        n = len(self.vertices)
        p1 = self.vertices[0]
        for i in range(1, n + 1):
            p2 = self.vertices[i % n]
            if p.y > min(p1.y, p2.y):
                if p.y <= max(p1.y, p2.y):
                    if p1.y != p2.y:
                        x_inters = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                        if p1.x == p2.x or p.x <= x_inters:
                            inside = not inside
            p1 = p2
        return inside

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return (min(xs), max(xs), min(ys), max(ys))


class TerrainObstacle(BaseObstacle):
    """
    Bottom terrain boundary defined by a polyline ground profile.
    Everything below the polyline is solid obstacle.
    """
    def __init__(self, points: List[Point2D], name: str = "terrain", color: str = "#374151"):
        super().__init__(name=name, color=color)
        if len(points) < 2:
            raise ValueError("Terrain must have at least 2 profile points")
        self.points = sorted(points, key=lambda p: p.x)

    def get_ground_y(self, x: float) -> Optional[float]:
        """Interpolate ground height at position x."""
        if x < self.points[0].x or x > self.points[-1].x:
            return None
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            if p1.x <= x <= p2.x:
                if p2.x == p1.x:
                    return max(p1.y, p2.y)
                t = (x - p1.x) / (p2.x - p1.x)
                return p1.y + t * (p2.y - p1.y)
        return None

    def contains_point(self, p: Point2D) -> bool:
        ground_y = self.get_ground_y(p.x)
        if ground_y is None:
            return False
        return p.y <= ground_y

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), max(xs), -100.0, max(ys))


# ---------------------------------------------------------------------------
# JSON (de)serialization shared by the scenario loader, UI and tools
# ---------------------------------------------------------------------------

def _bbox_dict(obs: BaseObstacle) -> Dict[str, float]:
    x_min, x_max, y_min, y_max = obs.get_bounding_box()
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def obstacle_to_dict(obs: BaseObstacle, obs_id: str) -> Dict[str, Any]:
    """
    Serialize an obstacle to the contract format. Box and circle are fully described
    by their bounding box; polygon and terrain also carry their exact geometry,
    since a bounding box alone would misdescribe them.
    """
    if isinstance(obs, BoxObstacle):
        obs_type = "box"
    elif isinstance(obs, CircleObstacle):
        obs_type = "circle"
    elif isinstance(obs, PolygonObstacle):
        obs_type = "polygon"
    elif isinstance(obs, TerrainObstacle):
        obs_type = "terrain"
    else:
        raise ValueError(f"Cannot serialize obstacle of type {type(obs).__name__}")

    data: Dict[str, Any] = {
        "id": obs_id,
        "type": obs_type,
        "name": obs.name,
        "bounding_box": _bbox_dict(obs),
    }
    if isinstance(obs, PolygonObstacle):
        data["vertices"] = [{"x": v.x, "y": v.y} for v in obs.vertices]
    elif isinstance(obs, TerrainObstacle):
        data["points"] = [{"x": p.x, "y": p.y} for p in obs.points]
    return data


def obstacle_from_dict(data: Dict[str, Any]) -> BaseObstacle:
    """Reconstruct an obstacle from its contract dict. Unknown types raise instead of being dropped."""
    obs_type = data.get("type", "box").lower()
    name = data.get("name", "obstacle")
    bbox = data.get("bounding_box", {})

    if obs_type == "box":
        return BoxObstacle(
            x_min=bbox["x_min"], x_max=bbox["x_max"],
            y_min=bbox["y_min"], y_max=bbox["y_max"],
            name=name,
        )
    if obs_type == "circle":
        if "center" in data and "radius" in data:
            center = Point2D(data["center"]["x"], data["center"]["y"])
            radius = data["radius"]
        else:
            center = Point2D((bbox["x_min"] + bbox["x_max"]) / 2.0, (bbox["y_min"] + bbox["y_max"]) / 2.0)
            radius = (bbox["x_max"] - bbox["x_min"]) / 2.0
        return CircleObstacle(center=center, radius=radius, name=name)
    if obs_type == "polygon":
        if "vertices" not in data:
            raise ValueError(f"Polygon obstacle '{name}' is missing 'vertices'")
        return PolygonObstacle([Point2D(v["x"], v["y"]) for v in data["vertices"]], name=name)
    if obs_type == "terrain":
        if "points" not in data:
            raise ValueError(f"Terrain obstacle '{name}' is missing 'points'")
        return TerrainObstacle([Point2D(p["x"], p["y"]) for p in data["points"]], name=name)

    raise ValueError(f"Unknown obstacle type '{obs_type}' for obstacle '{name}'")
