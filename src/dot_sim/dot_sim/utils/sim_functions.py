from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from geometry_msgs.msg import Point, TransformStamped, Twist
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
import rclpy
import pygame

Color = Tuple[int, int, int]
Point2D = Tuple[float, float]

def _as_color(value: Optional[Sequence[int]], fallback: Color) -> Color:
    if isinstance(value, Sequence) and len(value) >= 3:
        try:
            r, g, b = (max(0, min(255, int(v))) for v in value[:3])
            return int(r), int(g), int(b)
        except (TypeError, ValueError):
            pass
    return fallback


def _as_point2d(value: Optional[Sequence[float]], fallback: Point2D) -> Point2D:
    if isinstance(value, Sequence) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            pass
    return fallback


def _as_positive_float(value: Optional[float], fallback: float) -> float:
    try:
        v = float(value)
        return fallback if v <= 0 else v
    except (TypeError, ValueError):
        return fallback


def _point_in_polygon(point: Point2D, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting to determine if a point lies inside a polygon."""
    x, y = point
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _distance_point_to_segment(point: Point2D, a: Sequence[float], b: Sequence[float]) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    if vx == 0 and vy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * vx
    closest_y = ay + t * vy
    return math.hypot(px - closest_x, py - closest_y)


def _distance_point_to_polygon(point: Point2D, polygon: Sequence[Sequence[float]]) -> float:
    if not polygon:
        return float("inf")
    if len(polygon) == 1:
        return math.hypot(point[0] - polygon[0][0], point[1] - polygon[0][1])
    min_dist = float("inf")
    for idx in range(len(polygon)):
        a = polygon[idx]
        b = polygon[(idx + 1) % len(polygon)]
        dist = _distance_point_to_segment(point, a, b)
        if dist < min_dist:
            min_dist = dist
    return min_dist


@dataclass
class WindowSettings:
    width: Optional[int] = None
    height: Optional[int] = None
    width_ratio: float = 0.6
    height_ratio: float = 0.6
    min_width: int = 800
    min_height: int = 600
    resizable: bool = True
    fullscreen: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WindowSettings":
        settings = WindowSettings()
        if not isinstance(data, dict):
            return settings
        settings.width = data.get("width")
        settings.height = data.get("height")
        settings.width_ratio = float(data.get("width_ratio", settings.width_ratio))
        settings.height_ratio = float(data.get("height_ratio", settings.height_ratio))
        settings.min_width = int(data.get("min_width", settings.min_width))
        settings.min_height = int(data.get("min_height", settings.min_height))
        settings.resizable = bool(data.get("resizable", settings.resizable))
        settings.fullscreen = bool(data.get("fullscreen", settings.fullscreen))
        return settings


@dataclass
class RenderConfig:
    background_color: Color = (0, 179, 60)
    agent_color: Color = (0, 255, 0)
    alert_color: Color = (255, 0, 0)
    area_outline_color: Color = (255, 0, 0)
    drawing_region_color: Color = (0, 255, 255)
    waypoint_label_color: Color = (255, 255, 255)
    window: WindowSettings = field(default_factory=WindowSettings)
    pixels_per_meter: float = 50.0
    world_width_m: Optional[float] = None
    world_height_m: Optional[float] = None
    world_center_ratio: Point2D = (0.5, 0.5)
    font_name: str = "Comic Sans"
    font_scale: float = 0.02  # multiply by window min dimension

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RenderConfig":
        render = RenderConfig()
        if not isinstance(data, dict):
            return render

        render.background_color = _as_color(data.get("background_color"), render.background_color)
        render.agent_color = _as_color(data.get("agent_color"), render.agent_color)
        render.alert_color = _as_color(data.get("alert_color"), render.alert_color)
        render.area_outline_color = _as_color(data.get("area_outline_color"), render.area_outline_color)
        render.drawing_region_color = _as_color(data.get("drawing_region_color"), render.drawing_region_color)
        render.waypoint_label_color = _as_color(data.get("waypoint_label_color"), render.waypoint_label_color)
        render.window = WindowSettings.from_dict(data.get("window", {}))
        render.pixels_per_meter = _as_positive_float(data.get("pixels_per_meter"), render.pixels_per_meter)
        render.world_width_m = data.get("world_width_m")
        render.world_height_m = data.get("world_height_m")
        render.world_center_ratio = _as_point2d(data.get("world_center_ratio"), render.world_center_ratio)
        render.font_name = data.get("font_name", render.font_name)
        render.font_scale = _as_positive_float(data.get("font_scale"), render.font_scale)
        return render


@dataclass
class CollisionServiceConfig:
    enabled: bool = False
    name: str = "/sim/collision_event"
    type: str = "std_srvs/srv/Trigger"
    timeout_sec: float = 2.0
    payload_field: Optional[str] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CollisionServiceConfig":
        cfg = CollisionServiceConfig()
        if not isinstance(data, dict):
            return cfg
        cfg.enabled = bool(data.get("enabled", cfg.enabled))
        cfg.name = data.get("name", cfg.name)
        cfg.type = data.get("type", cfg.type)
        cfg.timeout_sec = _as_positive_float(data.get("timeout_sec"), cfg.timeout_sec)
        cfg.payload_field = data.get("payload_field", cfg.payload_field)
        return cfg


@dataclass
class CollisionConfig:
    waypoint_radius: float = 0.5
    polygon_margin: float = 0.0
    debounce_sec: float = 1.0
    service: CollisionServiceConfig = field(default_factory=CollisionServiceConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CollisionConfig":
        cfg = CollisionConfig()
        if not isinstance(data, dict):
            return cfg
        cfg.waypoint_radius = _as_positive_float(data.get("waypoint_radius"), cfg.waypoint_radius)
        cfg.polygon_margin = float(data.get("polygon_margin", cfg.polygon_margin))
        cfg.debounce_sec = _as_positive_float(data.get("debounce_sec"), cfg.debounce_sec)
        cfg.service = CollisionServiceConfig.from_dict(data.get("service", {}))
        return cfg


@dataclass
class SimConfig:
    agent_start: Point2D = (0.0, 0.0)
    agents: int = 1
    named_areas: List[Dict[str, Any]] = field(default_factory=list)
    render: RenderConfig = field(default_factory=RenderConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    update_hz: float = 20.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SimConfig":
        cfg = SimConfig()
        if not isinstance(data, dict):
            return cfg

        cfg.agent_start = _as_point2d(data.get("agent_start"), cfg.agent_start)
        cfg.agents = data.get("agents")
        areas = data.get("named_areas", cfg.named_areas)
        if isinstance(areas, list):
            cfg.named_areas = []
            for a in areas:
                if isinstance(a, dict):
                    area = copy.deepcopy(a)
                    if "color" in area:
                        area["color"] = _as_color(area.get("color"), (255, 0, 0))
                    cfg.named_areas.append(area)
        cfg.render = RenderConfig.from_dict(data.get("render", {}))
        cfg.collision = CollisionConfig.from_dict(data.get("collision", {}))
        cfg.update_hz = _as_positive_float(data.get("update_hz"), cfg.update_hz)
        return cfg


@dataclass
class CollisionEvent:
    kind: str  # "area" or "waypoint"
    name: str
    position: Point2D
    metadata: Dict[str, Any] = field(default_factory=dict)


class Viewport:
    """Utility for world<->screen coordinate conversions that adapts to window size."""

    def __init__(
        self,
        width: int,
        height: int,
        pixels_per_meter: float,
        world_center_ratio: Point2D,
    ) -> None:
        self.update_dimensions(width, height)
        self.pixels_per_meter = pixels_per_meter
        self.world_center_ratio = world_center_ratio

    def update_dimensions(self, width: int, height: int) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    @property
    def origin_px(self) -> Tuple[int, int]:
        ox = int(self.width * self.world_center_ratio[0])
        oy = int(self.height * self.world_center_ratio[1])
        return ox, oy

    def world_to_screen(self, world_xy: Sequence[float]) -> Tuple[int, int]:
        ox, oy = self.origin_px
        return (
            int(round(ox + world_xy[0] * self.pixels_per_meter)),
            int(round(oy + world_xy[1] * self.pixels_per_meter)),
        )

    def screen_to_world(self, pixel_xy: Sequence[float]) -> Tuple[float, float]:
        ox, oy = self.origin_px
        return (
            (float(pixel_xy[0]) - ox) / self.pixels_per_meter,
            (float(pixel_xy[1]) - oy) / self.pixels_per_meter,
        )
