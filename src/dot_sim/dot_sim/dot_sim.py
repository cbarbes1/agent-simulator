
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pygame
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from .utils.sim_functions import _as_color, _distance_point_to_polygon, _point_in_polygon
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, TransformStamped, Twist, Quaternion
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_service
from std_srvs.srv import SetBool
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from .utils.sim_functions import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import threading, queue, time
import subprocess
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry



class DotSim(Node):
    def __init__(self):
        super().__init__("dot_sim")

        # Get package share path
        pkg_share = Path(get_package_share_directory("dot_sim"))
        config_file = pkg_share / "config" / "default_config.yaml"

        # Load config
        config = yaml.safe_load(config_file.read_text()) or {}
        self.config = SimConfig.from_dict(config)

        self.pygame_is_initialized = False

        self.num_agents = self.config.agents
        self.agent_pos: List[List[float]] = [
            list(self.config.agent_start) for _ in range(self.num_agents)
        ]
        self.agent_theta: List[float] = [1 for _ in range(self.num_agents)]
        
        self.areas = copy.deepcopy(self.config.named_areas)
        self.config_areas = copy.deepcopy(self.config.named_areas)
        self.which_agent = 0

        self.drawing_region = []   # temp list of points being placed
        self.region_counter = len(self.areas)

        self.bridge = CvBridge()
        print("Loaded area:", self.areas)

        self.reentrant_group = ReentrantCallbackGroup()
        self.names_areas_pub = self.create_publisher(MarkerArray, f"/named_areas", 10)


        # ROS2 pubs/subs
        self.odom_pub = []
        
        
        for i in range(self.num_agents):
            self.agent_pos[i][1]+=i
            self.odom_pub.append(self.create_publisher(
                Odometry, 
                f"agent_{i}/odometry", 
                10))
            self.create_subscription(
                Twist,
                f"/agent_{i}/cmd_vel",
                lambda msg, idx=i: self.cmd_cb(msg, idx),
                10,
                callback_group=self.reentrant_group,
            )

        self.br = TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)

        for i in range(self.num_agents):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "world"
            t.child_frame_id = f"agent_{i}/map"
            t.transform.translation.x = self.agent_pos[i][0]
            t.transform.translation.y = self.agent_pos[i][1]
            t.transform.rotation.w = 1.0
            self.static_br.sendTransform(t)


        self.create_service(SetBool, "/set_color", self.set_color_cb, callback_group=self.reentrant_group)

        self.default_agent_color = self.config.render.agent_color
        self.color = self.default_agent_color

        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0

        # Timer
        update_period = 1.0 / self.config.update_hz if self.config.update_hz else 0.05
        self.create_timer(0.05, self._check_collisions, callback_group=self.reentrant_group)
        # start pygame loop on a *dedicated thread*
        self.ui_thread = threading.Thread(target=self.update, daemon=True)
        self.ui_thread.start()

        # Queue for images to publish
        self.image_queue: "queue.Queue[Tuple[int, Any]]" = queue.Queue(maxsize=8)
        threading.Thread(target=self.image_worker, daemon=True).start()

        self.image_pub_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image_publishers: List[Any] = [
            self.create_publisher(Image, f"/agent_{i}/image", self.image_pub_qos)
            for i in range(self.num_agents)
        ]

        self.media_mode = os.getenv("DOT_SIM_MEDIA_MODE", "image").strip().lower()
        if self.media_mode not in {"image", "bag"}:
            self.get_logger().warn(
                f"Unsupported media mode '{self.media_mode}', defaulting to 'image'"
            )
            self.media_mode = "image"

        self.image_dir = self._resolve_image_dir(pkg_share)
        self.image_cache: Dict[str, Optional[Any]] = {}
        self.missing_image_names: set[str] = set()
        self.missing_bag_names: set[str] = set()
        self.last_collision_emit: Dict[str, float] = {}
        self.bag_dir = self._resolve_bag_dir(pkg_share)
        self.bag_cache: Dict[str, Optional[Path]] = {}
        self.bag_processes: Dict[int, subprocess.Popen] = {}
        self.bag_lock = threading.Lock()
        self.get_logger().info(
            f"Media mode: {self.media_mode}; images dir: {self.image_dir}; bags dir: {self.bag_dir}"
        )
        

    def initialize_pygame(self):
        # Visualization
        pygame.init()
        pygame.display.set_caption("Agent Dot Sim")
        self.info_object = pygame.display.Info()

        self.screen_width = self.info_object.current_w or 1024
        self.screen_height = self.info_object.current_h or 920

        window_cfg = self.config.render.window
        if window_cfg.width and window_cfg.height:
            self.window_width = int(window_cfg.width)
            self.window_height = int(window_cfg.height)
        else:
            self.window_width = max(
                window_cfg.min_width, int(self.screen_width * window_cfg.width_ratio)
            )
            self.window_height = max(
                window_cfg.min_height, int(self.screen_height * window_cfg.height_ratio)
            )

        self.flags = 0
        if window_cfg.resizable:
            self.flags |= pygame.RESIZABLE
        if window_cfg.fullscreen:
            self.flags |= pygame.FULLSCREEN

        self.window_flags = self.flags
        pixels_per_meter = self._compute_pixels_per_meter(self.window_width, self.window_height)
        self.viewport = Viewport(
            self.window_width,
            self.window_height,
            pixels_per_meter,
            self.config.render.world_center_ratio,
        )

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), self.flags)
        self.screen.fill(self.config.render.background_color)

        self._update_font()
        self.clock = pygame.time.Clock()

    def _compute_pixels_per_meter(self, width: int, height: int) -> float:
        ppm_candidates = [max(1e-6, self.config.render.pixels_per_meter)]
        if self.config.render.world_width_m:
            ppm_candidates.append(width / float(self.config.render.world_width_m))
        if self.config.render.world_height_m:
            ppm_candidates.append(height / float(self.config.render.world_height_m))
        return max(1e-6, min(ppm_candidates))

    def _update_font(self) -> None:
        font_size = max(
            12,
            int(self.config.render.font_scale * min(self.window_width, self.window_height)),
        )
        self.system_font = pygame.font.SysFont(self.config.render.font_name, font_size)

    def _handle_resize(self, width: int, height: int) -> None:
        self.window_width = max(1, int(width))
        self.window_height = max(1, int(height))
        self.viewport.update_dimensions(self.window_width, self.window_height)
        self.viewport.pixels_per_meter = self._compute_pixels_per_meter(
            self.window_width, self.window_height
        )
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), self.window_flags
        )
        self.screen.fill(self.config.render.background_color)
        self._update_font()

    def _get_area_color(self, area: Dict[str, Any]) -> Color:
        return _as_color(area.get("color"), self.config.render.area_outline_color)

    def _get_waypoint_color(self, waypoint: Dict[str, Any]) -> Color:
        return _as_color(waypoint.get("color"), self.config.render.area_outline_color)

    def _store_new_region(self, name: str) -> None:
        if not self.drawing_region:
            return
        new_region = {
            "name": name,
            "points": copy.deepcopy(self.drawing_region),
            "color": self.config.render.area_outline_color,
        }
        self.areas.append(new_region)
        self.region_counter += 1
        self.get_logger().info(f"Closed region {new_region['name']}")

    def _prompt_region_name(self) -> Optional[str]:
        default_name = f"region{self.region_counter + 1}"
        colors = {
            "border": pygame.Color("black"),
            "background": pygame.Color("white"),
            "input": pygame.Color("lightgray"),
            "active": pygame.Color("dodgerblue2"),
            "text": pygame.Color("black"),
            "button": pygame.Color("gray"),
        }

        def layout() -> Tuple[pygame.Rect, pygame.Rect, pygame.Rect, Tuple[int, int], int]:
            popup_width = max(260, int(self.window_width * 0.35))
            popup_height = max(160, int(self.window_height * 0.25))
            popup_rect = pygame.Rect(
                (self.window_width - popup_width) // 2,
                (self.window_height - popup_height) // 2,
                popup_width,
                popup_height,
            )
            input_rect = pygame.Rect(
                popup_rect.x + 20,
                popup_rect.y + popup_height // 2 - 20,
                popup_width - 40,
                36,
            )
            enter_rect = pygame.Rect(
                popup_rect.right - 90,
                popup_rect.bottom - 50,
                70,
                32,
            )
            close_center = (popup_rect.right - 20, popup_rect.y + 20)
            close_radius = max(12, int(min(popup_width, popup_height) * 0.05))
            return popup_rect, input_rect, enter_rect, close_center, close_radius

        user_text = ""
        input_active = True
        cursor_visible = True
        cursor_timer = 0
        cursor_blink_time = 500  # ms
        clock = pygame.time.Clock()
        backdrop = self.screen.copy()

        while input_active:
            popup_rect, input_rect, enter_rect, close_center, close_radius = layout()

            font_size = max(24, self.system_font.get_height())
            font = pygame.font.Font(None, font_size)

            dt = clock.tick(30)
            cursor_timer += dt
            if cursor_timer >= cursor_blink_time:
                cursor_visible = not cursor_visible
                cursor_timer = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rclpy.shutdown()
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    backdrop = self.screen.copy()
                    continue
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        input_active = False
                    elif event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    elif len(user_text) < 24 and event.unicode.isprintable():
                        user_text += event.unicode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (event.pos[0] - close_center[0]) ** 2 + (event.pos[1] - close_center[1]) ** 2 <= close_radius ** 2:
                        return None
                    if enter_rect.collidepoint(event.pos):
                        input_active = False

            # Draw popup overlay
            self.screen.blit(backdrop, (0, 0))
            pygame.draw.rect(self.screen, colors["background"], popup_rect)
            pygame.draw.rect(self.screen, colors["border"], popup_rect, 2, border_radius=6)

            # Title bar
            title_surface = font.render("Enter Region Name", True, colors["text"])
            self.screen.blit(title_surface, (popup_rect.x + 20, popup_rect.y + 10))

            # Close button
            pygame.draw.circle(self.screen, colors["border"], close_center, close_radius)
            x_surface = font.render("X", True, colors["background"])
            x_rect = x_surface.get_rect(center=close_center)
            self.screen.blit(x_surface, x_rect)

            # Input box
            pygame.draw.rect(self.screen, colors["input"], input_rect)
            pygame.draw.rect(self.screen, colors["active"], input_rect, 2)
            text_surface = font.render(user_text, True, colors["text"])
            self.screen.blit(text_surface, (input_rect.x + 8, input_rect.y + 4))

            if cursor_visible and input_active:
                cursor_x = input_rect.x + 8 + text_surface.get_width() + 2
                cursor_y = input_rect.y + 4
                pygame.draw.line(
                    self.screen,
                    colors["text"],
                    (cursor_x, cursor_y),
                    (cursor_x, cursor_y + font.get_height() - 4),
                    2,
                )

            # Enter button
            pygame.draw.rect(self.screen, colors["button"], enter_rect, border_radius=4)
            enter_surface = font.render("OK", True, colors["text"])
            enter_rect_text = enter_surface.get_rect(center=enter_rect.center)
            self.screen.blit(enter_surface, enter_rect_text)

            pygame.display.flip()

        return user_text.strip() or default_name

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _resolve_image_dir(self, pkg_share: Path) -> Path:
        env_dir = os.getenv("DOT_SIM_IMAGE_DIR")
        workspace_dir = Path(os.getcwd()) / "src" / "dot_sim" / "images"
        legacy_package_dir = Path(os.getcwd()) / "src" / "dot_sim" / "dot_sim" / "images"
        fallback_package_dir = Path(os.getcwd()) / "src" / "dot_sim" / "dot_sim"
        share_dir = pkg_share / "images"

        candidates: List[Path] = []
        if env_dir:
            candidates.append(Path(env_dir).expanduser())

        # Prefer editable workspace assets so new images are picked up without reinstalling.
        candidates.extend(
            [
                workspace_dir,
                legacy_package_dir,
                fallback_package_dir,
                share_dir,
            ]
        )

        for candidate in candidates:
            if candidate and candidate.is_dir():
                return candidate

        fallback = Path(os.getcwd())
        self.get_logger().warn(
            f"No image directory found in {', '.join(str(c) for c in candidates)}; "
            f"falling back to {fallback}"
        )
        return fallback

    def _resolve_bag_dir(self, pkg_share: Path) -> Path:
        env_dir = os.getenv("DOT_SIM_BAG_DIR")
        workspace_dir = Path(os.getcwd()) / "src" / "dot_sim" / "bags"
        legacy_package_dir = Path(os.getcwd()) / "src" / "dot_sim" / "dot_sim" / "bags"
        share_dir = pkg_share / "bags"

        candidates: List[Path] = []
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.extend(
            [
                workspace_dir,
                legacy_package_dir,
                share_dir,
            ]
        )

        for candidate in candidates:
            if candidate and candidate.is_dir():
                return candidate

        fallback = workspace_dir
        self.get_logger().warn(
            f"No bag directory found in {', '.join(str(c) for c in candidates)}; "
            f"falling back to {fallback}"
        )
        return fallback

    @staticmethod
    def _sanitize_region_stem(region_name: str) -> List[str]:
        base = region_name.strip()
        if not base:
            return []
        variants = [base, base.replace(" ", "_")]
        variants.extend([v.lower() for v in variants])
        ordered: List[str] = []
        for variant in variants:
            if variant and variant not in ordered:
                ordered.append(variant)
        return ordered

    def _candidate_image_paths(self, region_name: str) -> List[Path]:
        stems = self._sanitize_region_stem(region_name)
        extensions = [".jpg", ".jpeg", ".png"]
        candidates: List[Path] = []
        for stem in stems:
            for extension in extensions:
                candidate = self.image_dir / f"{stem}{extension}"
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def _get_image_for_region(self, region_name: str) -> Optional[Any]:
        key = region_name.strip()
        if not key:
            return None
        if key in self.image_cache:
            return self.image_cache[key]

        for path in self._candidate_image_paths(key):
            if not path.exists():
                continue
            image = cv2.imread(str(path))
            if image is not None:
                self.image_cache[key] = image
                self.get_logger().info(f"Loaded image for region '{key}' from {path}")
                return image
            self.get_logger().error(f"Failed to read image file for region '{key}': {path}")

        if key not in self.missing_image_names:
            self.get_logger().warn(
                f"No image found for region '{key}'. Expected one of: "
                + ", ".join(str(p.name) for p in self._candidate_image_paths(key))
            )
            self.missing_image_names.add(key)
        self.image_cache[key] = None
        return None

    def _candidate_bag_paths(self, region_name: str) -> List[Path]:
        stems = self._sanitize_region_stem(region_name)
        candidates: List[Path] = []
        for stem in stems:
            dir_candidate = self.bag_dir / stem
            file_candidate = self.bag_dir / f"{stem}.db3"
            if dir_candidate not in candidates:
                candidates.append(dir_candidate)
            if file_candidate not in candidates:
                candidates.append(file_candidate)
        return candidates

    def _get_bag_for_region(self, region_name: str) -> Optional[Path]:
        key = region_name.strip()
        if not key:
            return None
        if key in self.bag_cache:
            return self.bag_cache[key]

        for candidate in self._candidate_bag_paths(key):
            if candidate.is_dir():
                metadata = candidate / "metadata.yaml"
                if metadata.exists() or any(candidate.glob("*.db3")):
                    self.bag_cache[key] = candidate
                    self.get_logger().info(f"Found bag directory for region '{key}': {candidate}")
                    return candidate
            elif candidate.exists():
                self.bag_cache[key] = candidate
                self.get_logger().info(f"Found bag file for region '{key}': {candidate}")
                return candidate

        if key not in self.missing_bag_names:
            self.get_logger().warn(
                f"No bag found for region '{key}'. Expected one of: "
                + ", ".join(str(p) for p in self._candidate_bag_paths(key))
            )
            self.missing_bag_names.add(key)
        self.bag_cache[key] = None
        return None

    def _should_emit_collision(self, agent_idx: int, region_name: str, now: float) -> bool:
        debounce = float(self.config.collision.debounce_sec or 0.0)
        key = f"{agent_idx}:{region_name}"
        last_emit = self.last_collision_emit.get(key)
        if last_emit is not None and now - last_emit < debounce:
            return False
        self.last_collision_emit[key] = now
        return True

    def _stop_bag_process(self, agent_idx: int) -> None:
        with self.bag_lock:
            proc = self.bag_processes.pop(agent_idx, None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.get_logger().warn(f"Bag play for agent_{agent_idx} did not terminate; killing")
                proc.kill()
            except Exception as exc:
                self.get_logger().error(f"Error stopping bag for agent_{agent_idx}: {exc}")

    def _stop_all_bag_processes(self) -> None:
        with self.bag_lock:
            agent_indices = list(self.bag_processes.keys())
        for agent_idx in agent_indices:
            self._stop_bag_process(agent_idx)

    def _play_bag_for_region(self, agent_idx: int, region_name: str) -> bool:
        bag_path = self._get_bag_for_region(region_name)
        if bag_path is None:
            return False

        self._stop_bag_process(agent_idx)

        cmd = ["ros2", "bag", "play", str(bag_path)]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.get_logger().error("Failed to start 'ros2 bag play'; is ROS 2 sourced?")
            return False
        except Exception as exc:
            self.get_logger().error(f"Failed to play bag '{bag_path}': {exc}")
            return False

        with self.bag_lock:
            self.bag_processes[agent_idx] = proc
        self.get_logger().info(f"Playing bag for agent_{agent_idx} from {bag_path}")
        return True

    def _publish_region_image(self, agent_idx: int, region_name: str) -> None:
        image = self._get_image_for_region(region_name)
        if image is None:
            return
        try:
            self.image_queue.put_nowait((agent_idx, image))
            self.get_logger().info(f"Queued image for agent_{agent_idx} in region '{region_name}'")
        except queue.Full:
            self.get_logger().warn("Image queue full; dropping image publish request")

    def _emit_region_media(self, agent_idx: int, region_name: str) -> None:
        if self.media_mode == "bag":
            if self._play_bag_for_region(agent_idx, region_name):
                return
        self._publish_region_image(agent_idx, region_name)
    
    def image_worker(self):
        while rclpy.ok():
            try:
                agent_idx, img = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not (0 <= agent_idx < len(self.image_publishers)):
                self.get_logger().warn(f"Dropping image for invalid agent index {agent_idx}")
                continue

            try:
                msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            except Exception as exc:
                self.get_logger().error(f"Failed to convert image for agent_{agent_idx}: {exc}")
                continue

            self.image_publishers[agent_idx].publish(msg)

    def _check_collisions(self) -> None:
        if not self.areas:
            return

        areas = copy.deepcopy(self.areas)
        now = monotonic()
        for agent_idx, agent_pos in enumerate(self.agent_pos):
            position = (float(agent_pos[0]), float(agent_pos[1]))
            for area in areas:
                points = area.get("points", [])
                region_name = str(area.get("name", "")).strip()
                if not points:
                    continue
                if not region_name:
                    continue

                if len(points) == 1:
                    distance = math.hypot(
                        position[0] - float(points[0][0]),
                        position[1] - float(points[0][1]),
                    )
                    if distance <= self.config.collision.waypoint_radius:
                        if self._should_emit_collision(agent_idx, region_name, now):
                            self._emit_region_media(agent_idx, region_name)
                elif len(points) >= 3:
                    inside = _point_in_polygon(position, points)
                    distance_to_edge = _distance_point_to_polygon(position, points)
                    if inside or distance_to_edge <= self.config.collision.polygon_margin:
                        if self._should_emit_collision(agent_idx, region_name, now):
                            self._emit_region_media(agent_idx, region_name)

    def cmd_cb(self, msg: Twist, agent_idx: int):
        if 0 <= agent_idx < len(self.agent_pos):
            self.agent_pos[agent_idx][0] += msg.linear.x * 0.1
            self.agent_pos[agent_idx][1] += msg.linear.y * 0.1

    def set_color_cb(self, req, res):
        self.color = self.config.render.alert_color if req.data else self.default_agent_color
        res.success = True
        res.message = "Color changed"
        return res

    def publish_waypoints_markers(self):
        ma = MarkerArray()
        for i, wp in enumerate(self.areas):
            if len(wp["points"]) == 1:
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "waypoints"
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD

                m.pose.position.x = wp["points"][0][0]
                m.pose.position.y = wp["points"][0][1]
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0

                m.scale.x = 0.2  # sphere size
                m.scale.y = 0.2
                m.scale.z = 0.2

                wp_color = self._get_waypoint_color(wp)
                m.color.r = wp_color[0] / 255.0
                m.color.g = wp_color[1] / 255.0
                m.color.b = wp_color[2] / 255.0
                m.color.a = 1.0
                m.text = wp["name"]

                ma.markers.append(m)

                # Optional text label above waypoint
                t = Marker()
                t.header.frame_id = "world"
                t.header.stamp = m.header.stamp
                t.ns = "waypoint_labels"
                t.id = 1000 + i
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD

                t.pose.position.x = wp["points"][0][0]
                t.pose.position.y = wp["points"][0][1]
                t.pose.position.z = 0.5
                t.pose.orientation.w = 1.0

                t.scale.z = 0.2  # text height
                label_color = self.config.render.waypoint_label_color
                t.color.r = label_color[0] / 255.0
                t.color.g = label_color[1] / 255.0
                t.color.b = label_color[2] / 255.0
                t.color.a = 1.0
                t.text = wp["name"]

                ma.markers.append(t)

        self.names_areas_pub.publish(ma)


    def publish_tf(self):
        for i in range(self.num_agents):
            
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = f"agent_{i}/map"
            t.child_frame_id = f"agent_{i}/odom"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.rotation.w = 1.0
            self.br.sendTransform(t)

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = f"agent_{i}/odom"
            t.child_frame_id = f"agent_{i}/base_link"
            t.transform.translation.x = self.agent_pos[i][0]
            t.transform.translation.y = self.agent_pos[i][1]
            t.transform.rotation.z = math.sin(self.agent_theta[i] / 2.0)
            t.transform.rotation.w = math.sin(self.agent_theta[i] / 2.0)
            self.br.sendTransform(t)

            """Timer callback to publish simulated odometry."""
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = f"agent_{i}/odom"
            odom.child_frame_id = f"agent_{i}/base_link"

            # --- Position ---
            odom.pose.pose.position.x = self.agent_pos[i][0]
            odom.pose.pose.position.y = self.agent_pos[i][1]
            odom.pose.pose.position.z = 0.0

            # --- Orientation (yaw → quaternion) ---
            qz = math.sin(self.agent_theta[i] * 0.5)
            qw = math.cos(self.agent_theta[i] * 0.5)
            odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

            # (Optional) velocity if you track Δx/Δt
            # odom.twist.twist.linear.x = self.vx
            # odom.twist.twist.angular.z = self.vtheta

            self.odom_pub[i].publish(odom)
            self.get_logger().debug(f"Published odom: x={self.agent_pos[i][0]:.2f}, y={self.agent_pos[i][1]:.2f}, θ={self.agent_theta[i]:.2f}")
        
    
    def publish_areas(self):
        ma = MarkerArray()
        for i, area in enumerate(self.areas):
            if len(area["points"]) > 1:
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = area["name"]
                m.id = i
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.05  # line width
                area_color = self._get_area_color(area)
                m.color.r = area_color[0] / 255.0
                m.color.g = area_color[1] / 255.0
                m.color.b = area_color[2] / 255.0
                m.color.a = 1.0
                
                for p in area["points"]:
                    pt = Point()
                    pt.x, pt.y = p
                    m.points.append(pt)

                # Close loop
                if m.points:
                    m.points.append(m.points[0])

                ma.markers.append(m)
                # Optional text label above waypoint
                t = Marker()
                t.header.frame_id = "world"
                t.header.stamp = m.header.stamp
                t.ns = "waypoint_labels"
                t.id = 1000 + i
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD

                t.pose.position.x = area["points"][0][0]
                t.pose.position.y = area["points"][0][1]
                t.pose.position.z = 0.5
                t.pose.orientation.w = 1.0

                t.scale.z = 0.2  # text height
                label_color = self.config.render.waypoint_label_color
                t.color.r = label_color[0] / 255.0
                t.color.g = label_color[1] / 255.0
                t.color.b = label_color[2] / 255.0
                t.color.a = 1.0
                t.text = area["name"]
                ma.markers.append(t)


        self.names_areas_pub.publish(ma)


    def draw_areas(self):
        for area in self.areas:
            points = area.get("points", [])
            if len(points) >= 2:
                pts = [self.viewport.world_to_screen(p) for p in points]
                pygame.draw.lines(
                    self.screen,
                    self._get_area_color(area),
                    True,
                    pts,
                    2,
                )
                if pts:
                    text_surface = self.system_font.render(
                        area.get("name", ""),
                        True,
                        self.config.render.waypoint_label_color,
                    )
                    tw, th = text_surface.get_size()
                    margin = max(6, int(0.08 * self.viewport.pixels_per_meter))

                    min_x = min(p[0] for p in pts)
                    max_x = max(p[0] for p in pts)
                    min_y = min(p[1] for p in pts)
                    label_x = int((min_x + max_x) / 2 - tw / 2)
                    label_y = int(min_y - margin - th)
                    label_y = max(0, label_y) # keep on-screen
                    self.screen.blit(text_surface, (label_x, label_y))

    def draw_drawing_region(self):
        if len(self.drawing_region) >= 1:
            pts = [self.viewport.world_to_screen(p) for p in self.drawing_region]
            point_radius = max(3, int(0.12 * self.viewport.pixels_per_meter))
            # Draw connecting lines
            if len(pts) > 1:
                pygame.draw.lines(self.screen, self.config.render.drawing_region_color, False, pts, 2)
            # Draw points
            for pt in pts:
                pygame.draw.circle(self.screen, self.config.render.drawing_region_color, pt, point_radius)



    def update(self):
        if not self.pygame_is_initialized:
            self.initialize_pygame()
            self.pygame_is_initialized = True
        while rclpy.ok():    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rclpy.shutdown()
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    continue
                
                elif event.type == pygame.KEYDOWN:
                    # Forward / Backward
                    if event.key in (pygame.K_w, pygame.K_UP):
                        self.cmd_vel_y = -1.0
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        self.cmd_vel_y = 1.0
                    # Left / Right
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        self.cmd_vel_x = -1.0
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        self.cmd_vel_x = 1.0
                    elif event.key == pygame.K_RETURN:  # Enter also finishes
                        if len(self.drawing_region) >= 3:
                            region_name = self._prompt_region_name()
                            if region_name:
                                self._store_new_region(region_name)
                        self.drawing_region.clear()
                    
                    elif event.key == pygame.K_r:  # 🔄 reset
                        self.get_logger().info("Resetting simulation...")
                        # Reset agents to configured start positions
                        self.agent_pos = [
                            list(self.config.agent_start) for _ in range(self.num_agents)
                        ]
                        self.agent_theta = [1 for _ in range(self.num_agents)]
                        self.color = self.default_agent_color
                        self._stop_all_bag_processes()
                        # Reset regions
                        self.areas.clear()
                        self.drawing_region.clear()
                        self.areas = copy.deepcopy(self.config_areas)
                        
                        self.region_counter = len(self.config_areas)
                        # Publish cleared areas
                        self.publish_areas()
                    
                    elif event.key == pygame.K_0:
                        if self.which_agent < self.num_agents-1:
                            self.which_agent+=1
                        else:
                            self.which_agent = 0

                elif event.type == pygame.KEYUP:
                    # Stop motion when key released
                    if event.key in (pygame.K_w, pygame.K_s, pygame.K_UP, pygame.K_DOWN):
                        self.cmd_vel_y = 0.0
                    if event.key in (pygame.K_a, pygame.K_d, pygame.K_LEFT, pygame.K_RIGHT):
                        self.cmd_vel_x = 0.0
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # left click = add point
                        wx, wy = self.viewport.screen_to_world(event.pos)
                        self.drawing_region.append([float(wx), float(wy)])
                        self.get_logger().info(f"Added point ({wx:.2f}, {wy:.2f})")

                    elif event.button == 3:  # right click = finish region
                        if len(self.drawing_region) >= 3:
                            region_name = self._prompt_region_name()
                            if region_name:
                                self._store_new_region(region_name)
                        self.drawing_region.clear()

            # Update position
            
            self.agent_pos[self.which_agent][0] += self.cmd_vel_x * 0.1
            self.agent_pos[self.which_agent][1] += self.cmd_vel_y * 0.1

            # Wrap around screen edges (toroidal space)
            sx, sy = self.viewport.world_to_screen(self.agent_pos[self.which_agent])
            sx %= self.window_width
            sy %= self.window_height
            wrapped = self.viewport.screen_to_world((sx, sy))
            self.agent_pos[self.which_agent][0] = wrapped[0]
            self.agent_pos[self.which_agent][1] = wrapped[1]


            #self._check_collisions()

            pixel_scale = self.viewport.pixels_per_meter
            waypoint_radius = max(6, int(0.25 * pixel_scale))
            agent_radius = max(6, int(0.3 * pixel_scale))

            self.screen.fill(self.config.render.background_color)
            # Draw waypoints
            for wp in self.areas:
                if len(wp["points"]) == 1:
                    screen_pos = self.viewport.world_to_screen(wp["points"][0])
                    pygame.draw.circle(
                        self.screen,
                        self._get_waypoint_color(wp),
                        screen_pos,
                        waypoint_radius,
                    )
                    text = self.system_font.render(
                        wp.get("name", ""),
                        True,
                        self.config.render.waypoint_label_color,
                    )

                    tw, th = text.get_size()
                    margin = max(6, int(0.08 * self.viewport.pixels_per_meter)) # screen‑px padding
                    cx, cy = screen_pos
                    text_rect = text.get_rect(center=(cx, cy - (waypoint_radius + margin + th / 2)))
                    self.screen.blit(text, text_rect)

            # Draw agent
            for i in range(self.num_agents):
                pygame.draw.circle(
                    self.screen,
                    self.color,
                    self.viewport.world_to_screen(self.agent_pos[i]),
                    agent_radius,
                )

            # Draw the areas
            self.draw_areas()
            self.draw_drawing_region()

            pygame.display.flip()
            self.clock.tick(int(max(1, round(self.config.update_hz))))

            self.publish_waypoints_markers()
            # draw waypoints + agent after this
            self.publish_areas()
            self.publish_tf()


    def destroy_node(self):
        self._stop_all_bag_processes()
        super().destroy_node()


def main():
    rclpy.init()
    node = DotSim()

    # Multi-threaded executor, but exclusive group ensures some callbacks share one thread
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
