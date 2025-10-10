import copy
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Pose, Point
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

import pygame
import yaml
import math
from pathlib import Path

class DotSim(Node):
    def __init__(self):
        super().__init__("dot_sim")

        # Get package share path
        pkg_share = Path(get_package_share_directory("dot_sim"))
        config_file = pkg_share / "config" / "default_config.yaml"

        # Load config
        config = yaml.safe_load(config_file.read_text())
        self.agent_pos = config.get("agent_start", [0.0, 0.0])
        self.areas = config.get("named_areas", [])
        self.config_areas = copy.deepcopy(self.areas)

        self.drawing_region = []   # temp list of points being placed
        self.region_counter = len(self.areas)


        print("Loaded area:", self.areas)


        # ROS2 pubs/subs
        self.create_subscription(Twist, "/sim/cmd_vel", self.cmd_cb, 10)
        self.names_areas_pub = self.create_publisher(MarkerArray, "/sim/named_areas", 10)

        self.br = TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)

        self.create_service(SetBool, "/sim/set_color", self.set_color_cb)

        # Visualization
        pygame.init()
        pygame.display.set_caption("Agent Dot Sim")
        self.screen = pygame.display.set_mode((1366, 768))
        self.screen.fill((0,179,60))   # black background

        self.waypoint_font = pygame.font.SysFont("Times New Roman", 10)

        self.clock = pygame.time.Clock()
        self.color = (0, 255, 0)

        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0

        # Timer
        self.create_timer(0.05, self.update)

    def go_to_region(self, region_name: str):
        pass

    def cmd_cb(self, msg: Twist):
        self.agent_pos[0] += msg.linear.x * 0.1
        self.agent_pos[1] += msg.linear.y * 0.1

    def set_color_cb(self, req, res):
        self.color = (255, 0, 0) if req.data else (0, 255, 0)
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

                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 1.0
                m.color.a = 1.0

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
                t.color.r = 1.0
                t.color.g = 1.0
                t.color.b = 1.0
                t.color.a = 1.0
                t.text = wp["name"]

                ma.markers.append(t)

        self.names_areas_pub.publish(ma)


    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "agent"
        t.transform.translation.x = self.agent_pos[0]
        t.transform.translation.y = self.agent_pos[1]
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)
    
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
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 1.0, 0.0, 1.0)
                
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
                t.color.r = 1.0
                t.color.g = 1.0
                t.color.b = 1.0
                t.color.a = 1.0
                t.text = area["name"]
                ma.markers.append(t)


        self.names_areas_pub.publish(ma)


    def draw_areas(self):
        for area in self.areas:
            if len(area["points"]) >= 2:
                pts = [(int(p[0]*50+300), int(p[1]*50+300)) for p in area["points"]]
                # Draw polygon outline
                pygame.draw.lines(self.screen, (255, 0, 0), True, pts, 2)
                # Optional: label with name
                font = pygame.font.SysFont(None, 20)
                text = font.render(area["name"], True, (255, 255, 255))
                if pts:
                    self.screen.blit(text, pts[0])  # place at first point
    
    def draw_drawing_region(self):
        if len(self.drawing_region) >= 1:
            pts = [(int(p[0]*50+300), int(p[1]*50+300)) for p in self.drawing_region]
            # Draw connecting lines
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (0, 255, 255), False, pts, 2)
            # Draw points
            for pt in pts:
                pygame.draw.circle(self.screen, (0, 255, 255), pt, 4)



    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
            
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
                        self.region_counter += 1
                        new_region = {
                            "name": f"region{self.region_counter}",
                            "points": self.drawing_region.copy()
                        }
                        self.areas.append(new_region)
                        self.get_logger().info(f"Closed region {new_region['name']}")
                    self.drawing_region.clear()
                
                elif event.key == pygame.K_r:  # 🔄 reset
                    self.get_logger().info("Resetting simulation...")
                    # Reset agent
                    self.agent_pos = [0.0, 0.0]  # or load from config
                    # Reset regions
                    self.areas.clear()
                    self.drawing_region.clear()
                    self.areas = copy.deepcopy(self.config_areas)
                    
                    self.region_counter = len(self.config_areas)
                    # Publish cleared areas
                    self.publish_areas()

            elif event.type == pygame.KEYUP:
                # Stop motion when key released
                if event.key in (pygame.K_w, pygame.K_s, pygame.K_UP, pygame.K_DOWN):
                    self.cmd_vel_y = 0.0
                if event.key in (pygame.K_a, pygame.K_d, pygame.K_LEFT, pygame.K_RIGHT):
                    self.cmd_vel_x = 0.0
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click = add point
                    x, y = event.pos
                    # Convert screen coords back to world coords
                    wx = (x - 300) / 50.0
                    wy = (y - 300) / 50.0
                    self.drawing_region.append([wx, wy])
                    self.get_logger().info(f"Added point ({wx:.2f}, {wy:.2f})")

                elif event.button == 3:  # right click = finish region
                    if len(self.drawing_region) >= 3:
                        font = pygame.font.Font(None, 32)
                        popup_rect = pygame.Rect(120, 100, 260, 120)
                        input_rect = pygame.Rect(popup_rect.x + 20, popup_rect.y + 50, 220, 32)
                        enter_rect = pygame.Rect(popup_rect.x + 180, popup_rect.y + 90, 50, 20)
                        close_rect = pygame.Rect(popup_rect.x + 230, popup_rect.y + 10, 20, 20)

                        color_border = pygame.Color('black')
                        color_bg = pygame.Color('white')
                        color_input = pygame.Color('lightgray')
                        color_active = pygame.Color('dodgerblue2')
                        color_text = pygame.Color('black')
                        color_button = pygame.Color('gray')

                        user_text = ""
                        input_active = True
                        cursor_visible = True
                        cursor_timer = 0
                        cursor_blink_time = 500  # ms

                        clock = pygame.time.Clock()

                        while input_active:
                            dt = clock.tick(30)
                            cursor_timer += dt
                            if cursor_timer >= cursor_blink_time:
                                cursor_visible = not cursor_visible
                                cursor_timer = 0

                            for e in pygame.event.get():
                                if e.type == pygame.QUIT:
                                    rclpy.shutdown()
                                elif e.type == pygame.MOUSEBUTTONDOWN:
                                    if close_rect.collidepoint(e.pos):
                                        input_active = False
                                        user_text = ""
                                    elif enter_rect.collidepoint(e.pos):
                                        if not user_text.strip():
                                            user_text = f"region{self.region_counter + 1}"
                                        input_active = False

                                elif e.type == pygame.KEYDOWN:
                                    if e.key == pygame.K_RETURN:
                                        if not user_text.strip():
                                            user_text = f"region{self.region_counter + 1}"
                                        input_active = False
                                    elif e.key == pygame.K_BACKSPACE:
                                        user_text = user_text[:-1]
                                    elif len(user_text) < 20:
                                        user_text += e.unicode

                            # --- Draw popup window ---
                            pygame.draw.rect(self.screen, color_bg, popup_rect, 2, 3)
                            pygame.draw.rect(self.screen, color_border, popup_rect, 2, 3)

                            # Title bar
                            pygame.draw.rect(self.screen, pygame.Color('darkgray'),
                                            (popup_rect.x, popup_rect.y, popup_rect.w, 30))
                            title_surface = font.render("Enter Region Name", True, color_text)
                            self.screen.blit(title_surface, (popup_rect.x + 10, popup_rect.y + 5))

                            # Close (X) button
                            pygame.draw.circle(self.screen, pygame.Color('black'), (popup_rect.x + 230, popup_rect.y + 10), 10)
                            x_surface = font.render("X", True, color_text)
                            text_rect = x_surface.get_rect(center=(popup_rect.x + 230, popup_rect.y + 10))
                            self.screen.blit(x_surface, text_rect)

                            # Input box
                            pygame.draw.rect(self.screen, color_input, input_rect)
                            pygame.draw.rect(self.screen, color_active, input_rect, 2)
                            text_surface = font.render(user_text, True, color_text)
                            self.screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))

                            # Blinking cursor
                            if cursor_visible and input_active:
                                cursor_x = input_rect.x + 5 + text_surface.get_width() + 2
                                cursor_y = input_rect.y + 5
                                pygame.draw.line(self.screen, color_text,
                                                (cursor_x, cursor_y),
                                                (cursor_x, cursor_y + 24), 2)

                            # Enter button
                            pygame.draw.rect(self.screen, color_button, enter_rect)
                            enter_surface = font.render("OK", True, color_text)
                            self.screen.blit(enter_surface, (enter_rect.x + 5, enter_rect.y - 3))

                            pygame.display.flip()

                        # === Save region only if a name was entered ===
                        if user_text.strip():
                            self.region_counter += 1
                            new_region = {
                                "name": user_text,
                                "points": self.drawing_region.copy()
                            }
                            self.areas.append(new_region)
                            self.get_logger().info(f"Closed region {new_region['name']}")

                    self.drawing_region.clear()

        # Update position
        self.agent_pos[0] += self.cmd_vel_x * 0.1
        self.agent_pos[1] += self.cmd_vel_y * 0.1

        self.screen.fill((0,179,64))
        # Draw waypoints
        for wp in self.areas:
            if len(wp["points"]) == 1:
                circle_pos = (int(wp["points"][0][0]*50+310), int(wp["points"][0][1]*50+310))
                pygame.draw.circle(self.screen, wp["color"], (int(wp["points"][0][0]*50+300), int(wp["points"][0][1]*50+300)), 5)
                # Text label
                text = self.waypoint_font.render(wp["name"], True, (255, 255, 255))
                text_rect = text.get_rect(center=circle_pos)
                self.screen.blit(text, text_rect)

        # Draw agent
        pygame.draw.circle(self.screen, self.color, (int(self.agent_pos[0]*50+300), int(self.agent_pos[1]*50+300)), 8)

        # Draw the areas
        self.draw_areas()
        self.draw_drawing_region()

        pygame.display.flip()
        self.clock.tick(30)

        self.publish_waypoints_markers()
        # draw waypoints + agent after this
        self.publish_areas()
        self.publish_tf()


def main():
    rclpy.init()
    sim = DotSim()
    rclpy.spin(sim)
    sim.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
