# DotSim Overview

`dot_sim` is a lightweight ROS 2 package that opens an interactive 2D playground for a point-sized robot ("the dot"). It uses Pygame for fast visualization while exposing ROS topics, services, and tf so that higher-level planners can treat the sim like a real robot.

- Move the agent with the keyboard or `/sim/cmd_vel`.
- Sketch polygonal regions and labelled waypoints with the mouse.
- Stream visualization markers, odometry, TF, and a camera-style image topic back to ROS 2.
- Trigger a configurable image publish whenever the agent enters a region or touches a waypoint.

The rest of this README focuses on how to install, run, and configure the `dot_sim` package. (Other packages in the repository, such as navigation behaviors, are documented separately.)

## Prerequisites

- ROS 2 (tested with Humble+ APIs; any recent rclpy-based distro should work).
- A working `cv_bridge` installation (ships with ROS desktop variants).
- Python dependencies that ship with the ROS 2 desktop install, plus:
  - `pygame`
  - `opencv-python`
  - `yaml` (PyYAML)
  - `numpy` (pulled indirectly by OpenCV)

If you are on Ubuntu with ROS 2 desktop, install the missing extras with:

```bash
sudo apt install python3-pygame python3-opencv
```

## Build & Install

```bash
cd /path/to/agent-simulator
colcon build --packages-select dot_sim
source install/setup.bash
```

The package installs its launchable script, default configuration, and the sample `Alpha.jpg` image into the ROS share directory (`install/dot_sim/share/dot_sim`).

## Running the Simulator

```bash
ros2 run dot_sim dot_sim
```

What happens:

- A Pygame window opens with the agent placed at the configured start position.
- ROS 2 nodes publish TF (`map → odom`), odometry (`/sim/odometry`), and named-area markers (`/sim/named_areas`).
- A background thread monitors for collisions and pushes the static `Alpha.jpg` frame onto `/sim/images` whenever the dot enters a polygonal region or waypoint.

> **Tip:** If the window fails to open, verify you are running under an X11/Wayland session with SDL video support and that `Alpha.jpg` exists at `share/dot_sim/Alpha.jpg`.

## Controls Inside the Pygame Window

- `W/A/S/D` or arrow keys: translate the dot in ±X/±Y.
- `R`: reset the simulation (agent pose, color, and regions).
- Left click: drop vertices for a new polygon or waypoint.
  - Single vertex → waypoint.
  - 3+ vertices → polygon.
- Right click or `Enter`: finish the shape, name it in the popup, and add it to the active region list.
- Window resize events rescale the world dynamically; the agent performs toroidal wrap-around to stay visible.

Keyboard motion commands adjust internal velocity variables, allowing you to blend keyboard inputs with external `/sim/cmd_vel` messages. New regions you draw are immediately published as visualization markers and become eligible for collision checks.

## ROS Interfaces

**Subscriptions**
- `/sim/cmd_vel` (`geometry_msgs/msg/Twist`): planar velocity command; affects internal agent position each frame.

**Publications**
- `/sim/named_areas` (`visualization_msgs/msg/MarkerArray`): markers for polygonal regions and waypoint labels.
- `/sim/odometry` (`nav_msgs/msg/Odometry`): agent pose in the `odom` frame (zero height; yaw derived from last command).
- `/tf` (`map → odom` transform) plus `/tf_static` initialization.
- `/sim/images` (`sensor_msgs/msg/Image`): static frame from `Alpha.jpg`, emitted whenever collision logic fires.

**Services**
- `/sim/set_color` (`std_srvs/srv/SetBool`): toggles the dot’s color between the normal color and the alert color.

## Configuration

The simulator reads `share/dot_sim/config/default_config.yaml` on startup. Key sections include:

- `agent_start`: initial `[x, y]` in meters.
- `update_hz`: update rate for the render loop and collision checks.
- `render`: colors, window sizing behavior, and the initial agent color.
- `collision`: waypoint radius, polygon margin, debounce timing, and (placeholder) service payload hooks.
- `named_areas`: pre-defined polygons (`points` with ≥3 vertices) and waypoints (`points` with a single vertex).

To override the defaults, copy the YAML to your workspace, adjust values, and export `DOT_SIM_CONFIG` before launching:

```bash
export DOT_SIM_CONFIG=/absolute/path/to/your_config.yaml
ros2 run dot_sim dot_sim
```

If the variable is unset, the node falls back to the packaged default.

## Extending the Simulator

- Replace `Alpha.jpg` with any image (keep the path consistent) to change the collision-triggered frame.
- Subscribe to `/sim/odometry` and `/tf` for downstream localization, and use `/sim/named_areas` to visualize regions in RViz.
- Integration hooks such as `go_to_region()` are scaffolded; additional behaviors can be wired in using ROS services or composition.

For additional packages in this repository, consult their individual documentation. The steps above are sufficient to explore and iterate on the `dot_sim` functionality.
