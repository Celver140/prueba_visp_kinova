# spacecraft_servoing

Model-based IBVS (ViSP) for on-orbit servicing with a Kinova Gen3 wrist camera
(ROS 2 Humble, Gazebo Classic).

```
image ─► model_tracker (ViSP vpMbGenericTracker, CAD .cao) ─► cMo
         targets (CAD points of screw / panel) + refiners (local image refinement) ─► s, s*
         mission_fsm (INIT ► HOMING ► ACQUIRE ► IDLE ► APPROACH ► ALIGN ► HOLD, RETREAT, LOST)
         ibvs_controller (ViSP vpServo) ─► camera twist
         robot_interface (KDL, DLS, joint_trajectory_controller) ─► robot
```

| module | responsibility | ROS? |
|---|---|---|
| `servoing_node.py` | wiring, parameters, topics, debug image | yes |
| `robot_interface.py` | joint states, KDL chain, FK, twist→q̇, commands, home | yes |
| `model_tracker.py` | ViSP model-based tracker wrapper, `.cao` parser | no |
| `targets.py` | target catalog (YAML) → feature points, desired features | no |
| `refiners.py` | ROI refinement (screw Hough, corner sub-pixel) | no |
| `ibvs_controller.py` | ViSP IBVS law, adaptive gain, saturation | no |
| `mission_fsm.py` | mission logic (what to do) | no |
| `geometry.py` | homogeneous transforms, projection | no |

## Build / run
```bash
cd ~/kinova_control_ws/src && cp -r <this folder> .   # next to kinova_control
cd .. && colcon build --packages-select spacecraft_servoing && source install/setup.bash
ros2 launch spacecraft_servoing spacecraft_servoing.launch.py                 # waits in IDLE
ros2 topic pub --once /spacecraft_servoing/select_target std_msgs/String "{data: screw_1}"
ros2 topic pub --once /spacecraft_servoing/select_target std_msgs/String "{data: solar_panel}"
ros2 topic pub --once /spacecraft_servoing/select_target std_msgs/String "{data: abort}"
ros2 topic echo /spacecraft_servoing/status
ros2 run rqt_image_view rqt_image_view /spacecraft_servoing/debug_image
```
Tests (no ROS needed): `python3 -m pytest test/ -q`

## Adding things
* new target kind → builder in `targets.py` + refiner in `refiners.py`
* new mission step (e.g. tool insertion, grasp) → state in `mission_fsm.py`
* real tracker initialisation → replace `ServoingNode.initial_pose_guess()`
* real robot → `command_mode: velocity`, `camera_convention: optical`
