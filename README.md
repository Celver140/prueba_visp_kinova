# kinova_control

Image-Based Visual Servoing (IBVS) with **ViSP** on a simulated **Kinova Gen3 6-DoF** arm
(**ROS 2 Humble**, **Gazebo Classic 11**, Ubuntu 22.04). The robot moves to a home pose,
detects a red cube with the wrist camera and servos until the camera is parallel to the
top face and close to it.

## Setup overview

```
~/kinova_control_ws/          ROS 2 workspace
└── src/
    ├── ros2_kortex/          Kinova ROS 2 packages (humble branch) — must be installed
    └── kinova_control/       this repository
~/visp-ws/                    ViSP source + build (C++ library and Python bindings)
```

## Requirements

| Component | Version |
|---|---|
| Ubuntu | 22.04 (jammy) |
| ROS 2 | Humble |
| Gazebo | Classic 11 (`gazebo_ros`, `gazebo_ros2_control`) |
| ViSP | 3.7.x built from source **with Python bindings** |
| Python | 3.10, numpy < 2 (required by ROS Humble `cv_bridge`) |

System packages (apt):

```bash
sudo apt install python3-colcon-common-extensions python3-vcstool python3-rosdep \
  ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-cv-bridge \
  ros-humble-control-msgs ros-humble-trajectory-msgs ros-humble-urdfdom-py \
  python3-pykdl python3-opencv
```


## 1. ROS 2 Kortex (Kinova)

Installed from the `humble` branch of https://github.com/Kinovarobotics/ros2_kortex/tree/humble :


> **Camera in Gazebo Classic:** `kinova_urdf.xacro` in this repository contains the
> Gazebo Classic camera plugin (`libgazebo_ros_camera.so`, topics
> `/wrist_mounted_camera/image/*`). It replaces the robot xacro of
> `kortex_description` (`ros2_kortex/kortex_description/robots/kinova.urdf.xacro`)
> after cloning `ros2_kortex`.

Modifications from original: 
> ros2_kortex/kortex_description/robots/kinova.urdf.xacro: updated to include the vision plugin for Gazebo classic
> ros2_kortex/kortex_description/arms/gen3/6dof/urdf/gen3_macro.xacro: updated the vision frames of references

## 2. ViSP (with Python bindings)

ViSP lives in its own workspace, `~/visp-ws`, following the official tutorial
https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-install-ubuntu.html
(prerequisites, recommended 3rd parties, source in `~/visp-ws/visp`). Additional
3rd parties used here:

```bash
sudo apt install libdmtx-dev libzbar-dev nlohmann-json3-dev   # libdmtx-dev avoids a binding/library mismatch
python3 -m pip install --user "pybind11[global]>=2.11" "numpy<2"
```

Build and install with Python bindings:

```bash
cd ~/visp-ws && mkdir -p visp-build-bindings && cd visp-build-bindings
cmake ../visp -DCMAKE_BUILD_TYPE=Release \
  -DALLOW_SYSTEM_PYTHON=ON -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc)
sudo make install && sudo ldconfig        # C++ libraries -> /usr/local/lib
make -j$(nproc) visp_python_bindings      # Python module -> ~/.local/lib/python3.10/site-packages/visp
```

Check: `python3 -c "import visp.vs; print('ViSP OK')"`

## 3. Environment (`~/.bashrc`)

The ViSP variables from the tutorial and the ROS sourcing were added at the **end** of
`~/.bashrc`, so every new terminal is ready:

```bash
export VISP_WS=$HOME/visp-ws
export VISP_DIR=$VISP_WS/visp-build-bindings
source /opt/ros/humble/setup.bash
[ -f ~/kinova_control_ws/install/setup.bash ] && source ~/kinova_control_ws/install/setup.bash
```

After editing `~/.bashrc`, open a new terminal or run `source ~/.bashrc` in the current one.
If a new terminal does not pick up the variables, check whether a `~/.bash_profile` exists
(it prevents `~/.bashrc` from being read in login shells).

## 4. Build and run

```bash
cd ~/kinova_control_ws
colcon build
source install/setup.bash

ros2 launch kinova_control kinova_sim_vision.launch.py      # Gazebo + Gen3 + red cube (+ IBVS node)
```


## Repository contents

| File | Purpose |
|---|---|
| `IBVS_node.py` | Current IBVS node (top-face detection, autonomous descent, recovery) |
| `PBVS_node.py` | Position-based visual servoing variant (not working) |
| `launch/kinova_sim_vision.launch.py` | Simulation + cube spawn + servoing node |
| `urdf/scene_cube.urdf` | Red cube (0.1 m) |
| `kortex_description_files/` | Modified files of the ros2_kortex package |
| `CMakeLists.txt`, `package.xml` | ROS 2 package (ament_cmake) |
