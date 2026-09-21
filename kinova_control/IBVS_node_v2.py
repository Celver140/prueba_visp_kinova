#!/usr/bin/env python3
"""
ibvs_cube_node.py

Image-Based Visual Servoing (IBVS) of a red cube using ViSP for the visual
control law and PyKDL (from the live robot_description) for the robot
Jacobian, so that the camera-frame velocity screw computed by ViSP is
mapped to joint velocities correctly.

Strategy (3 phases, as requested)
-----------------------------------
STATE 1 -- Center the centroid (lateral translation only, J1-J5)
    Only the average of the 4 corners' pixel position is regulated
    toward the image center. The full ViSP IBVS velocity v_c is still
    computed correctly from the 4-point interaction matrix (so signs/
    scale are right), but only its vx,vy components are kept -- vz and
    all rotation are masked to zero. This deliberately does NOT try to
    approach or align corners yet, so it can't fight itself the way a
    full 6-DOF command from a badly-off-center, badly-scaled initial
    view can.

STATE 2 -- Match the corners (full IBVS, J1-J5)
    Once the centroid is centered, the full 4-point IBVS law (vx, vy,
    vz, wx, wy, wz) drives the corners to their desired image positions
    -- this is what pulls the camera closer/tilts it so the square
    matches the target size and shape.

STATE 3 -- End effector parallel to the face (task priority, J1-J5)
    Once corners are matched, J4/J5 are driven toward a target
    orientation (target_j4/target_j5) meant to make the camera axis
    perpendicular to the cube's top face. J1-J3 cancel the camera motion
    that rotation would induce (feedforward decoupling) plus track
    residual IBVS feedback, so the cube stays matched while J4/J5
    finish rotating. J6 stays locked throughout.

Each phase can regress to an earlier one if its error grows past a
"lost the plot" threshold (e.g. state 3's J4/J5 motion knocks the
corners far enough off that centroid centering itself breaks down) --
see recenter_error_threshold.

Units bug fixed in this revision
-----------------------------------
The previous version compared `task.getError()` (ViSP's internal
normalized/metric-unit error, typically O(0.01-0.3)) directly against a
threshold written in pixels (15.0). That comparison was almost always
true, so the old "STATE 1" exited after a single control cycle
regardless of actual convergence -- visible in the log as switching to
STATE 2 within ~0.3s of reaching the seed pose. All phase-transition
decisions now use pixel-space error computed directly from detected vs.
desired pixel corners, independent of ViSP's internal feature units
(which are still used, correctly, inside computeControlLaw() itself).

Fixes applied vs. the previous version (see chat for the diagnosis)
---------------------------------------------------------------------
1. camera_frame is now AUTO-DETECTED from the image message's
   header.frame_id (like the PBVS node), instead of trusting a guessed
   parameter default. Using the wrong frame (e.g. a mechanical mount
   frame instead of the optical frame) silently rotates every velocity
   command and can prevent convergence entirely.
2. Camera intrinsics are read from the real CameraInfo topic instead of
   being hardcoded -- wrong intrinsics subtly corrupt every pixel-to-
   meter conversion feeding the interaction matrix.
3. Per-corner depth (Z) is sampled from the real depth image instead of
   guessed from contour pixel area via an unvalidated magic constant.
   That heuristic was very likely the reason it wasn't approaching the
   cube: IBVS's vz (approach) component is scaled by Z, and a
   miscalibrated/clipped Z estimate gives it no meaningful signal to
   act on.
4. Corner identity is now tracked frame-to-frame via nearest-neighbor
   matching to the previous frame's corners, instead of re-deriving
   "top-left/top-right/..." from raw geometry every frame (which can
   flip labels as the face rotates and cause the servo to fight itself).

IMPORTANT -- things you must still verify/adjust for your setup
-------------------------------------------------------------------
- base_link_frame: must match your URDF's root link name.
- camera_info_topic / depth_topic: best-guess defaults below -- confirm
  with `ros2 topic list | grep wrist_mounted_camera` and adjust if
  needed; there was visible ambiguity between the ros_gz bridge's
  advertised topic names and the Gazebo camera plugin's own log message
  in your launch output.
- depth image encoding is assumed to be 32FC1 (meters), the Ignition/
  Gazebo default. If your bridge produces something else, adjust
  `_sample_depth`.
- Depth and color frames are NOT hardware-synchronized here -- the node
  just uses the most recently received depth frame. For a sim with a
  static or slow-moving cube this is a fine approximation; for faster
  motion, use message_filters' ApproximateTimeSynchronizer instead.
- This assumes the camera is rigidly mounted directly after joint_6
  (6 movable joints in the chain). If your URDF differs, the column
  slicing (J[:, :5], J[:, :3], J[:, 3:5]) needs adjusting.
"""

import math
import numpy as np
import cv2
from itertools import permutations

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge

import PyKDL as kdl
from urdf_parser_py.urdf import URDF as UrdfModel

from visp.core import CameraParameters
from visp.visual_features import FeaturePoint
from visp.vs import Servo


def kdl_jacobian_to_np(jac: kdl.Jacobian) -> np.ndarray:
    m = np.zeros((jac.rows(), jac.columns()))
    for i in range(jac.rows()):
        for j in range(jac.columns()):
            m[i, j] = jac[i, j]
    return m


def kdl_rotation_to_np(R: kdl.Rotation) -> np.ndarray:
    return np.array([[R[0, 0], R[0, 1], R[0, 2]],
                      [R[1, 0], R[1, 1], R[1, 2]],
                      [R[2, 0], R[2, 1], R[2, 2]]])


def _make_fixed_kdl_joint(name: str) -> kdl.Joint:
    try:
        return kdl.Joint(name)
    except TypeError:
        pass
    for attr in ('Fixed', 'None'):
        try:
            jtype = getattr(kdl.Joint, attr)
            return kdl.Joint(name, jtype)
        except (AttributeError, TypeError):
            continue
    raise RuntimeError('Could not construct a fixed PyKDL.Joint with this PyKDL build.')


def _origin_to_kdl_frame(origin) -> kdl.Frame:
    rpy = list(origin.rpy) if (origin is not None and origin.rpy) else [0.0, 0.0, 0.0]
    xyz = list(origin.xyz) if (origin is not None and origin.xyz) else [0.0, 0.0, 0.0]
    return kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))


def build_kdl_chain_from_urdf(urdf_xml: str, base_link: str, tip_link: str) -> kdl.Chain:
    robot = UrdfModel.from_xml_string(urdf_xml)
    joint_names = robot.get_chain(base_link, tip_link, links=False, joints=True)

    chain = kdl.Chain()
    for jname in joint_names:
        joint = robot.joint_map[jname]
        F = _origin_to_kdl_frame(joint.origin)

        if joint.type in ('revolute', 'continuous'):
            axis = joint.axis if joint.axis else [1.0, 0.0, 0.0]
            axis_vec = F.M * kdl.Vector(*axis)
            kdl_joint = kdl.Joint(joint.name, F.p, axis_vec, kdl.Joint.RotAxis)
        elif joint.type == 'prismatic':
            axis = joint.axis if joint.axis else [1.0, 0.0, 0.0]
            axis_vec = F.M * kdl.Vector(*axis)
            kdl_joint = kdl.Joint(joint.name, F.p, axis_vec, kdl.Joint.TransAxis)
        else:
            kdl_joint = _make_fixed_kdl_joint(joint.name)

        segment = kdl.Segment(joint.child, kdl_joint, F)
        chain.addSegment(segment)

    return chain


class IBVSCubeNode(Node):

    def __init__(self):
        super().__init__('ibvs_cube_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('base_link_frame', 'base_link')
        # '' = auto-detect the camera's optical frame from the Image
        # message's header.frame_id. See module docstring, fix (1).
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('camera_info_topic', '/wrist_mounted_camera/image/camera_info')
        self.declare_parameter('depth_topic', '/wrist_mounted_camera/depth_image')

        self.declare_parameter('joint_names',
                                ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('lambda_gain', 0.25)
        self.declare_parameter('centroid_error_threshold_px', 15.0)
        self.declare_parameter('corner_error_threshold_px', 20.0)
        self.declare_parameter('recenter_error_threshold_px', 80.0)
        self.declare_parameter('joint_vel_limit', 0.25)  # rad/s, per joint
        self.declare_parameter('target_j4', 0.0)
        self.declare_parameter('target_j5', -1.57)
        self.declare_parameter('phase3_gain', 0.6)
        self.declare_parameter('desired_standoff_m', 0.15)  # depth of the desired feature points
        self.declare_parameter('fallback_depth_m', 0.3)     # used only if no valid depth sample exists yet

        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.camera_frame_param = self.get_parameter('camera_frame').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.lambda_gain = float(self.get_parameter('lambda_gain').value)
        self.centroid_error_threshold_px = float(self.get_parameter('centroid_error_threshold_px').value)
        self.corner_error_threshold_px = float(self.get_parameter('corner_error_threshold_px').value)
        self.recenter_error_threshold_px = float(self.get_parameter('recenter_error_threshold_px').value)
        self.joint_vel_limit = float(self.get_parameter('joint_vel_limit').value)
        self.target_j4 = float(self.get_parameter('target_j4').value)
        self.target_j5 = float(self.get_parameter('target_j5').value)
        self.phase3_gain = float(self.get_parameter('phase3_gain').value)
        self.Z_d = float(self.get_parameter('desired_standoff_m').value)
        self.fallback_depth_m = float(self.get_parameter('fallback_depth_m').value)

        self.n_joints = len(self.joint_names)

        # ---------------- ROS I/O ----------------
        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/wrist_mounted_camera/image/image_raw',
                                  self.image_cb, qos_img)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_img)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_img)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        qos_urdf = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, '/robot_description', self.urdf_cb, qos_urdf)

        self.traj_pub = self.create_publisher(JointTrajectory,
                                               '/joint_trajectory_controller/joint_trajectory', 10)
        self.traj_client = ActionClient(self, FollowJointTrajectory,
                                         '/joint_trajectory_controller/follow_joint_trajectory')

        self.bridge = CvBridge()

        # ---------------- Kinematics (filled once URDF + camera frame known) ----------------
        self.kdl_chain = None
        self.jac_solver = None
        self.fk_solver = None
        self._urdf_xml = None
        self._resolved_camera_frame = self.camera_frame_param or None

        # ---------------- Camera model (filled once CameraInfo arrives) ----------------
        self.cam = None
        self.latest_depth = None  # np.ndarray (H,W) float32 meters, or None

        # ---------------- ViSP IBVS task ----------------
        self.task = Servo()
        self.task.setServo(Servo.EYEINHAND_CAMERA)
        self.task.setInteractionMatrixType(Servo.CURRENT, Servo.PSEUDO_INVERSE)
        self.task.setLambda(self.lambda_gain)

        self.s = [FeaturePoint() for _ in range(4)]
        self.s_star = [FeaturePoint() for _ in range(4)]
        # Desired pixel targets, ordered [top-left, top-right, bottom-right,
        # bottom-left] -- deliberately near the image periphery so the
        # converged configuration corresponds to the cube filling most of
        # the frame (i.e. camera close to it).
        self.u_star = [100.0, 540.0, 540.0, 100.0]
        self.v_star = [80.0, 80.0, 400.0, 400.0]
        for i in range(4):
            self.task.addFeature(self.s[i], self.s_star[i])
        self._desired_features_built = False  # need cam intrinsics first

        # ---------------- State ----------------
        self.current_q = [0.0] * self.n_joints
        self.has_joint_states = False
        self.state = 0  # 0: seed pose, 1: center centroid, 2: match corners, 3: orthogonal align
        self.goal_sent = False
        self.last_stamp = None
        self.prev_corners = None  # for frame-to-frame correspondence tracking

        self.get_logger().info('IBVS node started. Waiting for /robot_description, '
                                'camera_info and joint states...')

    # ------------------------------------------------------------------
    def camera_info_cb(self, msg: CameraInfo):
        if self.cam is not None:
            return
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.cam = CameraParameters()
        self.cam.initPersProjWithoutDistortion(fx, fy, cx, cy)
        self.get_logger().info(f'Camera intrinsics received: fx={fx:.2f} fy={fy:.2f} '
                                f'cx={cx:.2f} cy={cy:.2f}')
        # Now that we have real intrinsics, build the desired features.
        for i in range(4):
            x_star = (self.u_star[i] - self.cam.get_u0()) / self.cam.get_px()
            y_star = (self.v_star[i] - self.cam.get_v0()) / self.cam.get_py()
            self.s_star[i].buildFrom(x_star, y_star, self.Z_d)
        self._desired_features_built = True

    # ------------------------------------------------------------------
    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception:
            # Some bridges publish 16UC1 in mm instead -- handle that too.
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth_raw.astype(np.float32) / 1000.0
        self.latest_depth = depth

    # ------------------------------------------------------------------
    def _sample_depth(self, u: float, v: float):
        """Returns depth in meters at pixel (u,v), or None if unavailable/invalid."""
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape[:2]
        iu = int(np.clip(round(u), 0, w - 1))
        iv = int(np.clip(round(v), 0, h - 1))
        z = float(self.latest_depth[iv, iu])
        if not np.isfinite(z) or z <= 0.01:
            return None
        return z

    # ------------------------------------------------------------------
    def urdf_cb(self, msg: String):
        if self.kdl_chain is not None:
            return
        self._urdf_xml = msg.data
        self._try_build_chain()

    def _try_build_chain(self):
        if self.kdl_chain is not None:
            return
        if self._urdf_xml is None or self._resolved_camera_frame is None:
            return
        try:
            self.kdl_chain = build_kdl_chain_from_urdf(
                self._urdf_xml, self.base_link_frame, self._resolved_camera_frame)
        except KeyError as e:
            self.get_logger().error(
                f'Could not build KDL chain {self.base_link_frame} -> '
                f'{self._resolved_camera_frame}: missing joint/link {e}.')
            return
        except Exception as e:
            self.get_logger().error(f'Failed to build KDL chain: {e}')
            return
        n = self.kdl_chain.getNrOfJoints()
        if n != self.n_joints:
            self.get_logger().warn(
                f'KDL chain has {n} movable joints but joint_names has {self.n_joints}. '
                f'Check base_link_frame/camera_frame and that the camera is rigidly '
                f'mounted directly after the last arm joint.')
        self.jac_solver = kdl.ChainJntToJacSolver(self.kdl_chain)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        self.get_logger().info(f'KDL chain ready: {self.base_link_frame} -> '
                                f'{self._resolved_camera_frame} ({n} joints)')

    # ------------------------------------------------------------------
    def joint_state_cb(self, msg: JointState):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                self.current_q[i] = msg.position[msg.name.index(name)]
        self.has_joint_states = True

    # ------------------------------------------------------------------
    def send_initial_trajectory(self):
        if self.goal_sent:
            return
        if not self.traj_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Waiting for joint_trajectory_controller action server...')
            return
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, -1.57, 0.0, -1.57, 0.0]
        point.time_from_start.sec = 10
        goal_msg.trajectory.points = [point]
        self.goal_sent = True
        fut = self.traj_client.send_goal_async(goal_msg)
        fut.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Initial trajectory rejected.')
            return
        handle.get_result_async().add_done_callback(self._traj_done_cb)

    def _traj_done_cb(self, future):
        self.get_logger().info('Seed pose reached -- starting IBVS (state 1: center centroid).')
        self.state = 1

    # ------------------------------------------------------------------
    def _detect_raw_corners(self, cv_image):
        """Returns the 4 detected corners in arbitrary (unlabeled) order,
        or None if no suitable contour was found."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 150:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    def _order_corners_initial(self, pts):
        """Sum/diff heuristic -- only used to seed labeling on first detection."""
        rect = np.zeros((4, 2), dtype='float32')
        ssum = pts.sum(axis=1)
        rect[0] = pts[np.argmin(ssum)]   # top-left
        rect[2] = pts[np.argmax(ssum)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        return rect

    def _match_corners_to_previous(self, pts, prev):
        """Finds the labeling of `pts` (any order) that best matches `prev`
        (fixed order), by minimizing total corner displacement."""
        best_perm, best_cost = None, float('inf')
        for perm in permutations(range(4)):
            cost = sum(np.linalg.norm(pts[perm[i]] - prev[i]) for i in range(4))
            if cost < best_cost:
                best_cost, best_perm = cost, perm
        return pts[list(best_perm)]

    def detect_red_cube_corners(self, cv_image):
        raw = self._detect_raw_corners(cv_image)
        if raw is None:
            self.prev_corners = None  # lost -- re-seed labeling on reacquire
            return None
        if self.prev_corners is None:
            ordered = self._order_corners_initial(raw)
        else:
            ordered = self._match_corners_to_previous(raw, self.prev_corners)
        self.prev_corners = ordered
        return ordered

    # ------------------------------------------------------------------
    def _jacobian_and_camera_rotation(self):
        q_kdl = kdl.JntArray(self.n_joints)
        for i in range(self.n_joints):
            q_kdl[i] = self.current_q[i]

        jac_kdl = kdl.Jacobian(self.n_joints)
        self.jac_solver.JntToJac(q_kdl, jac_kdl)
        J = kdl_jacobian_to_np(jac_kdl)

        frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)
        R = kdl_rotation_to_np(frame.M)
        return J, R

    def _solve_qdot(self, J_cols: np.ndarray, v_base: np.ndarray) -> np.ndarray:
        qdot, *_ = np.linalg.lstsq(J_cols, v_base, rcond=None)
        return np.clip(qdot, -self.joint_vel_limit, self.joint_vel_limit)

    # ------------------------------------------------------------------
    def image_cb(self, msg: Image):
        if self._resolved_camera_frame is None:
            self._resolved_camera_frame = msg.header.frame_id
            self.get_logger().info(
                f"Auto-detected camera optical frame from image header: "
                f"'{self._resolved_camera_frame}'")
            self._try_build_chain()

        if self.state == 0:
            self.send_initial_trajectory()
            return

        if not self.has_joint_states or self.kdl_chain is None:
            return
        if self.cam is None or not self._desired_features_built:
            return  # waiting for camera_info

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.05 if self.last_stamp is None else max(1e-3, min(0.2, stamp - self.last_stamp))
        self.last_stamp = stamp

        detected = self.detect_red_cube_corners(cv_image)
        if detected is None:
            # Fallback heuristic: Try to detect the largest red contour and move to center it.
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 50:
                    M = cv2.moments(largest_contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        self.get_logger().warn(f'Cube corners not found -- centering largest red blob at ({cx}, {cy}).',
                                               throttle_duration_sec=1.0)
                        cv2.circle(cv_image, (cx, cy), 8, (0, 165, 255), -1)

                        error_x = (cx - self.cam.get_u0()) / self.cam.get_px()
                        error_y = (cy - self.cam.get_v0()) / self.cam.get_py()

                        v_c_np = np.zeros(6)
                        # Proportional velocity to center the blob in the camera frame
                        v_c_np[0] = self.lambda_gain * error_x * self.fallback_depth_m
                        v_c_np[1] = self.lambda_gain * error_y * self.fallback_depth_m

                        J, R = self._jacobian_and_camera_rotation()
                        R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
                        v_base = R6 @ v_c_np

                        qdot = np.zeros(self.n_joints)
                        qdot[:5] = self._solve_qdot(J[:, :5], v_base)

                        self._publish_qdot(qdot, dt)
                        self._show(cv_image)
                        return

            # If no valid contour is found at all, hold position
            self.get_logger().warn('No red blob visible -- holding position.', throttle_duration_sec=1.0)
            self._publish_qdot(np.zeros(self.n_joints), dt)
            self._show(cv_image)
            return

        for i in range(4):
            u, v = float(detected[i][0]), float(detected[i][1])
            z = self._sample_depth(u, v)
            if z is None:
                z = self.fallback_depth_m
            x = (u - self.cam.get_u0()) / self.cam.get_px()
            y = (v - self.cam.get_v0()) / self.cam.get_py()
            self.s[i].buildFrom(x, y, z)
            cv2.circle(cv_image, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.circle(cv_image, (int(self.u_star[i]), int(self.v_star[i])), 5, (255, 0, 0), -1)

        det_u = np.array([detected[i][0] for i in range(4)], dtype=float)
        det_v = np.array([detected[i][1] for i in range(4)], dtype=float)
        star_u = np.array(self.u_star, dtype=float)
        star_v = np.array(self.v_star, dtype=float)

        centroid_error_px = math.hypot(det_u.mean() - star_u.mean(),
                                        det_v.mean() - star_v.mean())
        corner_error_px = math.sqrt(np.mean((det_u - star_u) ** 2 + (det_v - star_v) ** 2))

        v_c = self.task.computeControlLaw()  # camera-frame twist, vx..wz (correct units/scale)
        v_c_np = np.array([v_c[i] for i in range(6)], dtype=float)

        J, R = self._jacobian_and_camera_rotation()
        R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
        v_base = R6 @ v_c_np  # rotate camera-frame twist into base-frame axes

        qdot = np.zeros(self.n_joints)

        if self.state == 1:
            # Lateral centering only: keep vx,vy from the correctly-scaled
            # IBVS law, zero out vz and all rotation.
            v_lateral = np.zeros(6)
            v_lateral[0:2] = v_base[0:2]
            qdot[:5] = self._solve_qdot(J[:, :5], v_lateral)
            self.get_logger().info(
                f'[STATE 1] centering centroid, centroid_err={centroid_error_px:.1f}px',
                throttle_duration_sec=0.5)

            if centroid_error_px < self.centroid_error_threshold_px:
                self.get_logger().warn('Centroid centered -- switching to STATE 2 (match corners).')
                self.state = 2

        elif self.state == 2:
            qdot[:5] = self._solve_qdot(J[:, :5], v_base)
            self.get_logger().info(
                f'[STATE 2] matching corners, corner_err={corner_error_px:.1f}px '
                f'centroid_err={centroid_error_px:.1f}px', throttle_duration_sec=0.5)

            if centroid_error_px > self.recenter_error_threshold_px:
                self.get_logger().warn('Drifted off-center -- back to STATE 1.')
                self.state = 1
            elif corner_error_px < self.corner_error_threshold_px:
                self.get_logger().warn('Corners matched -- switching to STATE 3 (orthogonal alignment).')
                self.state = 3

        elif self.state == 3:
            q4, q5 = self.current_q[3], self.current_q[4]
            qdot4 = float(np.clip(self.phase3_gain * (self.target_j4 - q4),
                                   -self.joint_vel_limit, self.joint_vel_limit))
            qdot5 = float(np.clip(self.phase3_gain * (self.target_j5 - q5),
                                   -self.joint_vel_limit, self.joint_vel_limit))

            induced_v = J[:, 3:5] @ np.array([qdot4, qdot5])
            v_target = v_base - induced_v
            qdot[:3] = self._solve_qdot(J[:, :3], v_target)
            qdot[3] = qdot4
            qdot[4] = qdot5

            self.get_logger().info(
                f'[STATE 3] aligning J4/J5 (q4={q4:.2f}->{self.target_j4:.2f}, '
                f'q5={q5:.2f}->{self.target_j5:.2f}) corner_err={corner_error_px:.1f}px '
                f'centroid_err={centroid_error_px:.1f}px', throttle_duration_sec=0.5)

            if centroid_error_px > self.recenter_error_threshold_px:
                self.get_logger().warn('Drifted off-center -- back to STATE 1.')
                self.state = 1
            elif corner_error_px > self.recenter_error_threshold_px:
                self.get_logger().warn('Corners drifted apart -- back to STATE 2.')
                self.state = 2
            elif abs(self.target_j4 - q4) < 0.02 and abs(self.target_j5 - q5) < 0.02:
                self.get_logger().info('Orthogonal alignment reached.', throttle_duration_sec=2.0)

        self._publish_qdot(qdot, dt)
        self._show(cv_image)

    # ------------------------------------------------------------------
    def _publish_qdot(self, qdot: np.ndarray, dt: float):
        new_positions = [self.current_q[i] + float(qdot[i]) * dt for i in range(self.n_joints)]
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = new_positions
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(dt * 1e9)
        traj_msg.points = [point]
        self.traj_pub.publish(traj_msg)

    # ------------------------------------------------------------------
    def _show(self, cv_image):
        try:
            cv2.imshow('IBVS wrist camera', cv_image)
            cv2.waitKey(1)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = IBVSCubeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()