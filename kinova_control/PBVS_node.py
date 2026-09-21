#!/usr/bin/env python3
"""
pbvs_cube_node.py

Position-Based Visual Servoing (PBVS) of a red cube, using ViSP for pose
estimation + the visual control law, and PyKDL (built from the live
/robot_description, no kdl_parser_py dependency) for the robot Jacobian --
same integration approach as ibvs_cube_node.py, different control law.

Why PBVS needs no phase split (unlike the IBVS version)
---------------------------------------------------------
IBVS only measures 2D pixel error, so it can't tell "centered" apart from
"centered but tilted" -- that's why that node needed a two-phase,
task-priority scheme (center first, then rotate J4/J5 to perpendicular
while J1-J3 compensate).

PBVS instead estimates the actual 3D pose cMo of the cube's top face
(translation + orientation) from its 4 detected corners plus the known
face size. The full 6-DOF pose error (built from cdMc = cdMo * cMo^-1)
already encodes *both* "camera centered over the cube" and "camera
looking straight down at it" simultaneously. So a single PBVS task,
solved against the full J1-J5 Jacobian in one state, drives both at once
-- no hand-off between phases required.

Perception
----------
Same red-cube top-face corner detector as the IBVS node (OpenCV, HSV
threshold + 4-point polygon approx). The 4 corners are paired with their
known 3D positions on the cube's top face (a square of side
`cube_face_size_m`, in an object frame centered on the face) and PBVS's
usual vpPose (Dementhon+Lagrange init, refined with virtual visual
servoing) estimates cMo.

IMPORTANT -- things you must verify/adjust for your setup
-----------------------------------------------------------
- base_link_frame / camera_frame / cube_face_size_m / cam intrinsics:
  same caveats as the IBVS node -- verify against your URDF/camera_info.
- desired_translation / desired_rpy_deg define cdMo: where you want the
  camera relative to the cube's top face. Defaults: camera 25 cm above
  the face, looking straight down (180 deg flip about X so the camera's
  Z axis points into the face, opposite the face normal).
- Corner ordering: detect_red_cube_corners() returns [top-left, top-right,
  bottom-right, bottom-left] in the image. The object-frame 3D points
  below are defined in that same order -- if your cube/camera orientation
  makes this pairing ambiguous (e.g. face nearly square-on with symmetric
  detection), pose estimation can flip; watch the logged pose if
  servoing seems to fight itself.
"""

import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge

import PyKDL as kdl
from urdf_parser_py.urdf import URDF as UrdfModel

from visp.core import (
    CameraParameters,
    HomogeneousMatrix,
    RotationMatrix,
    TranslationVector,
    ThetaUVector,
    Point,
    Math as vpMath,
)
from visp.visual_features import FeatureTranslation, FeatureThetaU
from visp.vs import Servo
from visp.vision import Pose as vpPoseClass


# =======================================================================
# KDL helpers -- identical to ibvs_cube_node.py. See that file for the
# rationale (kdl_parser_py isn't installed, so the URDF->KDL chain is
# built directly from urdf_parser_py + PyKDL).
# =======================================================================
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


class PBVSCubeNode(Node):

    def __init__(self):
        super().__init__('pbvs_cube_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('base_link_frame', 'base_link')
        # Empty string = auto-detect the camera's optical frame from the
        # incoming Image message's header.frame_id. This matters: the
        # image Jacobian / vpServo velocity assumes the camera's OPTICAL
        # frame (Z along the optical axis, X-right, Y-down, matching
        # pixels), which is usually NOT the same frame as a mechanical
        # mount link like "camera_link" (REP-103: X-forward, Z-up). Using
        # the wrong one silently rotates every velocity command and the
        # servo will never converge. Only override this if you are sure
        # of your URDF's frame naming.
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('joint_names',
                                ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('lambda_gain', 0.25)
        self.declare_parameter('joint_vel_limit', 0.25)  # rad/s per joint
        self.declare_parameter('convergence_error_norm', 0.01)

        self.declare_parameter('cube_face_size_m', 0.05)

        # --- Cube-loss recovery ---
        self.declare_parameter('lost_frames_before_recovery', 5)
        self.declare_parameter('history_record_period_s', 0.2)
        self.declare_parameter('history_max_age_s', 6.0)
        self.declare_parameter('recovery_position_tolerance', 0.03)  # rad
        self.declare_parameter('recovery_gain', 0.8)

        # Desired camera pose relative to the cube's top face (cdMo).
        self.declare_parameter('desired_translation', [0.0, 0.0, 0.25])
        self.declare_parameter('desired_rpy_deg', [180.0, 0.0, 0.0])

        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.camera_frame_param = self.get_parameter('camera_frame').value  # '' = auto-detect
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.lambda_gain = float(self.get_parameter('lambda_gain').value)
        self.joint_vel_limit = float(self.get_parameter('joint_vel_limit').value)
        self.convergence_error_norm = float(self.get_parameter('convergence_error_norm').value)
        self.cube_face_size_m = float(self.get_parameter('cube_face_size_m').value)

        self.lost_frames_before_recovery = int(self.get_parameter('lost_frames_before_recovery').value)
        self.history_record_period_s = float(self.get_parameter('history_record_period_s').value)
        self.history_max_age_s = float(self.get_parameter('history_max_age_s').value)
        self.recovery_position_tolerance = float(self.get_parameter('recovery_position_tolerance').value)
        self.recovery_gain = float(self.get_parameter('recovery_gain').value)

        self.n_joints = len(self.joint_names)

        # ---------------- Desired pose cdMo ----------------
        dt = self.get_parameter('desired_translation').value
        drpy = self.get_parameter('desired_rpy_deg').value
        t = TranslationVector(dt[0], dt[1], dt[2])
        tu = ThetaUVector(vpMath.rad(drpy[0]), vpMath.rad(drpy[1]), vpMath.rad(drpy[2]))
        R = RotationMatrix(tu)
        self.cdMo = HomogeneousMatrix()
        self.cdMo.insert(R)
        self.cdMo.insert(t)

        # 3D corners of the cube's top face in its own object frame,
        # ordered [top-left, top-right, bottom-right, bottom-left] to
        # match detect_red_cube_corners()'s pixel ordering. Z=0 plane,
        # face normal along +Z.
        s = self.cube_face_size_m / 2.0
        self.object_points = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]

        # ---------------- ROS I/O ----------------
        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/wrist_mounted_camera/image/image_raw',
                                  self.image_cb, qos_img)
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

        # ---------------- Kinematics (filled once URDF + camera frame are known) ----------------
        self.kdl_chain = None
        self.jac_solver = None
        self.fk_solver = None
        self._urdf_xml = None
        self._resolved_camera_frame = self.camera_frame_param or None

        # ---------------- ViSP camera + PBVS task ----------------
        self.cam = CameraParameters(554.25469, 554.25469, 320.5, 240.5)  # VERIFY vs your sim camera_info

        self.task = Servo()
        self.task.setServo(Servo.EYEINHAND_CAMERA)
        self.task.setInteractionMatrixType(Servo.CURRENT, Servo.PSEUDO_INVERSE)
        self.task.setLambda(self.lambda_gain)

        self.s_t = FeatureTranslation(FeatureTranslation.cdMc)
        self.s_tu = FeatureThetaU(FeatureThetaU.cdRc)
        self.s_t_star = FeatureTranslation(FeatureTranslation.cdMc)   # stays zero
        self.s_tu_star = FeatureThetaU(FeatureThetaU.cdRc)            # stays zero
        self.task.addFeature(self.s_t, self.s_t_star)
        self.task.addFeature(self.s_tu, self.s_tu_star)

        # ---------------- State ----------------
        self.current_q = [0.0] * self.n_joints
        self.has_joint_states = False
        self.state = 0  # 0: move to seed pose, 1: PBVS active
        self.goal_sent = False
        self.last_stamp = None
        self.converged_logged = False

        # Cube-loss recovery: a stack of (t, q) snapshots recorded while the
        # cube was visible; on loss we pop them and retrace the path back.
        self.q_history = []
        self.rollback_target = None  # (t, q) currently being pursued
        self.lost_frame_count = 0
        self._last_history_stamp = None
        self.ever_seen_cube = False
        self.seed_positions = [0.0, 0.0, -1.57, 0.0, -1.57, 0.0]

        self.get_logger().info('PBVS node started. Waiting for /robot_description and joint states...')

    # ------------------------------------------------------------------
    def urdf_cb(self, msg: String):
        if self.kdl_chain is not None:
            return
        self._urdf_xml = msg.data
        self._try_build_chain()

    # ------------------------------------------------------------------
    def _try_build_chain(self):
        if self.kdl_chain is not None:
            return
        if self._urdf_xml is None or self._resolved_camera_frame is None:
            return  # still waiting on one of the two prerequisites
        try:
            self.kdl_chain = build_kdl_chain_from_urdf(
                self._urdf_xml, self.base_link_frame, self._resolved_camera_frame)
        except KeyError as e:
            self.get_logger().error(
                f'Could not build KDL chain {self.base_link_frame} -> '
                f'{self._resolved_camera_frame}: missing joint/link {e}. Check '
                f'base_link_frame and that the image frame_id exists as a link '
                f'in the URDF.')
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
        point.positions = self.seed_positions
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
        self.get_logger().info('Seed pose reached -- starting PBVS.')
        self.state = 1

    # ------------------------------------------------------------------
    def detect_red_cube_corners(self, cv_image):
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
                    pts = approx.reshape(4, 2).astype(np.float32)
                    rect = np.zeros((4, 2), dtype='float32')
                    ssum = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(ssum)]   # top-left
                    rect[2] = pts[np.argmax(ssum)]   # bottom-right
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]   # top-right
                    rect[3] = pts[np.argmax(diff)]   # bottom-left
                    return rect
        return None

    # ------------------------------------------------------------------
    def _pixel_to_meter(self, u, v):
        return ((u - self.cam.get_u0()) / self.cam.get_px(),
                (v - self.cam.get_v0()) / self.cam.get_py())

    def _estimate_pose(self, corners):
        """4 pixel corners (in object_points order) -> vpHomogeneousMatrix cMo,
        or None on failure."""
        pose = vpPoseClass()
        for (oX, oY, oZ), (u, v) in zip(self.object_points, corners):
            p = Point()
            p.set_oX(oX)
            p.set_oY(oY)
            p.set_oZ(oZ)
            x, y = self._pixel_to_meter(u, v)
            p.set_x(x)
            p.set_y(y)
            pose.addPoint(p)

        cMo = HomogeneousMatrix()
        try:
            ok = pose.computePose(vpPoseClass.DEMENTHON_LAGRANGE_VIRTUAL_VS, cMo)
        except TypeError:
            result = pose.computePose(vpPoseClass.DEMENTHON_LAGRANGE_VIRTUAL_VS)
            ok, cMo = result if isinstance(result, tuple) else (True, result)

        return cMo if ok else None

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

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.05 if self.last_stamp is None else max(1e-3, min(0.2, stamp - self.last_stamp))
        self.last_stamp = stamp

        detected = self.detect_red_cube_corners(cv_image)

        if detected is None:
            self._handle_cube_lost(dt)
            self._show(cv_image)
            return

        # Cube visible: reset recovery bookkeeping and record this pose as
        # a known-good waypoint for future recovery.
        if self.lost_frame_count >= self.lost_frames_before_recovery:
            self.get_logger().info('Cube reacquired -- resuming PBVS.')
        self.lost_frame_count = 0
        self.rollback_target = None
        self.ever_seen_cube = True
        self._record_history(stamp)

        for (u, v) in detected:
            cv2.circle(cv_image, (int(u), int(v)), 5, (0, 255, 0), -1)

        cMo = self._estimate_pose(detected)
        if cMo is None:
            self.get_logger().warn('Pose estimation failed -- holding position.',
                                    throttle_duration_sec=1.0)
            self._publish_qdot(np.zeros(self.n_joints), dt)
            self._show(cv_image)
            return

        # --- PBVS control law ---
        cdMc = self.cdMo * cMo.inverse()
        self.s_t.buildFrom(cdMc)
        self.s_tu.buildFrom(cdMc)

        v_c = self.task.computeControlLaw()
        err = self.task.getError()
        norm_e = err.frobeniusNorm() if hasattr(err, 'frobeniusNorm') else float(np.linalg.norm(np.array(err)))
        v_c_np = np.array([v_c[i] for i in range(6)], dtype=float)

        J, R = self._jacobian_and_camera_rotation()
        R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
        v_base = R6 @ v_c_np

        # Solve for J1-J5 jointly (J6 locked) -- see module docstring for
        # why PBVS doesn't need the IBVS node's two-phase split.
        qdot = np.zeros(self.n_joints)
        qdot5, *_ = np.linalg.lstsq(J[:, :5], v_base, rcond=None)
        qdot[:5] = np.clip(qdot5, -self.joint_vel_limit, self.joint_vel_limit)

        self._publish_qdot(qdot, dt)
        self._show(cv_image)

        if norm_e < self.convergence_error_norm:
            if not self.converged_logged:
                self.get_logger().info(f'PBVS converged, |e|={norm_e:.5f}')
                self.converged_logged = True
        else:
            self.converged_logged = False
            self.get_logger().info(f'[PBVS] |e|={norm_e:.5f}', throttle_duration_sec=0.5)

    # ------------------------------------------------------------------
    def _record_history(self, stamp: float):
        """Push the current joint configuration as a known-good waypoint,
        throttled, and trim anything older than history_max_age_s."""
        if self.q_history and (stamp - self.q_history[-1][0]) < self.history_record_period_s:
            return
        self.q_history.append((stamp, list(self.current_q)))
        cutoff = stamp - self.history_max_age_s
        while self.q_history and self.q_history[0][0] < cutoff:
            self.q_history.pop(0)

    # ------------------------------------------------------------------
    def _handle_cube_lost(self, dt: float):
        self.lost_frame_count += 1

        if self.lost_frame_count < self.lost_frames_before_recovery:
            # Brief flicker (occlusion, motion blur, a bad frame) -- don't
            # panic, just hold still and wait for it to come back.
            self._publish_qdot(np.zeros(self.n_joints), dt)
            return

        if self.rollback_target is None:
            if not self.q_history:
                if not self.ever_seen_cube:
                    # Never tracked it even once -- there's nothing to
                    # retrace to. Drive back to the known seed pose (where
                    # it's presumably visible) instead of freezing forever.
                    self.get_logger().warn(
                        'Cube never seen -- returning to seed pose to try to '
                        'reacquire it. If it stays lost from there, reposition '
                        'the cube or check camera_frame/cam intrinsics/HSV '
                        'thresholds.', throttle_duration_sec=2.0)
                    error = np.array(self.seed_positions) - np.array(self.current_q)
                    qdot = np.zeros(self.n_joints)
                    qdot[:5] = np.clip(self.recovery_gain * error[:5],
                                        -self.joint_vel_limit, self.joint_vel_limit)
                    self._publish_qdot(qdot, dt)
                else:
                    self.get_logger().warn(
                        'Cube lost and recovery history exhausted -- holding position.',
                        throttle_duration_sec=2.0)
                    self._publish_qdot(np.zeros(self.n_joints), dt)
                return
            self.rollback_target = self.q_history.pop()
            self.get_logger().warn(
                'Cube lost -- retracing path back toward where it was last seen.')

        target_q = np.array(self.rollback_target[1])
        current_q = np.array(self.current_q)
        error = target_q - current_q

        if np.max(np.abs(error)) < self.recovery_position_tolerance:
            # Reached this waypoint; step to the next (older) one.
            if self.q_history:
                self.rollback_target = self.q_history.pop()
                target_q = np.array(self.rollback_target[1])
                error = target_q - current_q
            else:
                self.get_logger().warn(
                    'Retraced back to the oldest recorded pose and still no '
                    'cube in view -- holding here.', throttle_duration_sec=2.0)
                self._publish_qdot(np.zeros(self.n_joints), dt)
                return

        qdot = np.zeros(self.n_joints)
        qdot[:5] = np.clip(self.recovery_gain * error[:5],
                            -self.joint_vel_limit, self.joint_vel_limit)
        self._publish_qdot(qdot, dt)
        self.get_logger().info('[RECOVERY] retracing toward last known-good configuration',
                                throttle_duration_sec=5.0)

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
            cv2.imshow('PBVS wrist camera', cv_image)
            cv2.waitKey(1)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PBVSCubeNode()
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