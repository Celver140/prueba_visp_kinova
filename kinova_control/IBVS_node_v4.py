#!/usr/bin/env python3
"""
ibvs_cube_node.py

Image-Based Visual Servoing (IBVS) con Segmentación por Profundidad
para aislar la cara superior del cubo y control de velocidad saturado.
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
    try: return kdl.Joint(name)
    except TypeError: pass
    for attr in ('Fixed', 'None'):
        try: return kdl.Joint(name, getattr(kdl.Joint, attr))
        except (AttributeError, TypeError): continue
    raise RuntimeError('Fallo PyKDL.Joint Fijo.')

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
            chain.addSegment(kdl.Segment(joint.child, kdl.Joint(joint.name, F.p, F.M * kdl.Vector(*(joint.axis or [1.,0.,0.])), kdl.Joint.RotAxis), F))
        elif joint.type == 'prismatic':
            chain.addSegment(kdl.Segment(joint.child, kdl.Joint(joint.name, F.p, F.M * kdl.Vector(*(joint.axis or [1.,0.,0.])), kdl.Joint.TransAxis), F))
        else:
            chain.addSegment(kdl.Segment(joint.child, _make_fixed_kdl_joint(joint.name), F))
    return chain

class IBVSCubeNode(Node):

    def __init__(self):
        super().__init__('ibvs_cube_node')

        self.declare_parameter('base_link_frame', 'base_link')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('camera_info_topic', '/wrist_mounted_camera/image/camera_info')
        self.declare_parameter('depth_topic', '/wrist_mounted_camera/depth_image')

        self.declare_parameter('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        
        # --- PARÁMETROS DE VELOCIDAD REDUCIDOS PARA ESTABILIDAD ---
        self.declare_parameter('lambda_gain', 0.10)  
        self.declare_parameter('corner_error_threshold_px', 25.0)
        self.declare_parameter('recenter_error_threshold_px', 80.0)
        self.declare_parameter('joint_vel_limit', 0.1) 
        self.declare_parameter('target_j4', 0.0)
        self.declare_parameter('target_j5', -1.57)
        self.declare_parameter('phase3_gain', 0.4)
        self.declare_parameter('desired_standoff_m', 0.20)
        self.declare_parameter('fallback_depth_m', 0.6)

        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.camera_frame_param = self.get_parameter('camera_frame').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.lambda_gain = float(self.get_parameter('lambda_gain').value)
        self.corner_error_threshold_px = float(self.get_parameter('corner_error_threshold_px').value)
        self.recenter_error_threshold_px = float(self.get_parameter('recenter_error_threshold_px').value)
        self.joint_vel_limit = float(self.get_parameter('joint_vel_limit').value)
        self.target_j4 = float(self.get_parameter('target_j4').value)
        self.target_j5 = float(self.get_parameter('target_j5').value)
        self.phase3_gain = float(self.get_parameter('phase3_gain').value)
        self.Z_d = float(self.get_parameter('desired_standoff_m').value)
        self.fallback_depth_m = float(self.get_parameter('fallback_depth_m').value)

        self.n_joints = len(self.joint_names)

        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/wrist_mounted_camera/image/image_raw', self.image_cb, qos_img)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_img)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, qos_img)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        qos_urdf = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, '/robot_description', self.urdf_cb, qos_urdf)

        self.traj_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.traj_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

        self.bridge = CvBridge()

        self.kdl_chain = None
        self.jac_solver = None
        self.fk_solver = None
        self._urdf_xml = None
        self._resolved_camera_frame = self.camera_frame_param or None

        self.cam = None
        self.latest_depth = None

        self.task = Servo()
        self.task.setServo(Servo.EYEINHAND_CAMERA)
        self.task.setInteractionMatrixType(Servo.CURRENT, Servo.PSEUDO_INVERSE)
        self.task.setLambda(self.lambda_gain)

        self.s = [FeaturePoint() for _ in range(4)]
        self.s_star = [FeaturePoint() for _ in range(4)]
        
        self.u_star = [150.0, 490.0, 490.0, 150.0]
        self.v_star = [110.0, 110.0, 370.0, 370.0]
        for i in range(4): self.task.addFeature(self.s[i], self.s_star[i])
        self._desired_features_built = False

        self.current_q = [0.0] * self.n_joints
        self.has_joint_states = False
        self.state = 0
        self.goal_sent = False
        self.last_stamp = None
        self.prev_corners = None
        
        self.last_dq = np.zeros(self.n_joints)
        self.last_corner_error = float('inf')
        self.recovery_mode = False

        self.R_opt_to_kdl = np.array([
            [ 0.0,  0.0,  1.0], 
            [-1.0,  0.0,  0.0], 
            [ 0.0, -1.0,  0.0]  
        ])
        self.R6_opt_to_kdl = np.block([[self.R_opt_to_kdl, np.zeros((3, 3))], 
                                       [np.zeros((3, 3)), self.R_opt_to_kdl]])

    def camera_info_cb(self, msg: CameraInfo):
        if self.cam is not None: return
        self.cam = CameraParameters()
        self.cam.initPersProjWithoutDistortion(msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        for i in range(4):
            self.s_star[i].buildFrom((self.u_star[i] - self.cam.get_u0()) / self.cam.get_px(),
                                     (self.v_star[i] - self.cam.get_v0()) / self.cam.get_py(), self.Z_d)
        self._desired_features_built = True

    def depth_cb(self, msg: Image):
        try: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32) / 1000.0

    def _sample_depth(self, u: float, v: float):
        if self.latest_depth is None: return None
        h, w = self.latest_depth.shape[:2]
        z = float(self.latest_depth[int(np.clip(round(v), 0, h - 1)), int(np.clip(round(u), 0, w - 1))])
        return z if (np.isfinite(z) and z > 0.01) else None

    def urdf_cb(self, msg: String):
        if self.kdl_chain is not None: return
        self._urdf_xml = msg.data
        if self._resolved_camera_frame: self._try_build_chain()

    def _try_build_chain(self):
        try:
            self.kdl_chain = build_kdl_chain_from_urdf(self._urdf_xml, self.base_link_frame, self._resolved_camera_frame)
            self.jac_solver, self.fk_solver = kdl.ChainJntToJacSolver(self.kdl_chain), kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        except Exception as e: pass

    def joint_state_cb(self, msg: JointState):
        for i, name in enumerate(self.joint_names):
            if name in msg.name: self.current_q[i] = msg.position[msg.name.index(name)]
        self.has_joint_states = True

    def send_initial_trajectory(self):
        if self.goal_sent or not self.traj_client.wait_for_server(timeout_sec=2.0): return
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, -1.57, 0.0, -1.57, 0.0]
        point.time_from_start.sec = 10
        goal_msg.trajectory.points = [point]
        self.goal_sent = True
        self.traj_client.send_goal_async(goal_msg).add_done_callback(lambda f: f.result().accepted and f.result().get_result_async().add_done_callback(lambda _: setattr(self, 'state', 1)))

    def get_target_features(self, cv_image):
        """
        Segmentación de la cara superior utilizando color y profundidad fusionados.
        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) | \
                   cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        
        # --- AISLAMIENTO DE LA CARA SUPERIOR VÍA PROFUNDIDAD ---
        mask = mask_red
        if self.latest_depth is not None:
            h, w = mask_red.shape
            depth_map = cv2.resize(self.latest_depth, (w, h), interpolation=cv2.INTER_NEAREST) if self.latest_depth.shape[:2] != (h, w) else self.latest_depth
            
            valid_depths = depth_map[mask_red > 0]
            valid_depths = valid_depths[np.isfinite(valid_depths) & (valid_depths > 0.05)]
            
            if len(valid_depths) > 0:
                min_z = np.min(valid_depths)
                # Máscara de profundidad: Solo píxeles rojos que estén a < 3.5 cm de la cara más alta
                depth_mask = ((depth_map >= min_z - 0.01) & (depth_map <= min_z + 0.035)).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask_red, depth_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 50: return None

        approx = cv2.approxPolyDP(largest, 0.04 * cv2.arcLength(largest, True), True)
        if len(approx) == 4:
            raw_corners = approx.reshape(4, 2).astype(np.float32)
            corners = self._match_corners_to_previous(raw_corners, self.prev_corners) if self.prev_corners is not None else self._order_corners_initial(raw_corners)
            self.prev_corners = corners
            return corners
        
        self.prev_corners = None
        return None

    def _order_corners_initial(self, pts):
        rect, ssum, diff = np.zeros((4, 2), dtype='float32'), pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(ssum)], pts[np.argmax(ssum)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def _match_corners_to_previous(self, pts, prev):
        best_perm, best_cost = None, float('inf')
        for perm in permutations(range(4)):
            cost = sum(np.linalg.norm(pts[perm[i]] - prev[i]) for i in range(4))
            if cost < best_cost: best_cost, best_perm = cost, perm
        return pts[list(best_perm)]

    def _solve_qdot(self, J_cols: np.ndarray, v_base: np.ndarray) -> np.ndarray:
        qdot, *_ = np.linalg.lstsq(J_cols, v_base, rcond=None)
        return np.clip(qdot, -self.joint_vel_limit, self.joint_vel_limit)

    def image_cb(self, msg: Image):
        if not self._resolved_camera_frame:
            self._resolved_camera_frame = msg.header.frame_id
            self._try_build_chain()

        if self.state == 0: self.send_initial_trajectory(); return
        if not self.has_joint_states or self.kdl_chain is None or not self._desired_features_built: return

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        dt = 0.05 if self.last_stamp is None else max(1e-3, min(0.2, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - self.last_stamp))
        self.last_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        corners = self.get_target_features(cv_image)

        if corners is None:
            self.get_logger().warn('>>> [ALERTA] Esquinas perdidas. Rollback... <<<', throttle_duration_sec=1.0)
            self._publish_qdot(-1.2 * (self.last_dq / dt), dt)
            try: cv2.imshow('IBVS wrist camera', cv_image); cv2.waitKey(1)
            except Exception: pass
            return

        corner_error_px = math.sqrt(np.mean((corners[:, 0] - np.array(self.u_star)) ** 2 + (corners[:, 1] - np.array(self.v_star)) ** 2))

        if corner_error_px > (self.last_corner_error + 35.0) and not self.recovery_mode:
            self.get_logger().warn(f'>>> [ALERTA] Discrepancia masiva. Rollback... <<<')
            self._publish_qdot(-1.0 * (self.last_dq / dt), dt)
            self.recovery_mode = True
            return

        self.recovery_mode, self.last_corner_error = False, corner_error_px

        for i in range(4):
            u_act, v_act = int(corners[i][0]), int(corners[i][1])
            u_tar, v_tar = int(self.u_star[i]), int(self.v_star[i])
            cv2.circle(cv_image, (u_act, v_act), 6, (0, 255, 0), -1)  
            cv2.circle(cv_image, (u_tar, v_tar), 6, (0, 0, 255), -1)  
            cv2.line(cv_image, (u_act, v_act), (u_tar, v_tar), (0, 150, 255), 2)

        if self.state == 1 and corner_error_px < self.recenter_error_threshold_px:
            self.state = 2
            self.get_logger().warn('>>> STATE 2: Esquinas aproximadas. Ajuste Fino Activo... <<<')
        elif self.state == 2 and corner_error_px > (self.recenter_error_threshold_px + 20.0):
            self.state = 1
        elif self.state == 2 and corner_error_px < self.corner_error_threshold_px:
            self.state = 3
            self.get_logger().warn('>>> STATE 3: Match completado. Ejecutando Ortogonalización... <<<')
        elif self.state == 3 and corner_error_px > self.recenter_error_threshold_px:
            self.state = 2

        for i in range(4):
            z_c = self._sample_depth(corners[i][0], corners[i][1]) or self.fallback_depth_m
            x, y = (corners[i][0] - self.cam.get_u0()) / self.cam.get_px(), (corners[i][1] - self.cam.get_v0()) / self.cam.get_py()
            self.s[i].buildFrom(x, y, z_c)
            
        v_c = self.task.computeControlLaw()
        v_opt = np.array([v_c[i] for i in range(6)], dtype=float)

        if self.state == 1:
            descent_factor = max(0.0, 1.0 - (corner_error_px / 120.0))
            v_opt[2] = v_opt[2] * descent_factor
            v_opt[0] = v_opt[0] * 1.5 
            v_opt[1] = v_opt[1] * 1.5
            self.get_logger().info(f'[STATE 1] Centrando y bajando | Error Esq: {corner_error_px:.1f}px', throttle_duration_sec=0.5)
        elif self.state == 2:
            self.get_logger().info(f'[STATE 2] Ajuste Angular Fino | Error Esq: {corner_error_px:.1f}px', throttle_duration_sec=0.5)

        # --- SATURACIÓN CARTESIANA DE SEGURIDAD ---
        v_trans_norm = np.linalg.norm(v_opt[:3])
        if v_trans_norm > 0.05: v_opt[:3] = v_opt[:3] * (0.05 / v_trans_norm)
        v_rot_norm = np.linalg.norm(v_opt[3:])
        if v_rot_norm > 0.20: v_opt[3:] = v_opt[3:] * (0.20 / v_rot_norm)

        v_kdl_frame = self.R6_opt_to_kdl @ v_opt

        q_kdl = kdl.JntArray(self.n_joints)
        for i in range(self.n_joints): q_kdl[i] = self.current_q[i]
        jac_kdl, frame = kdl.Jacobian(self.n_joints), kdl.Frame()
        self.jac_solver.JntToJac(q_kdl, jac_kdl)
        self.fk_solver.JntToCart(q_kdl, frame)

        R6_kdl_to_base = np.block([[kdl_rotation_to_np(frame.M), np.zeros((3, 3))], [np.zeros((3, 3)), kdl_rotation_to_np(frame.M)]])
        v_base = R6_kdl_to_base @ v_kdl_frame
        
        J = kdl_jacobian_to_np(jac_kdl)
        qdot = np.zeros(self.n_joints)

        if self.state in (1, 2): 
            qdot = self._solve_qdot(J, v_base)
        elif self.state == 3:
            qdot[3] = float(np.clip(self.phase3_gain * (self.target_j4 - self.current_q[3]), -self.joint_vel_limit, self.joint_vel_limit))
            qdot[4] = float(np.clip(self.phase3_gain * (self.target_j5 - self.current_q[4]), -self.joint_vel_limit, self.joint_vel_limit))

            induced_v = J[:, 3:5] @ np.array([qdot[3], qdot[4]])
            idx_free = [0, 1, 2, 5]
            qdot_free = self._solve_qdot(J[:, idx_free], v_base - induced_v)
            for i, idx in enumerate(idx_free): qdot[idx] = qdot_free[i]
            
            self.get_logger().info(f'[STATE 3] Ortogonalizando...', throttle_duration_sec=0.5)

        self.last_dq = qdot * dt
        self._publish_qdot(qdot, dt)
        
        try: cv2.imshow('IBVS wrist camera', cv_image); cv2.waitKey(1)
        except Exception: pass

    def _publish_qdot(self, qdot: np.ndarray, dt: float):
        traj_msg, point = JointTrajectory(), JointTrajectoryPoint()
        traj_msg.joint_names = self.joint_names
        point.positions = [self.current_q[i] + float(qdot[i]) * dt for i in range(self.n_joints)]
        point.time_from_start.nanosec = int(dt * 1e9)
        traj_msg.points = [point]
        self.traj_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(IBVSCubeNode())
    except KeyboardInterrupt: pass
    finally: cv2.destroyAllWindows(); rclpy.shutdown()

if __name__ == '__main__': main()