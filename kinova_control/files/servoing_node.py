#!/usr/bin/env python3
"""
servoing_node -- model-based IBVS for on-orbit servicing targets.

    image -> ModelTracker (cMo) -> target features (CAD points projected,
    locally refined) -> MissionFSM (what to do) -> IBVSController (twist)
    -> RobotInterface (joint commands)

Select a target at runtime:
    ros2 topic pub --once /spacecraft_servoing/select_target std_msgs/String "{data: screw_1}"
    ros2 topic pub --once /spacecraft_servoing/select_target std_msgs/String "{data: abort}"
Status (JSON):  ros2 topic echo /spacecraft_servoing/status
"""
import json
import math
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .geometry import inv_T, pose_to_T, project, transform
from .mission_fsm import FSMConfig, Inputs, MissionFSM, State
from .refiners import make_refiner
from .robot_interface import RobotInterface
from .targets import load_targets

PKG = 'spacecraft_servoing'


class ServoingNode(Node):

    def __init__(self):
        super().__init__('spacecraft_servoing')
        p = self.declare_parameter
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        # io / robot
        p('image_topic', '/wrist_mounted_camera/image/image_raw')
        p('camera_info_topic', '/wrist_mounted_camera/image/camera_info')
        p('base_link_frame', 'base_link')
        p('camera_frame', '')
        p('camera_convention', 'gazebo_link')
        p('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        p('home_positions', [0.0, 0.0, -1.57, 0.0, -1.57, 0.0])
        p('home_duration_s', 8)
        p('command_mode', 'trajectory')
        p('qdot_max', 0.5)
        # targets / model
        p('targets_file', '')
        p('cao_file', '')
        p('initial_target', '')
        p('target_x', 0.55)
        p('target_y', 0.0)
        p('target_z', 0.0)
        p('target_roll', 0.0)
        p('target_pitch', 0.0)
        p('target_yaw', 0.0)
        # tracker
        p('use_klt', False)
        p('me_range', 10)
        p('me_threshold', 20.0)
        p('max_projection_error_deg', 45.0)
        p('refine_features', True)
        # controller
        p('lambda_0', 1.0)
        p('lambda_inf', 0.4)
        p('lambda_slope', 5.0)
        p('v_max', 0.08)
        p('w_max', 0.4)
        p('interaction', 'mean')
        # fsm
        for k, v in FSMConfig().__dict__.items():
            p('fsm.' + k, v)
        p('show_window', True)

        share = get_package_share_directory(PKG)
        tfile = g('targets_file') or os.path.join(share, 'config', 'targets.yaml')
        model_cfg, self.targets = load_targets(tfile)
        cao = g('cao_file') or os.path.join(share, 'models', model_cfg.get('cao_file', ''))
        if not os.path.isfile(cao):
            raise FileNotFoundError(f'.cao model not found: {cao}')
        self.cao = cao
        self.bMo_guess = pose_to_T([g('target_x'), g('target_y'), g('target_z')],
                                   [g('target_roll'), g('target_pitch'), g('target_yaw')])
        self.get_logger().info(f'targets: {sorted(self.targets)} | model: {cao}')

        self.tracker_cfg = dict(use_klt=bool(g('use_klt')), me_range=int(g('me_range')),
                                me_threshold=float(g('me_threshold')),
                                max_projection_error_deg=float(g('max_projection_error_deg')))
        self.refine = bool(g('refine_features'))
        self.show = bool(g('show_window'))
        self.home = list(g('home_positions'))
        self.home_duration = int(g('home_duration_s'))

        # ViSP is mandatory here: fail early with a clear message
        try:
            from .ibvs_controller import IBVSController
            self.ctrl = IBVSController(g('lambda_0'), g('lambda_inf'), g('lambda_slope'),
                                       g('v_max'), g('w_max'), g('interaction'))
        except ImportError as e:
            cause = e.__cause__ or e.__context__
            self.get_logger().fatal(f'ViSP python bindings unavailable: {e} {cause or ""}')
            raise SystemExit(1)

        self.robot = RobotInterface(self, g('joint_names'), g('base_link_frame'), g('camera_frame'),
                                    g('camera_convention'), g('command_mode'), qdot_max=float(g('qdot_max')))
        fcfg = FSMConfig(**{k: type(v)(g('fsm.' + k)) for k, v in FSMConfig().__dict__.items()})
        self.fsm = MissionFSM(self.targets, fcfg, log=lambda m: self.get_logger().info(m))
        if g('initial_target'):
            self.fsm.request(g('initial_target'))

        self.K = None
        self.tracker = None
        self.last_stamp = None
        self.active = None             # (target name) for which yaw/refiner were set
        self.yaw = 0.0
        self.refiner = None
        self.bridge = CvBridge()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, g('image_topic'), self.image_cb, qos)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self.info_cb, qos)
        self.create_subscription(String, '~/select_target', self.select_cb, 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 1)

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        from .model_tracker import ModelTracker
        try:
            self.tracker = ModelTracker(self.K, self.cao, logger=lambda m: self.get_logger().warn(m),
                                        **self.tracker_cfg)
        except Exception as e:
            self.get_logger().fatal(f'could not create the model-based tracker: {e}')
            raise SystemExit(1)
        self.get_logger().info(f'tracker ready ({self.tracker.mode}, {self.tracker.me_info})')

    def select_cb(self, msg):
        ok, text = self.fsm.request(msg.data)
        (self.get_logger().info if ok else self.get_logger().warn)(text)

    # ------------------------------------------------------------------
    def initial_pose_guess(self):
        """cMo guess for (re)initialisation. Sim: known spawn pose + FK.
        Real mission: replace with a pose estimator / relative navigation."""
        return inv_T(self.robot.camera_pose()) @ self.bMo_guess

    def target_features(self, gray, cMo, tgt):
        """Returns (s Nx3 normalized, uv Nx2) or (None, None) if not servoable."""
        cMf = cMo @ tgt.oMf
        if np.dot(cMf[:3, 2], cMf[:3, 3]) >= 0.0:        # surface faces away from the camera
            return None, None
        P = transform(cMf, tgt.points_f)
        if np.any(P[:, 2] <= 0.02):
            return None, None
        uv, _, Z = project(P, self.K)
        if self.refiner is not None:
            ref = self.refiner.refine(gray, uv, tgt, self.K, Z)
            if ref is not None:
                uv = ref
        x = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]
        return np.column_stack([x, y, Z]), uv

    # ------------------------------------------------------------------
    def image_cb(self, msg):
        self.robot.set_camera_frame(msg.header.frame_id)
        if self.tracker is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.1 if self.last_stamp is None else float(np.clip(stamp - self.last_stamp, 1e-3, 0.5))
        self.last_stamp = stamp
        self.robot.set_period(dt)

        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cMo = self.tracker.track(gray) if self.tracker.ok else None

        # features for the target currently handled by the FSM
        tgt = self.fsm.current_target
        if tgt is not None and tgt.name != self.active and cMo is not None:
            self.active = tgt.name
            self.yaw = tgt.best_yaw(cMo)
            self.refiner = make_refiner(tgt.kind, self.refine)
            self.ctrl.reset(len(tgt.points_f))
            self.get_logger().info(f'target {tgt.name}: {len(tgt.points_f)} points, '
                                   f'yaw {math.degrees(self.yaw):.0f} deg, refiner {self.refiner.name}')
        s, uv, err_px = None, None, None
        if tgt is not None and cMo is not None and self.fsm.z_ref is not None:
            s, uv = self.target_features(gray, cMo, tgt)
            if s is not None:
                sd = tgt.desired_features(self.fsm.z_ref, self.yaw)
                e = np.concatenate([s[:, 0] - sd[:, 0], s[:, 1] - sd[:, 1]]) * self.K[0, 0]
                err_px = float(np.sqrt(np.mean(e ** 2)))

        homed = self.robot.homed
        cmd = self.fsm.step(Inputs(self.robot.ready(), homed, cMo is not None, err_px, dt))

        if cmd.mode == 'home':
            self.robot.send_home(self.home, self.home_duration)
        elif cmd.mode == 'reinit':
            self.robot.hold()
            try:
                self.tracker.init(gray, self.initial_pose_guess())
            except Exception as e:
                self.get_logger().error(f'tracker init failed: {e}', throttle_duration_sec=2.0)
        elif cmd.mode == 'hold':
            self.robot.hold()
        elif cmd.mode == 'servo':
            tgt = self.targets[cmd.target]
            if tgt.name != self.active or s is None:
                self.robot.hold()               # target just switched or not visible
            else:
                sd = tgt.desired_features(cmd.standoff, self.yaw)
                v, _ = self.ctrl.compute(s, sd, cmd.gain_scale)
                if v is None:
                    self.robot.hold()
                else:
                    self.robot.command_twist(v)
                self._draw_target(bgr, uv, sd)

        self._publish(bgr, msg.header, cmd, err_px)

    # ------------------------------------------------------------------
    def _draw_target(self, bgr, uv, sd):
        uvd = np.stack([self.K[0, 0] * sd[:, 0] + self.K[0, 2], self.K[1, 1] * sd[:, 1] + self.K[1, 2]], axis=1)
        for a, b in zip(uv, uvd):
            cv2.circle(bgr, (int(a[0]), int(a[1])), 5, (0, 255, 0), -1)
            cv2.circle(bgr, (int(b[0]), int(b[1])), 6, (0, 0, 255), 2)
            cv2.line(bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 150, 255), 1)

    def _publish(self, bgr, header, cmd, err_px):
        tr = self.tracker
        if tr.ok and tr.cMo is not None:
            P = transform(tr.cMo, tr.model_points)
            if np.all(P[:, 2] > 0.02):
                uvm, _, _ = project(P, self.K)
                for poly in tr.model_polys:
                    cv2.polylines(bgr, [uvm[poly].astype(np.int32)], len(poly) > 2, (255, 128, 0), 1)
        st = self.fsm.state.value
        txt = f'{st} {self.fsm.current or "-"} z_ref={self.fsm.z_ref or 0:.3f} ' \
              f'err={err_px if err_px is not None else float("nan"):.1f}px track_err={tr.last_err:.1f}'
        cv2.putText(bgr, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        status = {'state': st, 'target': self.fsm.current, 'z_ref': self.fsm.z_ref,
                  'err_px': err_px, 'tracking': tr.ok, 'tracker_err_deg': tr.last_err,
                  'command': cmd.mode}
        self.status_pub.publish(String(data=json.dumps(status)))
        if self.fsm.state in (State.APPROACH, State.ALIGN, State.HOLD, State.RETREAT, State.IDLE):
            self.get_logger().info(txt, throttle_duration_sec=1.0)
        out = self.bridge.cv2_to_imgmsg(bgr, 'bgr8')
        out.header = header
        self.debug_pub.publish(out)
        if self.show:
            try:
                cv2.imshow('spacecraft servoing', bgr)
                cv2.waitKey(1)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ServoingNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            try:
                node.robot.hold()
            except Exception:
                pass
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
