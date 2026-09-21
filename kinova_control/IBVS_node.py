#!/usr/bin/env python3
"""
IBVS_node_v4.py  --  Autonomous IBVS approach to the top face of a cube
=======================================================================

Kinova Gen3 6-DoF, ROS 2 Humble, ViSP (python bindings) + PyKDL.

Behaviour
---------
INIT     Move to the home pose (FollowJointTrajectory action).
SERVO    Single 4-point IBVS law (ViSP vpServo, all 6 joints).
         * Desired features = a CENTERED SQUARE. Reaching it implies that
           the camera is centered AND parallel to the face (no joint-space
           "orthogonalisation" phase needed).
         * Depth per corner from solvePnP(IPPE_SQUARE) with the known face
           size (no depth sensor needed, works identically on the real robot).
         * Autonomous descent: the reference depth Z_ref goes down only while
           the square is centered and the robot keeps up; it goes UP again if
           any corner approaches the image border (field-of-view guard).
SEARCH   Red blob visible but no clean quadrilateral: center the blob
         centroid (and back off if it touches the border).
LOST     Nothing visible: hold a few frames, then retrace the joint history
         to where the cube was last seen (or go back to home).
AXIS TEST  (param axis_test = vx|vy|vz|wz) publish a constant OPTICAL-frame
         twist to verify the camera frame convention before servoing:
           vz>0 -> blob grows,  vx>0 -> blob moves LEFT,
           vy>0 -> blob moves UP, wz>0 -> image rotates counter-clockwise.

Camera convention (param camera_convention)
-------------------------------------------
'gazebo_link' : image header frame is a Gazebo Classic sensor link
                (camera looks along +X of that link)  -> remap applied.
'optical'     : image header frame is a REP-103 optical frame
                (Z forward, X right, Y down)          -> identity.
"""

import math
from collections import deque
from enum import Enum

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import String, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

import PyKDL as kdl
from urdf_parser_py.urdf import URDF as UrdfModel



# ----------------------------------------------------------------------
# ViSP import with a numpy fallback implementing the SAME 4-point IBVS law
# (point interaction matrix, CURRENT or MEAN, pseudo-inverse). Lets you test
# the pipeline while the ViSP python bindings are being fixed.
# ----------------------------------------------------------------------
class _NpFeaturePoint:
    def __init__(self):
        self.x, self.y, self.Z = 0.0, 0.0, 1.0

    def buildFrom(self, x, y, Z):
        self.x, self.y, self.Z = float(x), float(y), float(Z)

    def L(self):
        x, y, Z = self.x, self.y, self.Z
        return np.array([[-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y],
                         [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x]])


class _NpServo:
    EYEINHAND_CAMERA, CURRENT, DESIRED, MEAN, PSEUDO_INVERSE = 0, 0, 1, 2, 0

    def __init__(self):
        self.pairs, self.lam, self.itype, self.err = [], 0.5, 0, np.zeros(0)

    def setServo(self, _):
        pass

    def setInteractionMatrixType(self, itype, _=None):
        self.itype = itype

    def setLambda(self, lam):
        self.lam = float(lam)

    def addFeature(self, s, sd):
        self.pairs.append((s, sd))

    def computeControlLaw(self):
        self.err = np.concatenate([[s.x - sd.x, s.y - sd.y] for s, sd in self.pairs])
        Ls = np.vstack([s.L() for s, _ in self.pairs])
        if self.itype == self.MEAN:
            Ls = 0.5 * (Ls + np.vstack([sd.L() for _, sd in self.pairs]))
        if not (np.all(np.isfinite(Ls)) and np.all(np.isfinite(self.err))):
            raise ValueError('non-finite interaction matrix')
        return -self.lam * np.linalg.pinv(Ls, rcond=1e-6) @ self.err

    def getError(self):
        return self.err


try:
    from visp.visual_features import FeaturePoint
    from visp.vs import Servo
    VISP_BACKEND = 'ViSP'
except ImportError as _visp_err:
    FeaturePoint, Servo = _NpFeaturePoint, _NpServo
    VISP_BACKEND = f'NUMPY FALLBACK (ViSP import failed: {_visp_err})'


# ======================================================================
# KDL helpers (URDF -> KDL chain without kdl_parser_py)
# ======================================================================
def kdl_jacobian_to_np(jac):
    return np.array([[jac[i, j] for j in range(jac.columns())] for i in range(jac.rows())])


def kdl_rotation_to_np(R):
    return np.array([[R[i, j] for j in range(3)] for i in range(3)])


def _make_fixed_kdl_joint(name):
    try:
        return kdl.Joint(name)
    except TypeError:
        pass
    for attr in ('Fixed', 'None'):
        try:
            return kdl.Joint(name, getattr(kdl.Joint, attr))
        except (AttributeError, TypeError):
            continue
    raise RuntimeError('Cannot build a fixed PyKDL.Joint with this PyKDL build.')


def _origin_to_kdl_frame(origin):
    rpy = list(origin.rpy) if (origin is not None and origin.rpy) else [0.0, 0.0, 0.0]
    xyz = list(origin.xyz) if (origin is not None and origin.xyz) else [0.0, 0.0, 0.0]
    return kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))


def build_kdl_chain_from_urdf(urdf_xml, base_link, tip_link):
    robot = UrdfModel.from_xml_string(urdf_xml)
    chain = kdl.Chain()
    for jname in robot.get_chain(base_link, tip_link, links=False, joints=True):
        joint = robot.joint_map[jname]
        F = _origin_to_kdl_frame(joint.origin)
        axis = joint.axis if joint.axis else [1.0, 0.0, 0.0]
        if joint.type in ('revolute', 'continuous'):
            kj = kdl.Joint(joint.name, F.p, F.M * kdl.Vector(*axis), kdl.Joint.RotAxis)
        elif joint.type == 'prismatic':
            kj = kdl.Joint(joint.name, F.p, F.M * kdl.Vector(*axis), kdl.Joint.TransAxis)
        else:
            kj = _make_fixed_kdl_joint(joint.name)
        chain.addSegment(kdl.Segment(joint.child, kj, F))
    return chain


def colvec_to_np(v):
    try:
        return np.array(v, dtype=float).ravel()
    except Exception:
        return np.array([v[i] for i in range(v.getRows())], dtype=float)


# Optical (Z fwd, X right, Y down) -> Gazebo Classic sensor link (X fwd, Y left, Z up)
R_LINK_OPT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])


# ======================================================================
# PERCEPTION (pure OpenCV/numpy) -- BEGIN
# ======================================================================
def _clockwise(pts):
    """Order a 4-point polygon clockwise on screen (image y down)."""
    x, y = pts[:, 0], pts[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        return pts[::-1].copy()
    return pts


class CubeTopFaceDetector:
    """
    Finds the TOP face of a uniformly coloured cube.

    1. Colour mask -> largest blob (the whole visible cube: top + sides).
    2. Faces of a single-colour cube differ in SHADING (each face has a
       different angle to the light). Edges of the V channel inside the blob
       split it into one region per visible face.
    3. Every region that is a convex quadrilateral is a candidate face.
       Since all cube faces are squares of known size, each candidate gets a
       6-DoF pose with solvePnP(IPPE_SQUARE) -> its outward normal.
    4. The top face is the candidate whose normal is closest to
         'base_z' : the robot base +Z (table / gravity scenario), or
         'camera' : the camera viewing direction (most fronto-parallel face,
                    use this where "up" is undefined, e.g. in orbit).
       While tracking, among the valid candidates the one closest to the
       previous corners is preferred (temporal consistency).
    """

    def __init__(self, K, face_size, hsv_ranges, min_blob_area=80.0, min_face_area=150.0,
                 poly_eps=0.03, min_side_ratio=0.3, canny_lo=15, canny_hi=45,
                 edge_dilate=1, top_reference='base_z', min_top_score=0.5, track_gate_px=40.0,
                 max_reproj_px=3.0):
        self.K = np.asarray(K, dtype=float)
        self.max_reproj_px = max_reproj_px
        s = face_size / 2.0
        # IPPE_SQUARE order; with clockwise image order the object +Z points
        # toward the camera (= outward normal of the observed face).
        self.obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])
        self.hsv_ranges = hsv_ranges
        self.min_blob_area = min_blob_area
        self.min_face_area = min_face_area
        self.poly_eps = poly_eps
        self.min_side_ratio = min_side_ratio
        self.canny_lo, self.canny_hi = canny_lo, canny_hi
        self.edge_dilate = edge_dilate
        self.top_reference = top_reference
        self.min_top_score = min_top_score
        self.track_gate_px = track_gate_px
        self.k3 = np.ones((3, 3), np.uint8)

    # ------------------------------------------------------------------
    def _quad(self, contour):
        peri = cv2.arcLength(contour, True)
        for eps in (self.poly_eps, 1.5 * self.poly_eps, 2.0 * self.poly_eps):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                break
        else:
            return None
        if not cv2.isContourConvex(approx):
            return None
        pts = approx.reshape(4, 2).astype(np.float64)
        sides = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        if sides.min() / max(sides.max(), 1e-6) < self.min_side_ratio:
            return None
        return _clockwise(pts)

    _JITTER = ((0, 0.25, 0.0), (1, 0.0, 0.25), (2, -0.25, 0.0), (3, 0.0, -0.25))

    def _reproj(self, rvec, t, pts):
        proj, _ = cv2.projectPoints(self.obj, rvec, t, self.K, None)
        return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - pts, axis=1)))

    def _valid(self, rvec, t):
        rvec = np.asarray(rvec, dtype=float).reshape(3)
        t = np.asarray(t, dtype=float).reshape(3)
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(t))) or t[2] <= 0.0:
            return None
        R, _ = cv2.Rodrigues(rvec)
        if not np.all(np.isfinite(R)) or np.dot(R[:, 2], t) >= 0.0:   # outward normal must face camera
            return None
        return R, t, rvec

    def _ippe(self, q):
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                self.obj, q.reshape(4, 1, 2), self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return []
        return [v for v in (self._valid(rvecs[k], tvecs[k]) for k in range(n)) if v is not None]

    def _iterative(self, q):
        """Levenberg-Marquardt from an analytic fronto-parallel guess.
        Always well conditioned, including the exactly fronto-parallel case
        where IPPE degenerates."""
        d = q[1] - q[0]
        th = math.atan2(d[1], d[0])
        c, s_ = math.cos(th), math.sin(th)
        R0 = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])
        side = float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))
        Z0 = self.K[0, 0] * 2.0 * self.obj[1, 0] / max(side, 1.0)
        cu, cv_ = q.mean(axis=0)
        t0 = np.array([(cu - self.K[0, 2]) / self.K[0, 0] * Z0,
                       (cv_ - self.K[1, 2]) / self.K[1, 1] * Z0, Z0]).reshape(3, 1)
        r0, _ = cv2.Rodrigues(R0)
        try:
            ok, rvec, tvec = cv2.solvePnP(self.obj, q.reshape(4, 1, 2), self.K, None,
                                          rvec=r0.copy(), tvec=t0.copy(), useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return []
        v = self._valid(rvec, tvec) if ok else None
        return [v] if v is not None else []

    def pnp_all(self, pts):
        """All valid square poses (R, t, rvec, reproj_px), best first.
        Two solutions are normal near fronto-parallel (IPPE ambiguity)."""
        base = pts.reshape(4, 2).astype(np.float64)
        if not np.all(np.isfinite(base)):
            return []
        def scored(sols):
            out = [(R, t, rvec, self._reproj(rvec, t, base)) for R, t, rvec in sols]
            return sorted([o for o in out if o[3] <= self.max_reproj_px], key=lambda x: x[3])

        out = scored(self._ippe(base))
        if not out:
            for jit in self._JITTER:
                q = base.copy()
                q[jit[0]] += jit[1:]
                out = scored(self._ippe(q))
                if out:
                    break
        if not out:
            out = scored(self._iterative(base))
        return out

    def pnp(self, pts):
        sols = self.pnp_all(pts)
        return sols[0][:3] if sols else None

    def _score(self, R, t, R_base_cam):
        n_cam = R[:, 2].copy()
        if np.dot(n_cam, t) > 0.0:      # outward normal must face the camera
            n_cam = -n_cam
        if self.top_reference == 'base_z' and R_base_cam is not None:
            return float((R_base_cam @ n_cam)[2])
        return float(-n_cam[2])

    @staticmethod
    def _aligned_dist(pts, ref):
        return min(float(np.mean(np.linalg.norm(np.roll(pts, -k, axis=0) - ref, axis=1)))
                   for k in range(4))

    # ------------------------------------------------------------------
    def detect(self, bgr, R_base_cam=None, prev=None):
        """Returns (blob | None, top | None, candidates).
        top/candidate = dict(pts, score, R, t, rvec)."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in self.hsv_ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else (mask | m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, []
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_blob_area:
            return None, None, []
        M = cv2.moments(c)
        blob = {'centroid': (M['m10'] / M['m00'], M['m01'] / M['m00']),
                'area': area, 'bbox': cv2.boundingRect(c)}

        blob_mask = np.zeros_like(mask)
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        V = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(V, self.canny_lo, self.canny_hi), self.k3,
                           iterations=self.edge_dilate)
        faces_mask = cv2.bitwise_and(blob_mask, cv2.bitwise_not(edges))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(faces_mask, connectivity=4)

        cands = []
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < self.min_face_area:
                continue
            comp = np.where(labels == k, 255, 0).astype(np.uint8)
            comp = cv2.bitwise_and(cv2.dilate(comp, self.k3, iterations=self.edge_dilate + 1), blob_mask)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            quad = self._quad(max(cs, key=cv2.contourArea))
            if quad is None:
                continue
            pose = self.pnp(quad)
            if pose is None:
                continue
            R, t, rvec = pose
            cands.append({'pts': quad, 'score': self._score(R, t, R_base_cam),
                          'R': R, 't': t, 'rvec': rvec})

        good = [cd for cd in cands if cd['score'] >= self.min_top_score]
        top = None
        if good:
            top = max(good, key=lambda cd: cd['score'])
            if prev is not None:
                near = [cd for cd in good if self._aligned_dist(cd['pts'], prev) < self.track_gate_px]
                if near:
                    top = min(near, key=lambda cd: self._aligned_dist(cd['pts'], prev))
        return blob, top, cands


def write_cube_cao(path, face_size, height):
    """ViSP .cao model. Object frame at the TOP-face centre, +Z = outward
    normal of the top face (same frame as CubeTopFaceDetector's PnP)."""
    s, h = face_size / 2.0, height
    pts = [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0),
           (-s, -s, -h), (s, -s, -h), (s, s, -h), (-s, s, -h)]
    # counter-clockwise seen from outside (outward normals, ViSP convention)
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    with open(path, 'w') as f:
        f.write('V1\n# 3D Points\n8\n')
        for p in pts:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        f.write('# 3D Lines\n0\n# Faces from 3D lines\n0\n# Faces from 3D points\n6\n')
        for fc in faces:
            f.write('4 ' + ' '.join(map(str, fc)) + '\n')
        f.write('# 3D cylinders\n0\n# 3D circles\n0\n')
    return np.array(pts, dtype=float), faces
# ======================================================================
# PERCEPTION -- END
# ======================================================================


class CubeModelTracker:
    """ViSP model-based edge tracker (vpMbGenericTracker) on the cube CAD model.
    Initialised from the top-face PnP pose, then tracks the full 6-DoF pose
    using ALL visible cube edges (silhouette + internal edges)."""

    def __init__(self, K, cao_path, me_range=8, me_threshold=20.0, max_proj_error_deg=40.0):
        from visp.core import CameraParameters, ImageGray, HomogeneousMatrix
        from visp.mbt import MbGenericTracker
        from visp.me import Me
        self._ImageGray, self._HM = ImageGray, HomogeneousMatrix
        self.max_err = max_proj_error_deg
        try:
            self.tracker = MbGenericTracker(MbGenericTracker.EDGE_TRACKER)
        except Exception:
            self.tracker = MbGenericTracker()          # default = edge tracker
        me = Me()
        for fn, arg in (('setMaskSize', 5), ('setMaskNumber', 180), ('setRange', int(me_range)),
                        ('setMu1', 0.5), ('setMu2', 0.5), ('setSampleStep', 4.0)):
            try:
                getattr(me, fn)(arg)
            except Exception:
                pass
        try:
            me.setLikelihoodThresholdType(Me.NORMALIZED_THRESHOLD)
        except Exception:
            pass
        try:
            me.setThreshold(float(me_threshold))
        except Exception:
            pass
        self.tracker.setMovingEdge(me)
        cam = CameraParameters()
        cam.initPersProjWithoutDistortion(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.tracker.setCameraParameters(cam)
        self.tracker.loadModel(cao_path)
        for fn, arg in (('setDisplayFeatures', False), ('setProjectionErrorComputation', True),
                        ('setAngleAppear', math.radians(70)), ('setAngleDisappear', math.radians(80))):
            try:
                getattr(self.tracker, fn)(arg)
            except Exception:
                pass
        self.ok = False
        self.last_err = float('nan')

    def _img(self, gray):
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        try:
            return self._ImageGray(gray)
        except Exception:
            I = self._ImageGray()
            I.resize(gray.shape[0], gray.shape[1])
            np.asarray(I.numpy())[:, :] = gray
            return I

    @staticmethod
    def _to_np(M):
        try:
            return np.array(M.numpy(), dtype=float).reshape(4, 4)
        except Exception:
            return np.array([[M[i, j] for j in range(4)] for i in range(4)], dtype=float)

    def init(self, gray, rvec, t):
        cMo = self._HM(float(t[0]), float(t[1]), float(t[2]),
                       float(rvec[0]), float(rvec[1]), float(rvec[2]))  # rvec = theta*u
        self.tracker.initFromPose(self._img(gray), cMo)
        self.ok = True

    def track(self, gray):
        """Returns 4x4 cMo or None (and sets ok=False) on failure."""
        if not self.ok:
            return None
        try:
            self.tracker.track(self._img(gray))
            T = self._to_np(self.tracker.getPose())
        except Exception:
            self.ok = False
            return None
        try:
            self.last_err = float(self.tracker.getProjectionError())
        except Exception:
            self.last_err = float('nan')
        if not np.all(np.isfinite(T)) or T[2, 3] <= 0.0 or \
                (np.isfinite(self.last_err) and self.last_err > self.max_err):
            self.ok = False
            return None
        return T


class State(Enum):
    INIT = 0
    SERVO = 1
    SEARCH = 2
    LOST = 3


class IBVSCubeNode(Node):

    def __init__(self):
        super().__init__('ibvs_cube_node_v4')
        p = self.declare_parameter

        # --- frames / topics ---
        p('base_link_frame', 'base_link')
        p('camera_frame', '')                    # '' = from image header
        p('camera_convention', 'gazebo_link')    # 'gazebo_link' | 'optical'
        p('image_topic', '/wrist_mounted_camera/image/image_raw')
        p('camera_info_topic', '/wrist_mounted_camera/image/camera_info')
        p('joint_names', ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        p('home_positions', [0.0, 0.0, -1.57, 0.0, -1.57, 0.0])
        p('home_duration_s', 8)

        # --- command output ---
        p('command_mode', 'trajectory')          # 'trajectory' | 'velocity'
        p('trajectory_topic', '/joint_trajectory_controller/joint_trajectory')
        p('velocity_topic', '/forward_velocity_controller/commands')
        p('command_horizon_s', 0.1)            # minimum; actual = max(this, 2*frame period)

        # --- target geometry / approach ---
        p('cube_face_size_m', 0.10)              # scene_cube: 0.1 m box
        p('fill_ratio', 0.75)                    # desired half-side / min(cx, cy)
        p('min_standoff_m', 0.15)
        p('max_standoff_m', 1.0)
        p('fov_margin_px', 25.0)
        p('descent_rate', 0.06)                  # m/s of Z_ref decrease far from the goal
        p('descent_rate_near', 0.015)            # m/s when Z_ref is within near_zone_m of Z_final
        p('near_zone_m', 0.10)
        p('retreat_rate', 0.03)                  # m/s of Z_ref increase near border
        p('center_gate_px', 100.0)                # no descent above this centroid error
        p('max_depth_lag_m', 0.08)

        # --- control law ---
        p('interaction', 'mean')                 # 'mean' | 'current'
        p('lambda_0', 1.0)                       # gain at zero error
        p('lambda_inf', 0.4)                     # gain at large error
        p('lambda_slope', 5.0)
        p('v_max', 0.10)                         # m/s
        p('w_max', 0.50)                         # rad/s
        p('qdot_max', 0.6)                       # rad/s
        p('dls_eps', 0.05)
        p('dls_lambda_max', 0.05)

        # --- perception ---
        p('hsv_lower1', [0, 70, 50])
        p('hsv_upper1', [10, 255, 255])
        p('hsv_lower2', [170, 70, 50])
        p('hsv_upper2', [180, 255, 255])
        p('min_blob_area', 80.0)
        p('poly_eps', 0.03)
        p('min_side_ratio', 0.3)
        p('detector', 'faces')                   # 'faces' | 'mbt' (ViSP model-based tracker)
        p('top_reference', 'base_z')             # 'base_z' | 'camera'
        p('min_top_score', 0.5)                  # cos(max angle) between face normal and reference
        p('min_face_area', 150.0)
        p('canny_lo', 15)
        p('canny_hi', 45)
        p('track_gate_px', 40.0)
        p('cube_height_m', -1.0)                 # <0 -> same as face size (cube)
        p('mbt_max_projection_error_deg', 40.0)
        p('me_range', 8)
        p('me_threshold', 20.0)
        p('draw_candidates', True)

        # --- recovery ---
        p('lost_hold_frames', 5)
        p('history_period_s', 0.2)
        p('history_max_len', 60)
        p('recovery_gain', 0.8)
        p('recovery_tolerance', 0.03)
        p('search_gain', 0.5)
        p('search_retreat_speed', 0.02)
        p('search_stuck_frames', 10)

        # --- convergence / debug ---
        p('converge_px', 5.0)
        p('converge_tilt_deg', 3.0)
        p('converge_frames', 10)
        p('axis_test', '')                       # '', 'vx', 'vy', 'vz', 'wz'
        p('axis_test_speed', 0.02)
        p('axis_test_duration_s', 3.0)
        p('show_window', True)

        g = lambda n: self.get_parameter(n).value
        self.base_link = g('base_link_frame')
        self.cam_frame = g('camera_frame') or None
        self.convention = g('camera_convention')
        self.joint_names = list(g('joint_names'))
        self.n = len(self.joint_names)
        self.home = [float(x) for x in g('home_positions')]
        self.home_duration = int(g('home_duration_s'))
        self.command_mode = g('command_mode')
        self.horizon = float(g('command_horizon_s'))
        self.half = float(g('cube_face_size_m')) / 2.0
        self.fill_ratio = float(g('fill_ratio'))
        self.min_standoff = float(g('min_standoff_m'))
        self.max_standoff = float(g('max_standoff_m'))
        self.margin = float(g('fov_margin_px'))
        self.descent_rate = float(g('descent_rate'))
        self.descent_rate_near = float(g('descent_rate_near'))
        self.near_zone = float(g('near_zone_m'))
        self.retreat_rate = float(g('retreat_rate'))
        self.center_gate = float(g('center_gate_px'))
        self.max_lag = float(g('max_depth_lag_m'))
        self.l0, self.linf, self.lslope = float(g('lambda_0')), float(g('lambda_inf')), float(g('lambda_slope'))
        self.v_max, self.w_max, self.qdot_max = float(g('v_max')), float(g('w_max')), float(g('qdot_max'))
        self.dls_eps, self.dls_lmax = float(g('dls_eps')), float(g('dls_lambda_max'))
        self.hsv = [(np.array(g('hsv_lower1')), np.array(g('hsv_upper1'))),
                    (np.array(g('hsv_lower2')), np.array(g('hsv_upper2')))]
        self.min_blob_area = float(g('min_blob_area'))
        self.poly_eps = float(g('poly_eps'))
        self.min_side_ratio = float(g('min_side_ratio'))
        self.detector_type = g('detector')
        # self.detector_type = g('mbt')
        self.top_reference = g('top_reference')
        self.min_top_score = float(g('min_top_score'))
        self.min_face_area = float(g('min_face_area'))
        self.canny = (int(g('canny_lo')), int(g('canny_hi')))
        self.track_gate = float(g('track_gate_px'))
        h_cube = float(g('cube_height_m'))
        self.cube_height = h_cube if h_cube > 0 else 2.0 * self.half
        self.mbt_max_err = float(g('mbt_max_projection_error_deg'))
        self.me_range = int(g('me_range'))
        self.me_threshold = float(g('me_threshold'))
        self.draw_candidates = bool(g('draw_candidates'))
        self.face_det = None
        self.mbt = None
        self.model_pts, self.model_faces = None, None
        self.lost_hold_frames = int(g('lost_hold_frames'))
        self.history_period = float(g('history_period_s'))
        self.history = deque(maxlen=int(g('history_max_len')))
        self.recovery_gain = float(g('recovery_gain'))
        self.recovery_tol = float(g('recovery_tolerance'))
        self.search_gain = float(g('search_gain'))
        self.search_retreat = float(g('search_retreat_speed'))
        self.search_stuck_frames = int(g('search_stuck_frames'))
        self.converge_px = float(g('converge_px'))
        self.converge_tilt = float(g('converge_tilt_deg'))
        self.converge_frames = int(g('converge_frames'))
        self.axis_test = g('axis_test')
        self.axis_speed = float(g('axis_test_speed'))
        self.axis_duration = float(g('axis_test_duration_s'))
        self.show = bool(g('show_window'))

        # --- ROS I/O ---
        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, g('image_topic'), self.image_cb, qos_img)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self.camera_info_cb, qos_img)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(String, '/robot_description', self.urdf_cb, qos_latched)

        self.traj_pub = self.create_publisher(JointTrajectory, g('trajectory_topic'), 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, g('velocity_topic'), 10)
        self.traj_client = ActionClient(self, FollowJointTrajectory,
                                        '/joint_trajectory_controller/follow_joint_trajectory')
        self.bridge = CvBridge()
        self.kernel = np.ones((3, 3), np.uint8)

        # --- kinematics / camera ---
        self.urdf_xml = None
        self.chain = None
        self.jac_solver = None
        self.fk_solver = None
        self.K = None
        self.Z_final = None
        self.obj_pts = np.array([[-self.half, self.half, 0.0],
                                 [self.half, self.half, 0.0],
                                 [self.half, -self.half, 0.0],
                                 [-self.half, -self.half, 0.0]])  # IPPE_SQUARE order

        # --- ViSP task ---
        self.task = Servo()
        self.task.setServo(Servo.EYEINHAND_CAMERA)
        itype = getattr(Servo, 'MEAN', Servo.CURRENT) if g('interaction') == 'mean' else Servo.CURRENT
        self.task.setInteractionMatrixType(itype, Servo.PSEUDO_INVERSE)
        self.task.setLambda(self.l0)
        self.s = [FeaturePoint() for _ in range(4)]
        self.s_star = [FeaturePoint() for _ in range(4)]
        for i in range(4):
            self.s[i].buildFrom(0.0, 0.0, 1.0)
            self.s_star[i].buildFrom(0.0, 0.0, 1.0)
            self.task.addFeature(self.s[i], self.s_star[i])

        # --- runtime state ---
        self.q = np.zeros(self.n)
        self.have_js = False
        self.state = State.INIT
        self.goal_sent = False
        self.last_stamp = None
        self.prev_corners = None
        self.des_shift = 0
        self.Z_ref = None
        self.lost_frames = 0
        self.rollback_target = None
        self.converged_count = 0
        self.converged_logged = False
        self.axis_t0 = None

        self.create_timer(1.0, self.home_timer_cb)
        if VISP_BACKEND == 'ViSP':
            self.get_logger().info('Control law backend: ViSP')            
        else:
            self.get_logger().warn(f'Control law backend: {VISP_BACKEND}')
        self.get_logger().info('IBVS started (waiting for URDF, camera_info, joint_states).')

    # ==================================================================
    # Setup callbacks
    # ==================================================================
    def camera_info_cb(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        fx, cx, cy = self.K[0, 0], self.K[0, 2], self.K[1, 2]
        half_img = min(cx, cy)
        half_target_px = min(self.fill_ratio * half_img, half_img - self.margin - 10.0)
        z_geom = fx * self.half / half_target_px
        self.Z_final = max(self.min_standoff, z_geom)
        side_px = 2.0 * fx * self.half / self.Z_final
        self.get_logger().info(
            f'Intrinsics fx={fx:.1f} cx={cx:.1f} cy={cy:.1f} | face={2*self.half:.3f} m | '
            f'Z_final={self.Z_final:.3f} m -> final square {side_px:.0f}px')
        if z_geom < self.min_standoff:
            self.get_logger().warn(
                f'Filling the image would need Z={z_geom:.3f} m (< min_standoff). '
                f'Final square limited to {side_px:.0f}px. Use a bigger target or lower min_standoff_m.')

        self.face_det = CubeTopFaceDetector(
            self.K, 2.0 * self.half, self.hsv, self.min_blob_area, self.min_face_area,
            self.poly_eps, self.min_side_ratio, self.canny[0], self.canny[1], 1,
            self.top_reference, self.min_top_score, self.track_gate)
        
        if self.detector_type == 'mbt':
            
            cao = '/home/huro/kinova_control_ws/src/urdf/cube_model.cao'
            # /tmp/ibvs_cube_model.cao'
            self.model_pts, self.model_faces = write_cube_cao(cao, 2.0 * self.half, self.cube_height)
            
            self.get_logger().info(f'Using ViSP model-based tracker (mbt) with model {cao}.')
            if VISP_BACKEND != 'ViSP':
                self.get_logger().error('detector=mbt needs ViSP; falling back to detector=faces.')
            else:
                try:
                    self.mbt = CubeModelTracker(self.K, cao, self.me_range, self.me_threshold, self.mbt_max_err)
                    self.get_logger().info(f'ViSP model-based tracker ready (model {cao}).')
                except Exception as e:
                    self.get_logger().error(f'Could not create vpMbGenericTracker ({e}); using detector=faces.')
                    self.mbt = None
        self.get_logger().info(f'Top-face detection: {"mbt" if self.mbt else "faces"}, '
                               f'reference={self.top_reference}')

    def urdf_cb(self, msg):
        if self.urdf_xml is None:
            self.urdf_xml = msg.data
            self.try_build_chain()

    def try_build_chain(self):
        if self.chain is not None or self.urdf_xml is None or self.cam_frame is None:
            return
        try:
            self.chain = build_kdl_chain_from_urdf(self.urdf_xml, self.base_link, self.cam_frame)
        except Exception as e:
            self.get_logger().error(f'KDL chain {self.base_link}->{self.cam_frame} failed: {e}')
            return
        nj = self.chain.getNrOfJoints()
        if nj != self.n:
            self.get_logger().error(f'Chain has {nj} joints, expected {self.n}.')
            self.chain = None
            return
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.get_logger().info(f'KDL chain ready: {self.base_link} -> {self.cam_frame} '
                               f'(convention: {self.convention})')

    def joint_state_cb(self, msg):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                self.q[i] = msg.position[msg.name.index(name)]
        self.have_js = True

    # ==================================================================
    # Home pose
    # ==================================================================
    def home_timer_cb(self):
        if self.goal_sent:
            return
        if not self.traj_client.server_is_ready():
            self.get_logger().info('Waiting for FollowJointTrajectory server...', throttle_duration_sec=5.0)
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = self.home
        pt.time_from_start = Duration(sec=self.home_duration)
        goal.trajectory.points = [pt]
        self.goal_sent = True
        self.traj_client.send_goal_async(goal).add_done_callback(self._goal_resp)
        self.get_logger().info('Moving to home pose...')

    def _goal_resp(self, fut):
        h = fut.result()
        if not h.accepted:
            self.get_logger().error('Home trajectory rejected.')
            self.goal_sent = False
            return
        h.get_result_async().add_done_callback(self._home_done)

    def _home_done(self, _):
        self.get_logger().info('Home reached -> visual servoing active.')
        self.state = State.SEARCH

    # ==================================================================
    # Perception
    # ==================================================================
    def _camera_rotation_base(self):
        """Rotation base <- OPTICAL camera frame (from FK)."""
        if self.fk_solver is None:
            return None
        q_kdl = kdl.JntArray(self.n)
        for i in range(self.n):
            q_kdl[i] = float(self.q[i])
        frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)
        R = kdl_rotation_to_np(frame.M)
        return R @ R_LINK_OPT if self.convention == 'gazebo_link' else R

    def _project_model(self, T, pts):
        P = pts @ T[:3, :3].T + T[:3, 3]
        if np.any(P[:, 2] <= 0.02):
            return None
        u = self.K[0, 0] * P[:, 0] / P[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * P[:, 1] / P[:, 2] + self.K[1, 2]
        return np.stack([u, v], axis=1)

    def perceive(self, bgr):
        """Returns (blob | None, top-face corners (4x2, clockwise) | None)."""
        blob, top, cands = self.face_det.detect(bgr, self._camera_rotation_base(), self.prev_corners)
        if self.draw_candidates:
            for cd in cands:
                cv2.polylines(bgr, [cd['pts'].astype(np.int32)], True, (0, 255, 255), 1)
                cx_, cy_ = cd['pts'].mean(axis=0)
                cv2.putText(bgr, f"{cd['score']:.2f}", (int(cx_), int(cy_)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if top is not None:
            cv2.polylines(bgr, [top['pts'].astype(np.int32)], True, (255, 0, 255), 2)

        corners = top['pts'] if top is not None else None
        if self.mbt is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            T = self.mbt.track(gray) if self.mbt.ok else None
            proj = self._project_model(T, self.model_pts) if T is not None else None
            if proj is not None:
                for fc in self.model_faces:
                    cv2.polylines(bgr, [proj[list(fc)].astype(np.int32)], True, (255, 128, 0), 1)
                corners = _clockwise(proj[:4].copy())
                cv2.putText(bgr, f'MBT err {self.mbt.last_err:.1f}deg', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            else:
                if self.mbt.ok or T is not None:
                    self.mbt.ok = False
                if top is not None:
                    try:
                        self.mbt.init(gray, top['rvec'], top['t'])
                        self.get_logger().info('MBT (re)initialised from top-face pose.')
                    except Exception as e:
                        self.get_logger().error(f'MBT init failed: {e}', throttle_duration_sec=2.0)

        if corners is not None and blob is None:
            c32 = corners.astype(np.float32)
            blob = {'centroid': tuple(corners.mean(axis=0)), 'area': float(cv2.contourArea(c32)),
                    'bbox': cv2.boundingRect(c32)}
        return blob, corners

    @staticmethod
    def _cyclic_align(pts, ref):
        costs = [np.sum(np.linalg.norm(np.roll(pts, -k, axis=0) - ref, axis=1)) for k in range(4)]
        return np.roll(pts, -int(np.argmin(costs)), axis=0)

    def _acquire(self, pts, Zmean):
        """(Re)initialise labels, desired-square rotation and Z_ref."""
        pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)  # start near top-left
        shape = pts - pts.mean(axis=0)
        shape *= math.sqrt(2.0) / max(np.mean(np.linalg.norm(shape, axis=1)), 1e-6)
        canon = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        costs = [np.sum(np.linalg.norm(shape - np.roll(canon, -k, axis=0), axis=1)) for k in range(4)]
        self.des_shift = int(np.argmin(costs))
        self.Z_ref = float(np.clip(Zmean, self.Z_final, self.max_standoff))
        self.converged_count = 0
        self.converged_logged = False
        self.get_logger().info(f'Square acquired: Z={Zmean:.3f} m, Z_ref={self.Z_ref:.3f} m')
        return pts

    def _estimate_depths(self, pts, area):
        """Per-corner depth from the best square pose. Tilt = the smallest tilt
        among the pose solutions that explain the image equally well (the IPPE
        pair near fronto-parallel), so ambiguity does not show up as fake tilt."""
        fx = self.K[0, 0]
        z_fallback = float(np.clip(fx * 2.0 * self.half / math.sqrt(max(area, 1.0)), 0.02, 5.0))
        sols = self.face_det.pnp_all(pts) if self.face_det is not None else []
        if not sols:
            return np.full(4, z_fallback), float('nan')
        R, t, _, best_err = sols[0]
        Z = (self.obj_pts @ R.T + t.reshape(1, 3))[:, 2]
        if not np.all(np.isfinite(Z)) or np.any(Z < 0.02) or np.any(Z > 5.0):
            return np.full(4, z_fallback), float('nan')
        tilts = [math.degrees(math.acos(min(1.0, abs(Rk[2, 2]))))
                 for Rk, _, _, e in sols if e <= best_err + 1.0]
        return Z, min(tilts)

    # ==================================================================
    # Main loop
    # ==================================================================
    def image_cb(self, msg):
        if self.cam_frame is None:
            self.cam_frame = msg.header.frame_id
            self.get_logger().info(f"Camera frame from image header: '{self.cam_frame}'")
            self.try_build_chain()
        if self.state == State.INIT or self.K is None or self.chain is None or not self.have_js:
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.1 if self.last_stamp is None else float(np.clip(stamp - self.last_stamp, 1e-3, 0.3))
        self.last_stamp = stamp
        self.cmd_horizon = max(self.horizon, 2.0 * dt)

        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        blob, corners = self.perceive(bgr)

        if self.axis_test:
            self.axis_test_step(bgr, blob, stamp)
        elif corners is not None:
            self.servo_step(bgr, corners, blob, dt, stamp)
        elif blob is not None:
            self.search_step(bgr, blob)
        else:
            self.lost_step(bgr)
        self._show(bgr)

    # ------------------------------------------------------------------
    def servo_step(self, bgr, corners, blob, dt, stamp):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        h, w = bgr.shape[:2]

        Z, tilt = self._estimate_depths(corners, blob['area'])
        if self.state != State.SERVO or self.prev_corners is None:
            corners = self._acquire(corners, float(np.mean(Z)))
            Z, tilt = self._estimate_depths(corners, blob['area'])
            self.state = State.SERVO
        else:
            corners = self._cyclic_align(corners, self.prev_corners)
            Z, tilt = self._estimate_depths(corners, blob['area'])
        self.prev_corners = corners
        self.lost_frames = 0
        self.rollback_target = None
        self._record_history(stamp)

        xn = (corners[:, 0] - cx) / fx
        yn = (corners[:, 1] - cy) / fy
        Zmean = float(np.mean(Z))

        # ---- autonomous reference-depth scheduling ----
        u, v = corners[:, 0], corners[:, 1]
        near_border = bool(np.any((u < self.margin) | (u > w - self.margin) |
                                  (v < self.margin) | (v > h - self.margin)))
        c_err = math.hypot(u.mean() - cx, v.mean() - cy)
        if near_border:
            self.Z_ref = min(self.Z_ref + self.retreat_rate * dt, self.max_standoff)
        elif Zmean - self.Z_ref < self.max_lag:
            gate = float(np.clip(1.0 - c_err / self.center_gate, 0.0, 1.0))
            a = float(np.clip((self.Z_ref - self.Z_final) / max(self.near_zone, 1e-3), 0.0, 1.0))
            rate = self.descent_rate_near + a * (self.descent_rate - self.descent_rate_near)
            self.Z_ref = max(self.Z_ref - rate * gate * dt, self.Z_final)

        # ---- desired features: centered square at Z_ref ----
        d = self.half / self.Z_ref
        canon = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])
        des = np.roll(canon, -self.des_shift, axis=0)

        err = np.concatenate([xn - des[:, 0], yn - des[:, 1]])
        e_norm = float(np.linalg.norm(err))
        if self.l0 > self.linf:
            lam = (self.l0 - self.linf) * math.exp(-self.lslope * e_norm / (self.l0 - self.linf)) + self.linf
        else:
            lam = self.l0
        self.task.setLambda(lam)

        for i in range(4):
            self.s[i].buildFrom(float(xn[i]), float(yn[i]), float(Z[i]))
            self.s_star[i].buildFrom(float(des[i, 0]), float(des[i, 1]), float(self.Z_ref))
        if not (np.all(np.isfinite(xn)) and np.all(np.isfinite(yn)) and np.all(np.isfinite(Z))):
            self.get_logger().warn('Non-finite features -> holding this frame.', throttle_duration_sec=1.0)
            self._send_qdot(np.zeros(self.n))
            return
        try:
            v_opt = colvec_to_np(self.task.computeControlLaw())[:6]
        except Exception as e:
            self.get_logger().warn(f'Control law failed ({e}) -> holding this frame.', throttle_duration_sec=1.0)
            self._send_qdot(np.zeros(self.n))
            return
        if v_opt.size < 6 or not np.all(np.isfinite(v_opt)):
            self.get_logger().warn('Non-finite velocity -> holding this frame.', throttle_duration_sec=1.0)
            self._send_qdot(np.zeros(self.n))
            return

        self._command_optical_twist(v_opt)

        # ---- monitoring ----
        ud = cx + fx * des[:, 0]
        vd = cy + fy * des[:, 1]
        corner_err = math.sqrt(np.mean((u - ud) ** 2 + (v - vd) ** 2))
        at_final = abs(self.Z_ref - self.Z_final) < 1e-3
        tilt_ok = (not math.isnan(tilt)) and tilt < self.converge_tilt
        if at_final and corner_err < self.converge_px and tilt_ok:
            self.converged_count += 1
        else:
            self.converged_count = 0
        if self.converged_count >= self.converge_frames and not self.converged_logged:
            self.get_logger().info(f'CONVERGED: err={corner_err:.1f}px, Z={Zmean:.3f} m, tilt={tilt:.1f} deg '
                                   '(keeps servoing to hold the pose)')
            self.converged_logged = True

        self.get_logger().info(
            f'[SERVO] err={corner_err:5.1f}px c_err={c_err:5.1f}px Z={Zmean:.3f} Z_ref={self.Z_ref:.3f} '
            f'tilt={tilt:4.1f}deg lam={lam:.2f}{" BORDER" if near_border else ""}',
            throttle_duration_sec=0.5)

        for i in range(4):
            cv2.circle(bgr, (int(u[i]), int(v[i])), 6, (0, 255, 0), -1)
            cv2.putText(bgr, str(i), (int(u[i]) + 6, int(v[i]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(bgr, (int(ud[i]), int(vd[i])), 6, (0, 0, 255), 2)
            cv2.line(bgr, (int(u[i]), int(v[i])), (int(ud[i]), int(vd[i])), (0, 150, 255), 1)
        cv2.putText(bgr, f'SERVO err {corner_err:.1f}px Z {Zmean:.2f}/{self.Z_ref:.2f} tilt {tilt:.1f}',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    def search_step(self, bgr, blob):
        if self.state != State.SEARCH:
            self.get_logger().warn('Blob visible but no clean square -> centering blob.')
            self.search_frames = 0
        self.state = State.SEARCH
        self.search_frames = getattr(self, 'search_frames', 0) + 1
        self.prev_corners = None
        self.lost_frames = 0
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        h, w = bgr.shape[:2]
        u, v = blob['centroid']
        Z = float(np.clip(fx * 2.0 * self.half / math.sqrt(blob['area']), 0.05, self.max_standoff))
        bx, by, bw, bh = blob['bbox']
        touches = bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2
        v_opt = np.zeros(6)
        v_opt[0] = self.search_gain * Z * (u - cx) / fx
        v_opt[1] = self.search_gain * Z * (v - cy) / fy
        stuck = self.search_frames > self.search_stuck_frames   # centered but still no square
        v_opt[2] = -self.search_retreat if (touches or stuck) else 0.0
        self._command_optical_twist(v_opt)
        cv2.circle(bgr, (int(u), int(v)), 8, (0, 165, 255), -1)
        cv2.putText(bgr, 'SEARCH', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # ------------------------------------------------------------------
    def lost_step(self, bgr):
        self.prev_corners = None
        if self.mbt is not None:
            self.mbt.ok = False
        self.lost_frames += 1
        cv2.putText(bgr, 'LOST', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.lost_frames < self.lost_hold_frames:
            self._send_qdot(np.zeros(self.n))
            return
        if self.state != State.LOST:
            self.get_logger().warn('Target lost -> retracing joint history.')
        self.state = State.LOST

        if self.rollback_target is None:
            if self.history:
                self.rollback_target = np.array(self.history.pop())
            else:
                self.rollback_target = np.array(self.home)
        err = self.rollback_target - self.q
        if np.max(np.abs(err)) < self.recovery_tol:
            if self.history:
                self.rollback_target = np.array(self.history.pop())
                err = self.rollback_target - self.q
            else:
                self.get_logger().warn('History exhausted, target not visible. Holding.',
                                       throttle_duration_sec=3.0)
                self._send_qdot(np.zeros(self.n))
                return
        self._send_qdot(self._scale(self.recovery_gain * err))

    def _record_history(self, stamp):
        if not hasattr(self, '_last_hist') or stamp - self._last_hist >= self.history_period:
            self.history.append(self.q.copy())
            self._last_hist = stamp

    # ------------------------------------------------------------------
    def axis_test_step(self, bgr, blob, stamp):
        if self.axis_t0 is None:
            self.axis_t0 = stamp
            self.get_logger().warn(f'AXIS TEST {self.axis_test} = +{self.axis_speed} for {self.axis_duration}s')
        idx = {'vx': 0, 'vy': 1, 'vz': 2, 'wx': 3, 'wy': 4, 'wz': 5}.get(self.axis_test)
        v_opt = np.zeros(6)
        if idx is not None and stamp - self.axis_t0 < self.axis_duration:
            v_opt[idx] = self.axis_speed
        self._command_optical_twist(v_opt)
        if blob is not None:
            self.get_logger().info(f'[AXIS TEST] centroid=({blob["centroid"][0]:.0f},{blob["centroid"][1]:.0f}) '
                                   f'area={blob["area"]:.0f}', throttle_duration_sec=0.5)

    # ==================================================================
    # Kinematics and commands
    # ==================================================================
    def _command_optical_twist(self, v_opt):
        v_opt = np.array(v_opt, dtype=float)
        tn = np.linalg.norm(v_opt[:3])
        if tn > self.v_max:
            v_opt[:3] *= self.v_max / tn
        rn = np.linalg.norm(v_opt[3:])
        if rn > self.w_max:
            v_opt[3:] *= self.w_max / rn

        if self.convention == 'gazebo_link':
            v_link = np.concatenate([R_LINK_OPT @ v_opt[:3], R_LINK_OPT @ v_opt[3:]])
        else:
            v_link = v_opt

        q_kdl = kdl.JntArray(self.n)
        for i in range(self.n):
            q_kdl[i] = float(self.q[i])
        jac = kdl.Jacobian(self.n)
        self.jac_solver.JntToJac(q_kdl, jac)
        frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)
        R = kdl_rotation_to_np(frame.M)
        v_base = np.concatenate([R @ v_link[:3], R @ v_link[3:]])

        J = kdl_jacobian_to_np(jac)  # base frame, ref point = camera origin
        if not (np.all(np.isfinite(J)) and np.all(np.isfinite(v_base))):
            self._send_qdot(np.zeros(self.n))
            return
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        smin = S[-1]
        lam2 = 0.0 if smin >= self.dls_eps else (1.0 - (smin / self.dls_eps) ** 2) * self.dls_lmax ** 2
        qdot = Vt.T @ np.diag(S / (S ** 2 + lam2)) @ U.T @ v_base
        self._send_qdot(self._scale(qdot))

    def _scale(self, qdot):
        m = float(np.max(np.abs(qdot)))
        return qdot * (self.qdot_max / m) if m > self.qdot_max else qdot

    def _send_qdot(self, qdot):
        if self.command_mode == 'velocity':
            self.vel_pub.publish(Float64MultiArray(data=[float(x) for x in qdot]))
            return
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        h = getattr(self, 'cmd_horizon', self.horizon)
        pt.positions = [float(self.q[i] + qdot[i] * h) for i in range(self.n)]
        sec = int(h)
        pt.time_from_start = Duration(sec=sec, nanosec=int((h - sec) * 1e9))
        msg.points = [pt]
        self.traj_pub.publish(msg)

    def _show(self, bgr):
        if not self.show:
            return
        try:
            cv2.imshow('IBVS v4 wrist camera', bgr)
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
        node._send_qdot(np.zeros(node.n))
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()