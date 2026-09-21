"""
Target catalog: servicing targets defined in the spacecraft MODEL frame (o),
the same frame as the ViSP .cao model and the Gazebo model.

Every target has a local frame f (given as oMf) whose +Z axis is the OUTWARD
normal of the surface the camera must face. IBVS features are 3D points
defined in f; they are projected with the tracked pose (current features) and
with the desired camera pose cdMf (desired features).

Adding a new kind of target = adding a builder in _BUILDERS.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import yaml

from .geometry import make_T, pose_to_T, rot_x, rot_z, rotation_angle, transform


@dataclass
class Target:
    name: str
    kind: str                      # 'screw' | 'planar_region' | ...
    oMf: np.ndarray                # target frame in model frame
    points_f: np.ndarray           # Nx3 feature points in target frame
    standoff: float                # final camera distance along the normal [m]
    approach_standoff: float       # distance for the first (coarse) approach [m]
    symmetry: int = 1              # n-fold symmetry of the feature pattern about f_z
    params: dict = field(default_factory=dict)

    @property
    def points_o(self):
        return transform(self.oMf, self.points_f)

    def cdMf(self, standoff, yaw=0.0):
        """Desired camera pose w.r.t. the target frame: on the normal, looking at it."""
        return make_T(rot_x(math.pi) @ rot_z(yaw), [0.0, 0.0, standoff])

    def desired_features(self, standoff, yaw=0.0):
        """Nx3 array (x, y, Z) of desired normalized features."""
        P = transform(self.cdMf(standoff, yaw), self.points_f)
        return np.column_stack([P[:, 0] / P[:, 2], P[:, 1] / P[:, 2], P[:, 2]])

    def best_yaw(self, cMo):
        """In-plane rotation of the desired pose (multiple of 2*pi/symmetry) closest
        to the current camera orientation -> smallest wz motion."""
        R_cf = (cMo @ self.oMf)[:3, :3]
        best, best_ang = 0.0, float('inf')
        for k in range(max(1, self.symmetry)):
            yaw = 2.0 * math.pi * k / max(1, self.symmetry)
            ang = rotation_angle(self.cdMf(1.0, yaw)[:3, :3].T @ R_cf)
            if ang < best_ang:
                best, best_ang = yaw, ang
        return best


def _screw(name, d):
    r = float(d.get('ring_radius', 0.03))
    n = int(d.get('ring_points', 4))
    ring = [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), 0.0) for k in range(n)]
    # point 0 = screw head centre (refined in the image), others = virtual ring on the surface
    pts = np.array([(0.0, 0.0, 0.0)] + ring)
    return pts, n, {'head_radius': float(d.get('head_radius', 0.005))}


def _planar_region(name, d):
    w, h = (float(v) for v in d['size'])
    pts = np.array([(-w / 2, -h / 2, 0.0), (w / 2, -h / 2, 0.0), (w / 2, h / 2, 0.0), (-w / 2, h / 2, 0.0)])
    return pts, 2, {'size': (w, h)}


_BUILDERS = {'screw': _screw, 'planar_region': _planar_region}


def load_targets(path):
    """Returns (model_cfg dict, {name: Target})."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    targets = {}
    for name, d in (cfg.get('targets') or {}).items():
        kind = d['type']
        if kind not in _BUILDERS:
            raise ValueError(f"target '{name}': unknown type '{kind}' (known: {list(_BUILDERS)})")
        pts, sym, params = _BUILDERS[kind](name, d)
        targets[name] = Target(
            name=name, kind=kind,
            oMf=pose_to_T(d.get('xyz', [0, 0, 0]), d.get('rpy', [0, 0, 0])),
            points_f=pts,
            standoff=float(d['standoff']),
            approach_standoff=float(d.get('approach_standoff', 2.0 * float(d['standoff']))),
            symmetry=int(d.get('symmetry', sym)),
            params=params,
        )
    return cfg.get('model', {}), targets
