"""
Mission state machine (pure python, no ROS -> unit-testable).

    INIT -> HOMING -> ACQUIRE -> IDLE --select--> APPROACH -> ALIGN -> HOLD
                         ^                           |          |        |
                         +--------- LOST <-----------+----------+--------+  (tracking lost)
    APPROACH/ALIGN/HOLD --select other / abort--> RETREAT -> APPROACH(new) / IDLE

APPROACH : servo at the target's approach_standoff (coarse, far, safe).
ALIGN    : servo while the reference distance ramps down to the final standoff.
HOLD     : keep servoing at the final standoff (target may drift/tumble).
RETREAT  : ramp back to approach_standoff before switching target or aborting.

The FSM only decides WHAT to do; the node decides HOW (perception, control).
"""
from dataclasses import dataclass
from enum import Enum


class State(Enum):
    INIT = 'INIT'
    HOMING = 'HOMING'
    ACQUIRE = 'ACQUIRE'
    IDLE = 'IDLE'
    APPROACH = 'APPROACH'
    ALIGN = 'ALIGN'
    HOLD = 'HOLD'
    RETREAT = 'RETREAT'
    LOST = 'LOST'


SERVO_STATES = (State.APPROACH, State.ALIGN, State.HOLD, State.RETREAT)


@dataclass
class Inputs:
    robot_ready: bool
    homed: bool
    tracking_ok: bool
    feature_err_px: float = None   # RMS image error for the current target (None = not available)
    dt: float = 0.05


@dataclass
class Command:
    mode: str                      # 'none' | 'home' | 'reinit' | 'hold' | 'servo'
    target: str = None
    standoff: float = None
    gain_scale: float = 1.0


@dataclass
class FSMConfig:
    acquire_stable_frames: int = 10
    lost_hold_frames: int = 5
    approach_tol_px: float = 15.0
    final_tol_px: float = 4.0
    settle_frames: int = 10
    descent_rate: float = 0.03       # m/s, ALIGN ramp
    retreat_rate: float = 0.05       # m/s, RETREAT ramp
    descent_gate_px: float = 30.0    # no descent while the image error is above this
    align_gain_scale: float = 0.7


class MissionFSM:

    def __init__(self, targets, cfg=None, log=None):
        self.targets = targets
        self.cfg = cfg or FSMConfig()
        self.log = log or (lambda msg: None)
        self.state = State.INIT
        self.current = None            # name of the target being serviced
        self.next_target = None        # pending target during RETREAT (None = go IDLE)
        self.pending = None            # request not yet handled
        self.z_ref = None
        self.resume = False            # resume APPROACH after re-acquisition
        self._count = 0

    # ------------------------------------------------------------------
    def request(self, name):
        """'abort' or a target name. Returns (accepted, message)."""
        name = (name or '').strip()
        if name != 'abort' and name not in self.targets:
            return False, f"unknown target '{name}'. Known: {sorted(self.targets)} or 'abort'"
        self.pending = name
        return True, f"request '{name}' queued (state {self.state.value})"

    @property
    def current_target(self):
        return self.targets.get(self.current) if self.current else None

    def _go(self, state, msg=''):
        if state != self.state:
            self.log(f'[FSM] {self.state.value} -> {state.value}' + (f' ({msg})' if msg else ''))
        self.state = state
        self._count = 0

    def _servo(self, gain=1.0):
        return Command('servo', self.current, self.z_ref, gain)

    def _counter(self, cond):
        self._count = self._count + 1 if cond else 0
        return self._count

    # ------------------------------------------------------------------
    def step(self, inp: Inputs) -> Command:
        c, s = self.cfg, self.state
        err = inp.feature_err_px

        if s == State.INIT:
            if inp.robot_ready:
                self._go(State.HOMING)
                return Command('home')
            return Command('none')

        if s == State.HOMING:
            if inp.homed:
                self._go(State.ACQUIRE)
            return Command('none')

        # tracking loss has priority in every servo state
        if s in SERVO_STATES and not inp.tracking_ok:
            self.resume = self.current is not None
            self._go(State.LOST, 'tracking lost')
            return Command('hold')

        if s == State.LOST:
            if self._counter(True) >= c.lost_hold_frames:
                self._go(State.ACQUIRE)
            return Command('hold')

        if s == State.ACQUIRE:
            if not inp.tracking_ok:
                self._count = 0
                return Command('reinit')
            if self._counter(True) >= c.acquire_stable_frames:
                if self.resume and self.current:
                    self.resume = False
                    self.z_ref = self.current_target.approach_standoff
                    self._go(State.APPROACH, f'resume {self.current}')
                else:
                    self._go(State.IDLE, 'tracking stable')
            return Command('hold')

        # ---- requests ----
        if self.pending is not None:
            req, self.pending = self.pending, None
            if s == State.IDLE:
                if req != 'abort':
                    self.current = req
                    self.z_ref = self.targets[req].approach_standoff
                    self._go(State.APPROACH, req)
            elif s in SERVO_STATES and req != self.current:
                self.next_target = None if req == 'abort' else req
                self._go(State.RETREAT, f'-> {req}')
            s = self.state

        if s == State.IDLE:
            return Command('hold')

        tgt = self.current_target
        if s == State.APPROACH:
            if self._counter(err is not None and err < c.approach_tol_px) >= c.settle_frames:
                self._go(State.ALIGN)
            return self._servo()

        if s == State.ALIGN:
            if err is not None and err < c.descent_gate_px:
                self.z_ref = max(tgt.standoff, self.z_ref - c.descent_rate * inp.dt)
            at_goal = abs(self.z_ref - tgt.standoff) < 1e-6
            if self._counter(at_goal and err is not None and err < c.final_tol_px) >= c.settle_frames:
                self._go(State.HOLD, f'{tgt.name} reached')
            return self._servo(c.align_gain_scale)

        if s == State.HOLD:
            return self._servo(c.align_gain_scale)

        if s == State.RETREAT:
            self.z_ref = min(tgt.approach_standoff, self.z_ref + c.retreat_rate * inp.dt)
            at_top = abs(self.z_ref - tgt.approach_standoff) < 1e-6
            if self._counter(at_top and err is not None and err < c.approach_tol_px) >= c.settle_frames:
                if self.next_target is None:
                    self.current = None
                    self._go(State.IDLE, 'aborted')
                    return Command('hold')
                self.current, self.next_target = self.next_target, None
                self.z_ref = self.current_target.approach_standoff
                self._go(State.APPROACH, self.current)
            return self._servo()

        return Command('hold')
