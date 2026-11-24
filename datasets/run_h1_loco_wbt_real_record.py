#!/usr/bin/env python3
"""Run Holosoma's real H1 Loco/WBT policy and record synchronized telemetry.

This launcher deliberately reuses Holosoma's native dual-policy controller.  It
does not alter policy actions or the control loop.  The only additions are a
MotionSwitcher preflight check and bounded, in-memory logging of LowCmd,
LowState, IMU, foot-force fields, and policy/input events.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
import types

import numpy as np

from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.policies.dual_mode import DualModePolicy
from holosoma_inference.sdk.unitree.h1_sysid_recorder import H1SysidRecorder
from holosoma_inference.utils.misc import restore_terminal_settings


CONFIRM_TOKEN = "H1_LOCO_WBT_RECORD"


class H1PolicyRecorder(H1SysidRecorder):
    """Extend the existing non-blocking recorder with H1 foot-force fields."""

    def __init__(self, robot_config) -> None:
        super().__init__(robot_config)
        self._augment_lock = threading.Lock()

    def start(self, *args, **kwargs):
        # Serialize start/record/stop so the DDS callback cannot observe a
        # partially constructed set of augmentation arrays.
        with self._augment_lock:
            result = super().start(*args, **kwargs)
            with self._lock:
                shape = (self._capacity, 4)
                self._arrays["foot_force"] = np.full(shape, np.nan, dtype=np.float32)
                self._arrays["foot_force_est"] = np.full(shape, np.nan, dtype=np.float32)
            return result

    def record_low_state(self, state, *, t_state_ns: int) -> None:
        try:
            foot_force = self._vector(getattr(state, "foot_force"), 4)
            foot_force_est = self._vector(getattr(state, "foot_force_est"), 4)
        except (AttributeError, TypeError, ValueError):
            foot_force = np.full(4, np.nan, dtype=np.float32)
            foot_force_est = np.full(4, np.nan, dtype=np.float32)

        with self._augment_lock:
            with self._lock:
                count_before = self._count
            super().record_low_state(state, t_state_ns=t_state_ns)
            with self._lock:
                if self._count == count_before + 1:
                    index = self._count - 1
                    self._arrays["foot_force"][index] = foot_force
                    self._arrays["foot_force_est"][index] = foot_force_est

    def stop(self, *args, **kwargs):
        with self._augment_lock:
            return super().stop(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--input", choices=("keyboard", "interface"), default="keyboard")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/h1/raw"),
    )
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def policy_config(args: argparse.Namespace):
    config = DEFAULTS["h1-19dof-loco-wbt-real"]
    primary_task = replace(
        config.task,
        interface=args.interface,
        domain_id=args.domain_id,
        velocity_input=args.input,
        state_input=args.input,
    )
    if config.secondary is None:
        raise RuntimeError("The real Loco/WBT preset has no secondary policy.")
    secondary_task = replace(
        config.secondary.task,
        interface=args.interface,
        domain_id=args.domain_id,
        velocity_input=args.input,
        state_input=args.input,
    )
    return replace(
        config,
        task=primary_task,
        secondary=replace(config.secondary, task=secondary_task),
    )


def install_event_hooks(policy: DualModePolicy, events: list[dict]) -> None:
    def mark(name: str, **fields) -> None:
        events.append({"t_ns": time.monotonic_ns(), "event": name, **fields})

    original_dispatch = policy.primary._dispatch_command

    def dispatch(command):
        mark(
            "input_command",
            command=getattr(command, "name", str(command)),
            active_before=policy.active_label,
        )
        result = original_dispatch(command)
        mark("input_command_complete", active_after=policy.active_label)
        return result

    # Only the active policy polls commands, but hook both instances so the
    # event stream remains continuous across policy switches.
    policy.primary._dispatch_command = dispatch
    policy.secondary._dispatch_command = dispatch

    original_switch = policy._switch_to_wbt

    def switch_to_wbt(self, auto_start_motion: bool = False):
        mark("loco_to_wbt_begin", auto_start_motion=bool(auto_start_motion))
        result = original_switch(auto_start_motion=auto_start_motion)
        mark("loco_to_wbt_complete", active=self.active_label)
        return result

    policy._switch_to_wbt = types.MethodType(switch_to_wbt, policy)

    original_return = policy._return_to_primary

    def return_to_primary(self):
        mark("wbt_to_loco_begin")
        result = original_return()
        mark("wbt_to_loco_complete", active=self.active_label)
        return result

    policy._return_to_primary = types.MethodType(return_to_primary, policy)


def write_events(path: Path, events: list[dict], recording_start_ns: int) -> None:
    payload = {
        "clock": "time.monotonic_ns",
        "recording_start_ns": recording_start_ns,
        "events": [
            {**event, "t_rel_s": (event["t_ns"] - recording_start_ns) * 1e-9}
            for event in events
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    plan = {
        "preset": "inference:h1-19dof-loco-wbt-real",
        "interface": args.interface,
        "domain_id": args.domain_id,
        "input": args.input,
        "max_duration_s": args.max_duration_s,
        "records": [
            "LowCmd q/dq/kp/kd/tau_ff",
            "LowState q/dq/tau_est/temperature/lost",
            "IMU quaternion/gyro/acceleration",
            "foot_force/foot_force_est",
            "input and Loco/WBT transition events",
        ],
        "safety": "overhead support, clear area, one LowCmd publisher, emergency stop ready",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print(f"Dry run only. Add --execute --confirm {CONFIRM_TOKEN} after review.")
        return
    if args.confirm != CONFIRM_TOKEN:
        raise SystemExit(f"Refusing execution: expected --confirm {CONFIRM_TOKEN}")
    if args.max_duration_s <= 0:
        raise ValueError("--max-duration-s must be positive.")

    config = policy_config(args)
    policy = DualModePolicy(config, config.secondary)
    interface = policy.primary.interface

    motion_mode = interface.get_motion_mode()
    print("MotionSwitcher: " + json.dumps(motion_mode), flush=True)
    if motion_mode["status"] != 0:
        raise RuntimeError(f"MotionSwitcher CheckMode failed: {motion_mode}")
    if motion_mode["name"]:
        raise RuntimeError(
            "A high-level motion service still owns control: "
            f"{motion_mode}. Release it before executing this launcher."
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{stamp}_h1_loco_wbt_real_closed_loop"
    recorder = H1PolicyRecorder(policy.primary.robot_config)
    # The DDS callback and LowCmd sender resolve this attribute dynamically.
    interface._sysid_recorder = recorder
    events: list[dict] = []
    install_event_hooks(policy, events)

    recording_start_ns = time.monotonic_ns()
    recorder.start(
        output_dir,
        metadata={
            "kind": "loco-wbt-real-closed-loop",
            "preset": "h1-19dof-loco-wbt-real",
            "primary_model": config.task.model_path,
            "secondary_model": config.secondary.task.model_path,
            "interface": args.interface,
            "domain_id": args.domain_id,
            "input": args.input,
            "events_path": str(output_dir / "events.json"),
        },
        max_duration_s=args.max_duration_s,
        state_rate_hz=500.0,
        command_available=True,
    )

    result = "completed"
    abort_reason = None
    saved = None
    try:
        events.append({"t_ns": time.monotonic_ns(), "event": "policy_loop_begin"})
        policy.run()
        events.append({"t_ns": time.monotonic_ns(), "event": "policy_loop_end"})
    except BaseException as exc:
        result = "aborted"
        abort_reason = f"{type(exc).__name__}: {exc}"
        events.append(
            {
                "t_ns": time.monotonic_ns(),
                "event": "policy_loop_exception",
                "reason": abort_reason,
            }
        )
        raise
    finally:
        restore_terminal_settings()
        try:
            saved = recorder.stop(result=result, abort_reason=abort_reason)
        finally:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_events(output_dir / "events.json", events, recording_start_ns)
        if saved is not None:
            print(json.dumps({**saved, "events_path": str(output_dir / "events.json")}, indent=2))


if __name__ == "__main__":
    main()
