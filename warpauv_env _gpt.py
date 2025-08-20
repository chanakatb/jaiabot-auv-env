"""
WarpAUV environment for IsaacLabs - Circular Path Following (FRAME-CONSISTENT)

- Standardizes control/observations/rewards around the BODY frame for consistency.
- Fixes TorchScript incompatibilities (no nested defs inside @script; no unsupported .norm overloads).
- Transforms desired WORLD tangent into BODY frame for heading alignment.
- Aligns initial yaw at reset with CLOCKWISE tangent by default.
- Adds cfg.tangent_sign (+1.0=CW, -1.0=CCW) and passes it to the JIT reward.

Author: adapted
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
    CUBOID_MARKER_CFG,
    VisualizationMarkers,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
import isaaclab.utils.math as math_utils

from .assets.warpauv import WARPAUV_CFG
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import (
    ConversionFunctionBasic,
    DynamicsFirstOrder,
    get_thruster_com_and_orientations,
)


# --------------------------- helpers (TorchScript-safe) ---------------------------
@torch.jit.script
def quat_inv(q: torch.Tensor) -> torch.Tensor:
    # q = [x, y, z, w]
    return torch.stack((-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]), dim=-1)


@torch.jit.script
def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2.0 * torch.pi) - torch.pi


class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the warp AUV env."""
    def __init__(self, env: "WarpAUVEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = WarpAUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=20.0, replicate_physics=True)
    debug_vis = True

    # Thruster control configuration - enable drive_right + both rear thrusters
    active_thrusters = [False, True, True, True, False, False]

    # env timing
    decimation = 4
    cap_episode_length = True
    episode_length_s = 500.0
    episode_length_before_reset = None

    # Actions: [forward in +X body, yaw]
    num_actions = 2
    num_observations = 20
    num_states = 0

    action_space = gym.spaces.Box(
        low=np.array([0.0, -1.0], dtype=np.float32),   # forward in [0,1], yaw in [-1,1]
        high=np.array([1.0,  1.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    state_space = gym.spaces.Box(low=0, high=0, shape=(0,), dtype=np.float32)

    use_boundaries = True
    max_auv_x = 15
    max_auv_y = 15
    max_auv_z = 2
    starting_depth = 0
    min_goal_steps = 100
    goal_completion_radius = 0.5
    goal_dims = 4
    eval_mode = False

    # Circular path parameters
    circle_radius: float = 7.0
    target_speed: float = 0.5  # m/s tangential target
    circle_center_x: float = 0.0
    circle_center_y: float = 0.0
    tangent_sign: float = +1.0  # +1.0=CW, -1.0=CCW

    init_guidance_rate = 0.8
    init_vel_max = 1.0

    # Reward scales
    rew_scale_position: float = 12.0
    rew_scale_speed: float = 6.0
    rew_scale_direction: float = 6.0
    rew_scale_ang: float = 4.0
    rew_scale_ang_vel: float = 0.1
    rew_scale_constraint_violation: float = 0.5
    rew_scale_actions: float = 0.001
    rew_scale_alive: float = 10.0

    # Visualization parameters
    target_radius: float = 7.0

    # dynamics (updated for surface operation)
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.05 / 100.0
    dyn_time_constant = 0.01
    volume = 1.252e-3
    mass = 1.248

    # Scaled hydrodynamic effects
    hydrodynamic_force_scale = 0.1
    buoyancy_force_scale = 0.2
    drag_force_scale = 0.05
    viscous_force_scale = 0.02

    # domain randomization
    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.01
        volume_range = [1.200e-3, 1.300e-3]
        mass_range = [1.200, 1.300]

    align_yaw_on_reset: bool = True


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._debug = True

        # Buffers
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._full_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins
        self._step_count = 0

        # Circle centers (WORLD)
        self._circle_centers = torch.zeros(self.num_envs, 2, device=self.device)
        self._circle_centers[:, 0] = self.scene.env_origins[:, 0] + self.cfg.circle_center_x
        self._circle_centers[:, 1] = self.scene.env_origins[:, 1] + self.cfg.circle_center_y

        # Thrusters
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        self.thruster_mask = torch.tensor(self.cfg.active_thrusters, device=self.device, dtype=torch.float)
        self.active_thruster_indices = torch.where(self.thruster_mask > 0)[0]

        torch.manual_seed(0)
        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug:
            print("mass: ", list(self._robot.root_physx_view._masses))

        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.inertia_tensors[:, 0] = 7.60e-4
        self.inertia_tensors[:, 1] = 1.04e-1
        self.inertia_tensors[:, 2] = 1.04e-1

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        if not isinstance(self.cfg.com_to_cob_offset, torch.Tensor):
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.clone()

        if not isinstance(self.cfg.volume, torch.Tensor):
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.clone()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True)

        self._init_thruster_dynamics()
        self._reset_idx(self._robot._ALL_INDICES)

        # Action smoothing
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.action_smoothing = 0.7
        self.max_action_change = 0.3

        print("=== CIRCULAR PATH FOLLOWING CONFIGURATION (BODY-FRAME CONSISTENT) ===")
        print(f"Circle radius: {self.cfg.circle_radius}m")
        print(f"Target speed: {self.cfg.target_speed} m/s")
        print(f"Tangent sign: {self.cfg.tangent_sign}  (+1=CW, -1=CCW)")
        print(f"Active thrusters: {self.cfg.active_thrusters}")

    # ---------- geometry helpers ----------
    def _get_circle_properties(self, pos_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        rel_pos = pos_w[:, :2] - self._circle_centers
        distance_from_center = torch.sqrt(torch.clamp(torch.sum(rel_pos * rel_pos, dim=1), min=1e-6))
        radial_error = distance_from_center - self.cfg.circle_radius
        current_angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])

        # Base CW tangent (sin, -cos), flipped by tangent_sign if needed
        desired_tangent = self.cfg.tangent_sign * torch.stack(
            (torch.sin(current_angle), -torch.cos(current_angle)), dim=1
        )

        if distance_from_center.min() > 1e-6:
            radial_unit = rel_pos / distance_from_center.unsqueeze(1)
            closest_point_on_circle = self._circle_centers + self.cfg.circle_radius * radial_unit
        else:
            closest_point_on_circle = self._circle_centers + torch.tensor(
                [self.cfg.circle_radius, 0.0], device=self.device
            )

        return {
            "distance_from_center": distance_from_center,
            "radial_error": radial_error,
            "current_angle": current_angle,
            "desired_tangent_world": desired_tangent,
            "closest_point_on_circle": closest_point_on_circle,
            "rel_pos": rel_pos,
        }


    # ---------- thrusters / dynamics ----------
    def _init_thruster_dynamics(self):
        if not isinstance(self.cfg.com_to_cob_offset, torch.Tensor):
            self.cfg.com_to_cob_offset = torch.tensor(
                self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False
            ).reshape(1, 3).repeat(self.num_envs, 1)

        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, 6, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------- RL API ----------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug:
            print("original actions vec: ", actions)

        # scale
        forward_raw = torch.clamp(actions[:, 0], 0, 1)
        forward_scaled = forward_raw * 0.6
        yaw_raw = torch.clamp(actions[:, 1], -1, 1)
        yaw_scaled = yaw_raw * 0.4

        # smooth
        alpha = 0.1
        current = torch.stack([forward_scaled, yaw_scaled], dim=1)
        smoothed = alpha * self._prev_actions + (1.0 - alpha) * current
        self._prev_actions = smoothed.clone()
        forward_final = smoothed[:, 0]
        yaw_final = smoothed[:, 1]

        # map to thrusters: drive_right (+X), differential rear for yaw
        self._full_actions[:] = 0.0
        self._full_actions[:, 1] = forward_final            # drive_right
        self._full_actions[:, 2] = -yaw_final * 0.8         # rear_left
        self._full_actions[:, 3] = +yaw_final * 0.8         # rear_right

        if self._debug:
            print(f"scaled actions - forward: {forward_final[0]:.4f}, yaw: {yaw_final[0]:.4f}")

    def _apply_action(self) -> None:
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._full_actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        circle_props = self._get_circle_properties(self._robot.data.root_pos_w)

        # WORLD and BODY velocities
        world_lin_vel = self._robot.data.root_lin_vel_w
        body_lin_vel = self._robot.data.root_lin_vel_b
        body_ang_vel = self._robot.data.root_ang_vel_b

        # Desired tangent (WORLD->BODY)
        tangent_world_2d = circle_props["desired_tangent_world"]
        tangent_world_3d = torch.zeros(self.num_envs, 3, device=self.device)
        tangent_world_3d[:, 0:2] = tangent_world_2d
        tangent_body_3d = quat_apply(quat_inv(self._robot.data.root_quat_w), tangent_world_3d)
        tangent_body_2d = tangent_body_3d[:, 0:2]

        # Tangential speed in BODY
        body_vel_2d = body_lin_vel[:, 0:2]
        tangential_speed = torch.sum(body_vel_2d * tangent_body_2d, dim=1)
        speed_error = (tangential_speed - self.cfg.target_speed) / (self.cfg.target_speed + 1e-6)

        # Direction alignment: BODY +X vs BODY tangent
        heading_body_2d = torch.tensor([1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        tangent_norm = torch.sqrt(torch.clamp(torch.sum(tangent_body_2d * tangent_body_2d, dim=1, keepdim=True), min=1e-6))
        direction_alignment = torch.sum(heading_body_2d * (tangent_body_2d / (tangent_norm + 1e-6)), dim=1)

        # Radial error normalized
        radial_error_normalized = circle_props["radial_error"] / (self.cfg.circle_radius + 1e-6)

        # Radial velocity (WORLD diagnostic)
        rel_pos = circle_props["rel_pos"]
        rel_n = rel_pos / (circle_props["distance_from_center"].unsqueeze(1) + 1e-6)
        world_vel_2d = world_lin_vel[:, :2]
        radial_velocity = torch.sum(world_vel_2d * rel_n, dim=1)

        # Attitude
        quat = self._robot.data.root_quat_w
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)).unsqueeze(1)
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0)).unsqueeze(1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).unsqueeze(1)

        # Position angle on circle
        angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        angle_sin = torch.sin(angle).unsqueeze(1)
        angle_cos = torch.cos(angle).unsqueeze(1)

        # Depth normalized
        depth_normalized = self._robot.data.root_pos_w[:, 2:3] / (self.cfg.max_auv_z + 1e-6)

        obs = torch.cat(
            [
                speed_error.unsqueeze(1),
                radial_error_normalized.unsqueeze(1),
                direction_alignment.unsqueeze(1),
                radial_velocity.unsqueeze(1),
                (tangential_speed / (self.cfg.target_speed + 1e-6)).unsqueeze(1),
                world_lin_vel / 2.0,      # 3
                body_lin_vel / 2.0,       # 3
                body_ang_vel / 3.0,       # 3
                tangent_body_2d,          # 2
                angle_sin, angle_cos,     # 2
                roll / torch.pi, pitch / torch.pi, yaw / torch.pi,  # 3
                depth_normalized,         # 1
            ],
            dim=-1,
        )
        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:
        reward_info = _compute_circular_path_reward(
            self.cfg.rew_scale_position,
            self.cfg.rew_scale_speed,
            self.cfg.rew_scale_direction,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_constraint_violation,
            self.cfg.rew_scale_actions,
            self.cfg.rew_scale_alive,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,   # BODY
            self._robot.data.root_ang_vel_b,   # BODY
            self._robot.data.root_lin_vel_w,   # WORLD
            self.cfg.target_speed,
            self.cfg.circle_radius,
            self._circle_centers,
            self._actions,
            float(self.cfg.tangent_sign),      # pass tangent_sign
        )

        if self._debug:
            self._print_circular_reward_debug(reward_info)

        return reward_info["total_reward"]


    def _print_circular_reward_debug(self, rc: Dict[str, torch.Tensor], env_idx: int = 0) -> None:
        i = env_idx
        try:
            print("================================")
            print(f"=== CIRCULAR PATH REWARD DEBUG (Env {i}) ===")
            print(f"Tangential speed: {rc['tangential_speed'][i].item():.4f} m/s (target: {self.cfg.target_speed} m/s)")
            print(f"Radial error: {rc['radial_error'][i].item():.4f} m (target: 0.0 m)")
            print(f"Direction alignment: {rc['direction_alignment'][i].item():.4f} (target: 1.0)")
            print("--- REWARD COMPONENTS ---")
            print(f"Position reward: {rc['position_reward'][i].item():.4f}")
            print(f"Speed reward: {rc['speed_reward'][i].item():.4f}")
            print(f"Direction reward: {rc['direction_reward'][i].item():.4f}")
            print(f"Upright reward: {rc['upright_reward'][i].item():.4f}")
            print(f"TOTAL: {rc['total_reward'][i].item():.4f}")
            print("================================")
        except KeyError as e:
            print(f"[DEBUG] Missing key: {e}")


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1 if self.cfg.cap_episode_length else torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._step_count += 1
        if self.cfg.episode_length_before_reset and self._step_count == self.cfg.episode_length_before_reset:
            time_out = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x)
                | (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y)
                | (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        quat = self._robot.data.root_quat_w
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        attitude_limit = math.radians(75.0)
        extreme_attitude = (torch.abs(roll) > attitude_limit) | (torch.abs(pitch) > attitude_limit)

        circle_props = self._get_circle_properties(self._robot.data.root_pos_w)
        too_far_from_circle = torch.abs(circle_props["radial_error"]) > 5.0

        out_of_bounds = out_of_bounds | extreme_attitude | too_far_from_circle
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        device = self.device
        env_ids = env_ids.to(device)
        root_state = self._default_root_state[env_ids].clone()

        # angle on circle
        rand_ang = torch.rand(len(env_ids), device=device) * 2 * torch.pi
        centers = self._circle_centers[env_ids]

        # position on circle (WORLD)
        root_state[:, 0] = centers[:, 0] + self.cfg.circle_radius * torch.cos(rand_ang)
        root_state[:, 1] = centers[:, 1] + self.cfg.circle_radius * torch.sin(rand_ang)
        root_state[:, 2] = 0.0

        # WORLD tangent per tangent_sign
        tan_x = self.cfg.tangent_sign * torch.sin(rand_ang)
        tan_y = self.cfg.tangent_sign * (-torch.cos(rand_ang))

        # yaw aligned with tangent
        yaw_angles = torch.atan2(tan_y, tan_x)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=device),
            torch.zeros(len(env_ids), device=device),
            yaw_angles,
        )
        root_state[:, 3:7] = rand_quat

        # initial velocity along tangent at target speed
        root_state[:, 7] = tan_x * self.cfg.target_speed
        root_state[:, 8] = tan_y * self.cfg.target_speed
        root_state[:, 9] = 0.0
        root_state[:, 10:13] = 0.0

        self._default_root_state[env_ids] = root_state
        if hasattr(self, "_actions"):
            self._actions[env_ids] = 0.0
        if hasattr(self, "_prev_actions"):
            self._prev_actions[env_ids] = 0.0

        if hasattr(self._robot, "write_root_state_to_sim"):
            self._robot.write_root_state_to_sim(self._default_root_state[env_ids], env_ids)
        else:
            self._robot.set_world_poses(root_state[:, 0:3], root_state[:, 3:7], env_ids)

        if self._debug and len(env_ids) > 0:
            print(f"=== CIRCULAR RESET DEBUG (Env {env_ids[0]}) ===")
            print(f"Reset angle: {math.degrees(rand_ang[0].item()):.1f}°")
            print(f"Position: ({root_state[0,0]:.3f}, {root_state[0,1]:.3f}, 0.0)")
            print(f"Heading aligned to tangent (yaw): {math.degrees(yaw_angles[0].item()):.1f}°")
            print(f"Initial tangent velocity: ({root_state[0,7]:.3f}, {root_state[0,8]:.3f})")


    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]
        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(
                len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius
            )
            lower, upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(lower, upper, self.volumes[env_ids].shape, self.device)

    def _sample_from_sphere(self, n: int, r: float):
        coords = torch.randn((n, 3), device=self.device)
        norms = torch.sqrt(torch.clamp(torch.sum(coords * coords, dim=1, keepdim=True), min=1e-9))
        coords = coords / norms
        radii = r * torch.pow(torch.rand((n, 1), device=self.device), 1.0 / 3.0)
        return radii * coords

    def _compute_dynamics(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        if self._debug:
            print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)

        masked_actions = actions * self.thruster_mask.unsqueeze(0)
        motorValues = torch.clone(masked_actions)

        motorValues[torch.abs(motorValues) < 0.01] = 0
        motorValues[motorValues >= 0.08] = -139.0 * (torch.pow(motorValues[motorValues >= 0.08], 2.0)) + 500 * motorValues[
            motorValues >= 0.08
        ] + 8.28
        motorValues[motorValues <= -0.08] = 161.0 * (torch.pow(motorValues[motorValues <= -0.08], 2.0)) + 517.86 * motorValues[
            motorValues <= -0.08
        ] - 5.72

        motorValues = self.thruster_dynamics.update(motorValues, self.episode_length_buf * self.sim.cfg.dt)
        motorValues = self.thruster_conversion.convert(motorValues)

        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * motorValues.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)

        # Hydro forces
        buoy_forces, buoy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, abs(self._gravity_magnitude), self.com_to_cob_offsets
        )
        dens_f, dens_t, visc_f, visc_t = self.force_calculation_functions.calculate_density_and_viscosity_forces(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            self.inertia_tensors,
            self.inertia_tensors_mean,
            self.cfg.water_beta,
            self.cfg.water_rho,
            self.masses,
        )

        # scales
        buoy_forces *= self.cfg.buoyancy_force_scale
        buoy_torques *= self.cfg.buoyancy_force_scale
        dens_f *= self.cfg.drag_force_scale
        dens_t *= self.cfg.drag_force_scale
        visc_f *= self.cfg.viscous_force_scale
        visc_t *= self.cfg.viscous_force_scale

        forces = dens_f + buoy_forces + visc_f + thruster_forces
        torques = dens_t + buoy_torques + visc_t + thruster_torques

        forces, torques = self._limit_forces_and_torques(forces, torques)

        if self._debug:
            print("thruster forces:", torch.norm(thruster_forces[0]).item())
            print("buoyancy forces:", torch.norm(buoy_forces[0]).item())
            print("final forces", forces)
            print("final torques", torques)

        return forces, torques

    def _limit_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_force = 20.0
        max_torque = 10.0
        return torch.clamp(forces, -max_force, max_force), torch.clamp(torques, -max_torque, max_torque)

    # --------------------------- visualization ---------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "circle_visualizer"):
                cfg = CUBOID_MARKER_CFG.copy()
                cfg.markers["cuboid"].size = (0.2, 0.2, 0.05)
                cfg.prim_path = "/Visuals/Command/circle_path"
                self.circle_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "global_frame_x_visualizer"):
                cfg = RED_ARROW_X_MARKER_CFG.copy()
                cfg.markers["arrow"].scale = (0.4, 0.4, 2.0)
                cfg.prim_path = "/Visuals/Command/global_frame_x"
                self.global_frame_x_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "global_frame_y_visualizer"):
                cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                cfg.markers["arrow"].scale = (0.4, 0.4, 2.0)
                cfg.prim_path = "/Visuals/Command/global_frame_y"
                self.global_frame_y_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "global_frame_z_visualizer"):
                cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                cfg.markers["arrow"].scale = (0.4, 0.4, 2.0)
                cfg.prim_path = "/Visuals/Command/global_frame_z"
                self.global_frame_z_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "body_frame_x_visualizer"):
                cfg = RED_ARROW_X_MARKER_CFG.copy()
                cfg.markers["arrow"].scale = (0.3, 0.3, 1.5)
                cfg.prim_path = "/Visuals/Command/body_frame_x"
                self.body_frame_x_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "body_frame_y_visualizer"):
                cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                cfg.markers["arrow"].scale = (0.3, 0.3, 1.5)
                cfg.prim_path = "/Visuals/Command/body_frame_y"
                self.body_frame_y_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "body_frame_z_visualizer"):
                cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                cfg.markers["arrow"].scale = (0.3, 0.3, 1.5)
                cfg.prim_path = "/Visuals/Command/body_frame_z"
                self.body_frame_z_visualizer = VisualizationMarkers(cfg)

            if not hasattr(self, "target_direction_visualizer"):
                cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/target_direction"
                cfg.markers["arrow"].scale = (0.25, 0.25, 2)
                cfg.markers["arrow"].visual_material.diffuse_color = (0.8, 0.2, 0.8)
                self.target_direction_visualizer = VisualizationMarkers(cfg)

            self.circle_visualizer.set_visibility(True)
            self.global_frame_x_visualizer.set_visibility(True)
            self.global_frame_y_visualizer.set_visibility(True)
            self.global_frame_z_visualizer.set_visibility(True)
            self.body_frame_x_visualizer.set_visibility(True)
            self.body_frame_y_visualizer.set_visibility(True)
            self.body_frame_z_visualizer.set_visibility(True)
            self.target_direction_visualizer.set_visibility(True)
        else:
            for name in [
                "circle_visualizer",
                "global_frame_x_visualizer",
                "global_frame_y_visualizer",
                "global_frame_z_visualizer",
                "body_frame_x_visualizer",
                "body_frame_y_visualizer",
                "body_frame_z_visualizer",
                "target_direction_visualizer",
            ]:
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "circle_visualizer") and hasattr(self, "_circle_centers"):
            num_points = 64
            angles = torch.linspace(0, 2 * torch.pi, num_points, device=self.device)
            circle_points = torch.zeros(self.num_envs * num_points, 3, device=self.device)
            for env_idx in range(self.num_envs):
                s, e = env_idx * num_points, (env_idx + 1) * num_points
                circle_points[s:e, 0] = self._circle_centers[env_idx, 0] + self.cfg.circle_radius * torch.cos(angles)
                circle_points[s:e, 1] = self._circle_centers[env_idx, 1] + self.cfg.circle_radius * torch.sin(angles)
                circle_points[s:e, 2] = 0.0
            self.circle_visualizer.visualize(translations=circle_points)

        if hasattr(self, "global_frame_x_visualizer"):
            centers = torch.zeros(self.num_envs, 3, device=self.device)
            centers[:, :2] = self._circle_centers
            centers[:, 2] = 0.0
            xq = torch.zeros(self.num_envs, 4, device=self.device)
            xq[:, 3] = 1.0
            self.global_frame_x_visualizer.visualize(translations=centers, orientations=xq)

        if hasattr(self, "global_frame_y_visualizer"):
            centers = torch.zeros(self.num_envs, 3, device=self.device)
            centers[:, :2] = self._circle_centers
            centers[:, 2] = 0.0
            yq = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi / 2, device=self.device),
            )
            self.global_frame_y_visualizer.visualize(translations=centers, orientations=yq)

        if hasattr(self, "global_frame_z_visualizer"):
            centers = torch.zeros(self.num_envs, 3, device=self.device)
            centers[:, :2] = self._circle_centers
            centers[:, 2] = 0.0
            zq = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi / 2, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )
            self.global_frame_z_visualizer.visualize(translations=centers, orientations=zq)

        if hasattr(self, "body_frame_x_visualizer"):
            self.body_frame_x_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w
            )

        if hasattr(self, "body_frame_y_visualizer"):
            y_off = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi / 2, device=self.device),
            )
            body_y = math_utils.quat_mul(self._robot.data.root_quat_w, y_off)
            self.body_frame_y_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=body_y
            )

        if hasattr(self, "body_frame_z_visualizer"):
            z_off = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi / 2, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )
            body_z = math_utils.quat_mul(self._robot.data.root_quat_w, z_off)
            self.body_frame_z_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=body_z
            )

        if hasattr(self, "target_direction_visualizer"):
            circle_props = self._get_circle_properties(self._robot.data.root_pos_w)
            tan_w = circle_props["desired_tangent_world"]
            tan_w3 = torch.zeros(self.num_envs, 3, device=self.device)
            tan_w3[:, :2] = tan_w
            tan_b3 = quat_apply(quat_inv(self._robot.data.root_quat_w), tan_w3)
            ang_b = torch.atan2(tan_b3[:, 1], tan_b3[:, 0])
            tan_q_b = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                ang_b,
            )
            self.target_direction_visualizer.visualize(
                translations=self._robot.data.root_pos_w,
                orientations=math_utils.quat_mul(self._robot.data.root_quat_w, tan_q_b),
            )

def get_tangent_sign(cw: bool) -> float:
    """+1.0 for CW, -1.0 for CCW — matches the JIT reward expectation."""
    return 1.0 if cw else -1.0

# ---------- JIT-safe reward for circular path following ----------
# =======================
# TorchScript REWARD (JIT)
# =======================
# Matches your env call signature and returns a dict with all keys that the env expects.
# NOTE: This version assumes the vehicle's FORWARD axis is +Y in the BODY frame
@torch.jit.script
def _compute_circular_path_reward(
    w_position: float,
    w_speed: float,
    w_direction: float,
    w_ang: float,
    w_ang_vel: float,
    w_constr: float,
    w_actions: float,
    w_alive: float,
    root_pos_w: torch.Tensor,        # (N,3)
    root_quat_w: torch.Tensor,       # (N,4)  (x,y,z,w)
    root_lin_vel_b: torch.Tensor,    # (N,3)
    root_ang_vel_b: torch.Tensor,    # (N,3)
    root_lin_vel_w: torch.Tensor,    # (N,3)
    target_speed: float,
    circle_radius: float,
    circle_centers: torch.Tensor,    # (N,3) or (N,2)
    actions: torch.Tensor,           # (N, A)
    tangent_sign: float,             # +1 = CW, -1 = CCW
) -> Dict[str, torch.Tensor]:
    eps: float = 1e-6
    N = root_pos_w.shape[0]
    dtype = root_pos_w.dtype
    device = root_pos_w.device

    center_xy = circle_centers[:, :2]
    rel_xy = root_pos_w[:, :2] - center_xy
    dist = torch.norm(rel_xy, dim=1)
    radial_error = dist - torch.as_tensor(circle_radius, dtype=dtype, device=device)

    angle = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])

    # Desired world tangent (CW base, flipped by sign)
    cw_tangent = torch.stack((torch.sin(angle), -torch.cos(angle)), dim=1)
    sign = torch.sign(torch.as_tensor(tangent_sign, dtype=dtype, device=device))
    desired_tangent_world = cw_tangent * sign

    # Yaw from quaternion (x,y,z,w)
    qx, qy, qz, qw = root_quat_w[:, 0], root_quat_w[:, 1], root_quat_w[:, 2], root_quat_w[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Rotate world->body (2D)
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)
    R_w2b = torch.stack([
        torch.stack([cos_yaw, -sin_yaw], dim=1),
        torch.stack([sin_yaw,  cos_yaw], dim=1)
    ], dim=1)
    desired_tangent_body_2d = torch.bmm(R_w2b, desired_tangent_world.unsqueeze(-1)).squeeze(-1)

    # Direction alignment (BODY +Y is forward here)
    heading_body_2d = torch.tensor([0.0, 1.0], device=device, dtype=dtype).repeat(N, 1)
    tan_norm = torch.norm(desired_tangent_body_2d, dim=1, keepdim=True)
    direction_alignment = torch.sum(heading_body_2d * (desired_tangent_body_2d / (tan_norm + eps)), dim=1)
    direction_reward = w_direction * direction_alignment

    # Speed reward (world progress along tangent)
    tangential_speed = torch.sum(root_lin_vel_w[:, :2] * desired_tangent_world, dim=1)
    pos_progress = torch.clamp(tangential_speed, min=0.0)
    speed_ratio = pos_progress / (torch.as_tensor(target_speed, dtype=dtype, device=device) + eps)
    speed_reward = w_speed * torch.clamp(speed_ratio, 0.0, 1.0)

    # Mild penalty for reverse motion
    wrong_way_pen = -0.5 * w_speed * torch.clamp(-tangential_speed, min=0.0) / (torch.as_tensor(target_speed, dtype=dtype, device=device) + eps)

    # Position reward (Gaussian in radial error)
    sigma = torch.as_tensor(0.5, dtype=dtype, device=device)
    position_reward = w_position * torch.exp(-0.5 * (radial_error / (sigma + eps)) ** 2)

    upright_reward = torch.zeros_like(position_reward)
    ang_reward = torch.zeros_like(position_reward)
    angvel_reward = torch.zeros_like(position_reward)
    constr_reward = torch.zeros_like(position_reward)
    action_penalty = torch.zeros_like(position_reward)
    alive_reward = torch.as_tensor(w_alive, dtype=dtype, device=device).repeat(N)

    total_reward = (
        position_reward
        + speed_reward
        + direction_reward
        + upright_reward
        + wrong_way_pen
        + ang_reward
        + angvel_reward
        + constr_reward
        + action_penalty
        + alive_reward
    )

    return {
        "position_reward": position_reward,
        "speed_reward": speed_reward,
        "direction_reward": direction_reward,
        "upright_reward": upright_reward,
        "wrong_way_penalty": wrong_way_pen,
        "ang_reward": ang_reward,
        "angvel_reward": angvel_reward,
        "constr_reward": constr_reward,
        "action_penalty": action_penalty,
        "alive_reward": alive_reward,
        "tangential_speed": tangential_speed,
        "radial_error": radial_error,
        "direction_alignment": direction_alignment,
        "total_reward": total_reward,
    }
