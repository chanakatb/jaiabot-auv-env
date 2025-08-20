"""
WarpAUV environment for IsaacLabs - Circular Trajectory Configuration

Author: Kevin Chang and Levi "Veevee" Cai (cail@mit.edu)
Modified for circular trajectory following + yaw integral bonus
"""

from __future__ import annotations

import gymnasium as gym
import random
import math
import torch
import numpy as np
from collections.abc import Sequence
from typing import Dict  # for hints used in methods above later imports

from .assets.warpauv import WARPAUV_CFG, SIMPLE_WARPAUV_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, normalize
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers, RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul
import isaaclab.utils.math as math_utils

##
# Hydrodynamic model
##
from isaaclab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations

class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the warpauvenv environment."""

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

    # Use the original USD version
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=20.0, replicate_physics=True)
    debug_vis = True

    # Circular trajectory parameters
    target_radius: float = 7.0
    target_speed: float = 1.0

    # Thruster control configuration - Only rear thrusters for forward thrust and yaw control
    active_thrusters = [False, True, False, True, False, False]  # drive_right + rear_right

    # env
    decimation = 4
    cap_episode_length = True
    episode_length_s = 120.0
    episode_length_before_reset = None
    
    # Action/Observation spaces
    num_actions = 2
    num_observations = 20
    num_states = 0
    
    action_space = gym.spaces.Box(
        low=np.array([0.0, -1.0], dtype=np.float32), 
        high=np.array([1.0, 1.0], dtype=np.float32), 
        shape=(2,), 
        dtype=np.float32
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

    goal_spawn_radius = 7.0
    init_guidance_rate = 0.8
    init_vel_max = 1.0

    # Reward scales
    rew_scale_circular_pos: float = 1.0
    rew_scale_speed: float = 15.0
    rew_scale_constraint_violation: float = 0.5
    rew_scale_completion: float = 100.0

    rew_scale_pos: float = 0.0
    rew_scale_ang: float = 8.0
    rew_scale_vel: float = 8.0
    rew_scale_ang_vel: float = 2.0
    rew_scale_lin_vel: float = 0.0
    rew_scale_actions: float = 0.05
    rew_scale_terminated: float = 0.0
    rew_scale_alive: float = 0.1

    # NEW: Yaw integral bonus knobs
    rew_scale_yaw_integral: float = 0.0     # weight of the integral in the total reward
    yaw_int_tau_s: float = 3.0              # decay time constant (s)
    yaw_int_gain: float = 1.0               # integration gain
    yaw_int_gate_speed_frac: float = 0.4    # only integrate if v_tan >= 0.4 * target_speed
    yaw_int_window_rad: float = math.radians(45.0)  # only integrate if |yaw_err| <= 45°

    # dynamics (updated for surface operation)
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.1 / 100.0
    dyn_time_constant = 0.05
    volume = 1.252e-3
    mass = 1.248

    # domain randomization
    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.01
        volume_range = [1.200e-3, 1.300e-3]
        mass_range = [1.200, 1.300]

    # allow aligning yaw on reset
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

        # NEW: yaw integral state
        self._yaw_int = torch.zeros(self.num_envs, device=self.device)
        
        # Each env's circle center = its origin
        self._circle_centers = self.scene.env_origins[:, :2].clone()
        
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

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.inertia_tensors[:, 0] = 7.60e-4
        self.inertia_tensors[:, 1] = 1.04e-1
        self.inertia_tensors[:, 2] = 1.04e-1

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        self._init_thruster_dynamics()
        
        self._reset_idx(self._robot._ALL_INDICES)

        # Add action smoothing
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.action_smoothing = 0.8  # Strong smoothing to prevent jittery control
        
        # Add action rate limiting
        self.max_action_change = 0.2  # Maximum change per step

        print("=== CIRCULAR TRAJECTORY THRUSTER CONFIGURATION ===")
        print(f"Active thrusters: {self.cfg.active_thrusters}")
        print(f"Target radius: {self.cfg.target_radius}m")
        print(f"Target speed: {self.cfg.target_speed} m/s")
        print("REWARD SCALES CHECK:")
        print(f"  Circular position: {self.cfg.rew_scale_circular_pos}")
        print(f"  Speed: {self.cfg.rew_scale_speed}")
        print(f"  Orientation: {self.cfg.rew_scale_ang}")
        print(f"  Constraint: {self.cfg.rew_scale_constraint_violation}")
        print(f"  Yaw integral: {self.cfg.rew_scale_yaw_integral}")
        print("THRUSTER CONSTRAINTS:")
        print("  Action 0 -> drive_right (index 1): FORWARD ONLY [0, 1]")
        print("  Action 1 -> rear_right (index 3): YAW CONTROL [-1, 1]")
        print()
        for i, name in enumerate(['drive_left', 'drive_right', 'rear_left', 'rear_right', 'front_left', 'front_right']):
            if self.cfg.active_thrusters[i]:
                pos = self.thruster_com_offsets[0, i]
                constraint = "FORWARD ONLY" if i == 1 else "BIDIRECTIONAL"
                print(f"{name:12s}: x={pos[0]:+7.3f}, y={pos[1]:+7.3f}, z={pos[2]:+7.3f} [ACTIVE - {constraint}]")
        
        

    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.cfg.com_to_cob_offset = torch.tensor(self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False).reshape(1,3).repeat(self.num_envs, 1)

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug: print("original actions vec: ", actions)
        
        # CRITICAL: Scale down forward action dramatically
        forward_raw = torch.clamp(actions[:, 0], 0, 1)
        forward_scaled = forward_raw * 0.25  # Scale down from 0.2 to 0.15
        
        yaw_raw = torch.clamp(actions[:, 1], -1, 1) 
        yaw_scaled = yaw_raw * 0.15  # Very small yaw authority
        
        # Strong smoothing
        if hasattr(self, '_prev_actions'):
            current = torch.stack([forward_scaled, yaw_scaled], dim=1)
            smoothed = 0.95 * self._prev_actions + 0.05 * current
            self._prev_actions = smoothed.clone()
            forward_final = smoothed[:, 0]
            yaw_final = smoothed[:, 1] 
        else:
            self._prev_actions = torch.stack([forward_scaled, yaw_scaled], dim=1)
            forward_final = forward_scaled
            yaw_final = yaw_scaled
        
        # Apply to thrusters
        self._full_actions[:] = 0.0
        self._full_actions[:, 1] = forward_final  # drive_right
        self._full_actions[:, 3] = yaw_final      # rear_right
        
        if self._debug: 
            print(f"scaled actions - forward: {forward_final[0]:.4f}, yaw: {yaw_final[0]:.4f}")
            print("final thruster actions: ", self._full_actions)

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._full_actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        robot_pos_xy = self._robot.data.root_pos_w[:, :2]
        distance_to_center = torch.norm(robot_pos_xy - self._circle_centers, dim=1, keepdim=True)
        radius_error = distance_to_center - self.cfg.target_radius
        relative_pos = robot_pos_xy - self._circle_centers
        current_angle = torch.atan2(relative_pos[:, 1], relative_pos[:, 0]).unsqueeze(1)

        desired_tangent = torch.cat([
            -torch.sin(current_angle.squeeze(1)).unsqueeze(1), 
            torch.cos(current_angle.squeeze(1)).unsqueeze(1)
        ], dim=1)
        
        forward_dir_w = quat_apply(self._robot.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        current_forward_xy = forward_dir_w[:, :2]
        forward_speed = self._robot.data.root_lin_vel_b[:, 0:1]
        speed_error = forward_speed - self.cfg.target_speed

        roll = torch.atan2(
            2 * (self._robot.data.root_quat_w[:, 3] * self._robot.data.root_quat_w[:, 0] + self._robot.data.root_quat_w[:, 1] * self._robot.data.root_quat_w[:, 2]),
            1 - 2 * (self._robot.data.root_quat_w[:, 0]**2 + self._robot.data.root_quat_w[:, 1]**2)
        ).unsqueeze(1)
        pitch = torch.asin(2 * (self._robot.data.root_quat_w[:, 3] * self._robot.data.root_quat_w[:, 1] - self._robot.data.root_quat_w[:, 2] * self._robot.data.root_quat_w[:, 0])).unsqueeze(1)

        obs = torch.cat([
            radius_error,
            speed_error,
            current_angle,
            desired_tangent,
            current_forward_xy,
            self._robot.data.root_pos_w[:, 2:3],
            roll,
            pitch,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.root_quat_w,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Fixed reward calculation matching the function signature"""
        reward_info = _compute_circular_trajectory(
            self.cfg.rew_scale_circular_pos,      # w_circ
            self.cfg.rew_scale_speed,             # w_speed
            self.cfg.rew_scale_ang,               # w_ang
            self.cfg.rew_scale_ang_vel,           # w_ang_vel 
            self.cfg.rew_scale_constraint_violation, # w_constr
            self.cfg.rew_scale_actions,           # w_actions
            self.cfg.rew_scale_alive,             # w_alive
            self._robot.data.root_pos_w,          # root_pos_w
            self._robot.data.root_quat_w,         # root_quat_w
            self._robot.data.root_lin_vel_b,      # root_lin_vel_b
            self._robot.data.root_ang_vel_b,      # root_ang_vel_b
            self._circle_centers,                 # circle_centers
            float(self.cfg.target_radius),        # target_radius
            float(self.cfg.target_speed),         # target_speed
            self._actions,                        # actions
            # REMOVE ALL THE EXTRA PARAMETERS - your function only takes 15!
        )

        # Keep yaw integral at zero
        self._yaw_int = torch.zeros_like(self._yaw_int)

        if self._debug:
            self._print_reward_debug_simple(reward_info)

        return reward_info["total_reward"]

    def _print_reward_debug_simple(self, reward_components: Dict[str, torch.Tensor], env_idx: int = 0) -> None:
        i = env_idx
        try:
            print("================================")
            print(f"=== REWARD COMPONENTS (Env {i}) ===")
            print(f"Distance to center: {reward_components['distance_to_center'][i].item():.4f}m (target: {self.cfg.target_radius}m)")
            print(f"Radius error: {reward_components['radius_error'][i].item():.4f}m") 
            print(f"Forward speed: {reward_components['forward_speed'][i].item():.4f} m/s (target: {self.cfg.target_speed} m/s)")
            print(f"Speed error: {reward_components['speed_error'][i].item():.4f} m/s")
            print(f"Yaw error: {reward_components['yaw_error'][i].item():.4f} rad ({torch.rad2deg(reward_components['yaw_error'][i]).item():.1f}°)")
            print(f"Yaw rate: {reward_components['yaw_rate'][i].item():.4f} rad/s")
            print("--- REWARD COMPONENTS ---")
            print(f"Speed reward: {reward_components['speed_reward'][i].item():.4f}")
            print(f"Radius reward: {reward_components['radius_reward'][i].item():.4f}")
            print(f"Orientation reward: {reward_components['orientation_reward'][i].item():.4f}")
            print(f"Overspeed penalty: {reward_components['overspeed_penalty'][i].item():.4f}")
            print(f"Underspeed penalty: {reward_components['underspeed_penalty'][i].item():.4f}")
            print(f"Angular velocity penalty: {reward_components['angular_velocity_penalty'][i].item():.4f}")
            print(f"Attitude penalty: {reward_components['attitude_penalty'][i].item():.4f}")
            print(f"Sway/heave penalty: {reward_components['sway_heave_penalty'][i].item():.4f}")
            print(f"Action penalty: {reward_components['action_penalty'][i].item():.4f}")
            print(f"Alive bonus: {reward_components['alive_bonus'][i].item():.4f}")
            print(f"TOTAL: {reward_components['total_reward'][i].item():.4f}")
            print("================================")
        except KeyError as e:
            print(f"[DEBUG] Missing key in reward_components: {e}")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._step_count = self._step_count + 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset selected envs and set initial WORLD velocity tangent to the target circle."""
        device = self.device if hasattr(self, "device") else self._device
        env_ids = env_ids.to(device)

        # Default root state
        root_state = self._default_root_state[env_ids].clone()
        root_state[:, 2] = 0.0
        root_state[:, 7:10] = 0.0
        root_state[:, 10:13] = 0.0

        # Tangent direction at current position
        centers_xy = self._circle_centers[env_ids]
        pos_w = root_state[:, 0:3]
        rel_xy = pos_w[:, :2] - centers_xy
        ang = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])
        t_hat = torch.stack([-torch.sin(ang), torch.cos(ang)], dim=1)

        target_speed = torch.as_tensor(self.cfg.target_speed, device=device)
        v_w = torch.zeros((env_ids.numel(), 3), device=device)
        v_w[:, :2] = target_speed * t_hat
        root_state[:, 7:10] = v_w

        if getattr(self.cfg, "align_yaw_on_reset", True):
            cos_yaw = torch.cos(ang * 0.5)
            sin_yaw = torch.sin(ang * 0.5)
            quat_yaw = torch.stack([cos_yaw, torch.zeros_like(ang), torch.zeros_like(ang), sin_yaw], dim=1)
            root_state[:, 3:7] = quat_yaw

        self._default_root_state[env_ids] = root_state

        if hasattr(self, "_default_dof_pos") and hasattr(self, "_default_dof_vel"):
            self._dof_pos[env_ids] = self._default_dof_pos[env_ids]
            self._dof_vel[env_ids] = self._default_dof_vel[env_ids]

        if hasattr(self, "_actions"):
            self._actions[env_ids] = 0.0

        # Reset yaw integral for these envs
        self._yaw_int[env_ids] = 0.0

        if hasattr(self._robot, "write_root_state_to_sim"):
            self._robot.write_root_state_to_sim(self._default_root_state[env_ids], env_ids)
        elif hasattr(self._robot, "set_world_poses"):
            self._robot.set_world_poses(root_state[:, 0:3], root_state[:, 3:7], env_ids)

        if hasattr(self._robot, "write_dof_state_to_sim") and hasattr(self, "_dof_pos") and hasattr(self, "_dof_vel"):
            self._robot.write_dof_state_to_sim(self._dof_pos[env_ids], self._dof_vel[env_ids], env_ids)

        if hasattr(self, "_obs_filter_state"):
            self._obs_filter_state[env_ids] = 0.0

        if hasattr(self, "_reset_goal"):
            self._reset_goal(env_ids)

    def _reset_goal(self, env_ids: Sequence[int]):
        """Place robot at random point on each robot's individual circle with tangential orientation and WORLD tangent velocity."""
        angles = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        circle_x = self.scene.env_origins[env_ids, 0] + self.cfg.target_radius * torch.cos(angles)
        circle_y = self.scene.env_origins[env_ids, 1] + self.cfg.target_radius * torch.sin(angles)
        
        self._default_root_state[env_ids, 0] = circle_x
        self._default_root_state[env_ids, 1] = circle_y
        self._default_root_state[env_ids, 2] = 0.0
        
        tangent_angles = angles + torch.pi/2
        self._default_root_state[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
            torch.zeros_like(tangent_angles),
            torch.zeros_like(tangent_angles),
            tangent_angles
        )
        
        t_hat = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=1)
        v_w = torch.zeros((len(env_ids), 3), device=self.device)
        v_w[:, :2] = self.cfg.target_speed * t_hat
        self._default_root_state[env_ids, 7:10] = v_w
        self._default_root_state[env_ids, 10:13] = 0.0
        
        # Reset yaw integral for these envs
        self._yaw_int[env_ids] = 0.0

        if self._debug and len(env_ids) > 0:
            print(f"=== RESET DEBUG (Env {env_ids[0]}) ===")
            print(f"Circle center: ({self.scene.env_origins[env_ids[0], 0]:.3f}, {self.scene.env_origins[env_ids[0], 1]:.3f})")
            print(f"Target position: ({circle_x[0]:.3f}, {circle_y[0]:.3f}, 0.0)")
            print(f"Distance to center: {torch.norm(torch.tensor([circle_x[0] - self.scene.env_origins[env_ids[0], 0], circle_y[0] - self.scene.env_origins[env_ids[0], 1]], device=self.device)):.3f}m")
            print(f"Target radius: {self.cfg.target_radius}m")
            print(f"Angle: {angles[0]:.3f} rad ({angles[0] * 180 / 3.14159:.1f}°)")
            print(f"Tangent angle: {tangent_angles[0]:.3f} rad ({tangent_angles[0] * 180 / 3.14159:.1f}°)")
            print("================================")

    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]

        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(vol_lower, vol_upper, self.volumes[env_ids].shape, self.device)

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms
        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1/3)
        return radii * coords

    def _compute_dynamics(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)

        masked_actions = actions * self.thruster_mask.unsqueeze(0)
        
        if self._debug: 
            print("Original actions:", actions[0])
            print("Thruster mask:", self.thruster_mask)
            print("Masked actions:", masked_actions[0])
        
        motorValues = torch.clone(masked_actions)

        if self._debug: print("motorValues: ", motorValues)

        motorValues[torch.abs(motorValues) < 0.08] = 0 
        motorValues[motorValues >= 0.08] = -139.0 * (torch.pow(motorValues[motorValues >= 0.08], 2.0)) + 500 * motorValues[motorValues >= 0.08] + 8.28
        motorValues[motorValues <= -0.08] = 161.0 * (torch.pow(motorValues[motorValues <= -0.08], 2.0)) + 517.86 * motorValues[motorValues <= -0.08] - 5.72

        motorValues = self.thruster_dynamics.update(motorValues, self.episode_length_buf * self.sim.cfg.dt)
        motorValues = self.thruster_conversion.convert(motorValues)

        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * motorValues.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)

        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, 
            abs(self._gravity_magnitude), self.com_to_cob_offsets)

        density_forces, density_torques, viscosity_forces, viscosity_torques = \
            self.force_calculation_functions.calculate_density_and_viscosity_forces(
                self._robot.data.root_quat_w, self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w, 
                self.inertia_tensors, self.inertia_tensors_mean, self.cfg.water_beta, self.cfg.water_rho, self.masses
            )

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques

        if self._debug: 
            print("final forces", forces)
            print("final torques", torques)

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "circle_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/circle_path"
                self.circle_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "global_frame_x_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.4, 0.4, 2.0)
                marker_cfg.prim_path = "/Visuals/Command/global_frame_x"
                self.global_frame_x_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "global_frame_y_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.4, 0.4, 2.0)
                marker_cfg.prim_path = "/Visuals/Command/global_frame_y"
                self.global_frame_y_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "global_frame_z_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.4, 0.4, 2.0)
                marker_cfg.prim_path = "/Visuals/Command/global_frame_z"
                self.global_frame_z_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "body_frame_x_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 1.5)
                marker_cfg.prim_path = "/Visuals/Command/body_frame_x"
                self.body_frame_x_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "body_frame_y_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 1.5)
                marker_cfg.prim_path = "/Visuals/Command/body_frame_y"
                self.body_frame_y_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "body_frame_z_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 1.5)
                marker_cfg.prim_path = "/Visuals/Command/body_frame_z"
                self.body_frame_z_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "target_direction_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/target_direction"
                marker_cfg.markers["arrow"].scale = (0.25, 0.25, 2)
                marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.8, 0.2, 0.8)
                self.target_direction_visualizer = VisualizationMarkers(marker_cfg)
            
            self.circle_visualizer.set_visibility(True)
            self.global_frame_x_visualizer.set_visibility(True)
            self.global_frame_y_visualizer.set_visibility(True)
            self.global_frame_z_visualizer.set_visibility(True)
            self.body_frame_x_visualizer.set_visibility(True)
            self.body_frame_y_visualizer.set_visibility(True)
            self.body_frame_z_visualizer.set_visibility(True)
            self.target_direction_visualizer.set_visibility(True)
        else:
            for attr_name in ["circle_visualizer", "global_frame_x_visualizer", "global_frame_y_visualizer", 
                             "global_frame_z_visualizer", "body_frame_x_visualizer", "body_frame_y_visualizer", 
                             "body_frame_z_visualizer", "target_direction_visualizer"]:
                if hasattr(self, attr_name):
                    getattr(self, attr_name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "circle_visualizer"):
            num_points = 32
            angles = torch.linspace(0, 2*torch.pi, num_points, device=self.device)
            circle_points = torch.zeros(self.num_envs * num_points, 3, device=self.device)
            for env_idx in range(self.num_envs):
                start_idx = env_idx * num_points
                end_idx = (env_idx + 1) * num_points
                circle_points[start_idx:end_idx, 0] = self._circle_centers[env_idx, 0] + self.cfg.target_radius * torch.cos(angles)
                circle_points[start_idx:end_idx, 1] = self._circle_centers[env_idx, 1] + self.cfg.target_radius * torch.sin(angles)
                circle_points[start_idx:end_idx, 2] = 0.0
            self.circle_visualizer.visualize(translations=circle_points)

        if hasattr(self, "global_frame_x_visualizer"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            x_orientations = torch.zeros(self.num_envs, 4, device=self.device)
            x_orientations[:, 3] = 1.0
            self.global_frame_x_visualizer.visualize(translations=center_positions, orientations=x_orientations)

        if hasattr(self, "global_frame_y_visualizer"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            y_orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi/2, device=self.device)
            )
            self.global_frame_y_visualizer.visualize(translations=center_positions, orientations=y_orientations)

        if hasattr(self, "global_frame_z_visualizer"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            z_orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi/2, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
            self.global_frame_z_visualizer.visualize(translations=center_positions, orientations=z_orientations)

        if hasattr(self, "body_frame_x_visualizer"):
            self.body_frame_x_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w)

        if hasattr(self, "body_frame_y_visualizer"):
            y_offset_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi/2, device=self.device)
            )
            body_y_orientations = math_utils.quat_mul(self._robot.data.root_quat_w, y_offset_quat)
            self.body_frame_y_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=body_y_orientations)

        if hasattr(self, "body_frame_z_visualizer"):
            z_offset_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi/2, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
            body_z_orientations = math_utils.quat_mul(self._robot.data.root_quat_w, z_offset_quat)
            self.body_frame_z_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=body_z_orientations)

        if hasattr(self, "target_direction_visualizer"):
            robot_pos_xy = self._robot.data.root_pos_w[:, :2]
            relative_pos = robot_pos_xy - self._circle_centers
            current_angle = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])
            tangent_angles = current_angle + torch.pi/2
            tangent_quats = math_utils.quat_from_euler_xyz(
                torch.zeros_like(tangent_angles),
                torch.zeros_like(tangent_angles), 
                tangent_angles
            )
            self.target_direction_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=tangent_quats)

# ---------- JIT-safe helpers ----------
@torch.jit.script
def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

# ---------- JIT-safe reward (free function; no inner defs) ----------
@torch.jit.script
def _compute_circular_trajectory(
    w_circ: float,
    w_speed: float,
    w_ang: float,
    w_ang_vel: float,
    w_constr: float,
    w_actions: float,
    w_alive: float,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    circle_centers: torch.Tensor,
    target_radius: float,
    target_speed: float,
    actions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    eps = 1e-6
    N = root_pos_w.shape[0]

    # Geometry
    pos_xy = root_pos_w[:, :2]
    rel_xy = pos_xy - circle_centers
    dist = torch.linalg.norm(rel_xy, dim=1)
    radius_err = torch.abs(dist - target_radius)
    
    # Orientation
    x = root_quat_w[:, 0]; y = root_quat_w[:, 1]; z = root_quat_w[:, 2]; w = root_quat_w[:, 3]
    siny_cosp = 2.0 * (w * z + x * y); cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    angle = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])
    tangent_yaw = wrap_to_pi(angle + 0.5 * torch.pi)
    yaw_err = wrap_to_pi(tangent_yaw - yaw)
    abs_yaw_err = torch.abs(yaw_err)

    # Velocities
    v_bx = root_lin_vel_b[:, 0]  # forward
    v_by = root_lin_vel_b[:, 1]  # sway  
    v_bz = root_lin_vel_b[:, 2]  # heave
    wz = root_ang_vel_b[:, 2]    # yaw rate

    # CRITICAL: Much stronger speed control
    speed_error = torch.abs(v_bx - target_speed)
    overspeed = torch.clamp(v_bx - target_speed, 0.0, float('inf'))
    underspeed = torch.clamp(target_speed - v_bx, 0.0, float('inf'))
    
    # Speed rewards - tight tolerance around target speed
    speed_tolerance = 0.1 * target_speed  # 10% tolerance
    rew_speed = w_speed * torch.exp(-(speed_error ** 2) / (2 * (speed_tolerance ** 2)))
    
    # MASSIVE overspeed penalty (this should dominate when overspeeding)
    pen_overspeed = -20.0 * w_speed * (overspeed ** 2)  # Massive penalty
    
    # Smaller underspeed penalty (encourage some forward motion)
    pen_underspeed = -2.0 * w_speed * (underspeed ** 2)
    
    # Position reward (radius tracking)
    radius_tolerance = 0.2 * target_radius
    rew_radius = w_circ * torch.exp(-(radius_err ** 2) / (2 * (radius_tolerance ** 2)))
    
    # Orientation reward
    yaw_tolerance = 0.2  # ~11 degrees
    rew_orientation = w_ang * torch.exp(-(abs_yaw_err ** 2) / (2 * (yaw_tolerance ** 2)))
    
    # Angular velocity penalty (anti-spinning)
    pen_angular_vel = -w_ang_vel * (torch.abs(wz) ** 2)
    
    # Constraint penalties
    pen_attitude = -w_constr * (torch.abs(root_pos_w[:, 2]) + 
                               torch.abs(torch.atan2(2*(root_quat_w[:,3]*root_quat_w[:,0] + root_quat_w[:,1]*root_quat_w[:,2]), 
                                                     1-2*(root_quat_w[:,0]**2 + root_quat_w[:,1]**2))))
    pen_sway_heave = -w_constr * (torch.abs(v_by) + torch.abs(v_bz))
    pen_actions = -w_actions * torch.sum(actions ** 2, dim=1)
    
    alive = torch.ones(N, device=root_pos_w.device) * w_alive
    
    total = (
        rew_speed + rew_radius + rew_orientation + alive +
        pen_overspeed + pen_underspeed + pen_angular_vel + 
        pen_attitude + pen_sway_heave + pen_actions
    )

    return {
        "speed_reward": rew_speed,
        "radius_reward": rew_radius, 
        "orientation_reward": rew_orientation,
        "overspeed_penalty": pen_overspeed,
        "underspeed_penalty": pen_underspeed,
        "angular_velocity_penalty": pen_angular_vel,
        "attitude_penalty": pen_attitude,
        "sway_heave_penalty": pen_sway_heave,
        "action_penalty": pen_actions,
        "alive_bonus": alive,
        "total_reward": total,
        
        # Debug info
        "forward_speed": v_bx,
        "speed_error": speed_error,
        "yaw_error": abs_yaw_err,
        "yaw_rate": wz,
        "distance_to_center": dist,
        "radius_error": radius_err,
    }
