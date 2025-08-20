"""
WarpAUV environment for IsaacLabs - Global X Movement Configuration

Author: Kevin Chang and Levi "Veevee" Cai (cail@mit.edu)
Modified for global X direction movement with stability system
"""

from __future__ import annotations

import gymnasium as gym
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
from isaaclab.utils.math import quat_apply
from isaaclab.markers import (
    CUBOID_MARKER_CFG,
    VisualizationMarkers,
    RED_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)
import isaaclab.utils.math as math_utils

##
# Hydrodynamic model
##
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import (
    DynamicsFirstOrder,
    ConversionFunctionBasic,
    get_thruster_com_and_orientations,
)


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
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=20.0, replicate_physics=True)
    debug_vis = True

    # Thruster control configuration - Only rear thrusters for forward thrust and yaw control
    active_thrusters = [False, True, False, True, False, False]  # drive_right + rear_right

    # env
    decimation = 4
    cap_episode_length = True
    episode_length_s = 120.0
    episode_length_before_reset = None

    # Action/Observation spaces - updated for global X movement task
    num_actions = 2
    num_observations = 16  # Updated count for global X movement
    num_states = 0

    action_space = gym.spaces.Box(
        low=np.array([0.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
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

    # Reward scales optimized for global X movement with stability
    rew_scale_speed: float = 10.0           # Reduced to balance with upright reward
    rew_scale_ang: float = 2.0            # Increased - upright is critical
    rew_scale_ang_vel: float = 0.1         # Very light
    rew_scale_constraint_violation: float = 0.5   # Very light
    rew_scale_actions: float = 0.001       # Very light
    rew_scale_alive: float = 10.0          # High value for staying alive

    # Target speed for global X movement
    target_body_x_speed: float = 0.5       # 1 m/s in global X direction
    
    # Visualization parameters (needed for coordinate frames)
    target_radius: float = 7.0
    target_speed: float = 0.5

    # dynamics (updated for surface operation)
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.05 / 100.0
    dyn_time_constant = 0.01
    volume = 1.252e-3
    mass = 1.248

    # Add these new scaling parameters to reduce hydrodynamic effects
    hydrodynamic_force_scale = 0.1     # Scale all hydro forces to 1% of original
    buoyancy_force_scale = 0.2           # Scale buoyancy forces to 10% of original
    drag_force_scale = 0.05             # Scale drag forces to 0.5% of original
    viscous_force_scale = 0.02        # Scale viscous forces to 0.1% of original

    # domain randomization
    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.01
        volume_range = [1.200e-3, 1.300e-3]
        mass_range = [1.200, 1.300]

    # allow aligning yaw on reset - keep as is for compatibility
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

        # Keep for compatibility with coordinate frame visualization
        self._yaw_int = torch.zeros(self.num_envs, device=self.device)
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
        self.action_smoothing = 0.5  # Strong smoothing to prevent jittery control

        # Add action rate limiting
        self.max_action_change = 0.4  # Maximum change per step

        print("=== GLOBAL X MOVEMENT CONFIGURATION ===")
        print(f"Active thrusters: {self.cfg.active_thrusters}")
        print(f"Target speed: 1.0 m/s (global X direction)")
        print("REWARD SCALES CHECK:")
        print(f"  Speed: {self.cfg.rew_scale_speed}")
        print(f"  Orientation/Stability: {self.cfg.rew_scale_ang}")
        print(f"  Angular velocity: {self.cfg.rew_scale_ang_vel}")
        print(f"  Constraint: {self.cfg.rew_scale_constraint_violation}")
        print(f"  Alive bonus: {self.cfg.rew_scale_alive}")
        print("THRUSTER CONSTRAINTS:")
        print("  Action 0 -> drive_right (index 1): FORWARD ONLY [0, 1]")
        print("  Action 1 -> rear_right (index 3): YAW CONTROL [-1, 1]")
        print()
        for i, name in enumerate(
            ["drive_left", "drive_right", "rear_left", "rear_right", "front_left", "front_right"]
        ):
            if self.cfg.active_thrusters[i]:
                pos = self.thruster_com_offsets[0, i]
                constraint = "FORWARD ONLY" if i == 1 else "BIDIRECTIONAL"
                print(
                    f"{name:12s}: x={pos[0]:+7.3f}, y={pos[1]:+7.3f}, z={pos[2]:+7.3f} [ACTIVE - {constraint}]"
                )
                
    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug:
            print("original actions vec: ", actions)

        # MUCH MORE CONSERVATIVE scaling to prevent 16 m/s speeds
        forward_raw = torch.clamp(actions[:, 0], 0, 1)
        forward_scaled = forward_raw * 1.5      # Reduced from 2.0 to 0.3 (massive reduction)

        # VERY LIGHT yaw control to prevent spinning
        yaw_raw = torch.clamp(actions[:, 1], -1, 1)
        yaw_scaled = yaw_raw * 0.05             # Reduced from 0.4 to 0.05 (8x weaker)

        # Strong smoothing to prevent jittery yaw control
        if hasattr(self, "_prev_actions"):
            current = torch.stack([forward_scaled, yaw_scaled], dim=1)
            smoothed = 0.8 * self._prev_actions + 0.2 * current  # Much more smoothing
            self._prev_actions = smoothed.clone()
            forward_final = smoothed[:, 0]
            yaw_final = smoothed[:, 1]
        else:
            self._prev_actions = torch.stack([forward_scaled, yaw_scaled], dim=1)
            forward_final = forward_scaled
            yaw_final = yaw_scaled

        # Apply to thrusters
        self._full_actions[:] = 0.0
        self._full_actions[:, 1] = forward_final  # drive_right (forward thrust)
        self._full_actions[:, 3] = yaw_final      # rear_right (yaw control)

        if self._debug:
            print(f"scaled actions - forward: {forward_final[0]:.4f}, yaw: {yaw_final[0]:.4f}")
            print("final thruster actions: ", self._full_actions)


    def _apply_action(self) -> None:
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._full_actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        """Get observations for global X direction movement task - FIXED SIGNS"""
        
        # World-frame velocities (with corrected signs)
        world_lin_vel = self._robot.data.root_lin_vel_w
        world_x_vel_corrected = -world_lin_vel[:, 0]  # FLIP SIGN: negative velocity = forward motion
        
        # Body-frame velocities (with corrected signs)
        body_lin_vel = self._robot.data.root_lin_vel_b
        body_x_vel_corrected = -body_lin_vel[:, 0]    # FLIP SIGN: negative velocity = forward motion
        body_ang_vel = self._robot.data.root_ang_vel_b
        
        # Speed error in global X direction (now using corrected velocity)
        target_speed = 1.0
        global_x_speed_error = world_x_vel_corrected - target_speed
        
        # Body orientation alignment with global X
        quat = self._robot.data.root_quat_w
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Body X direction in world frame
        body_x_in_world_x = (1 - 2 * (y**2 + z**2)).unsqueeze(1)
        
        # Attitude angles
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)).unsqueeze(1)
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0)).unsqueeze(1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).unsqueeze(1)
        
        # Position (z-coordinate for depth control)
        depth = self._robot.data.root_pos_w[:, 2:3]
        
        # Construct observation vector with corrected velocities
        obs = torch.cat([
            global_x_speed_error.unsqueeze(1),     # 1: speed error from target (corrected)
            torch.stack([world_x_vel_corrected, world_lin_vel[:, 1], world_lin_vel[:, 2]], dim=1),  # 3: world velocities (X corrected)
            torch.stack([body_x_vel_corrected, body_lin_vel[:, 1], body_lin_vel[:, 2]], dim=1),     # 3: body velocities (X corrected)
            body_ang_vel,                          # 3: body angular velocities
            body_x_in_world_x,                     # 1: alignment
            roll,                                  # 1: roll angle
            pitch,                                 # 1: pitch angle
            yaw,                                   # 1: yaw angle
            depth,                                 # 1: depth
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Updated reward calculation with stability bonuses"""
        reward_info = _compute_global_x_trajectory_with_stability(
            self.cfg.rew_scale_speed,         # w_speed
            self.cfg.rew_scale_ang,           # w_ang
            self.cfg.rew_scale_ang_vel,       # w_ang_vel
            self.cfg.rew_scale_constraint_violation,  # w_constr
            self.cfg.rew_scale_actions,       # w_actions
            self.cfg.rew_scale_alive,         # w_alive
            self._robot.data.root_pos_w,      # root_pos_w
            self._robot.data.root_quat_w,     # root_quat_w
            self._robot.data.root_lin_vel_b,  # root_lin_vel_b
            self._robot.data.root_ang_vel_b,  # root_ang_vel_b
            self._robot.data.root_lin_vel_w,  # root_lin_vel_w (world frame)
            1.0,                              # target_speed (1 m/s)
            self._actions,                    # actions
        )

        if self._debug:
            self._print_global_x_reward_debug_with_stability(reward_info)

        return reward_info["total_reward"]

    def _print_global_x_reward_debug_with_stability(self, reward_components: Dict[str, torch.Tensor], env_idx: int = 0) -> None:
        """Debug print with stability reward components"""
        i = env_idx
        try:
            print("================================")
            print(f"=== STABILITY-ENHANCED REWARD DEBUG (Env {i}) ===")
            print(f"CORRECTED Global X velocity: {reward_components['global_x_velocity'][i].item():.4f} m/s (target: 1.0 m/s)")
            print(f"Speed error: {reward_components['speed_error'][i].item():.4f} m/s")
            
            # Check which speed zone the robot is in
            error = reward_components['speed_error'][i].item()
            if error < 0.05:
                zone = "PERFECT (±5%)"
            elif error < 0.10:
                zone = "GOOD (±10%)"
            elif error < 0.15:
                zone = "ACCEPTABLE (±15%)"
            else:
                zone = "NEEDS IMPROVEMENT"
                
            print(f"Speed zone: {zone}")
            print("--- REWARD COMPONENTS ---")
            print(f"Base speed reward: {reward_components['speed_reward'][i].item():.4f}")
            print(f"NEW Stability bonus: {reward_components['stability_bonus'][i].item():.4f}")
            print(f"NEW Perfect zone bonus: {reward_components['perfect_zone_bonus'][i].item():.4f}")
            print(f"NEW Good zone bonus: {reward_components['good_zone_bonus'][i].item():.4f}")
            print(f"Forward bias: {reward_components['forward_bias'][i].item():.4f}")
            print(f"TOTAL: {reward_components['total_reward'][i].item():.4f}")
            print("================================")
        except KeyError as e:
            print(f"[DEBUG] Missing key: {e}")

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
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x)
                | (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y)
                | (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # INCREASED emergency reset threshold to be more forgiving
        quat = self._robot.data.root_quat_w
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        
        # Reset only if roll or pitch > 90 degrees (much more forgiving)
        attitude_limit = math.radians(90.0)  # Increased from 60° to 90°
        extreme_attitude = (torch.abs(roll) > attitude_limit) | (torch.abs(pitch) > attitude_limit)
        
        # Combine all reset conditions
        out_of_bounds = out_of_bounds | extreme_attitude

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset selected envs with guaranteed upright orientation."""
        device = self.device if hasattr(self, "device") else self._device
        env_ids = env_ids.to(device)

        # Default root state
        root_state = self._default_root_state[env_ids].clone()
        root_state[:, 2] = 0.0  # surface operation
        root_state[:, 7:10] = 0.0  # zero initial velocity
        root_state[:, 10:13] = 0.0  # zero initial angular velocity

        # CRITICAL: Force exactly upright orientation (no randomization initially)
        # Identity quaternion = [0, 0, 0, 1] = perfectly upright, no rotation
        root_state[:, 3] = 0.0  # qx = 0
        root_state[:, 4] = 0.0  # qy = 0  
        root_state[:, 5] = 0.0  # qz = 0
        root_state[:, 6] = 1.0  # qw = 1 (identity quaternion)

        # Start with small forward velocity in global X direction to help learning
        root_state[:, 7] = 0.0   # small initial X velocity
        root_state[:, 8] = 0.0   # no Y velocity
        root_state[:, 9] = 0.0   # no Z velocity

        self._default_root_state[env_ids] = root_state

        if hasattr(self, "_default_dof_pos") and hasattr(self, "_default_dof_vel"):
            self._dof_pos[env_ids] = self._default_dof_pos[env_ids]
            self._dof_vel[env_ids] = self._default_dof_vel[env_ids]

        if hasattr(self, "_actions"):
            self._actions[env_ids] = 0.0

        if hasattr(self, "_yaw_int"):
            self._yaw_int[env_ids] = 0.0

        if hasattr(self._robot, "write_root_state_to_sim"):
            self._robot.write_root_state_to_sim(self._default_root_state[env_ids], env_ids)
        elif hasattr(self._robot, "set_world_poses"):
            self._robot.set_world_poses(root_state[:, 0:3], root_state[:, 3:7], env_ids)

        if hasattr(self, "._robot") and hasattr(self._robot, "write_dof_state_to_sim") and hasattr(self, "_dof_pos") and hasattr(self, "_dof_vel"):
            self._robot.write_dof_state_to_sim(self._dof_pos[env_ids], self._dof_vel[env_ids], env_ids)

        if hasattr(self, "_obs_filter_state"):
            self._obs_filter_state[env_ids] = 0.0

        if hasattr(self, "_reset_goal"):
            self._reset_goal(env_ids)

    def _reset_goal(self, env_ids: Sequence[int]):
        """Reset goal ensuring perfectly upright start."""
        
        # Simple positioning - just put robot at environment origins
        env_origins = self.scene.env_origins[env_ids]
        
        self._default_root_state[env_ids, 0] = env_origins[:, 0]  # No random offset
        self._default_root_state[env_ids, 1] = env_origins[:, 1]  # No random offset  
        self._default_root_state[env_ids, 2] = 0.0  # surface operation
        
        # FORCE identity quaternion (perfectly upright)
        self._default_root_state[env_ids, 3] = 0.0  # qx = 0
        self._default_root_state[env_ids, 4] = 0.0  # qy = 0
        self._default_root_state[env_ids, 5] = 0.0  # qz = 0 
        self._default_root_state[env_ids, 6] = 1.0  # qw = 1
        
        # Small initial forward velocity
        self._default_root_state[env_ids, 7] = 0.2   # X velocity
        self._default_root_state[env_ids, 8] = 0.0   # Y velocity
        self._default_root_state[env_ids, 9] = 0.0   # Z velocity
        self._default_root_state[env_ids, 10:13] = 0.0  # No angular velocity

        if hasattr(self, "_yaw_int"):
            self._yaw_int[env_ids] = 0.0

        if self._debug and len(env_ids) > 0:
            print(f"=== FORCED UPRIGHT RESET DEBUG (Env {env_ids[0]}) ===")
            print(f"Position: ({self._default_root_state[env_ids[0], 0]:.3f}, {self._default_root_state[env_ids[0], 1]:.3f}, 0.0)")
            print("Quaternion: [0, 0, 0, 1] - PERFECTLY UPRIGHT")
            print("Initial velocity: [0.2, 0, 0] m/s in global frame")
            print("================================")

    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]

        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(
                len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius
            )
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
        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1 / 3)
        return radii * coords

    def _compute_stability_torques(self) -> torch.Tensor:
        """Compute VERY LIGHT stabilizing torques"""
        quat = self._robot.data.root_quat_w
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll and pitch angles
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        
        # Angular velocities in body frame
        roll_rate = self._robot.data.root_ang_vel_b[:, 0]
        pitch_rate = self._robot.data.root_ang_vel_b[:, 1]
        
        # MUCH LIGHTER stability gains - only for extreme situations
        stability_gain_pos = 10.0     # Reduced from 100.0 to 10.0
        stability_gain_vel = 2.0      # Reduced from 20.0 to 2.0
        
        # Only apply stabilization for angles > 15 degrees
        angle_threshold = math.radians(15.0)
        
        # Light restoring torques only when tilted significantly
        stability_torque_x = torch.where(
            torch.abs(roll) > angle_threshold,
            -stability_gain_pos * torch.sign(roll) * (torch.abs(roll) - angle_threshold) - stability_gain_vel * roll_rate,
            torch.zeros_like(roll)
        )
        
        stability_torque_y = torch.where(
            torch.abs(pitch) > angle_threshold,
            -stability_gain_pos * torch.sign(pitch) * (torch.abs(pitch) - angle_threshold) - stability_gain_vel * pitch_rate,
            torch.zeros_like(pitch)
        )
        
        stability_torque_z = torch.zeros_like(roll)  # No yaw stabilization
        
        stability_torques = torch.stack([stability_torque_x, stability_torque_y, stability_torque_z], dim=1)
        
        return stability_torques

    def _compute_dynamics(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        if self._debug:
            print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)

        masked_actions = actions * self.thruster_mask.unsqueeze(0)

        if self._debug:
            print("Original actions:", actions[0])
            print("Thruster mask:", self.thruster_mask)
            print("Masked actions:", masked_actions[0])

        motorValues = torch.clone(masked_actions)

        if self._debug:
            print("motorValues: ", motorValues)

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

        # Calculate hydrodynamic forces with SCALING
        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, abs(self._gravity_magnitude), self.com_to_cob_offsets
        )

        density_forces, density_torques, viscosity_forces, viscosity_torques = self.force_calculation_functions.calculate_density_and_viscosity_forces(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_w,
            self.inertia_tensors,
            self.inertia_tensors_mean,
            self.cfg.water_beta,
            self.cfg.water_rho,
            self.masses,
        )

        # APPLY SCALING TO REDUCE HYDRODYNAMIC FORCES
        buoyancy_forces *= self.cfg.buoyancy_force_scale
        buoyancy_torques *= self.cfg.buoyancy_force_scale
        
        density_forces *= self.cfg.drag_force_scale
        density_torques *= self.cfg.drag_force_scale
        
        viscosity_forces *= self.cfg.viscous_force_scale
        viscosity_torques *= self.cfg.viscous_force_scale

        # Remove stability torques initially to isolate the issue
        # stability_torques = self._compute_stability_torques()

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques
        # torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques + stability_torques

        forces, torques = self._limit_forces_and_torques(forces, torques)

        if self._debug:
            print("thruster forces:", torch.norm(thruster_forces[0]).item())
            print("buoyancy forces:", torch.norm(buoyancy_forces[0]).item())
            print("density forces:", torch.norm(density_forces[0]).item())
            print("viscosity forces:", torch.norm(viscosity_forces[0]).item())
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
            for attr_name in [
                "circle_visualizer",
                "global_frame_x_visualizer",
                "global_frame_y_visualizer",
                "global_frame_z_visualizer",
                "body_frame_x_visualizer",
                "body_frame_y_visualizer",
                "body_frame_z_visualizer",
                "target_direction_visualizer",
            ]:
                if hasattr(self, attr_name):
                    getattr(self, attr_name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "circle_visualizer") and hasattr(self, "_circle_centers"):
            num_points = 32
            angles = torch.linspace(0, 2 * torch.pi, num_points, device=self.device)
            circle_points = torch.zeros(self.num_envs * num_points, 3, device=self.device)
            for env_idx in range(self.num_envs):
                start_idx = env_idx * num_points
                end_idx = (env_idx + 1) * num_points
                # Use a default radius if target_radius doesn't exist
                radius = getattr(self.cfg, 'target_radius', 7.0)
                circle_points[start_idx:end_idx, 0] = self._circle_centers[env_idx, 0] + radius * torch.cos(angles)
                circle_points[start_idx:end_idx, 1] = self._circle_centers[env_idx, 1] + radius * torch.sin(angles)
                circle_points[start_idx:end_idx, 2] = 0.0
            self.circle_visualizer.visualize(translations=circle_points)

        if hasattr(self, "global_frame_x_visualizer") and hasattr(self, "_circle_centers"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            x_orientations = torch.zeros(self.num_envs, 4, device=self.device)
            x_orientations[:, 3] = 1.0
            self.global_frame_x_visualizer.visualize(translations=center_positions, orientations=x_orientations)

        if hasattr(self, "global_frame_y_visualizer") and hasattr(self, "_circle_centers"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            y_orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi / 2, device=self.device),
            )
            self.global_frame_y_visualizer.visualize(translations=center_positions, orientations=y_orientations)

        if hasattr(self, "global_frame_z_visualizer") and hasattr(self, "_circle_centers"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            z_orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi / 2, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )
            self.global_frame_z_visualizer.visualize(translations=center_positions, orientations=z_orientations)

        if hasattr(self, "body_frame_x_visualizer"):
            self.body_frame_x_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w
            )

        if hasattr(self, "body_frame_y_visualizer"):
            y_offset_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi / 2, device=self.device),
            )
            body_y_orientations = math_utils.quat_mul(self._robot.data.root_quat_w, y_offset_quat)
            self.body_frame_y_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=body_y_orientations
            )

        if hasattr(self, "body_frame_z_visualizer"):
            z_offset_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi / 2, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )
            body_z_orientations = math_utils.quat_mul(self._robot.data.root_quat_w, z_offset_quat)
            self.body_frame_z_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=body_z_orientations
            )

        if hasattr(self, "target_direction_visualizer"):
            # For global X movement, show global X direction (not body direction)
            global_x_quats = torch.zeros(self.num_envs, 4, device=self.device)
            global_x_quats[:, 3] = 1.0  # Identity quaternion pointing in global X
            self.target_direction_visualizer.visualize(
                translations=self._robot.data.root_pos_w, orientations=global_x_quats
            )
    
    def _limit_forces_and_torques(self, forces: torch.Tensor, torques: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Limit forces and torques to prevent instability"""
        max_force = 50.0   # Increased from 20.0 to 50.0
        max_torque = 10.0  # Increased from 2.0 to 10.0
        
        forces = torch.clamp(forces, -max_force, max_force)
        torques = torch.clamp(torques, -max_torque, max_torque)
        
        return forces, torques


# ---------- JIT-safe helpers ----------
@torch.jit.script
def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2.0 * torch.pi) - torch.pi


# ---------- JIT-safe reward function for global X movement ----------
@torch.jit.script
def _compute_global_x_trajectory_with_stability(
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
    root_lin_vel_w: torch.Tensor,
    target_speed: float,
    actions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    eps = 1e-6
    N = root_pos_w.shape[0]

    # CORRECTED: World-frame velocities with proper signs
    v_wx = -root_lin_vel_w[:, 0]     # FLIP SIGN: negative raw = forward motion
    v_wy = root_lin_vel_w[:, 1]      # Y unchanged
    v_wz = root_lin_vel_w[:, 2]      # Z unchanged
    
    # CORRECTED: Body-frame velocities with proper signs
    v_bx = -root_lin_vel_b[:, 0]     # FLIP SIGN: negative raw = forward motion
    
    # Angular velocities (unchanged)
    wx = root_ang_vel_b[:, 0]
    wy = root_ang_vel_b[:, 1]
    wz = root_ang_vel_b[:, 2]

    # === STEP 2: ENHANCED SPEED CONTROL WITH STABILITY ===
    speed_error = torch.abs(v_wx - target_speed)
    
    # Tighter tolerance for more precise control
    speed_tolerance = 0.15 * target_speed  # Reduced from 0.4 to 0.15
    
    # Primary speed reward (Gaussian)
    rew_speed = w_speed * torch.exp(-(speed_error ** 2) / (2 * (speed_tolerance ** 2) + eps))
    
    # NEW: Stability bonus for maintaining target speed consistently
    # This exponential function gives high rewards for very small errors
    speed_stability = torch.exp(-5.0 * speed_error)  # Very tight exponential
    stability_bonus = 0.5 * w_speed * speed_stability
    
    # NEW: Speed zone rewards (different rewards for different speed ranges)
    # Perfect zone: 0.95 - 1.05 m/s (5% tolerance)
    perfect_zone_mask = (torch.abs(speed_error) < 0.05 * target_speed)
    perfect_zone_bonus = 2.0 * w_speed * perfect_zone_mask.float()
    
    # Good zone: 0.90 - 1.10 m/s (10% tolerance) 
    good_zone_mask = (torch.abs(speed_error) < 0.10 * target_speed) & (~perfect_zone_mask)
    good_zone_bonus = 1.0 * w_speed * good_zone_mask.float()
    
    # Reduced forward bias since target is being achieved
    forward_bias = 1.0 * w_speed * torch.clamp(v_wx / target_speed, 0.0, 1.2)  # Reduced from 2.0
    
    # Speed penalties (unchanged)
    underspeed = torch.clamp(target_speed - v_wx, 0.0, float("inf"))
    pen_underspeed = -0.05 * w_speed * (underspeed ** 2)

    overspeed = torch.clamp(v_wx - 1.5 * target_speed, 0.0, float("inf"))
    pen_overspeed = -0.5 * w_speed * (overspeed ** 2)

    # Upright reward (unchanged)
    x = root_quat_w[:, 0]
    y = root_quat_w[:, 1] 
    z = root_quat_w[:, 2]
    w = root_quat_w[:, 3]
    
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    
    upright_reward = 2.0 * w_ang * (torch.cos(roll) + torch.cos(pitch))
    
    # Lateral motion penalty (unchanged)
    pen_lateral_motion = -0.2 * w_constr * (torch.abs(v_wy) + torch.abs(v_wz))
    
    # Other penalties (unchanged)
    pen_angular_vel = -0.01 * w_ang_vel * (torch.abs(wx) + torch.abs(wy) + torch.abs(wz))
    pen_actions = -0.0005 * w_actions * torch.sum(actions ** 2, dim=1)

    attitude_tolerance = 1.5
    extreme_roll_penalty = -5.0 * w_ang * torch.clamp(torch.abs(roll) - attitude_tolerance, 0.0, float("inf"))
    extreme_pitch_penalty = -5.0 * w_ang * torch.clamp(torch.abs(pitch) - attitude_tolerance, 0.0, float("inf"))
    pen_attitude = extreme_roll_penalty + extreme_pitch_penalty

    alive = torch.ones(N, device=root_pos_w.device) * w_alive

    # UPDATED: Total reward with new stability components
    total = (
        rew_speed +
        stability_bonus +        # NEW: Stability reward
        perfect_zone_bonus +     # NEW: Perfect zone bonus
        good_zone_bonus +        # NEW: Good zone bonus
        forward_bias +           # REDUCED: Lower forward bias
        upright_reward +
        alive +
        pen_overspeed +
        pen_underspeed +
        pen_lateral_motion +
        pen_angular_vel +
        pen_attitude +
        pen_actions
    )

    return {
        "speed_reward": rew_speed,
        "stability_bonus": stability_bonus,      # NEW
        "perfect_zone_bonus": perfect_zone_bonus, # NEW
        "good_zone_bonus": good_zone_bonus,      # NEW
        "forward_bias": forward_bias,
        "upright_reward": upright_reward,
        "overspeed_penalty": pen_overspeed,
        "underspeed_penalty": pen_underspeed,
        "lateral_motion_penalty": pen_lateral_motion,
        "angular_velocity_penalty": pen_angular_vel,
        "attitude_penalty": pen_attitude,
        "action_penalty": pen_actions,
        "alive_bonus": alive,
        "total_reward": total,
        # Debug info with corrected signs
        "global_x_velocity": v_wx,
        "global_y_velocity": v_wy,
        "global_z_velocity": v_wz,
        "body_x_velocity": v_bx,
        "speed_error": speed_error,
        "roll": roll,
        "pitch": pitch,
    }
