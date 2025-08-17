"""
WarpAUV environment for IsaacLabs - Circular Trajectory Configuration

Author: Kevin Chang and Levi "Veevee" Cai (cail@mit.edu)
Modified for circular trajectory following
"""

from __future__ import annotations

import gymnasium as gym
import random
import math
import torch
import numpy as np
from collections.abc import Sequence

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

    def __init__(self, env: WarpAUVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = WarpAUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # Use the original USD version
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=20.0, replicate_physics=True)  # Increased spacing for individual circles
    debug_vis = True

    # NEW: Circular trajectory parameters - each robot gets its own circle
    target_radius: float = 7.0        # Target radius in meters
    target_speed: float = 1.0         # Target speed in m/s

    # Thruster control configuration - Only rear thrusters for forward thrust and yaw control
    active_thrusters = [False, True, False, True, False, False]  # Only drive_right and rear_right active

    # env
    decimation = 2
    cap_episode_length = True
    episode_length_s = 30.0  # Longer episodes for circular trajectory
    episode_length_before_reset = None
    
    # Define action and observation spaces - reduced to 2 actions for 2 active thrusters
    num_actions = 2  # Only 2 active thrusters
    num_observations = 20  # Increased for circular trajectory state
    num_states = 0
    
    # Action space: Box space for 2 thruster commands
    # Action 0: drive_right thruster (forward only, 0 to 1)
    # Action 1: rear_right thruster (yaw control, -1 to 1)
    action_space = gym.spaces.Box(
        low=np.array([0.0, -1.0]), 
        high=np.array([1.0, 1.0]), 
        shape=(2,), 
        dtype=np.float32
    )
    
    # Observation space: Box space for observations
    observation_space = gym.spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(20,), 
        dtype=np.float32
    )
    
    # State space (empty for this environment)
    state_space = gym.spaces.Box(low=0, high=0, shape=(0,), dtype=np.float32)
    
    use_boundaries = True
    max_auv_x = 15  # Increased for 7m radius circle
    max_auv_y = 15
    max_auv_z = 2   # Small tolerance around surface
    starting_depth = 0  # Start at surface
    min_goal_steps = 100
    goal_completion_radius = 0.5
    goal_dims = 4
    eval_mode = False

    goal_spawn_radius = 7.0  # Match target radius
    init_guidance_rate = 0.8  # Higher guidance rate for circular motion
    init_vel_max = 1.0

    # NEW: Reward scales for circular trajectory
    rew_scale_circular_pos: float = 2.0      # Weight for circular position tracking
    rew_scale_speed: float = 1.5             # Weight for speed tracking
    rew_scale_constraint_violation: float = 5.0  # Weight for constraint violations
    rew_scale_completion: float = 100.0      # Completion bonus

    # UPDATED: Existing reward scales
    rew_scale_pos: float = 0.0               # Disable original position reward
    rew_scale_ang: float = 1.0               # Keep for tangent orientation
    rew_scale_vel: float = 0.0
    rew_scale_ang_vel: float = 0.0
    rew_scale_lin_vel: float = 0.0           # Handled by speed tracking
    rew_scale_actions: float = 0.1           # Reduce for energy efficiency
    rew_scale_terminated: float = 0.0
    rew_scale_alive: float = 0.1

    # dynamics (updated for surface operation)
    com_to_cob_offset = [0.0, 0.0, 0.01] # in meters, add this (xyz) to COM to get COB location
    water_rho = 997.0 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    rotor_constant = 0.1 / 100.0 # rotor constant used in Gazebo
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor 
    volume = 1.252e-3 # assuming cubic meters - NEUTRALLY BOUYANT
    mass = 1.248 # kg

    # domain randomization (reduced for more consistent circular motion)
    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.01 # Reduced randomization
        volume_range = [1.200e-3, 1.300e-3] # Tighter range around neutral buoyancy
        mass_range = [1.200, 1.300] # Tighter mass range

class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode
        self._debug = True  # Enable debug output for reward components

        # Initialize buffers with updated action size
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)  # Only 2 actions now
        self._full_actions = torch.zeros(self.num_envs, 6, device=self.device)  # For thruster dynamics
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins
        self._step_count = 0
        
        # Initialize circle centers - each environment gets its own circle at its origin
        # Circle centers are at the environment origins (scene.env_origins)
        self._circle_centers = self.scene.env_origins[:, :2].clone()  # Use each env's origin as circle center
        
        # Get thruster configurations
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        # Thruster mask for active thrusters
        self.thruster_mask = torch.tensor(self.cfg.active_thrusters, device=self.device, dtype=torch.float)
        self.active_thruster_indices = torch.where(self.thruster_mask > 0)[0]

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        # Get specific information about the AUV
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # Inertia tensors
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.inertia_tensors[:, 0] = 7.60e-4
        self.inertia_tensors[:, 1] = 1.04e-1
        self.inertia_tensors[:, 2] = 1.04e-1

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        # COM to COB offsets
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        # Initialize dynamics calculators
        self._init_thruster_dynamics()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)

        print("=== CIRCULAR TRAJECTORY THRUSTER CONFIGURATION ===")
        print(f"Active thrusters: {self.cfg.active_thrusters}")
        print(f"Target radius: {self.cfg.target_radius}m")
        print(f"Target speed: {self.cfg.target_speed} m/s")
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

        # get force calculation functions and rotor dynamics models
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
        
        # Don't clip the first action (drive_right) to maintain [0,1] range
        # Only clip the second action (rear_right) to [-1,1] range  
        self._actions[:, 0] = torch.clamp(actions[:, 0], 0, 1)  # drive_right: forward only
        self._actions[:, 1] = torch.clamp(actions[:, 1], -1, 1)  # rear_right: bidirectional
        self._actions = self._actions.to(self.device)
        
        # Map 2 actions to 6 thrusters using the mask
        self._full_actions[:] = 0.0
        active_count = 0
        for i, is_active in enumerate(self.cfg.active_thrusters):
            if is_active:
                self._full_actions[:, i] = self._actions[:, active_count]
                if self._debug and i == 1: 
                    print(f"drive_right (index {i}) forward-only: {self._full_actions[:, i]}")
                elif self._debug:
                    print(f"thruster {i} bidirectional: {self._full_actions[:, i]}")
                active_count += 1

        if self._debug: print("mapped full actions: ", self._full_actions)

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._full_actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        # Calculate circular trajectory specific observations
        robot_pos_xy = self._robot.data.root_pos_w[:, :2]
        distance_to_center = torch.norm(robot_pos_xy - self._circle_centers, dim=1, keepdim=True)
        radius_error = distance_to_center - self.cfg.target_radius
        
        # Current position relative to each robot's circle center
        relative_pos = robot_pos_xy - self._circle_centers
        
        # Current angle on circle
        current_angle = torch.atan2(relative_pos[:, 1], relative_pos[:, 0]).unsqueeze(1)
        
        # Desired tangent direction
        desired_tangent = torch.cat([
            -torch.sin(current_angle.squeeze(1)).unsqueeze(1), 
            torch.cos(current_angle.squeeze(1)).unsqueeze(1)
        ], dim=1)
        
        # Current forward direction in world frame
        forward_dir_w = quat_apply(self._robot.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        current_forward_xy = forward_dir_w[:, :2]
        
        # Forward speed
        forward_speed = self._robot.data.root_lin_vel_b[:, 0:1]
        speed_error = forward_speed - self.cfg.target_speed
        
        # Extract roll and pitch
        roll = torch.atan2(
            2 * (self._robot.data.root_quat_w[:, 3] * self._robot.data.root_quat_w[:, 0] + self._robot.data.root_quat_w[:, 1] * self._robot.data.root_quat_w[:, 2]),
            1 - 2 * (self._robot.data.root_quat_w[:, 0]**2 + self._robot.data.root_quat_w[:, 1]**2)
        ).unsqueeze(1)
        
        pitch = torch.asin(2 * (self._robot.data.root_quat_w[:, 3] * self._robot.data.root_quat_w[:, 1] - self._robot.data.root_quat_w[:, 2] * self._robot.data.root_quat_w[:, 0])).unsqueeze(1)

        obs = torch.cat([
            radius_error,                           # 1: Distance error from target radius
            speed_error,                            # 1: Speed error from target
            current_angle,                          # 1: Current angle on circle
            desired_tangent,                        # 2: Desired tangent direction (x, y)
            current_forward_xy,                     # 2: Current forward direction (x, y)
            self._robot.data.root_pos_w[:, 2:3],   # 1: Z position (should be 0)
            roll,                                   # 1: Roll (should be 0)
            pitch,                                  # 1: Pitch (should be 0)
            self._robot.data.root_lin_vel_b,       # 3: Body linear velocities
            self._robot.data.root_ang_vel_b,       # 3: Body angular velocities
            self._robot.data.root_quat_w,          # 4: Quaternion orientation
        ], dim=-1)
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Compute rewards
        reward_components = _compute_circular_trajectory_rewards(
            self.cfg.rew_scale_circular_pos,
            self.cfg.rew_scale_speed,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_constraint_violation,
            self.cfg.rew_scale_actions,
            self.cfg.rew_scale_alive,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._circle_centers,
            self.cfg.target_radius,
            self.cfg.target_speed,
            self._actions
        )

        # Debug printing (outside JIT function)
        if self._debug:
            self._print_reward_debug(reward_components)

        return reward_components["total_reward"]
    
    def _print_reward_debug(self, reward_components: dict):
        """Print detailed reward component information for debugging"""
        env_idx = 0  # Print for first environment
        
        print("=== REWARD COMPONENTS (Env 0) ===")
        print(f"Distance to center: {reward_components['distance_to_center'][env_idx].item():.4f}m (target: {self.cfg.target_radius}m)")
        print(f"Radius error: {reward_components['radius_error'][env_idx].item():.4f}m")
        print(f"Forward speed: {reward_components['forward_speed'][env_idx].item():.4f} m/s (target: {self.cfg.target_speed} m/s)")
        print(f"Speed error: {reward_components['speed_error'][env_idx].item():.4f} m/s")
        print(f"Yaw error: {reward_components['yaw_error'][env_idx].item():.4f} rad ({reward_components['yaw_error'][env_idx].item() * 180 / 3.14159:.1f}°)")
        print(f"Z position: {reward_components['z_position'][env_idx].item():.4f}m (should be 0)")
        print(f"Roll: {reward_components['roll'][env_idx].item():.4f} rad ({reward_components['roll'][env_idx].item() * 180 / 3.14159:.1f}°)")
        print(f"Pitch: {reward_components['pitch'][env_idx].item():.4f} rad ({reward_components['pitch'][env_idx].item() * 180 / 3.14159:.1f}°)")
        print(f"Sway velocity: {reward_components['sway_velocity'][env_idx].item():.4f} m/s")
        print(f"Heave velocity: {reward_components['heave_velocity'][env_idx].item():.4f} m/s")
        print(f"Roll rate: {reward_components['roll_rate'][env_idx].item():.4f} rad/s")
        print(f"Pitch rate: {reward_components['pitch_rate'][env_idx].item():.4f} rad/s")
        print("--- REWARD COMPONENTS ---")
        print(f"Circular position: {reward_components['rew_circular_pos'][env_idx].item():.4f} (scale: {self.cfg.rew_scale_circular_pos})")
        print(f"Speed tracking: {reward_components['rew_speed'][env_idx].item():.4f} (scale: {self.cfg.rew_scale_speed})")
        print(f"Orientation: {reward_components['rew_tangent_orientation'][env_idx].item():.4f} (scale: {self.cfg.rew_scale_ang})")
        print(f"Surface constraint: {reward_components['rew_surface_constraint'][env_idx].item():.4f}")
        print(f"Attitude constraint: {reward_components['rew_attitude_constraint'][env_idx].item():.4f}")
        print(f"Sway/heave constraint: {reward_components['rew_sway_heave_constraint'][env_idx].item():.4f}")
        print(f"Angular constraint: {reward_components['rew_angular_constraint'][env_idx].item():.4f}")
        print(f"Action penalty: {reward_components['rew_action'][env_idx].item():.4f}")
        print(f"Alive bonus: {self.cfg.rew_scale_alive}")
        print(f"TOTAL REWARD: {reward_components['total_reward'][env_idx].item():.4f}")
        print("================================")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs)

        self._step_count = self._step_count + 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        # Update circle centers to match environment origins
        self._circle_centers[env_ids] = self.scene.env_origins[env_ids, :2]

        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        self._step_count = 0
        
        # Apply domain randomization
        self._reset_domain(env_ids)

        # Reset goals with circular placement
        self._reset_goal(env_ids)

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)

    def _reset_goal(self, env_ids: Sequence[int]):
        """Place robot at random point on each robot's individual circle with tangential orientation"""
        # Random angles on circle
        angles = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        
        # Position on each robot's individual circle at surface (z=0)
        circle_x = self._circle_centers[env_ids, 0] + self.cfg.target_radius * torch.cos(angles)
        circle_y = self._circle_centers[env_ids, 1] + self.cfg.target_radius * torch.sin(angles)
        
        self._default_root_state[env_ids, 0] = circle_x
        self._default_root_state[env_ids, 1] = circle_y
        self._default_root_state[env_ids, 2] = 0.0  # At surface
        
        # Tangential orientation (perpendicular to radius)
        tangent_angles = angles + torch.pi/2
        self._default_root_state[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
            torch.zeros_like(tangent_angles),  # Roll = 0
            torch.zeros_like(tangent_angles),  # Pitch = 0
            tangent_angles                     # Yaw = tangent direction
        )
        
        # Initial velocity (forward at target speed)
        self._default_root_state[env_ids, 7] = self.cfg.target_speed  # Forward velocity
        self._default_root_state[env_ids, 8] = 0.0  # No sway
        self._default_root_state[env_ids, 9] = 0.0  # No heave
        self._default_root_state[env_ids, 10:13] = 0.0  # No initial angular velocities

    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]

        # Randomize COM to COB offset
        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)

        # Randomize volume
        if self.cfg.domain_randomization.use_custom_randomization:
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
        """ Compute dynamics from actions for circular trajectory following """
        
        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)

        # Apply thruster mask
        masked_actions = actions * self.thruster_mask.unsqueeze(0)
        
        if self._debug: 
            print("Original actions:", actions[0])
            print("Thruster mask:", self.thruster_mask)
            print("Masked actions:", masked_actions[0])
        
        motorValues = torch.clone(masked_actions)

        if self._debug: print("motorValues: ", motorValues)

        # Convert PWM commands to rad/s
        motorValues[torch.abs(motorValues) < 0.08] = 0 
        motorValues[motorValues >= 0.08] = -139.0 * (torch.pow(motorValues[motorValues >= 0.08], 2.0)) + 500 * motorValues[motorValues >= 0.08] + 8.28
        motorValues[motorValues <= -0.08] = 161.0 * (torch.pow(motorValues[motorValues <= -0.08], 2.0)) + 517.86 * motorValues[motorValues <= -0.08] - 5.72

        # Get current motor velocities using thruster dynamics
        motorValues = self.thruster_dynamics.update(motorValues, self.episode_length_buf * self.sim.cfg.dt)

        # Get thruster forces from their speeds
        motorValues = self.thruster_conversion.convert(motorValues)

        # Calculate thruster forces and torques
        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * motorValues.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        # Sum forces and torques
        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)

        ## Calculate hydrodynamics
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
        # Create circle visualization markers
        if debug_vis:
            if not hasattr(self, "circle_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/circle_path"
                self.circle_visualizer = VisualizationMarkers(marker_cfg)

            # Global coordinate frames at circle centers (X=red, Y=green, Z=blue)
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

            # Robot body coordinate frames (X=red, Y=green, Z=blue)
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
                # Create purple arrow for target direction
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()  # Start with green as base
                marker_cfg.prim_path = "/Visuals/Command/target_direction"
                marker_cfg.markers["arrow"].scale = (0.25, 0.25, 2)
                # Override color to purple
                marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.8, 0.2, 0.8)  # Purple color (R=0.8, G=0.2, B=0.8)
                self.target_direction_visualizer = VisualizationMarkers(marker_cfg)
            
            # Set visibility
            self.circle_visualizer.set_visibility(True)
            self.global_frame_x_visualizer.set_visibility(True)
            self.global_frame_y_visualizer.set_visibility(True)
            self.global_frame_z_visualizer.set_visibility(True)
            self.body_frame_x_visualizer.set_visibility(True)
            self.body_frame_y_visualizer.set_visibility(True)
            self.body_frame_z_visualizer.set_visibility(True)
            self.target_direction_visualizer.set_visibility(True)
        else:
            # Hide all visualizers
            for attr_name in ["circle_visualizer", "global_frame_x_visualizer", "global_frame_y_visualizer", 
                             "global_frame_z_visualizer", "body_frame_x_visualizer", "body_frame_y_visualizer", 
                             "body_frame_z_visualizer", "target_direction_visualizer"]:
                if hasattr(self, attr_name):
                    getattr(self, attr_name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "circle_visualizer"):
            # Visualize individual circle paths for each environment
            num_points = 32
            angles = torch.linspace(0, 2*torch.pi, num_points, device=self.device)
            circle_points = torch.zeros(self.num_envs * num_points, 3, device=self.device)
            
            for env_idx in range(self.num_envs):
                start_idx = env_idx * num_points
                end_idx = (env_idx + 1) * num_points
                # Each environment gets its own circle centered at its origin
                circle_points[start_idx:end_idx, 0] = self._circle_centers[env_idx, 0] + self.cfg.target_radius * torch.cos(angles)
                circle_points[start_idx:end_idx, 1] = self._circle_centers[env_idx, 1] + self.cfg.target_radius * torch.sin(angles)
                circle_points[start_idx:end_idx, 2] = 0.0  # At surface
            
            self.circle_visualizer.visualize(translations=circle_points)

        # Visualize Global Coordinate Frames at Circle Centers (X=red, Y=green, Z=blue)
        if hasattr(self, "global_frame_x_visualizer"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0  # At surface
            
            # X-axis (red) - identity orientation (points in +X direction)
            x_orientations = torch.zeros(self.num_envs, 4, device=self.device)
            x_orientations[:, 3] = 1.0  # w=1 for identity quaternion
            self.global_frame_x_visualizer.visualize(
                translations=center_positions,
                orientations=x_orientations
            )

        if hasattr(self, "global_frame_y_visualizer"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            
            # Y-axis (green) - 90 degree rotation around Z to point in +Y direction
            y_orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi/2, device=self.device)
            )
            self.global_frame_y_visualizer.visualize(
                translations=center_positions,
                orientations=y_orientations
            )

        if hasattr(self, "global_frame_z_visualizer"):
            center_positions = torch.zeros(self.num_envs, 3, device=self.device)
            center_positions[:, :2] = self._circle_centers
            center_positions[:, 2] = 0.0
            
            # Z-axis (blue) - 90 degree rotation around Y to point in +Z direction
            z_orientations = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi/2, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
            self.global_frame_z_visualizer.visualize(
                translations=center_positions,
                orientations=z_orientations
            )

        # Visualize Robot Body Coordinate Frames (X=red, Y=green, Z=blue)
        if hasattr(self, "body_frame_x_visualizer"):
            # X-axis (red) - robot's forward direction
            self.body_frame_x_visualizer.visualize(
                translations=self._robot.data.root_pos_w,
                orientations=self._robot.data.root_quat_w
            )

        if hasattr(self, "body_frame_y_visualizer"):
            # Y-axis (green) - robot's left direction (90 deg rotation around Z from X)
            y_offset_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), torch.pi/2, device=self.device)
            )
            body_y_orientations = math_utils.quat_mul(self._robot.data.root_quat_w, y_offset_quat)
            self.body_frame_y_visualizer.visualize(
                translations=self._robot.data.root_pos_w,
                orientations=body_y_orientations
            )

        if hasattr(self, "body_frame_z_visualizer"):
            # Z-axis (blue) - robot's down direction (90 deg rotation around Y from X)
            z_offset_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(self.num_envs, device=self.device),
                torch.full((self.num_envs,), -torch.pi/2, device=self.device),
                torch.zeros(self.num_envs, device=self.device)
            )
            body_z_orientations = math_utils.quat_mul(self._robot.data.root_quat_w, z_offset_quat)
            self.body_frame_z_visualizer.visualize(
                translations=self._robot.data.root_pos_w,
                orientations=body_z_orientations
            )

        if hasattr(self, "target_direction_visualizer"):
            # Visualize target tangent direction for each robot (purple arrows)
            robot_pos_xy = self._robot.data.root_pos_w[:, :2]
            relative_pos = robot_pos_xy - self._circle_centers
            current_angle = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])
            
            # Tangent direction quaternion
            tangent_angles = current_angle + torch.pi/2
            tangent_quats = math_utils.quat_from_euler_xyz(
                torch.zeros_like(tangent_angles),
                torch.zeros_like(tangent_angles), 
                tangent_angles
            )
            
            self.target_direction_visualizer.visualize(
                translations=self._robot.data.root_pos_w, 
                orientations=tangent_quats
            )


@torch.jit.script
def _compute_circular_trajectory_rewards(
    rew_scale_circular_pos: float,
    rew_scale_speed: float,
    rew_scale_ang: float,
    rew_scale_constraint_violation: float,
    rew_scale_actions: float,
    rew_scale_alive: float,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    circle_centers: torch.Tensor,
    target_radius: float,
    target_speed: float,
    actions: torch.Tensor,
):
    """Reward function for circular trajectory following"""
    
    # 1. Circular Position Reward
    robot_pos_xy = root_pos[:, :2]
    distance_to_center = torch.norm(robot_pos_xy - circle_centers, dim=1)
    radius_error = torch.abs(distance_to_center - target_radius)
    rew_circular_pos = rew_scale_circular_pos * torch.exp(-radius_error**2 / (0.5**2))
    
    # 2. Speed Tracking Reward
    forward_speed = lin_vel_b[:, 0]
    speed_error = torch.abs(forward_speed - target_speed)
    rew_speed = rew_scale_speed * torch.exp(-speed_error**2 / (0.2**2))
    
    # 3. Tangential Direction Reward
    robot_to_center = circle_centers - robot_pos_xy
    desired_tangent_angle = torch.atan2(robot_to_center[:, 0], -robot_to_center[:, 1])
    
    current_yaw = torch.atan2(
        2 * (root_quat[:, 3] * root_quat[:, 2] + root_quat[:, 0] * root_quat[:, 1]),
        1 - 2 * (root_quat[:, 1]**2 + root_quat[:, 2]**2)
    )
    
    yaw_error = torch.abs(torch.atan2(torch.sin(desired_tangent_angle - current_yaw), 
                                      torch.cos(desired_tangent_angle - current_yaw)))
    rew_tangent_orientation = rew_scale_ang * torch.exp(-yaw_error**2 / (0.2**2))
    
    # 4. Surface Constraint (z = 0)
    z_position = root_pos[:, 2]
    surface_violation = torch.abs(z_position)
    rew_surface_constraint = -rew_scale_constraint_violation * surface_violation**2
    
    # 5. Roll and Pitch Constraints
    roll = torch.atan2(
        2 * (root_quat[:, 3] * root_quat[:, 0] + root_quat[:, 1] * root_quat[:, 2]),
        1 - 2 * (root_quat[:, 0]**2 + root_quat[:, 1]**2)
    )
    pitch = torch.asin(torch.clamp(2 * (root_quat[:, 3] * root_quat[:, 1] - root_quat[:, 2] * root_quat[:, 0]), -1, 1))
    
    roll_pitch_violation = torch.abs(roll) + torch.abs(pitch)
    rew_attitude_constraint = -rew_scale_constraint_violation * roll_pitch_violation**2
    
    # 6. Sway and Heave Velocity Constraints
    sway_velocity = torch.abs(lin_vel_b[:, 1])
    heave_velocity = torch.abs(lin_vel_b[:, 2])
    rew_sway_heave_constraint = -rew_scale_constraint_violation * (sway_velocity**2 + heave_velocity**2)
    
    # 7. Angular Velocity Constraints (only yaw rate allowed)
    roll_rate_violation = torch.abs(ang_vel_b[:, 0])
    pitch_rate_violation = torch.abs(ang_vel_b[:, 1])
    rew_angular_constraint = -rew_scale_constraint_violation * (roll_rate_violation**2 + pitch_rate_violation**2)
    
    # 8. Action Smoothness
    rew_action = -rew_scale_actions * torch.norm(actions, dim=1)**2
    
    # 9. Alive bonus
    rew_alive = rew_scale_alive
    
    # Total reward
    total_rew = (
        rew_circular_pos +
        rew_speed +
        rew_tangent_orientation +
        rew_surface_constraint +
        rew_attitude_constraint +
        rew_sway_heave_constraint +
        rew_angular_constraint +
        rew_action +
        rew_alive
    )
    
    # Return components as a dictionary-like structure (using named tuple approach)
    return {
        "total_reward": total_rew,
        "distance_to_center": distance_to_center,
        "radius_error": radius_error,
        "forward_speed": forward_speed,
        "speed_error": speed_error,
        "yaw_error": yaw_error,
        "z_position": z_position,
        "roll": roll,
        "pitch": pitch,
        "sway_velocity": sway_velocity,
        "heave_velocity": heave_velocity,
        "roll_rate": roll_rate_violation,
        "pitch_rate": pitch_rate_violation,
        "rew_circular_pos": rew_circular_pos,
        "rew_speed": rew_speed,
        "rew_tangent_orientation": rew_tangent_orientation,
        "rew_surface_constraint": rew_surface_constraint,
        "rew_attitude_constraint": rew_attitude_constraint,
        "rew_sway_heave_constraint": rew_sway_heave_constraint,
        "rew_angular_constraint": rew_angular_constraint,
        "rew_action": rew_action,
    }