"""
Surface Vehicle environment for IsaacLabs - GPS-based navigation

Author: Modified from WarpAUV for surface vehicle control
"""

from __future__ import annotations

import random
import math
import torch
from collections.abc import Sequence

# Import the surface vehicle configuration
from .assets.surface_vehicle import SURFACE_VEHICLE_CFG

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
import gymnasium as gym
import numpy as np

##
# Surface vehicle dynamics
##
from .surface_vehicle_dynamics import SurfaceVehicleDynamics
from .surface_hydrodynamics import SurfaceVehicleHydrodynamics

class SurfaceVehicleEnvWindow(BaseEnvWindow):
    """Window manager for the surface vehicle environment."""

    def __init__(self, env: SurfaceVehicleEnv, window_name: str = "IsaacLab"):
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
class SurfaceVehicleEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = SurfaceVehicleEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 60)  # 60Hz for surface vehicle

    # robot - use surface vehicle configuration
    robot_cfg: RigidObjectCfg = SURFACE_VEHICLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=50.0, replicate_physics=True)
    debug_vis = True

    # GPS and navigation parameters
    gps_noise_std = 0.1  # meters (realistic GPS accuracy)
    compass_noise_std = 0.05  # radians (~3 degrees)
    
    # Goal generation
    goal_spawn_radius_min = 15.0   # Minimum distance to goal (meters)
    goal_spawn_radius_max = 20.0  # Maximum distance to goal (meters)
    
    # Observation space: [goal_x_body, goal_y_body, vehicle_heading, velocity_x, velocity_y, yaw_rate]
    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
    num_observations = 6
    
    # Action space: [thrust_command (0 to 1), rudder_angle (-1 to 1)]
    action_space: gym.spaces.Space = gym.spaces.Box(
        low=np.array([0.0, -1.0]), 
        high=np.array([1.0, 1.0]), 
        shape=(2,), 
        dtype=np.float64
    )
    num_actions = 2
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

    # env
    decimation = 2
    cap_episode_length = True
    episode_length_s = 50.0  # Longer episodes for navigation
    episode_length_before_reset = None
    num_states = 0
    use_boundaries = True
    max_vehicle_x = 40
    max_vehicle_y = 40
    max_vehicle_z = 5  # Keep near surface
    starting_height = 0.0  # Surface level
    eval_mode = False

    # rewards - REBALANCED to discourage spinning
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 100.0

    rew_scale_goal = 3.0        # Increased: Primary objective
    rew_scale_heading = 2.0     # INCREASED: Critical for straight-line navigation
    rew_scale_forward = 0.2     # REDUCED: Don't overprioritize speed
    rew_scale_actions = 0.02    # Slightly increased: Penalize excessive control

    # Surface vehicle dynamics - BALANCED for controlled movement
    surface_drag_linear = 1.5   # Moderate drag to prevent spinning
    surface_drag_quad = 3.0     # Moderate drag to control speed
    yaw_damping = 5.0           # Higher damping to resist spinning
    time_constant = 0.1


class SurfaceVehicleEnv(DirectRLEnv):
    cfg: SurfaceVehicleEnvCfg

    def __init__(self, cfg: SurfaceVehicleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)  # [thrust, rudder]
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Initialize GPS and compass noise
        self._gps_noise_std = cfg.gps_noise_std
        self._compass_noise_std = cfg.compass_noise_std
        
        # Goal positions (global coordinates)
        self._goal_positions_global = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Vehicle origin positions (for relative coordinate system)
        self._vehicle_origins = torch.zeros(self.num_envs, 2, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)

        # Speed monitoring
        self._speed_history = torch.zeros(self.num_envs, device=self.device)
        self._speed_violations = torch.zeros(self.num_envs, device=self.device)

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize surface vehicle dynamics
        self._init_surface_dynamics()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)

    def _init_surface_dynamics(self):
        """Initialize surface vehicle dynamics and hydrodynamics"""
        self.surface_dynamics = SurfaceVehicleDynamics(self.num_envs, self.device, self.cfg.time_constant)
        self.surface_hydro = SurfaceVehicleHydrodynamics(self.num_envs, self.device)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_height))
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

        self._actions[:] = actions
        
        # CONSTRAINT: Only allow forward thrust (no reverse)
        self._actions[:, 0] = torch.clamp(self._actions[:, 0], 0.0, 1.0)  # Thrust: 0 to +1 only
        self._actions[:, 1] = torch.clamp(self._actions[:, 1], -1.0, 1.0)  # Rudder: -1 to +1 (normal)
        
        # CRITICAL: Minimum thrust to overcome static friction (your robot needs ≥0.5 m/s)
        # If thrust command is small but non-zero, boost it to minimum effective level
        small_thrust = (self._actions[:, 0] > 0.05) & (self._actions[:, 0] < 0.3)
        self._actions[small_thrust, 0] = 0.4  # Minimum effective thrust
        
        # OPTIONAL: Limit max thrust to prevent speed spikes (more realistic)
        self._actions[:, 0] = torch.clamp(self._actions[:, 0], 0.0, 0.8)  # Max 80% thrust
        
        self._actions = self._actions.to(self.device)

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _simulate_gps_reading(self, true_position: torch.Tensor) -> torch.Tensor:
        """Simulate GPS sensor with noise"""
        gps_noise = torch.randn_like(true_position) * self._gps_noise_std
        return true_position + gps_noise
    
    def _simulate_compass_reading(self, true_heading: torch.Tensor) -> torch.Tensor:
        """Simulate compass/IMU heading with noise"""
        compass_noise = torch.randn_like(true_heading) * self._compass_noise_std
        return true_heading + compass_noise
    
    def _get_vehicle_heading(self) -> torch.Tensor:
        """Extract heading (yaw) from vehicle quaternion"""
        # Convert quaternion to euler angles and extract yaw
        euler_angles = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        return euler_angles[2]  # Yaw angle
    
    def _global_to_body_frame(self, global_vector: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """Convert global coordinates to vehicle body frame"""
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        
        # Rotation matrix application
        body_x = global_vector[:, 0] * cos_h + global_vector[:, 1] * sin_h
        body_y = -global_vector[:, 0] * sin_h + global_vector[:, 1] * cos_h
        
        return torch.stack([body_x, body_y], dim=-1)

    def _get_observations(self) -> dict:
        """Generate observations using simulated GPS and IMU data"""
        
        # Get true vehicle state
        true_position = self._robot.data.root_pos_w[:, :2]  # [x, y] global position
        true_heading = self._get_vehicle_heading()
        
        # Simulate GPS reading (with noise)
        gps_position = self._simulate_gps_reading(true_position)
        
        # Simulate compass reading (with noise) 
        compass_heading = self._simulate_compass_reading(true_heading)
        
        # Calculate goal relative to current position (in global frame)
        goal_vector_global = self._goal_positions_global - gps_position
        
        # Convert goal vector to body frame using compass heading
        goal_vector_body = self._global_to_body_frame(goal_vector_global, compass_heading)
        
        # Get vehicle velocities (these come from IMU integration in real system)
        vehicle_velocity_body = self._robot.data.root_lin_vel_b[:, :2]  # [surge, sway]
        vehicle_yaw_rate = self._robot.data.root_ang_vel_b[:, 2]        # yaw rate
        
        # Construct observation vector
        obs = torch.cat([
            goal_vector_body,           # [2] Goal position in body frame (meters)
            compass_heading.unsqueeze(-1),  # [1] Vehicle heading (radians)
            vehicle_velocity_body,      # [2] Vehicle velocity in body frame (m/s)
            vehicle_yaw_rate.unsqueeze(-1), # [1] Yaw rate (rad/s)
        ], dim=-1)
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on navigation performance"""
        
        # Get current position (with GPS noise for realism)
        current_pos = self._simulate_gps_reading(self._robot.data.root_pos_w[:, :2])
        
        # Distance to goal
        distance_to_goal = torch.norm(self._goal_positions_global - current_pos, dim=-1)
        
        # Goal reaching reward (exponential decay)
        goal_reward = torch.exp(-distance_to_goal / 10.0)
        
        # IMPROVED: Heading alignment reward (more sensitive to direction errors)
        goal_direction_global = self._goal_positions_global - current_pos
        goal_distance = torch.norm(goal_direction_global, dim=-1)
        
        # Only apply heading reward when goal is far enough (avoid spinning when close)
        desired_heading = torch.atan2(goal_direction_global[:, 1], goal_direction_global[:, 0])
        current_heading = self._get_vehicle_heading()
        heading_error = torch.abs(math_utils.wrap_to_pi(desired_heading - current_heading))
        
        # More aggressive heading reward - exponential decay
        heading_reward = torch.where(
            goal_distance > 3.0,  # Only when >3m from goal
            torch.exp(-heading_error / 0.3),  # Sharper penalty for misalignment (was 0.5)
            torch.ones_like(heading_error)   # No heading penalty when close to goal
        )
        
        # CRITICAL: Your robot cannot exceed 1.5 m/s - STRONG speed limiting
        current_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        
        # VERY strong rewards for your robot's actual speed range
        speed_reward = torch.zeros_like(current_speed)
        perfect_speed = (current_speed >= 0.5) & (current_speed <= 1.2)   # Your robot's sweet spot
        good_speed = (current_speed >= 0.3) & (current_speed <= 1.5)      # Your robot's range
        speed_reward[perfect_speed] = 3.0  # VERY strong bonus for perfect speeds
        speed_reward[good_speed & ~perfect_speed] = 1.0  # Good bonus for acceptable speeds
        
        # VERY strong penalties for impossible speeds
        speed_penalty = torch.zeros_like(current_speed)
        too_fast_mild = (current_speed > 1.5) & (current_speed <= 2.0)    # Slightly too fast
        too_fast_bad = (current_speed > 2.0) & (current_speed <= 3.0)     # Way too fast
        too_fast_impossible = current_speed > 3.0                         # Impossible for your robot
        
        speed_penalty[too_fast_mild] = -1.0       # Moderate penalty
        speed_penalty[too_fast_bad] = -3.0        # Strong penalty  
        speed_penalty[too_fast_impossible] = -10.0  # MASSIVE penalty for impossible speeds
        
        barely_moving = current_speed < 0.1
        speed_penalty[barely_moving] = -0.5   # Small penalty for not moving
        
        # Encourage forward motion but LIMIT to realistic speeds
        forward_velocity = self._robot.data.root_lin_vel_b[:, 0]  # Surge velocity  
        forward_reward = torch.clamp(forward_velocity / 1.2, 0.0, 1.0)  # Cap reward at 1.2 m/s
        
        # Monitor speed compliance for your robot's requirements
        self._speed_history = current_speed  # Store for debugging
        
        # More realistic violation thresholds
        too_fast = current_speed > 2.0   # Allow some overspeed (was 1.6)
        too_slow = current_speed < 0.3   # Only count very slow speeds (was 0.5)
        self._speed_violations += (too_fast | too_slow).float()
        
        # Print every 100 steps
        if self.common_step_counter % 100 == 0:
            avg_speed = current_speed.mean().item()
            max_speed = current_speed.max().item()
            violations = self._speed_violations.mean().item()
            # Better usability metric: percentage in your robot's preferred range  
            in_optimal_range = ((current_speed >= 0.5) & (current_speed <= 1.2)).float().mean().item() * 100
            in_usable_range = ((current_speed >= 0.3) & (current_speed <= 1.5)).float().mean().item() * 100
            too_fast_pct = (current_speed > 1.5).float().mean().item() * 100
            print(f"Speed: avg={avg_speed:.2f} m/s, max={max_speed:.2f} m/s, violations={violations:.1f}, optimal={in_optimal_range:.0f}%, usable={in_usable_range:.0f}%, too_fast={too_fast_pct:.0f}%")
        
        # GENTLER action penalty - don't discourage control inputs
        action_penalty = -torch.norm(self._actions, dim=-1) * 0.005  # Much smaller penalty
        
        # MUCH STRONGER anti-spinning penalty
        current_yaw_rate = torch.abs(self._robot.data.root_ang_vel_b[:, 2])  # |yaw rate|
        spinning_penalty = torch.zeros_like(current_yaw_rate)
        spinning_penalty[current_yaw_rate > 0.3] = -0.5   # Penalty for spinning >0.3 rad/s (~17°/s)
        spinning_penalty[current_yaw_rate > 0.7] = -1.5   # Strong penalty for spinning >0.7 rad/s (~40°/s)
        spinning_penalty[current_yaw_rate > 1.2] = -3.0   # VERY strong penalty for spinning >1.2 rad/s (~70°/s)
        
        # STRONG reward for straight-line navigation
        forward_vel = self._robot.data.root_lin_vel_b[:, 0]
        total_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        efficiency = torch.where(total_speed > 0.1, forward_vel / (total_speed + 1e-6), torch.zeros_like(total_speed))
        efficiency_reward = torch.clamp(efficiency, 0.0, 1.0) * 1.0  # Strong bonus for straight movement
        
        # PENALTY for erratic rudder movement
        rudder_penalty = torch.zeros_like(current_yaw_rate)
        large_rudder = torch.abs(self._actions[:, 1]) > 0.5  # Large rudder commands
        rudder_penalty[large_rudder] = -0.2  # Penalty for excessive rudder use
        
        # Goal completion bonus
        goal_reached = distance_to_goal < 2.0
        completion_bonus = goal_reached.float() * 10.0
        
        total_reward = (self.cfg.rew_scale_goal * goal_reward + 
                    self.cfg.rew_scale_heading * heading_reward + 
                    2.0 * speed_reward +                             # HIGHEST priority: correct speed
                    0.5 * forward_reward +                           # Lower priority: forward motion  
                    speed_penalty +                                  # STRONG speed penalties
                    action_penalty +                                 # Minimal action penalty  
                    spinning_penalty +                               # STRONG anti-spinning penalty
                    efficiency_reward +                              # STRONG straight-line bonus
                    rudder_penalty +                                 # Rudder penalty
                    completion_bonus)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_vehicle_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_vehicle_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_height) > self.cfg.max_vehicle_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Goal reached condition
        current_pos = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.norm(self._goal_positions_global - current_pos, dim=-1)
        goal_reached = distance_to_goal < 2.0  # Within 2 meters

        return out_of_bounds | goal_reached, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environment and generate new goals"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # Reset vehicle to origin
        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]
        
        self._vehicle_origins[env_ids] = self.scene.env_origins[env_ids, :2]
        
        # Generate random goals at realistic distances
        self._generate_goals(env_ids)
        
        # Reset vehicle state
        default_state = self._robot.data.default_root_state[env_ids].clone()
        default_state[:, :2] = self._vehicle_origins[env_ids]  # Set x, y position
        default_state[:, 2] = self.cfg.starting_height  # Surface level
        
        # Optional: randomize initial heading
        if not self.cfg.eval_mode:
            random_headings = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
            default_state[:, 3:7] = math_utils.quat_from_euler_xyz(
                torch.zeros_like(random_headings), 
                torch.zeros_like(random_headings), 
                random_headings
            )
        
        self._robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)
    
    def _generate_goals(self, env_ids: Sequence[int]):
        """Generate random goal positions in global coordinates"""
        num_goals = len(env_ids)
        
        # Generate random distances and angles for goals
        distances = torch.rand(num_goals, device=self.device) * \
                   (self.cfg.goal_spawn_radius_max - self.cfg.goal_spawn_radius_min) + \
                   self.cfg.goal_spawn_radius_min
        
        angles = torch.rand(num_goals, device=self.device) * 2 * math.pi
        
        # Convert to global coordinates relative to vehicle origin
        goal_x = self._vehicle_origins[env_ids, 0] + distances * torch.cos(angles)
        goal_y = self._vehicle_origins[env_ids, 1] + distances * torch.sin(angles)
        
        self._goal_positions_global[env_ids, 0] = goal_x
        self._goal_positions_global[env_ids, 1] = goal_y

    def _compute_dynamics(self, actions) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamics from actions for surface vehicle"""
        
        if self._debug: 
            print("actions: ", actions)
        
        # Get forces and torques from surface vehicle dynamics
        actuator_forces, actuator_torques = self.surface_dynamics.compute_forces_and_torques(
            actions, self.episode_length_buf * self.sim.cfg.dt
        )
        
        # Get hydrodynamic forces with TEMPORARY minimal drag
        hydro_forces, hydro_torques = self.surface_hydro.calculate_surface_forces(
            self._robot.data.root_lin_vel_b, 
            self._robot.data.root_ang_vel_b
        )
        
        # TEMPORARY: Reduce drag forces by 90% to test movement
        hydro_forces *= 0.1  # Apply only 10% of drag forces
        hydro_torques *= 0.1  # Apply only 10% of drag torques
        
        # Apply surface constraints (no vertical forces/torques)
        total_forces, total_torques = self.surface_hydro.apply_surface_constraints(
            actuator_forces + hydro_forces,
            actuator_torques + hydro_torques
        )
        
        if self._debug:
            print("actuator forces: ", actuator_forces)
            print("actuator torques: ", actuator_torques)
            print("hydro forces: ", hydro_forces)
            print("hydro torques: ", hydro_torques)
            print("total forces: ", total_forces)
            print("total torques: ", total_torques)
        
        return total_forces, total_torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization"""
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (2.0, 2.0, 0.5)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "heading_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/vehicle_heading"
                marker_cfg.markers["arrow"].scale = (0.25, 0.25, 2)
                self.heading_visualizer = VisualizationMarkers(marker_cfg)
            
            # Set visibility
            self.goal_visualizer.set_visibility(True)
            self.heading_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
            if hasattr(self, "heading_visualizer"):
                self.heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize goals and vehicle state for debugging"""
        if hasattr(self, "goal_visualizer"):
            # Show goal positions
            goal_positions_3d = torch.cat([
                self._goal_positions_global, 
                torch.zeros(self.num_envs, 1, device=self.device)
            ], dim=-1)
            self.goal_visualizer.visualize(translations=goal_positions_3d)
            
            # Show vehicle heading
            vehicle_positions = self._robot.data.root_pos_w
            vehicle_orientations = self._robot.data.root_quat_w
            self.heading_visualizer.visualize(
                translations=vehicle_positions, 
                orientations=vehicle_orientations
            )


@torch.jit.script
def _compute_surface_vehicle_rewards(
    goal_positions_global: torch.Tensor,
    current_position: torch.Tensor,
    current_heading: torch.Tensor,
    forward_velocity: torch.Tensor,
    actions: torch.Tensor,
    rew_scale_goal: float,
    rew_scale_heading: float,
    rew_scale_forward: float,
    rew_scale_actions: float,
):
    """JIT compiled reward function for performance"""
    
    # Distance to goal
    distance_to_goal = torch.norm(goal_positions_global - current_position, dim=-1)
    goal_reward = torch.exp(-distance_to_goal / 10.0)
    
    # Heading alignment
    goal_direction = goal_positions_global - current_position
    desired_heading = torch.atan2(goal_direction[:, 1], goal_direction[:, 0])
    heading_error = torch.abs(math_utils.wrap_to_pi(desired_heading - current_heading))
    heading_reward = torch.exp(-heading_error / 0.5)
    
    # Forward motion
    forward_reward = torch.clamp(forward_velocity / 5.0, 0.0, 1.0)
    
    # Action penalty
    action_penalty = -torch.norm(actions, dim=-1)
    
    total_reward = (rew_scale_goal * goal_reward + 
                   rew_scale_heading * heading_reward + 
                   rew_scale_forward * forward_reward + 
                   rew_scale_actions * action_penalty)
    
    return total_reward