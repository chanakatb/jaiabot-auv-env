"""
WarpAUV environment for IsaacLabs - Modified for Circular Motion with Origin Visualization
"""

from __future__ import annotations

import random
import math
import torch
from collections.abc import Sequence
from typing import Tuple

from .assets.warpauv import WARPAUV_CFG

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

from isaaclab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations

class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the warpauvenv environment."""

    def __init__(self, env: WarpAUVEnv, window_name: str = "IsaacLab"):
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=25.0, replicate_physics=True)
    debug_vis = True

    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
    action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
    
    # env
    decimation = 1
    cap_episode_length = True
    episode_length_s = 20.0  # Longer episodes for circular motion
    episode_length_before_reset = None
    use_boundaries = True
    max_auv_x = 10
    max_auv_y = 10
    max_auv_z = 2  # Small tolerance for Z
    starting_depth = 0.0  # Start at Z=0 for planar motion
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False

    # Circular motion parameters
    target_radius = 7.0  # Changed from 3.0 to 7.0 meters
    target_speed = 1.0   # m/s tangential speed
    
    goal_spawn_radius = 0.5  # Smaller variation for circular motion
    init_guidance_rate = 0.1
    init_vel_max = 1.0

    # Modified rewards for circular motion
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 0.0
    
    rew_scale_radius = 2.0      # For maintaining radius
    rew_scale_z_plane = 1.0     # For staying in Z=0 plane
    rew_scale_speed = 2.0       # For tangential speed
    rew_scale_heading = 1.0     # For proper heading
    rew_scale_level = 1.0       # For staying level
    rew_scale_actions = 0.1     # Energy penalty

    # Legacy reward scales (kept for compatibility but not used)
    rew_scale_pos = 0.0
    rew_scale_ang = 0.0
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0

    # dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.1 / 100.0
    dyn_time_constant = 0.05
    volume = 0.022747843530591776
    mass = 2.2701e+01

    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.05
        volume_range = [0.019747843530591773, 0.02574784353059178]
        mass_range = [2.2701e+01, 2.2701e+01]


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins
        self._step_count = 0
        
        # Get thruster configurations
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19

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
        if self._debug: print("concatenated actions shape: ", self._actions)

        self._actions[:] = actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        # Get position relative to origin
        offset_from_origin = self._robot.data.root_pos_w - self._default_env_origins
        
        # Compute radial and tangential components
        xy_distance = torch.sqrt(offset_from_origin[:, 0]**2 + offset_from_origin[:, 1]**2) + 1e-6
        
        # Radial unit vector
        radial_unit = torch.zeros_like(offset_from_origin)
        radial_unit[:, 0] = offset_from_origin[:, 0] / xy_distance
        radial_unit[:, 1] = offset_from_origin[:, 1] / xy_distance
        
        # Tangential unit vector
        tangent_unit = torch.zeros_like(offset_from_origin)
        tangent_unit[:, 0] = -radial_unit[:, 1]
        tangent_unit[:, 1] = radial_unit[:, 0]
        
        # Project velocities
        radial_vel = torch.sum(self._robot.data.root_lin_vel_w * radial_unit, dim=1, keepdim=True)
        tangent_vel = torch.sum(self._robot.data.root_lin_vel_w * tangent_unit, dim=1, keepdim=True)
        
        # Distance from target radius
        radius_error = (xy_distance - self.cfg.target_radius).unsqueeze(-1)
        
        # Speed error
        speed_error = (tangent_vel - self.cfg.target_speed)
        
        obs = torch.cat(
            [
                offset_from_origin,                    # 3D position relative to origin
                self._robot.data.root_quat_w,          # Orientation
                self._robot.data.root_lin_vel_w,       # Linear velocity in world frame
                self._robot.data.root_ang_vel_b,       # Angular velocity in body frame
                radius_error,                          # Distance from target radius
                radial_vel,                            # Radial velocity
                tangent_vel,                           # Tangential velocity
                speed_error,                           # Speed error from target
                radial_unit[:, :2],                    # Radial direction (x,y)
                tangent_unit[:, :2],                   # Tangent direction (x,y)
            ],
            dim=-1
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        offset_from_origin = self._robot.data.root_pos_w - self._default_env_origins
        
        total_reward = _compute_circular_rewards(
            self.cfg.rew_scale_radius,
            self.cfg.rew_scale_z_plane,
            self.cfg.rew_scale_speed,
            self.cfg.rew_scale_heading,
            self.cfg.rew_scale_level,
            self.cfg.rew_scale_actions,
            self._robot.data.root_lin_vel_w,
            self._robot.data.root_ang_vel_b,
            offset_from_origin,
            self._robot.data.root_quat_w,
            self._actions,
            self.cfg.target_radius,
            self.cfg.target_speed,
        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, device=self.device)

        self._step_count = self._step_count + 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs, device=self.device)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, device=self.device)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        if not self.cfg.eval_mode:
            # Start robots at random positions on the circle
            angles = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
            self._default_root_state[env_ids, 0] = self._default_env_origins[env_ids, 0] + self.cfg.target_radius * torch.cos(angles)
            self._default_root_state[env_ids, 1] = self._default_env_origins[env_ids, 1] + self.cfg.target_radius * torch.sin(angles)
            self._default_root_state[env_ids, 2] = self._default_env_origins[env_ids, 2]  # Keep at Z=0
            
            # Set initial heading tangent to circle (perpendicular to radius)
            initial_yaw = angles + np.pi/2
            self._default_root_state[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),  # roll = 0
                torch.zeros(len(env_ids), device=self.device),  # pitch = 0
                initial_yaw  # yaw = tangent to circle
            )
            
            # Set initial tangential velocity
            self._default_root_state[env_ids, 7] = -self.cfg.target_speed * torch.sin(angles)  # vx
            self._default_root_state[env_ids, 8] = self.cfg.target_speed * torch.cos(angles)   # vy
            self._default_root_state[env_ids, 9] = 0.0  # vz = 0
            self._default_root_state[env_ids, 10:13] = 0.0  # No initial angular velocity

        self._step_count = 0
        
        self._reset_domain(env_ids)
        self._reset_goal(env_ids)

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)

    def _reset_goal(self, env_ids: Sequence[int]):
        # For circular motion, we don't need orientation goals
        # But keeping for compatibility - fixed dtype issue
        self._goal[env_ids, 0:4] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)

    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]

        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)

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

    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motorValues = torch.clone(actions)

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

        if self._debug: print("gravity magnitude: ", self._gravity_magnitude) 
        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, 
            abs(self._gravity_magnitude), self.com_to_cob_offsets
        )

        density_forces, density_torques, viscosity_forces, viscosity_torques = self.force_calculation_functions.calculate_density_and_viscosity_forces(
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
            # Create coordinate system markers at origin
            if not hasattr(self, "origin_x_axis"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Origin/x_axis"
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 2.0)  # 2m long arrow
                self.origin_x_axis = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "origin_y_axis"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Origin/y_axis"
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 2.0)  # 2m long arrow
                self.origin_y_axis = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "origin_z_axis"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Origin/z_axis"
                marker_cfg.markers["arrow"].scale = (0.2, 0.2, 2.0)  # 2m long arrow
                self.origin_z_axis = VisualizationMarkers(marker_cfg)
            
            # Create circle marker to show target radius
            if not hasattr(self, "target_circle_markers"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Origin/target_circle"
                self.target_circle_markers = VisualizationMarkers(marker_cfg)

            # Robot body frame markers with consistent colors
            # X-axis (RED)
            if not hasattr(self, "robot_x_axis"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Robot/x_axis"
                self.robot_x_axis = VisualizationMarkers(marker_cfg)

            # Y-axis (GREEN)
            if not hasattr(self, "robot_y_axis"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Robot/y_axis"
                self.robot_y_axis = VisualizationMarkers(marker_cfg)

            # Z-axis (BLUE)
            if not hasattr(self, "robot_z_axis"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Robot/z_axis"
                self.robot_z_axis = VisualizationMarkers(marker_cfg)
            
            # Set visibility
            self.origin_x_axis.set_visibility(True)
            self.origin_y_axis.set_visibility(True)
            self.origin_z_axis.set_visibility(True)
            self.target_circle_markers.set_visibility(True)
            self.robot_x_axis.set_visibility(True)
            self.robot_y_axis.set_visibility(True)
            self.robot_z_axis.set_visibility(True)

        else:
            if hasattr(self, "origin_x_axis"):
                self.origin_x_axis.set_visibility(False)
            if hasattr(self, "origin_y_axis"):
                self.origin_y_axis.set_visibility(False)
            if hasattr(self, "origin_z_axis"):
                self.origin_z_axis.set_visibility(False)
            if hasattr(self, "target_circle_markers"):
                self.target_circle_markers.set_visibility(False)
            if hasattr(self, "robot_x_axis"):
                self.robot_x_axis.set_visibility(False)
            if hasattr(self, "robot_y_axis"):
                self.robot_y_axis.set_visibility(False)
            if hasattr(self, "robot_z_axis"):
                self.robot_z_axis.set_visibility(False)

    def _rotate_quat_by_euler_xyz(self, q: torch.tensor, x: float|torch.tensor, y: float|torch.tensor, z: float|torch.tensor, device=None):
        num_envs = q.shape[0]
        if device == None:
            device = self.device

        if type(x) == float:
            x = torch.zeros(num_envs, device=device) + x
        if type(y) == float:
            y = torch.zeros(num_envs, device=device) + y
        if type(z) == float:
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)

    def _debug_vis_callback(self, event):
        # Visualize coordinate axes at each environment's origin
        # X-axis (Red) - pointing in +X direction
        x_orientations = torch.tensor([[0, 0, 0, 1]], device=self.device).repeat(self.num_envs, 1)
        self.origin_x_axis.visualize(
            translations=self._default_env_origins,
            orientations=x_orientations
        )
        
        # Y-axis (Green) - pointing in +Y direction (rotate 90° around Z)
        y_orientations = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.full((self.num_envs,), torch.pi/2, device=self.device)
        )
        self.origin_y_axis.visualize(
            translations=self._default_env_origins,
            orientations=y_orientations
        )
        
        # Z-axis (Blue) - pointing in +Z direction (rotate -90° around Y)
        z_orientations = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.full((self.num_envs,), -torch.pi/2, device=self.device),
            torch.zeros(self.num_envs, device=self.device)
        )
        self.origin_z_axis.visualize(
            translations=self._default_env_origins,
            orientations=z_orientations
        )
        
        # Visualize target circle with points
        num_circle_points = 32
        angles = torch.linspace(0, 2*torch.pi, num_circle_points, device=self.device)
        circle_points = []
        for env_id in range(self.num_envs):
            env_circle_points = torch.zeros((num_circle_points, 3), device=self.device)
            env_circle_points[:, 0] = self._default_env_origins[env_id, 0] + self.cfg.target_radius * torch.cos(angles)
            env_circle_points[:, 1] = self._default_env_origins[env_id, 1] + self.cfg.target_radius * torch.sin(angles)
            env_circle_points[:, 2] = self._default_env_origins[env_id, 2]
            circle_points.append(env_circle_points)
        
        all_circle_points = torch.cat(circle_points, dim=0)
        self.target_circle_markers.visualize(translations=all_circle_points)

        # Visualize robot's coordinate system with consistent colors
        # Robot X-axis (RED)
        robot_x_orientations = self._robot.data.root_quat_w
        robot_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        self.robot_x_axis.visualize(
            translations=self._robot.data.root_pos_w, 
            orientations=robot_x_orientations, 
            scales=robot_scales
        )

        # Robot Y-axis (GREEN) - rotate 90° around Z from X
        robot_y_orientations = self._rotate_quat_by_euler_xyz(
            self._robot.data.root_quat_w, 0.0, 0.0, torch.pi/2
        )
        self.robot_y_axis.visualize(
            translations=self._robot.data.root_pos_w, 
            orientations=robot_y_orientations, 
            scales=robot_scales
        )

        # Robot Z-axis (BLUE) - rotate -90° around Y from X
        robot_z_orientations = self._rotate_quat_by_euler_xyz(
            self._robot.data.root_quat_w, 0.0, -torch.pi/2, 0.0
        )
        self.robot_z_axis.visualize(
            translations=self._robot.data.root_pos_w, 
            orientations=robot_z_orientations, 
            scales=robot_scales
        )


@torch.jit.script
def _compute_circular_rewards(
    rew_scale_radius: float,
    rew_scale_z_plane: float,
    rew_scale_speed: float,
    rew_scale_heading: float,
    rew_scale_level: float,
    rew_scale_actions: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    root_pos: torch.Tensor,  # Position relative to origin
    root_quat: torch.Tensor,
    actions: torch.Tensor,
    target_radius: float,
    target_speed: float,
):
    # 1. Reward for maintaining radius r from origin (in XY plane)
    xy_distance = torch.sqrt(root_pos[:, 0]**2 + root_pos[:, 1]**2) + 1e-6
    radius_error = torch.abs(xy_distance - target_radius)
    rew_radius = rew_scale_radius * torch.exp(-2.0 * radius_error**2)
    
    # 2. Reward for maintaining Z = 0 (staying in plane)
    z_error = torch.abs(root_pos[:, 2])
    rew_z_plane = rew_scale_z_plane * torch.exp(-5.0 * z_error**2)
    
    # 3. Calculate radial and tangential components
    radial_unit = torch.zeros_like(root_pos)
    radial_unit[:, 0] = root_pos[:, 0] / xy_distance
    radial_unit[:, 1] = root_pos[:, 1] / xy_distance
    
    tangent_unit = torch.zeros_like(root_pos)
    tangent_unit[:, 0] = -radial_unit[:, 1]
    tangent_unit[:, 1] = radial_unit[:, 0]
    
    # 4. Reward for tangential velocity
    tangential_speed = torch.sum(lin_vel * tangent_unit, dim=1)
    speed_error = torch.abs(tangential_speed - target_speed)
    rew_speed = rew_scale_speed * torch.exp(-2.0 * speed_error**2)
    
    # 5. Penalize radial velocity
    radial_vel = torch.sum(lin_vel * radial_unit, dim=1)
    rew_radial_vel = rew_scale_speed * torch.exp(-5.0 * radial_vel**2)
    
    # 6. Reward for proper heading
    expected_heading = torch.atan2(tangent_unit[:, 1], tangent_unit[:, 0])
    
    # Extract yaw from quaternion
    yaw = torch.atan2(2.0 * (root_quat[:, 3] * root_quat[:, 2] + root_quat[:, 0] * root_quat[:, 1]),
                       1.0 - 2.0 * (root_quat[:, 1]**2 + root_quat[:, 2]**2))
    
    heading_error = torch.abs(torch.atan2(torch.sin(yaw - expected_heading), 
                                          torch.cos(yaw - expected_heading)))
    rew_heading = rew_scale_heading * torch.exp(-2.0 * heading_error**2)
    
    # 7. Penalize roll and pitch
    roll = torch.atan2(2.0 * (root_quat[:, 3] * root_quat[:, 0] + root_quat[:, 1] * root_quat[:, 2]),
                       1.0 - 2.0 * (root_quat[:, 0]**2 + root_quat[:, 1]**2))
    pitch = torch.asin(torch.clamp(2.0 * (root_quat[:, 3] * root_quat[:, 1] - root_quat[:, 2] * root_quat[:, 0]), -1.0, 1.0))
    
    rew_level = rew_scale_level * torch.exp(-5.0 * (roll**2 + pitch**2))
    
    # 8. Penalize vertical velocity
    rew_z_vel = rew_scale_speed * torch.exp(-10.0 * lin_vel[:, 2]**2)
    
    # 9. Penalize energy consumption
    rew_action = rew_scale_actions * torch.exp(-1.0 * torch.norm(actions, dim=1)**2)
    
    total_rew = (rew_radius + rew_z_plane + rew_speed + rew_radial_vel + 
                 rew_heading + rew_level + rew_z_vel + rew_action)
    
    return total_rew