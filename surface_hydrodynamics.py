# /home/lucky-champ/isaac-auv-env/surface_hydrodynamics.py

import torch
from .rigid_body_hydrodynamics import HydrodynamicForceModels

class SurfaceVehicleHydrodynamics:
    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        
        # UPDATED: Higher drag coefficients to limit speed to 0.5-1.5 m/s
        self.surface_drag_linear = 3.0   # Increased from 5.0
        self.surface_drag_quad = 8.0     # Increased from 15.0
        self.yaw_damping = 8.0
        
    def calculate_surface_forces(self, root_lin_vel_b: torch.Tensor, root_ang_vel_b: torch.Tensor):
        """Calculate hydrodynamic forces for surface operation"""
        
        # Surface drag forces (only X and Y, no Z)
        drag_forces = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Linear + quadratic drag for surge and sway
        linear_drag = -self.surface_drag_linear * root_lin_vel_b[:, :2]
        quad_drag = -self.surface_drag_quad * torch.abs(root_lin_vel_b[:, :2]) * root_lin_vel_b[:, :2]
        
        drag_forces[:, :2] = linear_drag + quad_drag
        drag_forces[:, 2] = 0.0
        
        # Add speed limiting drag for your robot's constraints
        current_speed = torch.norm(root_lin_vel_b[:, :2], dim=-1, keepdim=True)
        
        # Strong drag above 1.5 m/s
        excess_speed = torch.clamp(current_speed - 1.5, min=0.0)
        speed_limiting_drag = -100.0 * excess_speed * root_lin_vel_b[:, :2]
        drag_forces[:, :2] += speed_limiting_drag
        
        # Rotational damping
        drag_torques = torch.zeros((self.num_envs, 3), device=self.device)
        drag_torques[:, 2] = -self.yaw_damping * root_ang_vel_b[:, 2]
        
        return drag_forces, drag_torques
    
    def apply_surface_constraints(self, forces: torch.Tensor, torques: torch.Tensor):
        """
        Apply surface vehicle constraints:
        - No vertical forces/motions
        - No roll/pitch moments
        """
        constrained_forces = forces.clone()
        constrained_torques = torques.clone()
        
        # Eliminate vertical components
        constrained_forces[:, 2] = 0.0      # No heave
        constrained_torques[:, 0] = 0.0     # No roll moment
        constrained_torques[:, 1] = 0.0     # No pitch moment
        
        return constrained_forces, constrained_torques