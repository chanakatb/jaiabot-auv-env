# /home/lucky-champ/isaac-auv-env/surface_vehicle_dynamics.py

import torch
from isaaclab.utils.math import quat_apply
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic

def get_surface_vehicle_thruster_config(device):
    """
    Configure only 2 thrusters from the existing 6-thruster setup:
    - Main thruster (for surge)
    - One side thruster (for yaw control via differential thrust)
    
    We'll use thrusters 0 and 1 (drive_left and drive_right) from the original config
    """
    
    # Use only the rear-facing thrusters for surge
    # Thruster 0: Main propulsion
    # Thruster 1: Differential thrust for steering (like a rudder effect)
    
    thruster_positions = torch.tensor([
        [-0.4127, 0.0, -0.0889],    # Main thruster (centered)
        [-0.4127, 0.1506, -0.0889] # Side thruster for steering
    ], dtype=torch.float32, device=device, requires_grad=False)
    
    thruster_orientations = torch.tensor([
        [1, 0, 0, 0],  # Forward thrust
        [1, 0, 0, 0],  # Forward thrust
    ], dtype=torch.float32, device=device, requires_grad=False)
    
    return thruster_positions, thruster_orientations

class SurfaceVehicleDynamics:
    def __init__(self, num_envs: int, device: torch.device, time_constant: float = 0.05):
        self.num_envs = num_envs
        self.device = device
        
        # Get thruster configuration
        self.thruster_positions, self.thruster_orientations = get_surface_vehicle_thruster_config(device)
        self.thruster_positions = self.thruster_positions.unsqueeze(0).repeat(num_envs, 1, 1)
        self.thruster_orientations = self.thruster_orientations.repeat(num_envs, 1)
        
        # Dynamics models (reuse existing classes)
        self.thruster_dynamics = DynamicsFirstOrder(num_envs, 2, time_constant, device)  # Only 2 thrusters
        self.thruster_conversion = ConversionFunctionBasic(0.1 / 100.0)  # Same rotor constant
    
    def compute_forces_and_torques(self, actions: torch.Tensor, episode_time: torch.Tensor):
        """
        Convert [thrust_command, rudder_command] to forces and torques
        
        Args:
            actions: [num_envs, 2] where actions[:, 0] = thrust, actions[:, 1] = rudder
            episode_time: Current simulation time for thruster dynamics
            
        Returns:
            forces, torques in body frame
        """
        
        # Convert control inputs to individual thruster commands
        thrust_cmd = actions[:, 0]   # Main thrust (-1 to 1)
        rudder_cmd = actions[:, 1]   # Rudder command (-1 to 1)
        
        # Map to two thrusters:
        # Thruster 0 (main): Always follows thrust command
        # Thruster 1 (steering): thrust + differential for steering
        thruster_commands = torch.zeros((self.num_envs, 2), device=self.device)
        thruster_commands[:, 0] = thrust_cmd  # Main thruster
        thruster_commands[:, 1] = thrust_cmd * 0.3 + rudder_cmd * 0.7  # Steering thruster
        
        # Apply thruster dynamics and conversion (reuse existing logic)
        thruster_commands = torch.clamp(thruster_commands, -1, 1)
        
        # Convert PWM to motor velocities (same as original)
        motor_values = thruster_commands.clone()
        motor_values[torch.abs(motor_values) < 0.08] = 0
        motor_values[motor_values >= 0.08] = (-139.0 * (motor_values[motor_values >= 0.08] ** 2.0) + 
                                              500 * motor_values[motor_values >= 0.08] + 8.28)
        motor_values[motor_values <= -0.08] = (161.0 * (motor_values[motor_values <= -0.08] ** 2.0) + 
                                               517.86 * motor_values[motor_values <= -0.08] - 5.72)
        
        # Apply first-order dynamics
        motor_values = self.thruster_dynamics.update(motor_values, episode_time)
        
        # Convert to thrust forces
        thrust_forces = self.thruster_conversion.convert(motor_values)
        
        # Calculate force vectors
        thruster_force_vectors = torch.zeros((self.num_envs, 2, 3), device=self.device)
        thruster_force_vectors[..., 0] = 1.0  # Forward direction
        
        # Apply thrust magnitudes
        thruster_force_vectors = thruster_force_vectors * thrust_forces.unsqueeze(-1)
        
        # Apply thruster orientations (forward thrust)
        thruster_force_vectors = quat_apply(self.thruster_orientations, thruster_force_vectors)
        
        # Calculate torques: T = r × F
        thruster_torques = torch.cross(self.thruster_positions, thruster_force_vectors, dim=-1)
        
        # Sum forces and torques
        total_forces = torch.sum(thruster_force_vectors, dim=-2)
        total_torques = torch.sum(thruster_torques, dim=-2)
        
        return total_forces, total_torques