# /home/lucky-champ/isaac-auv-env/surface_vehicle_dynamics.py

import torch
from isaaclab.utils.math import quat_apply
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic

def get_rudder_thruster_config(device):
    """
    Configure single thruster + rudder setup:
    - Single main thruster for propulsion
    - Rudder behind thruster for steering
    """
    
    thruster_positions = torch.tensor([
        [-0.4127, 0.0, -0.0889],   # Single main thruster (centered at stern)
    ], dtype=torch.float32, device=device, requires_grad=False)
    
    thruster_orientations = torch.tensor([
        [1, 0, 0, 0],  # Forward thrust direction
    ], dtype=torch.float32, device=device, requires_grad=False)
    
    rudder_position = torch.tensor([
        -0.5, 0.0, -0.0889      # Rudder behind thruster (single position)
    ], dtype=torch.float32, device=device, requires_grad=False)
    
    return thruster_positions, thruster_orientations, rudder_position

class SurfaceVehicleDynamics:
    def __init__(self, num_envs: int, device: torch.device, time_constant: float = 0.05):
        self.num_envs = num_envs
        self.device = device
        
        # Get thruster and rudder configuration
        self.thruster_positions, self.thruster_orientations, self.rudder_position = get_rudder_thruster_config(device)
        self.thruster_positions = self.thruster_positions.unsqueeze(0).repeat(num_envs, 1, 1)
        self.thruster_orientations = self.thruster_orientations.repeat(num_envs, 1)
        self.rudder_position = self.rudder_position.unsqueeze(0).repeat(num_envs, 1)
        
        # Dynamics models (now only 1 thruster)
        self.thruster_dynamics = DynamicsFirstOrder(num_envs, 1, time_constant, device)  # Only 1 thruster
        self.thruster_conversion = ConversionFunctionBasic(0.25 / 100.0)  # Same rotor constant
        
        # Rudder parameters
        self.rudder_effectiveness = 0.4    # How effective rudder is
        self.rudder_moment_arm = 0.5       # Distance from rudder to center of mass
        
    def compute_rudder_forces(self, thrust_cmd: torch.Tensor, rudder_angle: torch.Tensor):
        """
        Compute forces from rudder interaction with propeller wash
        
        Args:
            thrust_cmd: Thrust command magnitude [num_envs]
            rudder_angle: Rudder deflection angle in radians [num_envs]
            
        Returns:
            rudder_forces, rudder_torques in body frame
        """
        # Rudder only works when there's propeller wash (thrust)
        # Side force is proportional to thrust and sin(rudder_angle)
        rudder_side_force = torch.sin(rudder_angle) * torch.abs(thrust_cmd) * self.rudder_effectiveness
        
        # Rudder forces in body frame (side force in Y direction)
        rudder_forces = torch.zeros((self.num_envs, 3), device=self.device)
        rudder_forces[:, 1] = rudder_side_force  # Side force (sway)
        
        # Rudder creates yaw moment = side_force * moment_arm
        rudder_yaw_moment = rudder_side_force * self.rudder_moment_arm
        
        rudder_torques = torch.zeros((self.num_envs, 3), device=self.device)
        rudder_torques[:, 2] = rudder_yaw_moment  # Yaw torque
        
        return rudder_forces, rudder_torques
    
    def compute_forces_and_torques(self, actions: torch.Tensor, episode_time: torch.Tensor):
        """
        Convert [thrust_command, rudder_angle] to forces and torques
        
        Args:
            actions: [num_envs, 2] where actions[:, 0] = thrust, actions[:, 1] = rudder_angle
            episode_time: Current simulation time for thruster dynamics
            
        Returns:
            forces, torques in body frame
        """
        
        # Extract commands
        thrust_cmd = actions[:, 0]   # Main thrust (-1 to 1)
        rudder_cmd = actions[:, 1]   # Rudder angle command (-1 to 1)
        
        # Convert rudder command to angle (e.g., ±30 degrees max)
        max_rudder_angle = torch.pi / 6  # 30 degrees in radians
        rudder_angle = rudder_cmd * max_rudder_angle
        
        # Process thruster command
        thruster_commands = thrust_cmd.unsqueeze(-1)  # [num_envs, 1] for single thruster
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
        
        # Calculate thruster force vectors
        thruster_force_vectors = torch.zeros((self.num_envs, 1, 3), device=self.device)
        thruster_force_vectors[..., 0] = 1.0  # Forward direction
        
        # Apply thrust magnitudes
        thruster_force_vectors = thruster_force_vectors * thrust_forces.unsqueeze(-1)
        
        # Apply thruster orientations (forward thrust)
        thruster_force_vectors = quat_apply(self.thruster_orientations, thruster_force_vectors)
        
        # Calculate thruster torques: T = r × F
        thruster_torques = torch.cross(self.thruster_positions, thruster_force_vectors, dim=-1)
        
        # Sum thruster forces and torques
        thruster_total_forces = torch.sum(thruster_force_vectors, dim=-2)
        thruster_total_torques = torch.sum(thruster_torques, dim=-2)
        
        # Calculate rudder forces and torques
        rudder_forces, rudder_torques = self.compute_rudder_forces(thrust_forces.squeeze(-1), rudder_angle)
        
        # Combine thruster and rudder effects
        total_forces = thruster_total_forces + rudder_forces
        total_torques = thruster_total_torques + rudder_torques
        
        return total_forces, total_torques