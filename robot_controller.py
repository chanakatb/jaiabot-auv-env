# robot_controller.py - For your real robot
import torch
import numpy as np
import time

class TrainedPolicyController:
    def __init__(self, policy_path='deployed_policy.pt'):
        # Load trained policy
        checkpoint = torch.load(policy_path, map_location='cpu')
        self.policy = checkpoint['policy_architecture']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        
        # Your robot's sensor interfaces (replace with actual APIs)
        self.gps_sensor = GPSSensor()  # Your GPS interface
        self.imu_sensor = IMUSensor()  # Your IMU interface
        self.motor_controller = MotorController()  # Your motor interface
        
        # Target waypoint
        self.target_lat = None
        self.target_lon = None
        
    def set_waypoint(self, lat, lon):
        """Set navigation target"""
        self.target_lat = lat
        self.target_lon = lon
        
    def get_observation_from_sensors(self):
        """Convert real sensor data to policy input format"""
        # Get GPS position
        current_lat, current_lon = self.gps_sensor.get_position()
        
        # Get IMU data
        heading = self.imu_sensor.get_yaw()  # radians
        velocity = self.imu_sensor.get_velocity()  # [surge, sway] m/s
        yaw_rate = self.imu_sensor.get_yaw_rate()  # rad/s
        
        # Convert GPS to local coordinates
        goal_x_global = (self.target_lat - current_lat) * 111320  # rough conversion
        goal_y_global = (self.target_lon - current_lon) * 111320 * np.cos(current_lat)
        
        # Convert to body frame
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        goal_x_body = goal_x_global * cos_h + goal_y_global * sin_h
        goal_y_body = -goal_x_global * sin_h + goal_y_global * cos_h
        
        # Format observation (same as training)
        obs = np.array([
            goal_x_body,      # Goal X in body frame
            goal_y_body,      # Goal Y in body frame  
            heading,          # Vehicle heading
            velocity[0],      # Surge velocity
            velocity[1],      # Sway velocity
            yaw_rate         # Yaw rate
        ], dtype=np.float32)
        
        return obs
        
    def run_navigation_step(self):
        """Single navigation control step"""
        # Get current observation
        obs = self.get_observation_from_sensors()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # Add batch dimension
        
        # Get action from trained policy
        with torch.no_grad():
            action = self.policy(obs_tensor)
            thrust_cmd = action[0, 0].item()  # [0 to 1]
            rudder_cmd = action[0, 1].item()  # [-1 to 1]
        
        # Convert to your robot's actuator commands
        motor_power = thrust_cmd * 100  # Scale to 0-100% power
        rudder_angle = rudder_cmd * 30  # Scale to ±30 degrees
        
        # Send commands to robot
        self.motor_controller.set_thrust(motor_power)
        self.motor_controller.set_rudder(rudder_angle)
        
        # Debug info
        distance_to_goal = np.sqrt(obs[0]**2 + obs[1]**2)
        print(f"Distance to goal: {distance_to_goal:.1f}m, "
              f"Thrust: {motor_power:.0f}%, "
              f"Rudder: {rudder_angle:.1f}°")
        
        return distance_to_goal < 2.0  # Goal reached
        
    def navigate_to_waypoint(self, target_lat, target_lon):
        """Complete waypoint navigation"""
        self.set_waypoint(target_lat, target_lon)
        
        print(f"Navigating to: {target_lat:.6f}, {target_lon:.6f}")
        
        while True:
            goal_reached = self.run_navigation_step()
            
            if goal_reached:
                print("Goal reached!")
                self.motor_controller.stop()
                break
                
            time.sleep(0.1)  # 10Hz control loop

# Example usage
if __name__ == "__main__":
    controller = TrainedPolicyController('deployed_policy.pt')
    
    # Navigate to specific GPS coordinates
    target_lat = 40.7589  # Example coordinates
    target_lon = -73.9851
    
    controller.navigate_to_waypoint(target_lat, target_lon)