# test_trained_model.py
import gymnasium as gym
import torch
import isaaclab_tasks
from rsl_rl.runners import OnPolicyRunner
import numpy as np

def test_trained_policy():
    # Load trained model
    model_path = "/home/lucky-champ/IsaacLab/logs/rsl_rl/surface_vehicle_direct/2025-07-18_00-48-15/model_1000.pt"
    
    # Create environment
    env = gym.make("Isaac-SurfaceVehicle-Direct-v1", num_envs=4, render_mode="human")
    
    # Load the trained policy
    runner = OnPolicyRunner(env, None, None)
    runner.load(model_path)
    
    # Test scenarios
    test_scenarios = [
        {"goal_distance": 10, "goal_angle": 0},      # Straight ahead
        {"goal_distance": 20, "goal_angle": 90},     # Right turn
        {"goal_distance": 15, "goal_angle": -90},    # Left turn  
        {"goal_distance": 25, "goal_angle": 180},    # Behind (U-turn)
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario}")
        obs, _ = env.reset()
        
        # Override goal position for this test
        distance = scenario["goal_distance"]
        angle = np.radians(scenario["goal_angle"])
        env.unwrapped._goal_positions_global[:, 0] = distance * np.cos(angle)
        env.unwrapped._goal_positions_global[:, 1] = distance * np.sin(angle)
        
        total_reward = 0
        speeds = []
        
        for step in range(1000):  # Max 1000 steps
            # Get action from trained policy
            with torch.no_grad():
                actions = runner.policy(obs["policy"])
            
            # Step environment
            obs, rewards, dones, _ = env.step(actions)
            total_reward += rewards.mean().item()
            
            # Monitor speed
            current_speed = torch.norm(env.unwrapped._robot.data.root_lin_vel_b[:, :2], dim=-1)
            speeds.append(current_speed.mean().item())
            
            # Check if goals reached
            if dones.any():
                print(f"  Goal reached in {step} steps!")
                break
        
        avg_speed = np.mean(speeds)
        print(f"  Average speed: {avg_speed:.2f} m/s")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Speed in range (0.5-1.5): {np.mean([(s >= 0.5) and (s <= 1.5) for s in speeds]) * 100:.0f}%")
    
    env.close()

if __name__ == "__main__":
    test_trained_policy()