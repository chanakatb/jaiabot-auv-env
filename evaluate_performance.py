# evaluate_performance.py
import gymnasium as gym
import torch
import isaaclab_tasks
import numpy as np

def evaluate_model_performance(num_episodes=100):
    env = gym.make("Isaac-SurfaceVehicle-Direct-v1", num_envs=16)
    
    # Load your trained model
    # ... (load model code) ...
    
    metrics = {
        "success_rate": [],
        "episode_length": [],
        "average_speed": [],
        "speed_compliance": [],
        "navigation_efficiency": []
    }
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = []
        episode_speeds = []
        
        for step in range(1500):  # Max episode length
            # Get actions from trained policy
            actions = policy(obs["policy"])
            obs, rewards, dones, _ = env.step(actions)
            
            # Track metrics
            episode_rewards.append(rewards.mean().item())
            current_speed = torch.norm(env.unwrapped._robot.data.root_lin_vel_b[:, :2], dim=-1)
            episode_speeds.append(current_speed.mean().item())
            
            if dones.any():
                # Goal reached
                metrics["success_rate"].append(1.0)
                metrics["episode_length"].append(step)
                break
        else:
            # Timeout
            metrics["success_rate"].append(0.0)
            metrics["episode_length"].append(1500)
        
        # Calculate episode metrics
        avg_speed = np.mean(episode_speeds)
        metrics["average_speed"].append(avg_speed)
        
        speed_compliance = np.mean([(s >= 0.5) and (s <= 1.5) for s in episode_speeds])
        metrics["speed_compliance"].append(speed_compliance)
        
        # Navigation efficiency (reward per step)
        efficiency = np.sum(episode_rewards) / len(episode_rewards)
        metrics["navigation_efficiency"].append(efficiency)
    
    # Print results
    print("=== Trained Model Performance ===")
    print(f"Success Rate: {np.mean(metrics['success_rate']) * 100:.1f}%")
    print(f"Average Episode Length: {np.mean(metrics['episode_length']):.1f} steps")
    print(f"Average Speed: {np.mean(metrics['average_speed']):.2f} m/s")
    print(f"Speed Compliance (0.5-1.5 m/s): {np.mean(metrics['speed_compliance']) * 100:.1f}%")
    print(f"Navigation Efficiency: {np.mean(metrics['navigation_efficiency']):.2f}")

if __name__ == "__main__":
    evaluate_model_performance()