import gymnasium as gym
import isaaclab_tasks

# This should work without errors
env = gym.make("Isaac-SurfaceVehicle-Direct-v1")
print("Environment created successfully!")