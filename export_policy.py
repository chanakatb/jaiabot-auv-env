# export_policy.py
import torch
from rsl_rl.runners import OnPolicyRunner

def export_trained_policy():
    # Load trained model
    model_path = "/home/lucky-champ/IsaacLab/logs/rsl_rl/surface_vehicle_direct/2025-07-18_00-48-15/model_1000.pt"
    
    # Create dummy environment to get observation/action spaces
    env = gym.make("Isaac-SurfaceVehicle-Direct-v1", num_envs=1)
    
    # Load runner
    runner = OnPolicyRunner(env, None, None)
    runner.load(model_path)
    
    # Extract just the policy network
    policy_net = runner.policy
    
    # Save for deployment
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'obs_dim': 6,  # [goal_x, goal_y, heading, vel_x, vel_y, yaw_rate]
        'action_dim': 2,  # [thrust, rudder]
        'policy_architecture': policy_net
    }, 'deployed_policy.pt')
    
    print("Policy exported to: deployed_policy.pt")

if __name__ == "__main__":
    export_trained_policy()
    