# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SurfaceVehiclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48  # Longer episodes for navigation
    max_iterations = 300    # Fewer iterations needed
    save_interval = 50
    experiment_name = "surface_vehicle_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[32, 32],    # Smaller network
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Slightly lower for surface vehicle
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )