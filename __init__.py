# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Underwater and Surface Vehicle environments.
"""

import gymnasium as gym

from . import agents
from .warpauv_env import WarpAUVEnv, WarpAUVEnvCfg
from .surface_vehicle_env import SurfaceVehicleEnv, SurfaceVehicleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-WarpAUV-Direct-v1",
    entry_point="isaaclab_tasks.direct.isaac-auv-env:WarpAUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg
    },
)

gym.register(
    id="Isaac-SurfaceVehicle-Direct-v1",
    entry_point="isaaclab_tasks.direct.isaac-auv-env:SurfaceVehicleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SurfaceVehicleEnvCfg,
        "rsl_rl_cfg_entry_point": agents.surface_vehicle_ppo_cfg.SurfaceVehiclePPORunnerCfg
    },
)