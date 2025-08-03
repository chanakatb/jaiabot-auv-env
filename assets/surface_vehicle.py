# /home/lucky-champ/isaac-auv-env/assets/surface_vehicle.py

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
import os

# Reuse the existing WarpAUV USD file
USD_PATH = os.path.join(os.path.dirname(__file__), "../data/warpauv/warpauv.usd")

SURFACE_VEHICLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,  # Use existing model
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Surface level instead of underwater
    )
)
"""Configuration for the Surface Vehicle (using WarpAUV model)."""