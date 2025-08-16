import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
import os

USD_PATH = os.path.join(os.path.dirname(__file__), "../data/warpauv/warpauv.usd")

# WarpAUV configuration with proper visual asset handling
WARPAUV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        # Disable articulation to treat as rigid body
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=22.701,
        ),
        copy_from_source=False,
        activate_contact_sensors=False,
        # Important: Set the scale to ensure proper loading
        scale=(1.0, 1.0, 1.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
    )
)

# Alternative: Simple geometric AUV (backup)
SIMPLE_WARPAUV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot", 
    spawn=sim_utils.CuboidCfg(
        size=(0.7, 0.4, 0.2),  # Length, width, height in meters
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=22.701,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.4, 0.8),  # Blue color
            metallic=0.3,
            roughness=0.5,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
    ),
)

"""Configuration for the WarpAUV."""