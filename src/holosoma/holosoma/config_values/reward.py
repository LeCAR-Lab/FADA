"""Default reward manager configurations."""

from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_fast_sac
from holosoma.config_values.loco.g1.reward import (
    g1_29dof_loco_unitree,
    g1_29dof_loco_unitree_gym,
    g1_29dof_loco_unitree_slope,
    g1_29dof_loco_unitree_slope_ar05,
    g1_29dof_loco_unitree_slope_sf,
    g1_29dof_loco_unitree_slope_sf_ar05,
)
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac
from holosoma.config_values.loco.t1.reward import t1_23dof_loco, t1_23dof_loco_waist50
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_reward_w_object,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_23dof_loco": t1_23dof_loco,
    "t1_23dof_loco_waist50": t1_23dof_loco_waist50,
    "t1_29dof_loco": t1_29dof_loco,
    "t1_29dof_loco_fast_sac": t1_29dof_loco_fast_sac,
    "g1_29dof_loco": g1_29dof_loco,
    "g1_29dof_loco_fast_sac": g1_29dof_loco_fast_sac,
    "g1_29dof_loco_unitree": g1_29dof_loco_unitree,
    "g1_29dof_loco_unitree_gym": g1_29dof_loco_unitree_gym,
    "g1_29dof_loco_unitree_slope": g1_29dof_loco_unitree_slope,
    "g1_29dof_loco_unitree_slope_ar05": g1_29dof_loco_unitree_slope_ar05,
    "g1_29dof_loco_unitree_slope_sf": g1_29dof_loco_unitree_slope_sf,
    "g1_29dof_loco_unitree_slope_sf_ar05": g1_29dof_loco_unitree_slope_sf_ar05,
    "g1_29dof_wbt": g1_29dof_wbt_reward,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_reward_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac_reward,
}
