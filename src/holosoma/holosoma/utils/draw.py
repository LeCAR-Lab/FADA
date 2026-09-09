"""Drawing utilities that adapt to the configured simulator."""

from holosoma.utils.simulator_config import SimulatorType, get_simulator_type


def _not_initialized(*args, **kwargs):
    """Default function that raises when drawing functions aren't initialized."""
    raise NotImplementedError(
        "Drawing functions not initialized. Must set simulator type before importing draw module."
    )


def _make_noop(*args, **kwargs):
    """No-op function that accepts any arguments."""


# Start with not_initialized functions
clear_lines = _not_initialized
draw_sphere = _not_initialized
draw_line = _not_initialized
draw_points = _not_initialized
draw_height_points = _not_initialized
draw_foot_height_points = _not_initialized

# Initialize based on current simulator type
simulator_type = get_simulator_type()  # Will raise if not set

if simulator_type == SimulatorType.ISAACGYM:
    # Import IsaacGym drawing functions
    from holosoma.utils.adapters.isaacgym_draw_adapter import (
        clear_lines,
        draw_foot_height_points,
        draw_height_points,
        draw_line,
        draw_sphere,
    )
elif simulator_type == SimulatorType.MUJOCO:
    # Import MuJoCo drawing functions (including logging stubs)
    from holosoma.utils.adapters.mujoco_draw_adapter import (
        clear_lines,
        draw_foot_height_points,
        draw_height_points,
        draw_line,
        draw_points,
        draw_sphere,
    )
elif simulator_type == SimulatorType.ISAACSIM:
    # Import IsaacSim drawing functions
    from holosoma.utils.adapters.isaacsim_draw_adapter import (
        clear_lines,
        draw_foot_height_points,
        draw_height_points,
        draw_line,
        draw_points,
        draw_sphere,
    )
else:
    raise ValueError(f"Unsupported simulator type: {simulator_type}")


# Helper functions remain unchanged
def draw_debug_heights(simulator, env_id, base_pos, height_points, heights):
    """Draw height measurement points for debugging."""
    draw_height_points(simulator, base_pos, height_points, heights, env_id)


def draw_debug_feet(simulator, env_id, feet_positions, feet_heights):
    """Draw foot positions and heights for debugging."""
    for foot_pos, foot_height in zip(feet_positions, feet_heights):
        terrain_height = foot_pos[2] - foot_height
        draw_foot_height_points(simulator, foot_pos, terrain_height, env_id)
