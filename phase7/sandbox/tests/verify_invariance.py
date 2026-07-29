import numpy as np

def test_rotation_invariance(metric_func, x, y, u, v):
    """
    Rotates field 90 degrees counter-clockwise; verifies metric delta.
    """
    val_orig = metric_func(x, y, u, v)
    
    # 90-degree CCW rotation: u_rot = -v, v_rot = u, coordinates rotated
    u_rot = np.rot90(-v)
    v_rot = np.rot90(u)
    
    val_rot = metric_func(x, y, u_rot, v_rot)
    rel_diff = np.abs(val_orig - val_rot) / (np.abs(val_orig) + 1e-9)
    return rel_diff < 1e-3, rel_diff

def test_translation_invariance(metric_func, x, y, u, v, shift_px=20):
    """
    Translates field spatially; verifies core metric remains unchanged.
    """
    val_orig = metric_func(x, y, u, v)
    
    # Shift field by rolling array axes
    u_shift = np.roll(u, shift_px, axis=(0, 1))
    v_shift = np.roll(v, shift_px, axis=(0, 1))
    
    val_shift = metric_func(x, y, u_shift, v_shift)
    rel_diff = np.abs(val_orig - val_shift) / (np.abs(val_orig) + 1e-9)
    return rel_diff < 1e-3, rel_diff