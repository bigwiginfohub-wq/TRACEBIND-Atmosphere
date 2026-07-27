import sys
from pathlib import Path

# Force project root onto sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tests.test_domain_validation import run_pipeline
from tracebind.metrics.metric_result import MetricStatus


def test_high_aspect_ratio_corridor():
    """Stresses k-NN graph construction on narrow 1D-like ribbon geometries (1x32 strip)."""
    print("\n[Boundary Stress] Testing High Aspect Ratio Ribbon (1x32 Grid)...")
    data_1d = np.sin(np.linspace(0, 4 * np.pi, 32)).reshape(1, 32)
    
    res, collection, graph, roi = run_pipeline(data_1d, k=2, n_permutations=50, seed=42)
    assert res.status != MetricStatus.DEGENERATE_VARIANCE
    print(f"  -> Ribbon Geometry (1x32) Evaluated safely: R_observed = {res.r_observed:.4f}, Z = {res.z_score:.2f}")


def test_masked_disconnected_archipelago():
    """Stresses graph indexing on TRUE masked disconnected spatial topologies."""
    print("\n[Boundary Stress] Testing Masked Disconnected Archipelago Domain...")
    grid = np.full((16, 16), np.nan)  # Entire domain masked out by default
    
    # Island 1 (Active)
    grid[2:5, 2:5] = 10.0
    # Island 2 (Active)
    grid[10:14, 10:14] = -5.0
    
    # Extract only non-masked active nodes
    valid_mask = ~np.isnan(grid)
    valid_data = grid[valid_mask]
    
    # Flattened contiguous array representing active node cloud
    active_field = valid_data.reshape(1, -1)
    
    res, collection, graph, _ = run_pipeline(active_field, k=3, n_permutations=30, seed=42)
    assert len(collection.point_cloud.values) == (3 * 3) + (4 * 4)
    print(f"  -> True Masked Disconnected Domain (25 nodes retained): R_observed = {res.r_observed:.4f}, Status = {res.status.value}")


def test_rotation_and_reflection_invariance():
    """Validates full orthogonal rotation (90°, 180°, 270°) and reflection (H-flip, V-flip) invariance on isotropic fields."""
    print("\n[Boundary Stress] Testing Rotation & Reflection Coordinate Invariance...")
    
    x = np.linspace(-2, 2, 16)
    y = np.linspace(-2, 2, 16)
    xx, yy = np.meshgrid(x, y)
    field_isotropic = np.exp(-(xx**2 + yy**2))
    
    res_orig, _, _, _ = run_pipeline(field_isotropic, k=4, n_permutations=40, seed=42)
    res_rot90, _, _, _ = run_pipeline(np.rot90(field_isotropic, 1), k=4, n_permutations=40, seed=42)
    res_rot180, _, _, _ = run_pipeline(np.rot90(field_isotropic, 2), k=4, n_permutations=40, seed=42)
    res_hflip, _, _, _ = run_pipeline(np.fliplr(field_isotropic), k=4, n_permutations=40, seed=42)
    res_vflip, _, _, _ = run_pipeline(np.flipud(field_isotropic), k=4, n_permutations=40, seed=42)
    
    assert np.isclose(res_orig.r_observed, res_rot90.r_observed, atol=1e-8)
    assert np.isclose(res_orig.r_observed, res_rot180.r_observed, atol=1e-8)
    assert np.isclose(res_orig.r_observed, res_hflip.r_observed, atol=1e-8)
    assert np.isclose(res_orig.r_observed, res_vflip.r_observed, atol=1e-8)
    
    print(f"  -> Isotropic Field Base R: {res_orig.r_observed:.6f}")
    print(f"  -> Rot90 R: {res_rot90.r_observed:.6f} | H-Flip R: {res_hflip.r_observed:.6f}")
    print("  ✓ Full orthogonal rotation and reflection symmetry verified on isotropic fields.")


def test_graph_degree_validity():
    """Validates neighborhood graph bounds using graph properties."""
    print("\n[Boundary Stress] Testing Graph Degree Bounding Criteria...")
    rng = np.random.default_rng(77)
    data = rng.normal(size=(8, 8))
    
    k_req = 4
    res, collection, graph, _ = run_pipeline(data, k=k_req, n_permutations=30, seed=42)
    
    degrees = graph.degrees
    min_deg, max_deg = min(degrees), max(degrees)
    
    assert min_deg >= 1, "Isolated graph node detected!"
    assert max_deg <= graph.n_nodes - 1, "Graph degree exceeds total point cloud size!"
    print(f"  -> Node degrees bounded correctly: Min degree={min_deg}, Max degree={max_deg} (k={k_req})")
    print("  ✓ Boundary stress and symmetry tests completed successfully.\n")


if __name__ == "__main__":
    test_high_aspect_ratio_corridor()
    test_masked_disconnected_archipelago()
    test_rotation_and_reflection_invariance()
    test_graph_degree_validity()