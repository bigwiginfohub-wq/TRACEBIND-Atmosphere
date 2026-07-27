# tests/test_domain_validation.py
import sys
import tracemalloc
from pathlib import Path
import hashlib

# Force project root onto sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tracebind.core.region_of_interest import (
    RegionOfInterest, 
    ROIStrategy, 
    AffineCoordinateTransform,
    PointCloud
)
from tracebind.core.roi_extractor import ROIExtractor
from tracebind.core.scale_deriver import ScaleDeriver, MeanPooling
from tracebind.graph.neighborhood_graph import GraphBuilder, KNNStrategy, EuclideanDistance
from tracebind.graph.neighbourhood_collection import NeighbourhoodCollection
from tracebind.stats.null_model import NullModelEngine, GlobalPermutationNull
from tracebind.metrics.metrics_core import MetricsCore
from tracebind.metrics.metric_result import MetricStatus


class SyntheticParentGrid:
    """Mock parent observation grid satisfying ROIExtractor interface requirements."""
    def __init__(self, data: np.ndarray, fingerprint: str = "SYNTH_PARENT_HASH_001"):
        self.values = np.asarray(data, dtype=np.float64)
        self.height, self.width = self.values.shape
        self.mask = np.ones((self.height, self.width), dtype=bool)
        self.fingerprint = fingerprint
        self.transform = AffineCoordinateTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        self.coord_system = "EPSG:4326"
        self.metadata = {"sensor_id": "SYNTH_GRID_01"}

def compute_metric_fingerprint(data: np.ndarray, r_obs: float, z_score: float, p_val: float, config_str: str) -> str:
    """Generates a unique cryptographic fingerprint for a specific run execution."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(data.tobytes())
    hasher.update(f"{r_obs:.12f}:{z_score:.12f}:{p_val:.12f}".encode("utf-8"))
    hasher.update(config_str.encode("utf-8"))
    return hasher.hexdigest()

def run_pipeline(
    data: np.ndarray, 
    shift_x: float = 0.0, 
    shift_y: float = 0.0, 
    k: int = 4, 
    n_permutations: int = 50, 
    seed: int = 42,
    drop_nan: bool = True
):
    """
    Executes full pipeline and returns (result, collection, graph, roi).
    
    Parameters:
        drop_nan (bool): If True, missing/non-finite observations are excluded prior to 
                         PointCloud construction, resulting in graph construction over valid nodes only.
    """
    parent = SyntheticParentGrid(data, fingerprint="PARENT_HASH_99")
    parent.transform = AffineCoordinateTransform((shift_x, 1.0, 0.0, shift_y, 0.0, 1.0))
    
    roi = ROIExtractor.extract(
        parent=parent, strategy=ROIStrategy.BOUNDING_BOX,
        y_min=0, y_max=parent.height, x_min=0, x_max=parent.width,
        roi_id="ROI_VALIDATION_01"
    )
    
    rows, cols = np.indices(roi.values.shape)
    rows_flat = rows.ravel()
    cols_flat = cols.ravel()
    vals_flat = roi.values.ravel()

    # Explicit missing-observation filtering prior to PointCloud build
    if drop_nan:
        valid_mask = np.isfinite(vals_flat)
        rows_flat = rows_flat[valid_mask]
        cols_flat = cols_flat[valid_mask]
        vals_flat = vals_flat[valid_mask]

    coords = roi.transform.to_canonical(rows_flat, cols_flat)
    point_cloud = PointCloud(coordinates=coords, values=vals_flat)

    graph = GraphBuilder(strategy=KNNStrategy(k=k), distance_strategy=EuclideanDistance()).build(point_cloud)
    collection = NeighbourhoodCollection(point_cloud=point_cloud, graph=graph)
    null_engine = NullModelEngine(collection=collection, strategy=GlobalPermutationNull(), seed=seed)
    null_set = null_engine.build_realizations(n_permutations=n_permutations)
    result = MetricsCore.evaluate(collection=collection, null_realizations=null_set)
    
    # --- BIND Cryptographic Fingerprint ---
    config_str = f"k={k}:perms={n_permutations}:seed={seed}:shift_x={shift_x}:shift_y={shift_y}:drop_nan={drop_nan}"
    fp = compute_metric_fingerprint(
        data=data,
        r_obs=result.r_observed,
        z_score=result.z_score,
        p_val=result.p_value,
        config_str=config_str
    )
    
    if hasattr(result, "_fingerprint"):
        result._fingerprint = fp
    else:
        object.__setattr__(result, "fingerprint", fp)
    
    return result, collection, graph, roi

def test_1_constant_field_degenerate_variance():
    """1. Constant Field Test: Verifies zero-variance protection."""
    print("\n[Test 1] Constant Field Test (Degenerate Variance)...")
    constant_data = np.full((16, 16), 5.0)
    result, _, _, _ = run_pipeline(constant_data)
    
    assert result.r_observed == 0.0 or result.status == MetricStatus.DEGENERATE_VARIANCE
    print(f"  -> Handled safely: Status={result.status.value}, R_observed={result.r_observed}")


def test_2_white_noise_field():
    """2. White Noise Test: Deterministic tolerance bounds (R ≈ 0, Z ≈ 0)."""
    print("\n[Test 2] White Noise Field Test...")
    rng = np.random.default_rng(42)
    noise_data = rng.normal(loc=0.0, scale=1.0, size=(16, 16))
    result, _, _, _ = run_pipeline(noise_data, n_permutations=100, seed=42)
    
    assert -0.25 <= result.r_observed <= 0.25, f"Expected R ≈ 0, got {result.r_observed}"
    assert -2.0 <= result.z_score <= 2.0, f"Expected |Z| < 2.0, got {result.z_score}"
    print(f"  -> R_observed: {result.r_observed:.4f}, Z-score: {result.z_score:.2f}, p-value: {result.p_value:.4f}")


def test_3_checkerboard_anti_correlation():
    """3. Checkerboard Test: Validates detection of negative spatial correlation (R < 0)."""
    print("\n[Test 3] Checkerboard Test (Negative Predictability)...")
    row = np.array([1.0, -1.0] * 8)
    checker_data = np.array([row if i % 2 == 0 else -row for i in range(16)])
    result, _, _, _ = run_pipeline(checker_data)
    
    assert result.r_observed < 0.0, f"Expected R < 0 for checkerboard, got {result.r_observed}"
    print(f"  -> R_observed: {result.r_observed:.4f} (Negative spatial correlation confirmed)")


def test_4_scale_invariance_smoothness():
    """4. Scale Invariance Test: Smooth degradation across resolution steps."""
    print("\n[Test 4] Scale Invariance Test...")
    x = np.linspace(0, 10, 32)
    y = np.linspace(0, 10, 32)
    xx, yy = np.meshgrid(x, y)
    smooth_32 = np.sin(xx) + np.cos(yy)
    
    parent = SyntheticParentGrid(smooth_32)
    roi = ROIExtractor.extract(parent=parent, strategy=ROIStrategy.BOUNDING_BOX, y_min=0, y_max=32, x_min=0, x_max=32)
    
    roi_scale2 = ScaleDeriver.derive(roi, target_shape=(16, 16), strategy=MeanPooling())
    roi_scale4 = ScaleDeriver.derive(roi, target_shape=(8, 8), strategy=MeanPooling())
    
    r_scale2, _, _, _ = run_pipeline(roi_scale2.values)
    r_scale4, _, _, _ = run_pipeline(roi_scale4.values)
    
    print(f"  -> R (16x16 downscale): {r_scale2.r_observed:.4f}")
    print(f"  -> R (8x8 downscale)  : {r_scale4.r_observed:.4f}")
    assert r_scale2.r_observed > 0.4
    assert r_scale4.r_observed > 0.2


def test_5_determinism_and_provenance():
    """5. Determinism & Provenance Test: Same seed produces bitwise identical fingerprints & statistics."""
    print("\n[Test 5] Determinism & Provenance Test...")
    rng = np.random.default_rng(101)
    data = rng.normal(size=(16, 16))
    
    res1, _, graph1, roi1 = run_pipeline(data, seed=12345)
    res2, _, graph2, roi2 = run_pipeline(data, seed=12345)
    
    # Provenance assertion
    assert roi1.parent_fingerprint == "PARENT_HASH_99"
    assert hasattr(roi1, "fingerprint") and roi1.fingerprint is not None
    assert res1.fingerprint is not None
    
    # Determinism assertion
    assert res1.fingerprint == res2.fingerprint, "Fingerprints failed deterministic match!"
    assert np.isclose(res1.r_observed, res2.r_observed, atol=1e-12)
    assert np.isclose(res1.z_score, res2.z_score, atol=1e-12)
    print("  -> Determinism verified: Bitwise match for fingerprints and observed metrics.")


def test_6_spatial_translation_invariance():
    """6. Spatial Invariance Test: Translating coordinate space preserves structural metric."""
    print("\n[Test 6] Spatial Translation Invariance Test...")
    x = np.linspace(0, 10, 16)
    y = np.linspace(0, 10, 16)
    xx, yy = np.meshgrid(x, y)
    field = np.sin(xx) + np.cos(yy)
    
    res_orig, _, _, _ = run_pipeline(field, shift_x=0.0, shift_y=0.0)
    res_shift, _, _, _ = run_pipeline(field, shift_x=500.0, shift_y=1200.0)
    
    assert np.isclose(res_orig.r_observed, res_shift.r_observed, atol=1e-9)
    print(f"  -> Original R: {res_orig.r_observed:.6f} | Shifted R: {res_shift.r_observed:.6f}")
    print("  -> Spatial Invariance verified (Coordinate translation does not alter spatial correlation).")


if __name__ == "__main__":
    test_1_constant_field_degenerate_variance()
    test_2_white_noise_field()
    test_3_checkerboard_anti_correlation()
    test_4_scale_invariance_smoothness()
    test_5_determinism_and_provenance()
    test_6_spatial_translation_invariance()
    print("\n✅ PASSED: All TRACEBIND Domain Validation & Architectural Tests Succeeded!")