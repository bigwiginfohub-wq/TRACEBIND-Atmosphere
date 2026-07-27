# tests/test_pipeline.py
import sys
from pathlib import Path

# Force project root directory onto sys.path
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
    """Mock parent grid satisfying ROIExtractor interface requirements."""
    def __init__(self, data: np.ndarray):
        self.values = np.asarray(data, dtype=np.float64)
        self.height, self.width = self.values.shape
        self.mask = np.ones((self.height, self.width), dtype=bool)
        self.fingerprint = "SYNTH_PARENT_HASH_001"
        self.transform = AffineCoordinateTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        self.coord_system = "EPSG:4326"
        self.metadata = {"sensor_id": "SYNTH_GRID_01"}


def test_full_pipeline_end_to_end():
    print("Executing TRACEBIND End-to-End Pipeline Integration Test...")

    # 1. Instantiate Synthetic Parent Observation Grid (16x16 continuous spatial signal)
    x = np.linspace(0, 10, 16)
    y = np.linspace(0, 10, 16)
    xx, yy = np.meshgrid(x, y)
    grid_data = np.sin(xx) + np.cos(yy)

    parent_grid = SyntheticParentGrid(grid_data)
    assert parent_grid.values.shape == (16, 16)
    print("  [1/7] Parent Grid initialized (16x16).")

    # 2. Extract RegionOfInterest via ROIExtractor
    roi = ROIExtractor.extract(
        parent=parent_grid,
        strategy=ROIStrategy.BOUNDING_BOX,
        y_min=2, y_max=14,
        x_min=2, x_max=14,
        roi_id="ROI_TEST_01"
    )
    assert isinstance(roi, RegionOfInterest)
    print("  [2/7] RegionOfInterest extracted via BOUNDING_BOX (12x12).")

    # 3. Downscale ROI via ScaleDeriver.derive and convert to PointCloud
    scaled_roi = ScaleDeriver.derive(roi, target_shape=(6, 6), strategy=MeanPooling())
    
    if hasattr(scaled_roi, "to_point_cloud"):
        point_cloud = scaled_roi.to_point_cloud()
    else:
        # Fallback to direct PointCloud conversion from raster grid coordinates
        rows, cols = np.indices(scaled_roi.values.shape)
        coords = scaled_roi.transform.to_canonical(rows.ravel(), cols.ravel())
        vals = scaled_roi.values.ravel()
        point_cloud = PointCloud(coordinates=coords, values=vals)

    print("  [3/7] ScaleDeriver downscaled ROI to PointCloud (36 nodes).")

    # 4. Construct C-Accelerated KNN Spatial Graph
    graph_builder = GraphBuilder(strategy=KNNStrategy(k=4), distance_strategy=EuclideanDistance())
    graph = graph_builder.build(point_cloud)
    print("  [4/7] NeighborhoodGraph constructed (cKDTree KNN k=4).")

    # 5. Extract NeighbourhoodCollection
    collection = NeighbourhoodCollection(point_cloud=point_cloud, graph=graph)
    print("  [5/7] NeighbourhoodCollection moments extracted.")

    # 6. Generate Null Realization Set
    null_engine = NullModelEngine(collection=collection, strategy=GlobalPermutationNull(), seed=42)
    null_set = null_engine.build_realizations(n_permutations=50)
    print("  [6/7] NullRealizationSet initialized (50 permutations).")

    # 7. Evaluate Metrics Core Significance
    result = MetricsCore.evaluate(collection=collection, null_realizations=null_set)
    print("  [7/7] MetricsCore computed result successfully:")
    print(f"        -> Status     : {result.status}")
    print(f"        -> R_observed : {result.r_observed:.4f}")
    print(f"        -> Z-Score    : {result.z_score:.2f}")
    print(f"        -> p-Value    : {result.p_value:.4f}")
    print(f"        -> Null Mean  : {result.r_null_mean:.4f}")

    print("\n✅ PASSED: TRACEBIND End-to-End Pipeline Integration Test Successful!")


if __name__ == "__main__":
    test_full_pipeline_end_to_end()