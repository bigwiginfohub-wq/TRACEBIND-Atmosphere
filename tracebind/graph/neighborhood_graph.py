# tracebind/graph/neighborhood_graph.py
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, Any, Optional, Tuple, List
from types import MappingProxyType
from scipy.spatial import cKDTree
from tracebind.core.region_of_interest import PointCloud


class DistanceStrategy(ABC):
    """Abstract strategy for distance metric computation."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def p_norm(self) -> float:
        pass


class EuclideanDistance(DistanceStrategy):
    @property
    def name(self) -> str: return "euclidean"
    @property
    def p_norm(self) -> float: return 2.0


class ManhattanDistance(DistanceStrategy):
    @property
    def name(self) -> str: return "manhattan"
    @property
    def p_norm(self) -> float: return 1.0


class NeighborhoodGraph:
    """
    Immutable spatial adjacency graph built over spatial coordinates.
    Decoupled from node observation values, storing topological adjacency,
    distances, and node degrees.
    """
    def __init__(self,
                 point_cloud_id: str,
                 strategy_name: str,
                 search_param: float,
                 distance_strategy: DistanceStrategy,
                 indices_flat: np.ndarray,
                 offsets: np.ndarray,
                 distances_flat: np.ndarray,
                 coordinates: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self._point_cloud_id = str(point_cloud_id)
        self._strategy_name = str(strategy_name)
        self._search_param = float(search_param)
        self._distance_strategy = distance_strategy
        self._n_nodes = len(coordinates)

        # CSR-style topological flat buffers (memory-efficient, no padding)
        self._indices_flat = np.asarray(indices_flat, dtype=np.int64)
        self._indices_flat.flags.writeable = False

        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._offsets.flags.writeable = False

        self._distances_flat = np.asarray(distances_flat, dtype=np.float64)
        self._distances_flat.flags.writeable = False

        self._coordinates = np.asarray(coordinates, dtype=np.float64)
        self._coordinates.flags.writeable = False

        # Precompute integer degree vector
        self._degrees = np.diff(self._offsets)
        self._degrees.flags.writeable = False

        # Metadata Audit
        provenance = {
            "source_point_cloud_id": self._point_cloud_id,
            "strategy": self._strategy_name,
            "search_param": self._search_param,
            "distance_metric": self._distance_strategy.name,
            "n_nodes": self._n_nodes,
            "contract_version": "1.1.0",
            **(metadata or {})
        }
        self._metadata = MappingProxyType(provenance)

        # Strict identity fingerprint hashing topology and coordinates
        fp_bytes = (f"{self._point_cloud_id}_{self._strategy_name}_{self._search_param}_"
                    f"{hashlib.sha256(self._coordinates.tobytes()).hexdigest()[:8]}_"
                    f"{hashlib.sha256(self._indices_flat.tobytes()).hexdigest()[:8]}").encode('utf-8')
        self._fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]

    @property
    def fingerprint(self) -> str: return self._fingerprint
    @property
    def n_nodes(self) -> int: return self._n_nodes
    @property
    def degrees(self) -> np.ndarray: return self._degrees
    @property
    def coordinates(self) -> np.ndarray: return self._coordinates
    @property
    def metadata(self) -> MappingProxyType: return self._metadata

    def neighbours(self, node_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (neighbor_indices, neighbor_distances) for a target node ID."""
        if node_id < 0 or node_id >= self._n_nodes:
            raise IndexError(f"Graph Error: Node ID {node_id} out of bounds [0, {self._n_nodes - 1}].")
        
        start = self._offsets[node_id]
        end = self._offsets[node_id + 1]
        return self._indices_flat[start:end], self._distances_flat[start:end]

    @cached_property
    def mean_degree(self) -> float:
        return float(np.mean(self._degrees))

    @cached_property
    def mean_edge_distance(self) -> float:
        return float(np.mean(self._distances_flat)) if len(self._distances_flat) > 0 else 0.0

    def __repr__(self) -> str:
        return (f"NeighborhoodGraph(strategy='{self._strategy_name}', param={self._search_param}, "
                f"nodes={self._n_nodes}, mean_deg={self.mean_degree:.2f}, "
                f"mean_dist={self.mean_edge_distance:.4f}, fp='{self._fingerprint}')")


class GraphStrategy(ABC):
    """Abstract Strategy interface for building spatial adjacency graphs."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def build_graph(self, point_cloud: PointCloud, distance_strategy: DistanceStrategy) -> NeighborhoodGraph:
        pass


class KNNStrategy(GraphStrategy):
    """k-Nearest Neighbor Graph Strategy using scipy.spatial.cKDTree."""
    def __init__(self, k: int = 8):
        if k <= 1:
            raise ValueError(f"Graph Error: k must be >= 2 (got k={k}).")
        self._k = k

    @property
    def name(self) -> str: return "knn"

    def build_graph(self, point_cloud: PointCloud, distance_strategy: DistanceStrategy) -> NeighborhoodGraph:
        n_pts = point_cloud.n_points
        if n_pts < self._k:
            raise ValueError(f"Graph Error: PointCloud has {n_pts} points, insufficient for k={self._k} graph.")

        tree = cKDTree(point_cloud.coordinates)
        dists, idxs = tree.query(point_cloud.coordinates, k=self._k, p=distance_strategy.p_norm)

        # Exclude self-loop (first column)
        dists_no_self = dists[:, 1:]
        idxs_no_self = idxs[:, 1:]

        indices_flat = idxs_no_self.ravel()
        distances_flat = dists_no_self.ravel()

        offsets = np.arange(0, n_pts * (self._k - 1) + 1, self._k - 1, dtype=np.int64)

        return NeighborhoodGraph(
            point_cloud_id=str(point_cloud.metadata.get("roi_id", "PC-UNKNOWN")),
            strategy_name=self.name,
            search_param=float(self._k),
            distance_strategy=distance_strategy,
            indices_flat=indices_flat,
            offsets=offsets,
            distances_flat=distances_flat,
            coordinates=point_cloud.coordinates,
            metadata={"k_neighbors": self._k}
        )


class RadiusStrategy(GraphStrategy):
    """Fixed interaction radius strategy using scipy.spatial.cKDTree."""
    def __init__(self, radius: float):
        if radius <= 0.0:
            raise ValueError(f"Graph Error: Radius must be strictly positive (got radius={radius}).")
        self._radius = radius

    @property
    def name(self) -> str: return "radius"

    def build_graph(self, point_cloud: PointCloud, distance_strategy: DistanceStrategy) -> NeighborhoodGraph:
        tree = cKDTree(point_cloud.coordinates)
        neighbor_list = tree.query_ball_point(point_cloud.coordinates, r=self._radius, p=distance_strategy.p_norm)

        n_pts = point_cloud.n_points
        offsets = [0]
        indices_list: List[int] = []
        distances_list: List[float] = []

        coords = point_cloud.coordinates

        for i, nbrs in enumerate(neighbor_list):
            # Exclude self-loop
            valid_nbrs = [n for n in nbrs if n != i]
            offsets.append(offsets[-1] + len(valid_nbrs))
            indices_list.extend(valid_nbrs)

            if len(valid_nbrs) > 0:
                diffs = coords[valid_nbrs] - coords[i]
                dists = np.linalg.norm(diffs, ord=distance_strategy.p_norm, axis=1)
                distances_list.extend(dists)

        return NeighborhoodGraph(
            point_cloud_id=str(point_cloud.metadata.get("roi_id", "PC-UNKNOWN")),
            strategy_name=self.name,
            search_param=float(self._radius),
            distance_strategy=distance_strategy,
            indices_flat=np.array(indices_list, dtype=np.int64),
            offsets=np.array(offsets, dtype=np.int64),
            distances_flat=np.array(distances_list, dtype=np.float64),
            coordinates=coords,
            metadata={"interaction_radius": self._radius}
        )


class GraphBuilder:
    """Unified Factory executor for spatial graph construction."""
    def __init__(self, strategy: Optional[GraphStrategy] = None, distance_strategy: Optional[DistanceStrategy] = None):
        self._strategy = strategy or KNNStrategy(k=8)
        self._distance_strategy = distance_strategy or EuclideanDistance()

    def build(self, point_cloud: PointCloud) -> NeighborhoodGraph:
        return self._strategy.build_graph(point_cloud, self._distance_strategy)