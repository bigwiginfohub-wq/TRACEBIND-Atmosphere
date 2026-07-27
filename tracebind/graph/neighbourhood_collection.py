# tracebind/graph/neighbourhood_collection.py
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Protocol, runtime_checkable
from types import MappingProxyType
from tracebind.core.region_of_interest import PointCloud
from tracebind.graph.neighborhood_graph import NeighborhoodGraph


@runtime_checkable
class LocalNeighborhoodProtocol(Protocol):
    """Minimal protocol defining neighbor access interface for NullModelEngine and Metrics Core."""
    @property
    def n_nodes(self) -> int: ...
    def neighbour_indices(self, node_id: int) -> np.ndarray: ...
    def neighbour_values(self, node_id: int) -> np.ndarray: ...
    def neighbour_distances(self, node_id: int) -> np.ndarray: ...


class LocalStatisticStrategy(ABC):
    """Abstract Strategy interface for localized neighborhood statistics."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, cached_neighbour_values: List[np.ndarray]) -> np.ndarray:
        pass


class LocalMeanStatistic(LocalStatisticStrategy):
    @property
    def name(self) -> str: return "mean"

    def compute(self, cached_neighbour_values: List[np.ndarray]) -> np.ndarray:
        n_nodes = len(cached_neighbour_values)
        results = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            vals = cached_neighbour_values[i]
            results[i] = np.mean(vals) if len(vals) > 0 else 0.0
        return results


class LocalVarianceStatistic(LocalStatisticStrategy):
    @property
    def name(self) -> str: return "variance"

    def compute(self, cached_neighbour_values: List[np.ndarray]) -> np.ndarray:
        n_nodes = len(cached_neighbour_values)
        results = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            vals = cached_neighbour_values[i]
            results[i] = np.var(vals) if len(vals) > 1 else 0.0
        return results


class LocalMedianAbsoluteDeviationStatistic(LocalStatisticStrategy):
    """Calculates Median Absolute Deviation from Median (Robust Spread)."""
    @property
    def name(self) -> str: return "median_absolute_deviation"

    def compute(self, cached_neighbour_values: List[np.ndarray]) -> np.ndarray:
        n_nodes = len(cached_neighbour_values)
        results = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            vals = cached_neighbour_values[i]
            if len(vals) > 0:
                med = np.median(vals)
                results[i] = np.median(np.abs(vals - med))
            else:
                results[i] = 0.0
        return results


class NeighbourhoodCollection(LocalNeighborhoodProtocol):
    """
    Immutable working set and statistical computation engine connecting PointCloud with NeighborhoodGraph.
    Caches topological slices and provides pluggable spatial statistic evaluation.
    """
    def __init__(self, point_cloud: PointCloud, graph: NeighborhoodGraph):
        if point_cloud.n_points != graph.n_nodes:
            raise ValueError(
                f"Collection Error: PointCloud length ({point_cloud.n_points}) "
                f"does not match Graph node count ({graph.n_nodes})."
            )

        self._point_cloud = point_cloud
        self._graph = graph
        self._n_nodes = graph.n_nodes

        # Pre-extract single-pass working set cache to avoid repetitive indexing loops
        self._cached_indices: List[np.ndarray] = []
        self._cached_distances: List[np.ndarray] = []
        self._cached_values: List[np.ndarray] = []

        pc_vals = self._point_cloud.values

        for i in range(self._n_nodes):
            idxs, dists = self._graph.neighbours(i)
            self._cached_indices.append(idxs)
            self._cached_distances.append(dists)
            self._cached_values.append(pc_vals[idxs])

        # Registry for calculated statistical fields
        self._computed_stats: Dict[str, np.ndarray] = {}

        # Provenance Tracking
        provenance = {
            "source_point_cloud_id": str(point_cloud.metadata.get("roi_id", "PC-UNKNOWN")),
            "source_graph_fingerprint": graph.fingerprint,
            "n_nodes": self._n_nodes,
            "contract_version": "1.1.0"
        }
        self._metadata = MappingProxyType(provenance)

        # Cryptographic Identity Fingerprint
        fp_bytes = f"{graph.fingerprint}_{self._point_cloud.metadata.get('roi_id', 'PC')}".encode('utf-8')
        self._fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]

    @property
    def fingerprint(self) -> str: return self._fingerprint
    @property
    def n_nodes(self) -> int: return self._n_nodes
    @property
    def point_cloud(self) -> PointCloud: return self._point_cloud
    @property
    def graph(self) -> NeighborhoodGraph: return self._graph
    @property
    def metadata(self) -> MappingProxyType: return self._metadata

    # --- LocalNeighborhoodProtocol Implementations ---
    def neighbour_indices(self, node_id: int) -> np.ndarray:
        return self._cached_indices[node_id]

    def neighbour_values(self, node_id: int) -> np.ndarray:
        return self._cached_values[node_id]

    def neighbour_distances(self, node_id: int) -> np.ndarray:
        return self._cached_distances[node_id]

    # --- Pluggable Statistic Execution ---
    def compute(self, strategy: LocalStatisticStrategy) -> np.ndarray:
        """
        Computes or retrieves a cached localized neighborhood statistic field.
        """
        stat_name = strategy.name
        if stat_name not in self._computed_stats:
            stat_values = strategy.compute(self._cached_values)
            stat_values.flags.writeable = False
            self._computed_stats[stat_name] = stat_values
        return self._computed_stats[stat_name]

    def __repr__(self) -> str:
        return (f"NeighbourhoodCollection(nodes={self._n_nodes}, "
                f"cached_stats={list(self._computed_stats.keys())}, fp='{self._fingerprint}')")