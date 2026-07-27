# tracebind/stats/null_model.py
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, Tuple, List
from types import MappingProxyType
from tracebind.graph.neighbourhood_collection import LocalNeighborhoodProtocol


class NeighborhoodSelectionStrategy(ABC):
    """Abstract Strategy for selecting neighbor swap targets during localized permutations."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select_partner(self, 
                       node: int, 
                       collection: LocalNeighborhoodProtocol, 
                       available_mask: np.ndarray, 
                       rng: np.random.Generator) -> Optional[int]:
        """Returns an eligible neighbor index for swapping, or None if no valid candidate exists."""
        pass


class UniformNeighborhoodSelection(NeighborhoodSelectionStrategy):
    """Uniformly samples an available neighbor from the node's local neighborhood graph."""
    @property
    def name(self) -> str: return "uniform"

    def select_partner(self, 
                       node: int, 
                       collection: LocalNeighborhoodProtocol, 
                       available_mask: np.ndarray, 
                       rng: np.random.Generator) -> Optional[int]:
        nbr_idxs = collection.neighbour_indices(node)
        valid_candidates = nbr_idxs[available_mask[nbr_idxs]]
        if len(valid_candidates) == 0:
            return None
        return int(rng.choice(valid_candidates))


class NullStrategy(ABC):
    """Abstract Strategy defining a statistical null hypothesis and permutation generator."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def null_hypothesis(self) -> str:
        """Explicit mathematical statement of the null hypothesis being tested."""
        pass

    @abstractmethod
    def generate_single(self, 
                        collection: LocalNeighborhoodProtocol, 
                        rng: np.random.Generator) -> np.ndarray:
        """Generates a single 1D surrogate state vector (shape: n_nodes)."""
        pass


class GlobalPermutationNull(NullStrategy):
    """
    Global Monte Carlo permutation of observation values across fixed spatial coordinates.
    Null Hypothesis: Observed attribute values are globally exchangeable over spatial locations.
    Destroys all spatial autocorrelation while preserving the empirical 1D marginal distribution.
    """
    @property
    def name(self) -> str: return "global_permutation"

    @property
    def null_hypothesis(self) -> str:
        return "Observed attribute values are exchangeable over all spatial coordinates."

    def generate_single(self, 
                        collection: LocalNeighborhoodProtocol, 
                        rng: np.random.Generator) -> np.ndarray:
        base_vals = collection.point_cloud.values
        return rng.permutation(base_vals)


class LocalNeighborhoodPermutation(NullStrategy):
    """
    Spatially constrained permutation using disjoint local node swaps.
    Null Hypothesis: Local organization beyond immediate neighborhood-scale exchangeability is absent.
    Ensures every node participates in at most ONE swap per realization for well-defined perturbation.
    """
    def __init__(self, 
                 swap_fraction: float = 0.5, 
                 selection_strategy: Optional[NeighborhoodSelectionStrategy] = None):
        if not (0.0 < swap_fraction <= 1.0):
            raise ValueError(f"NullModel Error: swap_fraction must be in (0, 1], got {swap_fraction}.")
        self._swap_fraction = swap_fraction
        self._selection_strategy = selection_strategy or UniformNeighborhoodSelection()

    @property
    def name(self) -> str: return "local_neighborhood_permutation"

    @property
    def null_hypothesis(self) -> str:
        return "Attribute values are exchangeable within local graph neighborhoods; coarse-scale structures are preserved."

    def generate_single(self, 
                        collection: LocalNeighborhoodProtocol, 
                        rng: np.random.Generator) -> np.ndarray:
        
        base_vals = collection.point_cloud.values
        n_nodes = collection.n_nodes
        state = base_vals.copy()

        available_mask = np.ones(n_nodes, dtype=bool)
        target_swaps = int(np.floor(n_nodes * self._swap_fraction / 2.0))

        shuffled_nodes = rng.permutation(n_nodes)
        swaps_done = 0

        for node in shuffled_nodes:
            if swaps_done >= target_swaps:
                break
            if not available_mask[node]:
                continue

            partner = self._selection_strategy.select_partner(node, collection, available_mask, rng)
            if partner is not None:
                # Execute single disjoint swap
                state[node], state[partner] = state[partner], state[node]
                available_mask[node] = False
                available_mask[partner] = False
                swaps_done += 1

        return state


class NullRealizationSet:
    """
    Immutable container wrapping surrogate realizations, RNG state, strategy metadata, and provenance.
    Supports both low-memory streaming iterations O(N) and full matrix materialization O(M*N).
    """
    def __init__(self,
                 collection: LocalNeighborhoodProtocol,
                 strategy: NullStrategy,
                 n_permutations: int,
                 seed: Optional[int] = None):
        
        if n_permutations < 1:
            raise ValueError(f"NullModel Error: n_permutations must be >= 1, got {n_permutations}.")

        self._collection = collection
        self._strategy = strategy
        self._n_permutations = n_permutations
        self._seed = seed
        self._n_nodes = collection.n_nodes

        # Provenance Tracking
        provenance = {
            "source_collection_fingerprint": str(getattr(collection, "fingerprint", "UNKNOWN")),
            "strategy_name": strategy.name,
            "null_hypothesis": strategy.null_hypothesis,
            "n_permutations": n_permutations,
            "seed": seed,
            "n_nodes": self._n_nodes,
            "contract_version": "1.1.0"
        }
        self._metadata = MappingProxyType(provenance)

        fp_bytes = f"{strategy.name}_{seed}_{n_permutations}_{self._n_nodes}".encode('utf-8')
        self._fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]

    @property
    def fingerprint(self) -> str: return self._fingerprint
    @property
    def n_permutations(self) -> int: return self._n_permutations
    @property
    def strategy(self) -> NullStrategy: return self._strategy
    @property
    def metadata(self) -> MappingProxyType: return self._metadata

    def iter_surrogates(self) -> Iterator[np.ndarray]:
        """Streams surrogate 1D vectors one by one, keeping memory usage at O(N)."""
        rng = np.random.default_rng(self._seed)
        for _ in range(self._n_permutations):
            surrogate = self._strategy.generate_single(self._collection, rng)
            surrogate.flags.writeable = False
            yield surrogate

    def materialize_matrix(self) -> np.ndarray:
        """Materializes all surrogates into a 2D matrix of shape (n_permutations, n_nodes)."""
        matrix = np.zeros((self._n_permutations, self._n_nodes), dtype=np.float64)
        for i, surrogate in enumerate(self.iter_surrogates()):
            matrix[i] = surrogate
        matrix.flags.writeable = False
        return matrix

    def __repr__(self) -> str:
        return (f"NullRealizationSet(strategy='{self._strategy.name}', "
                f"perms={self._n_permutations}, nodes={self._n_nodes}, fp='{self._fingerprint}')")


class NullModelEngine:
    """Factory engine for initializing NullRealizationSet instances."""
    def __init__(self, 
                 collection: LocalNeighborhoodProtocol, 
                 strategy: Optional[NullStrategy] = None, 
                 seed: Optional[int] = None):
        self._collection = collection
        self._strategy = strategy or GlobalPermutationNull()
        self._seed = seed

    def build_realizations(self, n_permutations: int = 100) -> NullRealizationSet:
        return NullRealizationSet(
            collection=self._collection,
            strategy=self._strategy,
            n_permutations=n_permutations,
            seed=self._seed
        )