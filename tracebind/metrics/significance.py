# tracebind/metrics/significance.py
import numpy as np
from tracebind.graph.neighbourhood_collection import LocalNeighborhoodProtocol, LocalMeanStatistic
from tracebind.stats.null_model import NullRealizationSet
from tracebind.metrics.r_metric import RMetric
from tracebind.metrics.metric_result import MetricResult, MetricStatus


class SignificanceAnalyzer:
    """
    Evaluates observed metric significance against NullRealizationSet streaming surrogates
    using O(1) space Welford online accumulation.
    """

    @classmethod
    def evaluate(cls, 
                 collection: LocalNeighborhoodProtocol, 
                 null_realizations: NullRealizationSet) -> MetricResult:
        
        # 1. Evaluate Observed State
        if not hasattr(collection, "point_cloud"):
            raise AttributeError("SignificanceAnalyzer Error: Collection must expose point_cloud.")

        obs_vals = collection.point_cloud.values
        
        if hasattr(collection, "compute"):
            local_means = collection.compute(LocalMeanStatistic())
        else:
            n_nodes = collection.n_nodes
            local_means = np.zeros(n_nodes, dtype=np.float64)
            for i in range(n_nodes):
                local_means[i] = np.mean(collection.neighbour_values(i))

        r_obs, is_degenerate = RMetric.compute(obs_vals, local_means)

        warnings_list = []
        status = MetricStatus.SUCCESS

        if is_degenerate:
            status = MetricStatus.DEGENERATE_VARIANCE
            warnings_list.append("Observed vector exhibits zero variance across all spatial nodes.")

        # 2. Welford Online Accumulation for O(1) Memory Null Moment Calculations
        M = null_realizations.n_permutations
        n_count = 0
        mean = 0.0
        M2 = 0.0
        n_extreme = 0

        for surrogate_vals in null_realizations.iter_surrogates():
            # Vectorized local mean calculation for surrogate realization
            n_nodes = len(surrogate_vals)
            surr_local_means = np.zeros(n_nodes, dtype=np.float64)
            for i in range(n_nodes):
                nbr_idxs = collection.neighbour_indices(i)
                if len(nbr_idxs) > 0:
                    surr_local_means[i] = np.mean(surrogate_vals[nbr_idxs])
                else:
                    surr_local_means[i] = surrogate_vals[i]

            r_null, _ = RMetric.compute(surrogate_vals, surr_local_means)

            # Empirical upper-tail count
            if r_null >= r_obs:
                n_extreme += 1

            # Welford Online Variance Update Step
            n_count += 1
            delta = r_null - mean
            mean += delta / n_count
            delta2 = r_null - mean
            M2 += delta * delta2

        r_null_mean = float(mean)
        r_null_variance = float(M2 / (n_count - 1)) if n_count > 1 else 0.0
        r_null_std = float(np.sqrt(r_null_variance))

        # Standardized Z-score calculation
        epsilon = 1e-12
        z_score = float((r_obs - r_null_mean) / (r_null_std + epsilon))
        p_value = float((n_extreme + 1) / (M + 1))

        col_fp = getattr(collection, "fingerprint", "COL_UNKNOWN")
        null_fp = null_realizations.fingerprint

        return MetricResult(
            r_observed=r_obs,
            z_score=z_score,
            p_value=p_value,
            r_null_mean=r_null_mean,
            r_null_std=r_null_std,
            metric_name=RMetric.NAME,
            metric_version=RMetric.VERSION,
            status=status,
            collection_fingerprint=col_fp,
            null_fingerprint=null_fp,
            warnings=warnings_list
        )