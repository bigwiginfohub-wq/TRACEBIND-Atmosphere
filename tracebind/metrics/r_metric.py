# tracebind/metrics/r_metric.py
import numpy as np
from typing import Tuple


class RMetric:
    """
    Stateless mathematical kernel for Localized Predictability Ratio (R).
    """
    NAME = "LocalizedPredictabilityRatio"
    VERSION = "1.0.0"

    @classmethod
    def compute(cls, values: np.ndarray, local_means: np.ndarray) -> Tuple[float, bool]:
        """
        Calculates R_observed. Returns (r_value, is_degenerate_variance).
        """
        global_mean = np.mean(values)
        total_ss = np.sum((values - global_mean) ** 2)

        if total_ss == 0.0:
            return 0.0, True  # Degenerate constant variance field

        residual_ss = np.sum((values - local_means) ** 2)
        r_val = 1.0 - (residual_ss / total_ss)
        return float(r_val), False