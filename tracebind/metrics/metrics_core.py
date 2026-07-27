# tracebind/metrics/metrics_core.py
from tracebind.graph.neighbourhood_collection import LocalNeighborhoodProtocol
from tracebind.stats.null_model import NullRealizationSet
from tracebind.metrics.significance import SignificanceAnalyzer
from tracebind.metrics.metric_result import MetricResult


class MetricsCore:
    """Unified Facade for executing spatial metric evaluation and significance analysis."""

    @staticmethod
    def evaluate(collection: LocalNeighborhoodProtocol, null_realizations: NullRealizationSet) -> MetricResult:
        return SignificanceAnalyzer.evaluate(collection, null_realizations)