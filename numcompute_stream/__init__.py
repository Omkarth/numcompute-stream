"""numcompute_stream — streaming machine learning framework."""

__version__ = "0.1.0"

from .drift import ADWIN
from .ensemble import OnlineRandomForestClassifier
from .metrics import (
    Accuracy,
    PrecisionRecallF1,
    RollingMetric,
    StreamingAUC,
    StreamingConfusionMatrix,
)
from .pipeline import Pipeline
from .preprocessing import Imputer, OneHotEncoder, StandardScaler
from .stream import StreamTrainer
from .tree import HoeffdingTreeClassifier
from .visualise import (
    compare_models,
    plot_confusion_matrix,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

__all__ = [
    "ADWIN",
    "Accuracy",
    "HoeffdingTreeClassifier",
    "Imputer",
    "OneHotEncoder",
    "OnlineRandomForestClassifier",
    "Pipeline",
    "PrecisionRecallF1",
    "RollingMetric",
    "StandardScaler",
    "StreamingAUC",
    "StreamingConfusionMatrix",
    "StreamTrainer",
    "compare_models",
    "plot_confusion_matrix",
    "plot_metric_over_time",
    "plot_predictions_vs_ground_truth",
]
