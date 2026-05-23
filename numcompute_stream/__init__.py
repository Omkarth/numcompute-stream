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
from .preprocessing import Imputer, OneHotEncoder, StandardScaler
from .tree import HoeffdingTreeClassifier

__all__ = [
    "ADWIN",
    "Accuracy",
    "HoeffdingTreeClassifier",
    "Imputer",
    "OneHotEncoder",
    "OnlineRandomForestClassifier",
    "PrecisionRecallF1",
    "RollingMetric",
    "StandardScaler",
    "StreamingAUC",
    "StreamingConfusionMatrix",
]
