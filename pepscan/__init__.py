"""PepScan 1.0 — Multi-task peptide bioactivity prediction.

Five-model comparative framework:
  A: Traditional features (AAC+DPC, 420d) + LightGBM
  B: ESM-2 embeddings (640d) + LightGBM
  C: ESM-2 embeddings (640d) + Single-task MLP
  D: ESM-2 embeddings (640d) + Multi-task shared backbone
  E: ESM-2 embeddings (640d) + Multi-task + Transformer cross-task interaction
"""

__version__ = "1.0.0"

from pepscan.models import ModelA, ModelB, ModelC, ModelD, ModelE
from pepscan.tasks import TASK_NAMES, TASK_TO_INDEX, THRESHOLDS

__all__ = [
    "ModelA", "ModelB", "ModelC", "ModelD", "ModelE",
    "TASK_NAMES", "TASK_TO_INDEX", "THRESHOLDS",
]
