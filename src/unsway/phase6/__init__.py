"""Fresh-holdout causal steering experiment."""

from unsway.phase6.activations import extract_multilayer_activations
from unsway.phase6.baseline import run_phase6_baseline
from unsway.phase6.config import Phase6Config, load_phase6_config
from unsway.phase6.dataset import build_phase6_dataset, write_protocol_manifest

__all__ = [
    "Phase6Config",
    "build_phase6_dataset",
    "extract_multilayer_activations",
    "load_phase6_config",
    "run_phase6_baseline",
    "write_protocol_manifest",
]
