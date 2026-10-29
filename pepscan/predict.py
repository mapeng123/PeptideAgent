"""Inference interface for PepScan 2.0."""

import torch
import numpy as np

from pepscan.models import ModelD
from pepscan.data import ESM2Encoder
from pepscan.tasks import (
    TASK_NAMES, TASK_TO_INDEX, THRESHOLDS,
    validate_sequence, resolve_tasks,
)


class PepScanPredictor:
    """High-level predictor using Model D (best multi-task model).

    Loads ESM-2 encoder + Model D weights, provides simple predict API.

    Example:
        >>> from pepscan.predict import PepScanPredictor
        >>> pred = PepScanPredictor(weights_path="model_d_final.pt")
        >>> result = pred.predict("KWKLFKKIGAVLKVL", tasks=["AMP", "Toxicity"])
        >>> print(result)
    """

    def __init__(self, weights_path=None, device=None):
        """
        Args:
            weights_path: Path to Model D weights (.pt file).
            device: Device string ('cuda', 'cpu', or None for auto).
        """
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.encoder = ESM2Encoder(device=self.device)
        self.model = ModelD(input_dim=640, num_tasks=14).to(self.device)

        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device,
                                    weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    @torch.no_grad()
    def predict(self, sequence, tasks="all"):
        """Predict bioactivity for a single peptide sequence.

        Args:
            sequence: Amino acid sequence (single-letter codes).
            tasks: "all", a single task name, or list of task names.

        Returns:
            Dict mapping task name → {"probability": float, "prediction": str}
        """
        seq = validate_sequence(sequence)
        task_list = resolve_tasks(tasks)

        embedding = self.encoder.encode_single(seq)
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)

        logits = self.model(x)  # (1, 14)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        results = {}
        for task in task_list:
            idx = TASK_TO_INDEX[task]
            prob = float(probs[idx])
            threshold = THRESHOLDS[task]
            pred = "positive" if prob >= threshold else "negative"
            results[task] = {
                "probability": round(prob, 4),
                "prediction": pred,
            }
        return results

    @torch.no_grad()
    def predict_batch(self, sequences, tasks="all", batch_size=8):
        """Predict bioactivity for multiple sequences.

        Args:
            sequences: List of amino acid sequences.
            tasks: "all", a single task name, or list of task names.
            batch_size: Batch size for ESM-2 encoding.

        Returns:
            List of dicts (same format as predict()).
        """
        clean_seqs = [validate_sequence(s) for s in sequences]
        task_list = resolve_tasks(tasks)

        embeddings = self.encoder.encode(clean_seqs, batch_size=batch_size)
        x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        logits = self.model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_results = []
        for i in range(len(clean_seqs)):
            results = {}
            for task in task_list:
                idx = TASK_TO_INDEX[task]
                prob = float(probs[i, idx])
                threshold = THRESHOLDS[task]
                pred = "positive" if prob >= threshold else "negative"
                results[task] = {
                    "probability": round(prob, 4),
                    "prediction": pred,
                }
            all_results.append(results)

        return all_results

    @torch.no_grad()
    def predict_proba(self, sequence, tasks="all"):
        """Return only probabilities (no threshold-based prediction).

        Args:
            sequence: Amino acid sequence.
            tasks: "all", a single task name, or list of task names.

        Returns:
            Dict mapping task name → probability (float).
        """
        full = self.predict(sequence, tasks=tasks)
        return {task: info["probability"] for task, info in full.items()}
