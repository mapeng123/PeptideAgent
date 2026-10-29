"""Model definitions for PepScan 2.0 — five-model comparative framework.

Model A: Traditional features (AAC+DPC, 420d) + LightGBM
Model B: ESM-2 embeddings (640d) + LightGBM
Model C: ESM-2 embeddings (640d) + Single-task MLP
Model D: ESM-2 embeddings (640d) + Multi-task shared backbone
Model E: ESM-2 embeddings (640d) + Multi-task + Transformer cross-task interaction
"""

import numpy as np
import torch
import torch.nn as nn

from pepscan.tasks import (
    AA_LIST, AA_TO_IDX, DPC_TO_IDX,
    LGB_PARAMS, DL_HYPERPARAMS, TASK_NAMES, NUM_TASKS,
)


# ============================================================
# Model A: Traditional Features + LightGBM
# ============================================================

class ModelA:
    """Traditional features (AAC + DPC, 420d) + LightGBM classifier.

    Per-task binary classifier. AAC (20d amino acid composition) and
    DPC (400d dipeptide composition) are handcrafted sequence features
    fed into a gradient-boosted decision tree ensemble.

    This is a sklearn-style wrapper: fit / predict / predict_proba.
    """

    def __init__(self, task, params=None):
        """
        Args:
            task: Task name (e.g. 'AMP', 'Toxicity').
            params: LightGBM hyperparameters. Defaults to LGB_PARAMS.
        """
        import lightgbm as lgb

        self.task = task
        self.params = {**(params or LGB_PARAMS)}
        self.model = lgb.LGBMClassifier(**self.params)

    @staticmethod
    def extract_features(sequences):
        """Extract AAC (20d) + DPC (400d) = 420d features from sequences.

        Args:
            sequences: List of amino acid sequences (uppercase).

        Returns:
            np.ndarray of shape (N, 420).
        """
        n = len(sequences)
        aac = np.zeros((n, 20), dtype=np.float32)
        dpc = np.zeros((n, 400), dtype=np.float32)

        for i, seq in enumerate(sequences):
            slen = len(seq)
            if slen == 0:
                continue
            # AAC
            aa_counts = np.zeros(20, dtype=np.float32)
            for aa in seq:
                if aa in AA_TO_IDX:
                    aa_counts[AA_TO_IDX[aa]] += 1
            aac[i] = aa_counts / slen
            # DPC
            if slen >= 2:
                dipep_counts = np.zeros(400, dtype=np.float32)
                for j in range(slen - 1):
                    p = seq[j:j + 2]
                    if p in DPC_TO_IDX:
                        dipep_counts[DPC_TO_IDX[p]] += 1
                dpc[i] = dipep_counts / (slen - 1)

        return np.hstack([aac, dpc])

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LightGBM classifier.

        Args:
            X_train: Training features (N, 420).
            y_train: Training labels (N,).
            X_val: Validation features for early stopping (optional).
            y_val: Validation labels (optional).
        """
        import lightgbm as lgb

        eval_set = None
        callbacks = [lgb.log_evaluation(0)]
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(50, verbose=False))

        self.model.fit(X_train, y_train, eval_set=eval_set,
                       callbacks=callbacks)
        return self

    def predict_proba(self, X):
        """Predict probabilities for the positive class.

        Args:
            X: Feature matrix (N, 420).

        Returns:
            np.ndarray of shape (N,) — positive class probabilities.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Predict binary labels.

        Args:
            X: Feature matrix (N, 420).
            threshold: Classification threshold.

        Returns:
            np.ndarray of shape (N,) — binary predictions.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        """Save model to LightGBM text file."""
        self.model.booster_.save_model(path)

    def load(self, path):
        """Load model from LightGBM text file."""
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=path)


# ============================================================
# Model B: ESM-2 Embeddings + LightGBM
# ============================================================

class ModelB:
    """ESM-2 embeddings (640d) + LightGBM classifier.

    Same LightGBM backbone as Model A, but using ESM-2 protein language
    model embeddings instead of handcrafted features. Isolates the
    contribution of learned sequence representations.
    """

    def __init__(self, task, params=None):
        """
        Args:
            task: Task name.
            params: LightGBM hyperparameters. Defaults to LGB_PARAMS.
        """
        import lightgbm as lgb

        self.task = task
        self.params = {**(params or LGB_PARAMS)}
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train on ESM-2 embedding features.

        Args:
            X_train: ESM-2 embeddings (N, 640).
            y_train: Labels (N,).
            X_val: Validation embeddings (optional).
            y_val: Validation labels (optional).
        """
        import lightgbm as lgb

        eval_set = None
        callbacks = [lgb.log_evaluation(0)]
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(50, verbose=False))

        self.model.fit(X_train, y_train, eval_set=eval_set,
                       callbacks=callbacks)
        return self

    def predict_proba(self, X):
        """Predict probabilities.

        Args:
            X: ESM-2 embeddings (N, 640).

        Returns:
            np.ndarray of shape (N,) — positive class probabilities.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        self.model.booster_.save_model(path)

    def load(self, path):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=path)


# ============================================================
# Model C: ESM-2 + Single-task MLP
# ============================================================

class ModelC(nn.Module):
    """ESM-2 embeddings + Single-task MLP.

    Architecture:
        640 → 256 (BN + ReLU + Dropout) → 64 (BN + ReLU + Dropout) → 1

    Each task gets an independent MLP — no parameter sharing.
    This is the single-task deep learning baseline.
    """

    def __init__(self, input_dim=640, hidden_dims=None, dropout=0.3):
        """
        Args:
            input_dim: ESM-2 embedding dimension (default 640).
            hidden_dims: Hidden layer sizes. Default [256, 64].
            dropout: Dropout rate.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass. Returns raw logits (before sigmoid).

        Args:
            x: ESM-2 embeddings (B, 640).

        Returns:
            Tensor of shape (B,) — logits.
        """
        return self.net(x).squeeze(-1)


# ============================================================
# Model D: Multi-task Shared Backbone
# ============================================================

class ModelD(nn.Module):
    """ESM-2 embeddings + Multi-task shared backbone.

    Architecture:
        Shared backbone: 640 → 256 → 128 (BN + ReLU + Dropout)
        Task heads: 128 → 64 → 1 (per task, with ReLU + Dropout)

    All 14 tasks share the backbone but have independent prediction
    heads. This is the core multi-task model — transfer happens
    implicitly through the shared representation.
    """

    def __init__(self, input_dim=640, hidden_dim=256, shared_dim=128,
                 num_tasks=14, dropout=0.3):
        """
        Args:
            input_dim: ESM-2 embedding dimension.
            hidden_dim: First hidden layer size.
            shared_dim: Shared representation dimension.
            num_tasks: Number of prediction tasks.
            dropout: Dropout rate.
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Same head architecture as Model E for fair comparison
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            ) for _ in range(num_tasks)
        ])

    def forward(self, x, task_idx=None):
        """Forward pass.

        Args:
            x: ESM-2 embeddings (B, 640).
            task_idx: If given, return logits for a single task (B,).
                      If None, return all-task logits (B, num_tasks).

        Returns:
            Tensor of shape (B,) if task_idx given, else (B, num_tasks).
        """
        h = self.backbone(x)
        if task_idx is not None:
            return self.heads[task_idx](h).squeeze(-1)
        return torch.cat([head(h) for head in self.heads], dim=1)


# ============================================================
# Model E: Multi-task + Cross-Task Interaction
# ============================================================

class ModelE(nn.Module):
    """ESM-2 embeddings + Multi-task shared backbone + Transformer interaction.

    Architecture:
        Shared backbone: 640 → 256 → 128 (BN + ReLU + Dropout)
        Task embeddings: Learnable (num_tasks, 128) added to backbone output
        Transformer: 2-layer TransformerEncoder over task slots
        Task heads: 128 → 64 → 1 (per task)

    The Transformer allows explicit cross-task information exchange.
    Task embeddings differentiate task slots so the Transformer can
    attend to task-specific patterns.
    """

    def __init__(self, input_dim=640, hidden_dim=256, shared_dim=128,
                 num_tasks=14, dropout=0.3, n_heads=8, n_layers=2,
                 ff_dim=512):
        """
        Args:
            input_dim: ESM-2 embedding dimension.
            hidden_dim: First hidden layer size.
            shared_dim: Shared representation dimension (also Transformer d_model).
            num_tasks: Number of prediction tasks.
            dropout: Dropout rate.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer layers.
            ff_dim: Feed-forward dimension in Transformer.
        """
        super().__init__()
        self.shared_dim = shared_dim
        self.num_tasks = num_tasks

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable task embeddings
        self.task_embeddings = nn.Parameter(
            torch.randn(num_tasks, shared_dim) * 0.02
        )

        # Transformer cross-task interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Same head architecture as Model D for fair comparison
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            ) for _ in range(num_tasks)
        ])

    def forward(self, x, task_idx=None):
        """Forward pass.

        Args:
            x: ESM-2 embeddings (B, 640).
            task_idx: Not used in training (always outputs all tasks).
                      Kept for API consistency with Model D.

        Returns:
            Tensor of shape (B, num_tasks) — raw logits.
        """
        B = x.size(0)
        h = self.backbone(x)  # (B, shared_dim)

        # Add task embeddings: (B, 1, shared_dim) + (1, num_tasks, shared_dim)
        h_expanded = h.unsqueeze(1).expand(-1, self.num_tasks, -1)
        task_embs = self.task_embeddings.unsqueeze(0).expand(B, -1, -1)
        task_repr = h_expanded + task_embs  # (B, num_tasks, shared_dim)

        # Cross-task interaction
        interacted = self.interaction(task_repr)  # (B, num_tasks, shared_dim)

        # Per-task predictions
        logits = torch.cat([
            self.heads[i](interacted[:, i, :]) for i in range(self.num_tasks)
        ], dim=1)  # (B, num_tasks)

        return logits
