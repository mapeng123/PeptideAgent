"""Data loading and preprocessing utilities for PepScan 2.0."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pepscan.tasks import (
    TASK_NAMES, NUM_TASKS, TASK_TO_INDEX,
    ESM2_MODEL, ESM2_EMB_DIM, ESM2_REPR_LAYER,
)


# ============================================================
# ESM-2 Encoder
# ============================================================

class ESM2Encoder:
    """Encode peptide sequences using ESM-2 (esm2_t30_150M_UR50D).

    Produces 640-dim mean-pooled embeddings. First call downloads the
    model (~560MB) to ~/.cache/esm/.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    def _load(self):
        """Lazy-load ESM-2 model."""
        if self._model is not None:
            return
        try:
            import esm
        except ImportError:
            raise ImportError(
                "fair-esm is required. Install: pip install fair-esm"
            )
        self._model, self._alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self._model = self._model.to(self.device)
        self._model.eval()
        self._batch_converter = self._alphabet.get_batch_converter()

    @torch.no_grad()
    def encode(self, sequences, batch_size=8):
        """Encode sequences into ESM-2 mean-pooled embeddings.

        Args:
            sequences: List of amino acid sequences.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (N, 640).
        """
        self._load()
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = self._batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            results = self._model(
                batch_tokens, repr_layers=[ESM2_REPR_LAYER],
                return_contacts=False,
            )
            token_reprs = results["representations"][ESM2_REPR_LAYER]

            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                # tokens: BOS + seq + EOS → take [1:1+seq_len]
                emb = token_reprs[j, 1:1 + seq_len].mean(dim=0)
                all_embeddings.append(emb.cpu().numpy())

        return np.array(all_embeddings, dtype=np.float32)

    def encode_single(self, sequence):
        """Encode a single sequence. Returns (640,) array."""
        return self.encode([sequence])[0]


# ============================================================
# Data Loading from Pre-computed Embeddings
# ============================================================

def load_task_embeddings(embed_dir, task):
    """Load pre-computed ESM-2 embeddings for one task.

    Expects file: {embed_dir}/{task}_esm2.npz with keys:
        train_X, train_y, val_X, val_y, test_X, test_y

    Args:
        embed_dir: Directory containing {task}_esm2.npz files.
        task: Task name.

    Returns:
        Dict with keys: train_X, train_y, val_X, val_y, test_X, test_y
    """
    fp = os.path.join(embed_dir, f'{task}_esm2.npz')
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Embedding file not found: {fp}")

    data = np.load(fp)
    return {
        'train_X': data['train_X'].astype(np.float32),
        'train_y': data['train_y'].astype(np.float32),
        'val_X':   data['val_X'].astype(np.float32),
        'val_y':   data['val_y'].astype(np.float32),
        'test_X':  data['test_X'].astype(np.float32),
        'test_y':  data['test_y'].astype(np.float32),
    }


def load_all_embeddings(embed_dir, tasks=None):
    """Load pre-computed embeddings for all tasks.

    Args:
        embed_dir: Directory containing {task}_esm2.npz files.
        tasks: List of task names. Default: all 14 tasks.

    Returns:
        Dict mapping task name → embedding dict.
    """
    if tasks is None:
        tasks = TASK_NAMES
    ml_data = {}
    for task in tasks:
        ml_data[task] = load_task_embeddings(embed_dir, task)
        n_tr = len(ml_data[task]['train_y'])
        print(f"  {task:20s}  train={n_tr}")
    return ml_data


def load_traditional_features(embed_dir, task):
    """Load pre-computed AAC+DPC features for one task.

    Expects file: {embed_dir}/{task}_traditional.npz
    """
    fp = os.path.join(embed_dir, f'{task}_traditional.npz')
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Feature file not found: {fp}")

    data = np.load(fp)
    return {
        'train_X': data['train_X'].astype(np.float32),
        'train_y': data['train_y'].astype(np.float32),
        'val_X':   data['val_X'].astype(np.float32),
        'val_y':   data['val_y'].astype(np.float32),
        'test_X':  data['test_X'].astype(np.float32),
        'test_y':  data['test_y'].astype(np.float32),
    }


# ============================================================
# Multi-task Dataset
# ============================================================

class MultiTaskDataset(Dataset):
    """Dataset for multi-task models (D, E).

    Each sample has:
        - embedding: (640,) ESM-2 embedding
        - labels: (num_tasks,) with -1 for missing tasks
        - mask: (num_tasks,) 1.0 for observed tasks, 0.0 for missing
    """

    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings: np.ndarray (N, 640).
            labels: np.ndarray (N, num_tasks), -1 for unobserved.
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
        self.mask = (self.labels >= 0).float()
        # Replace -1 with 0 in labels (masked out anyway)
        self.labels = self.labels.clamp(min=0)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.mask[idx]


def build_multitask_data(ml_data, embed_dir=None, data_dir=None):
    """Build multi-label dataset from per-task embeddings.

    Merges all 14 tasks into a single multi-label dataset where each
    peptide is labeled for all tasks it appears in (unobserved = -1).

    Requires either:
        - ml_data['_multitask'] pre-built, OR
        - embed_dir + data_dir to build from scratch

    Args:
        ml_data: Per-task embedding dict from load_all_embeddings().
        embed_dir: Directory with ESM-2 npz files (for sequence mapping).
        data_dir: Directory with raw CSV files (for multi-label construction).

    Returns:
        Dict with keys: train, val, test — each containing
        'embeddings' and 'labels' arrays.
    """
    # If already built, return directly
    if '_multitask' in ml_data:
        return ml_data['_multitask']

    # Build from scratch using CSV + embeddings
    if embed_dir is None or data_dir is None:
        raise ValueError(
            "Need either ml_data['_multitask'] or embed_dir + data_dir"
        )

    # Map sequences to embeddings
    seq_to_emb = {}
    for task in TASK_NAMES:
        if task not in ml_data:
            continue
        for split in ['train', 'val', 'test']:
            csv_fp = os.path.join(data_dir, f'{task}_{split}.csv')
            if not os.path.exists(csv_fp):
                continue
            df = pd.read_csv(csv_fp)
            col_map = {}
            for c in df.columns:
                if 'seq' in c.lower():
                    col_map[c] = 'sequence'
            if col_map:
                df = df.rename(columns=col_map)
            if 'sequence' not in df.columns:
                continue
            df['sequence'] = df['sequence'].astype(str).str.strip()
            embs = ml_data[task][f'{split}_X']
            if len(embs) != len(df):
                continue
            for i, seq in enumerate(df['sequence']):
                if seq not in seq_to_emb:
                    seq_to_emb[seq] = embs[i]

    # Load raw CSVs for multi-label mapping
    raw_splits = {'train': [], 'val': [], 'test': []}
    for task in TASK_NAMES:
        for split in ['train', 'val', 'test']:
            fp = os.path.join(data_dir, f'{task}_{split}.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            col_map = {}
            for c in df.columns:
                if 'seq' in c.lower():
                    col_map[c] = 'sequence'
                elif c.lower() in ('label', 'class', 'target'):
                    col_map[c] = 'label'
            if col_map:
                df = df.rename(columns=col_map)
            if 'sequence' not in df.columns or 'label' not in df.columns:
                continue
            df['sequence'] = df['sequence'].astype(str).str.strip()
            df['label'] = df['label'].astype(int)
            df['task_id'] = TASK_TO_INDEX[task]
            raw_splits[split].append(df[['sequence', 'label', 'task_id']])

    for split in raw_splits:
        if raw_splits[split]:
            raw_splits[split] = pd.concat(raw_splits[split],
                                          ignore_index=True)
        else:
            raw_splits[split] = pd.DataFrame()

    # Build multi-label records
    mt_data = {}
    for split in ['train', 'val', 'test']:
        df = raw_splits[split]
        if df.empty:
            mt_data[split] = {
                'embeddings': np.array([]).reshape(0, ESM2_EMB_DIM),
                'labels': np.array([]).reshape(0, NUM_TASKS),
            }
            continue

        records = []
        for seq, group in df.groupby('sequence'):
            if seq not in seq_to_emb:
                continue
            labels = np.full(NUM_TASKS, -1.0, dtype=np.float32)
            for _, row in group.iterrows():
                labels[int(row['task_id'])] = row['label']
            records.append({
                'embedding': seq_to_emb[seq],
                'labels': labels,
            })

        if records:
            mt_data[split] = {
                'embeddings': np.stack([r['embedding'] for r in records]
                                       ).astype(np.float32),
                'labels': np.stack([r['labels'] for r in records]),
            }
        else:
            mt_data[split] = {
                'embeddings': np.array([]).reshape(0, ESM2_EMB_DIM),
                'labels': np.array([]).reshape(0, NUM_TASKS),
            }
        print(f"  {split}: {len(records)} samples")

    return mt_data
