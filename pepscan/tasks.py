"""Task definitions, thresholds, and constants for PepScan 2.0."""

# 14 prediction tasks in canonical order (matches model head indices)
TASK_NAMES = [
    'AMP', 'Neuropeptide', 'Anti-inflammatory', 'Toxicity',
    'Anticancer', 'Antiviral', 'AntiHIV', 'ACE-inhibitory',
    'DPP-IV', 'Antioxidant', 'CPP', 'Immunoactive',
    'Thrombolytic', 'Osteogenic'
]

NUM_TASKS = len(TASK_NAMES)

# Task index mapping (task name → head index)
TASK_TO_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}
INDEX_TO_TASK = {i: name for name, i in TASK_TO_INDEX.items()}

# Youden's J optimal thresholds (5-seed mean, Model D)
THRESHOLDS = {
    'AMP': 0.540,
    'Neuropeptide': 0.500,
    'Anti-inflammatory': 0.370,
    'Toxicity': 0.450,
    'Anticancer': 0.560,
    'Antiviral': 0.480,
    'AntiHIV': 0.420,
    'ACE-inhibitory': 0.440,
    'DPP-IV': 0.590,
    'Antioxidant': 0.490,
    'CPP': 0.520,
    'Immunoactive': 0.470,
    'Thrombolytic': 0.560,
    'Osteogenic': 0.250,
}

# Training set sizes (v2.1 hybrid-split)
N_TRAIN = {
    'AMP': 38423, 'Neuropeptide': 8060, 'Anti-inflammatory': 7244,
    'Toxicity': 7095, 'Anticancer': 4996, 'Antiviral': 2364,
    'AntiHIV': 1350, 'ACE-inhibitory': 2203, 'DPP-IV': 796,
    'Antioxidant': 634, 'CPP': 1270, 'Immunoactive': 564,
    'Thrombolytic': 124, 'Osteogenic': 96
}

# LightGBM hyperparameters (shared by Model A and Model B)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# DL training hyperparameters (shared by Model C, D, E)
DL_HYPERPARAMS = {
    'max_epochs': 50,
    'batch_size': 64,
    'lr': 1e-3,
    'patience': 8,
    'dropout': 0.3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
}

# Sample size groups
LARGE_TASKS = ['AMP', 'Neuropeptide', 'Anti-inflammatory', 'Toxicity']
MEDIUM_TASKS = ['Anticancer', 'Antiviral', 'AntiHIV', 'ACE-inhibitory', 'CPP']
SMALL_TASKS = ['DPP-IV', 'Antioxidant', 'Immunoactive']
TINY_TASKS = ['Thrombolytic', 'Osteogenic']

# Amino acid alphabet
VALID_AAs = set('ACDEFGHIKLMNPQRSTVWY')
AA_LIST = sorted(VALID_AAs)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Dipeptide indices
DPC_LIST = [a + b for a in AA_LIST for b in AA_LIST]
DPC_TO_IDX = {dp: i for i, dp in enumerate(DPC_LIST)}

# ESM-2 configuration
ESM2_MODEL = "esm2_t30_150M_UR50D"
ESM2_EMB_DIM = 640
ESM2_REPR_LAYER = 30

# 5 seeds for multi-run experiments
SEEDS = [42, 7, 13, 15, 2026]


def validate_sequence(seq):
    """Validate a peptide sequence. Returns cleaned uppercase sequence.

    Raises:
        ValueError: If sequence is empty or contains invalid amino acids.
    """
    seq = seq.strip().upper()
    seq = ''.join(seq.split())  # remove whitespace
    if not seq:
        raise ValueError("Empty sequence")
    invalid = set(seq) - VALID_AAs
    if invalid:
        raise ValueError(f"Invalid amino acids in sequence: {invalid}")
    return seq


def resolve_tasks(tasks):
    """Resolve task specification to a list of task names.

    Args:
        tasks: "all", a single task name (str), or list of task names.

    Returns:
        List of task names.

    Raises:
        ValueError: If any task name is not recognized.
    """
    if tasks == "all":
        return TASK_NAMES[:]
    if isinstance(tasks, str):
        tasks = [tasks]
    for t in tasks:
        if t not in TASK_TO_INDEX:
            raise ValueError(
                f"Unknown task '{t}'. Available tasks: {TASK_NAMES}"
            )
    return tasks
