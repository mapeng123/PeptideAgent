#!/usr/bin/env python3
"""Sensitivity analysis of composition-control gates in the original T40 funnel.

Candidates are all developable variants passing both mechanism percentiles at
0.3, nesting the 0.4 and 0.5 gates. The same fixed WH shuffle gate (z >= 1.5,
seed SD <= 0.15) is applied at every mechanism threshold without retuning it.
"""
import copy
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = "/root/autodl-tmp/pepscan"
OUT = f"{BASE}/output/wh_blind"
GRID = f"{BASE}/output/newfunc_nm/WoundHealing_uniprot"
SEEDS = [42, 7, 13, 15, 2026]
N_SHUFFLE = 20
sys.path.insert(0, BASE)
from pepscan.data import ESM2Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.shared_backbone(x)).squeeze(-1)


def adapt(theta, Xtr, ytr, seed):
    model = copy.deepcopy(theta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    n, bs = len(yt), min(32, len(yt))
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(10):
        model.train()
        perm = torch.randperm(n, generator=generator, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            scores, labels = model(Xt[idx]), yt[idx]
            pos, neg = scores[labels > 0.5], scores[labels < 0.5]
            if len(pos) == 0 or len(neg) == 0:
                continue
            opt.zero_grad()
            torch.relu(1.0 - (pos[:, None] - neg[None, :])).mean().backward()
            opt.step()
    model.eval()
    return model


def kmer_set(seq, k=3):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard(a, b):
    aa, bb = kmer_set(a), kmer_set(b)
    return len(aa & bb) / max(1, len(aa | bb))


def pick_diverse(frame, n=10):
    picked = []
    for _, row in frame.sort_values("WH", ascending=False).iterrows():
        if any(jaccard(row["seq"], chosen) >= 0.5 for chosen in picked):
            continue
        picked.append(row["seq"])
        if len(picked) >= n:
            break
    return picked


X0, TARGET = "SLDSPARGAPWP", "SLDSTHTHAPWP"
POS = [i for i, (a, b) in enumerate(zip(X0, TARGET)) if a != b]
AA = "ACDEFGHIKLMNPQRSTVWY"
variants = []
for combo in itertools.product(AA, repeat=4):
    seq = list(X0)
    for pos, aa in zip(POS, combo):
        seq[pos] = aa
    variants.append("".join(seq))
scores = np.load(f"{OUT}/funnel_t40_scores.npz")
candidate_idx = np.array([
    i for i in scores["sub"]
    if scores["PCT_AI"][i] >= 0.3 and scores["PCT_AO"][i] >= 0.3
], dtype=int)
seqs = [X0] + [variants[i] for i in candidate_idx]
names = ["parent"] + [f"variant_{i}" for i in candidate_idx]
print(f"nested mechanism pool: {len(candidate_idx)} candidates", flush=True)

full, owner, shuffle_id = list(seqs), list(range(len(seqs))), [0] * len(seqs)
rng = np.random.RandomState(7)
for i, seq in enumerate(seqs):
    for j in range(N_SHUFFLE):
        chars = list(seq)
        rng.shuffle(chars)
        full.append("".join(chars))
        owner.append(i)
        shuffle_id.append(j + 1)
owner = np.asarray(owner)
print(f"{len(seqs)} peptides -> {len(full)} sequences with controls", flush=True)
encoder = ESM2Encoder(device=device)
embeddings = encoder.encode(full, batch_size=64)
Xt = torch.tensor(embeddings, dtype=torch.float32, device=device)

pool = pd.read_csv(f"{GRID}/pool.csv")
z = np.load(f"{GRID}/emb.npz")
X, y = z["pool"], pool.label.values.astype(np.float32)
rng_adapt = np.random.RandomState(0)
pi = rng_adapt.choice(int((y == 1).sum()), int((y == 1).sum()), replace=False)
ni = rng_adapt.choice(int((y == 0).sum()), int((y == 0).sum()), replace=False)
Xp, Xn = X[y == 1][pi], X[y == 0][ni]
Xtr = np.concatenate([Xp, Xn])
ytr = np.array([1] * len(Xp) + [0] * len(Xn), dtype=np.float32)
print(f"WH adaptation set: {len(ytr)} ({int(ytr.sum())} positives)", flush=True)

per_seed = []
for seed in SEEDS:
    theta = MetaModel().to(device)
    theta.load_state_dict(torch.load(
        f"{BASE}/output/nm_meta_v4/T40c/meta_seed{seed}.pt",
        map_location=device, weights_only=False))
    torch.manual_seed(seed * 1000 + 3 * 7919)
    np.random.seed(seed * 1000 + 3 * 7919)
    model = adapt(theta, Xtr, ytr, seed * 5 + 1)
    with torch.no_grad():
        per_seed.append(torch.sigmoid(model(Xt)).cpu().numpy())
    print(f"seed {seed} scored", flush=True)
    del model, theta
per_seed = np.stack(per_seed)
score_all = per_seed.mean(axis=0)
raw = score_all[:len(seqs)]
seed_sd = per_seed[:, :len(seqs)].std(axis=0)

zscore, pvalue, shmean, shsd = [], [], [], []
for i in range(len(seqs)):
    shuffled = score_all[owner == i][1:]
    mu, sigma = shuffled.mean(), shuffled.std()
    shmean.append(mu)
    shsd.append(sigma)
    if sigma < 1e-4:
        zscore.append(0.0)
        pvalue.append(1.0)
    else:
        zscore.append((raw[i] - mu) / sigma)
        pvalue.append((1 + (shuffled >= raw[i]).sum()) / (1 + N_SHUFFLE))

variant_index = [-1] + candidate_idx.tolist()
result = pd.DataFrame({
    "name": names,
    "variant_index": variant_index,
    "seq": seqs,
    "WH": raw,
    "WH_seed_sd": seed_sd,
    "WH_shuffle_mean": shmean,
    "WH_shuffle_sd": shsd,
    "WH_z": zscore,
    "WH_p": pvalue,
    "pctAI": [np.nan] + scores["PCT_AI"][candidate_idx].tolist(),
    "pctAO": [np.nan] + scores["PCT_AO"][candidate_idx].tolist(),
    "TOX": [np.nan] + scores["S_TOX"][candidate_idx].tolist(),
    "HEM": [np.nan] + scores["S_HEM"][candidate_idx].tolist(),
    "pctTOX": [np.nan] + scores["PCT_TOX"][candidate_idx].tolist(),
    "pctHEM": [np.nan] + scores["PCT_HEM"][candidate_idx].tolist(),
})
result["pass_shuffle_gate"] = (result["WH_z"] >= 1.5) & (result["WH_seed_sd"] <= 0.15)
result.to_csv(f"{OUT}/wh_shuffle_expanded_all.csv", index=False)

summary = {"n_shuffle": N_SHUFFLE, "z_min": 1.5, "sd_max": 0.15, "thresholds": {}}
for threshold in (0.5, 0.4, 0.3):
    eligible = result.iloc[1:][
        (result.iloc[1:]["pctAI"] >= threshold)
        & (result.iloc[1:]["pctAO"] >= threshold)
        & result.iloc[1:]["pass_shuffle_gate"]
    ].copy().sort_values("WH", ascending=False)
    picked = pick_diverse(eligible, 10)
    top = eligible[eligible["seq"].isin(picked)].copy()
    top["new_rank"] = range(1, len(top) + 1)
    top.to_csv(f"{OUT}/wh_shuffle_top10_mechanism_{threshold:.1f}.csv", index=False)
    summary["thresholds"][str(threshold)] = {
        "n_mechanism_survivors": int(((result.iloc[1:]["pctAI"] >= threshold)
                                      & (result.iloc[1:]["pctAO"] >= threshold)).sum()),
        "n_pass_shuffle": int(len(eligible)),
        "n_after_diversity": len(picked),
        "selected": picked,
    }
    print(f"threshold {threshold:.1f}: mechanism={summary['thresholds'][str(threshold)]['n_mechanism_survivors']} "
          f"shuffle_pass={len(eligible)} selected={len(picked)}", flush=True)
    print(top[["seq", "WH", "WH_z", "WH_p", "WH_seed_sd", "pctAI", "pctAO"]].to_string(index=False), flush=True)
with open(f"{OUT}/wh_shuffle_expanded_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)
print("WH_EXPANDED_SHUFFLE_DONE", flush=True)
