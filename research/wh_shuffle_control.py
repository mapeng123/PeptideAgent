#!/usr/bin/env python3
"""Composition-preserved shuffle control on the original T40 wound-healing funnel.

Input is the 14 mechanism-gate survivors from funnel_t40_top14.csv plus the
parent. The original wound-healing adaptation data and T40 initialization are
used. Twenty shuffles per peptide are scored by the same five-seed ensemble.
Eligible candidates have WH z >= 1.5 and WH seed SD <= 0.15. They are ranked
by WH ensemble score, with optional 3-mer Jaccard family de-duplication.
"""
import copy
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
    n = len(yt)
    bs = min(32, n)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(10):
        model.train()
        perm = torch.randperm(n, generator=generator, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            scores = model(Xt[idx])
            labels = yt[idx]
            pos = scores[labels > 0.5]
            neg = scores[labels < 0.5]
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


funnel = pd.read_csv(f"{OUT}/funnel_t40_top14.csv")
seqs = ["SLDSPARGAPWP"] + funnel["seq"].tolist()
names = ["parent"] + [f"candidate_{i:02d}" for i in range(1, len(funnel) + 1)]
full = list(seqs)
owner = list(range(len(seqs)))
shuffle_id = [0] * len(seqs)
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
cap = int(1e9)
rng_adapt = np.random.RandomState(0)
pi = rng_adapt.choice(int((y == 1).sum()), min(cap, int((y == 1).sum())), replace=False)
ni = rng_adapt.choice(int((y == 0).sum()), min(cap, int((y == 0).sum())), replace=False)
Xp, Xn = X[y == 1][pi], X[y == 0][ni]
Xtr = np.concatenate([Xp, Xn])
ytr = np.array([1] * len(Xp) + [0] * len(Xn), dtype=np.float32)
print(f"WH adaptation set: {len(ytr)} ({int(ytr.sum())} positives)", flush=True)

per_seed = []
for seed in SEEDS:
    theta = MetaModel().to(device)
    weight_path = f"{BASE}/output/nm_meta_v4/T40c/meta_seed{seed}.pt"
    theta.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
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

zscore = np.zeros(len(seqs))
pvalue = np.ones(len(seqs))
for i in range(len(seqs)):
    shuffled = score_all[owner == i][1:]
    mu, sigma = shuffled.mean(), shuffled.std()
    if sigma < 1e-4:
        zscore[i], pvalue[i] = 0.0, 1.0
    else:
        zscore[i] = (raw[i] - mu) / sigma
        pvalue[i] = (1 + (shuffled >= raw[i]).sum()) / (1 + N_SHUFFLE)

result = pd.DataFrame({
    "name": names,
    "seq": seqs,
    "WH": raw,
    "WH_seed_sd": seed_sd,
    "WH_shuffle_mean": [score_all[owner == i][1:].mean() for i in range(len(seqs))],
    "WH_shuffle_sd": [score_all[owner == i][1:].std() for i in range(len(seqs))],
    "WH_z": zscore,
    "WH_p": pvalue,
    "pass_z": zscore >= 1.5,
    "pass_sd": seed_sd <= 0.15,
})
result = result.merge(funnel[["seq", "pctAI", "pctAO", "TOX", "HEM", "pctTOX", "pctHEM"]],
                      on="seq", how="left")
result["pass_shuffle_gate"] = result["pass_z"] & result["pass_sd"]
result["WH_rank_all14"] = result.loc[1:, "WH"].rank(ascending=False, method="min")
result.to_csv(f"{OUT}/wh_shuffle_controls.csv", index=False)

all_scores = pd.DataFrame({
    "name": [names[i] for i in owner],
    "parent_seq": [seqs[i] for i in owner],
    "seq": full,
    "shuffle_id": shuffle_id,
    "WH": score_all,
})
all_scores.to_csv(f"{OUT}/wh_shuffle_all_scores.csv", index=False)

eligible = result.iloc[1:][result.iloc[1:]["pass_shuffle_gate"]].copy()
eligible = eligible.sort_values("WH", ascending=False)
picked = []
for _, row in eligible.iterrows():
    if any(jaccard(row["seq"], chosen) >= 0.5 for chosen in picked):
        continue
    picked.append(row["seq"])
    if len(picked) >= 10:
        break

result["selected_top10"] = result["seq"].isin(picked)
result.to_csv(f"{OUT}/wh_shuffle_controls.csv", index=False)
top10 = eligible[eligible["seq"].isin(picked)].copy()
top10["new_rank"] = range(1, len(top10) + 1)
top10.to_csv(f"{OUT}/wh_shuffle_top10.csv", index=False)
summary = {
    "n_input_funnel": len(funnel),
    "n_shuffle_per_peptide": N_SHUFFLE,
    "z_min": 1.5,
    "sd_max": 0.15,
    "p_filter": None,
    "n_pass_shuffle_gate": int(eligible.shape[0]),
    "n_selected": len(picked),
    "selected_in_wh_rank_order": picked,
    "parent": result.iloc[0].to_dict(),
}
with open(f"{OUT}/wh_shuffle_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)
print(result[["name", "seq", "WH", "WH_seed_sd", "WH_z", "WH_p",
              "pass_shuffle_gate", "selected_top10"]].to_string(index=False), flush=True)
print("SELECTED", picked, flush=True)
print("WH_SHUFFLE_CONTROL_DONE", flush=True)
