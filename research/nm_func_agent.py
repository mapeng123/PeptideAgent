#!/usr/bin/env python3
"""Generic deployment agent builder + de novo scorer. Usage: python nm_func_agent.py <Func>"""
import os, sys, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = "/root/autodl-tmp/pepscan"
sys.path.insert(0, BASE)
import nm_v4clean as G

SEEDS = [42, 7, 13, 15, 2026]
FUNC = sys.argv[1]
ADIR = f"{BASE}/output/nm_agents/{FUNC}"
os.makedirs(ADIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class MetaModel(nn.Module):
    def __init__(s):
        super().__init__()
        s.shared_backbone = nn.Sequential(
            nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        s.head = nn.Linear(64, 1)
    def forward(s, x):
        return s.head(s.shared_backbone(x)).squeeze(-1)


def adapt(theta, Xtr, ytr, seed):
    m = copy.deepcopy(theta)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    n, bs = len(ytr), min(32, len(ytr))
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(10):
        m.train()
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            sc = m(Xt[idx]); yb = yt[idx]
            sp, sn = sc[yb > 0.5], sc[yb < 0.5]
            if len(sp) == 0 or len(sn) == 0:
                continue
            opt.zero_grad()
            torch.relu(1.0 - (sp[:, None] - sn[None, :])).mean().backward()
            opt.step()
    m.eval()
    return m


pos = G.load_positives(FUNC)
neg = G.uniprot_negatives(pos, seed=0)
print(f"pos={len(pos)} neg={len(neg)}", flush=True)
from pepscan.data import ESM2Encoder
enc = ESM2Encoder(device=device)
cache = f"{ADIR}/deploy_emb.npz"
if os.path.exists(cache):
    Xall = np.load(cache)["X"]
else:
    Xall = enc.encode(pos + neg, batch_size=64)
    np.savez(cache, X=Xall)
yall = np.array([1] * len(pos) + [0] * len(neg))
print("embeddings ready", Xall.shape, flush=True)

for s in SEEDS:
    wp = f"{ADIR}/agent_seed{s}.pt"
    if os.path.exists(wp):
        continue
    base = MetaModel().to(device)
    base.load_state_dict(torch.load(f"{BASE}/output/nm_meta_v4/T40c/meta_seed{s}.pt",
                                    map_location=device, weights_only=False))
    base.eval()
    torch.manual_seed(s * 1000 + 7919)
    agent = adapt(base, Xall, yall, s * 5 + 1)
    torch.save({k: v.cpu() for k, v in agent.state_dict().items()}, wp)
    print(f"agent seed {s} saved", flush=True)

# calibration on training sequences
sc = []
for s in SEEDS:
    m = MetaModel().to(device)
    m.load_state_dict(torch.load(f"{ADIR}/agent_seed{s}.pt", map_location=device,
                                 weights_only=False))
    m.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xall, dtype=torch.float32, device=device)
        sc.append(torch.sigmoid(m(Xt)).cpu().numpy())
sc = np.mean(sc, axis=0)
calib = {"n_pos": len(pos), "n_neg": len(neg),
         "pos_score_mean": float(np.mean(sc[:len(pos)])),
         "pos_score_median": float(np.median(sc[:len(pos)])),
         "neg_score_mean": float(np.mean(sc[len(pos):]))}
json.dump(calib, open(f"{ADIR}/calibration.json", "w"), indent=1)
print("calibration", calib, flush=True)

# score cached de novo embeddings
emb_file = f"{BASE}/data_denovo/embeddings.npy"
seqs = [l.strip().upper() for l in open(f"{BASE}/data_denovo/peptides_unique.txt") if l.strip()]
X = np.load(emb_file)
Xt = torch.tensor(X, dtype=torch.float32, device=device)
per_seed = []
for s in SEEDS:
    m = MetaModel().to(device)
    m.load_state_dict(torch.load(f"{ADIR}/agent_seed{s}.pt", map_location=device,
                                 weights_only=False))
    m.eval()
    with torch.no_grad():
        per_seed.append(torch.sigmoid(m(Xt)).cpu().numpy())
per_seed = np.stack(per_seed)
score = per_seed.mean(axis=0)
out = pd.DataFrame({"seq": seqs, f"{FUNC}_score": score, f"{FUNC}_seed_sd": per_seed.std(axis=0)})
out[f"{FUNC}_rank"] = out[f"{FUNC}_score"].rank(ascending=False, method="min").astype(int)
out.to_csv(f"{BASE}/data_denovo/denovo_{FUNC}_scores.csv", index=False)
print(FUNC, "top10:", flush=True)
print(out.nlargest(10, f"{FUNC}_score").to_string(index=False), flush=True)
print(FUNC + "_DONE", flush=True)
