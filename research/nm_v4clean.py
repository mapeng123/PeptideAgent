#!/usr/bin/env python3
"""Clean validation grid rebuild + re-evaluation for NM V4.

12 validation functions (WoundHealing excluded -> case study):
  - cluster-level split (mmseqs easy-cluster 30% id, cov 0.8), test_frac 0.2
  - POST-SPLIT PURGE: drop any test sequence with >=30% identity hit vs pool
    (easy-search, cov-mode 1, -c 0.8, alignment-mode 3, -s 7.5) -> strict disjoint
  - UniProt-fragment negatives 1:1 (same protocol as old _uniprot grids)

Eval per function:
  single  - SingleMLP from scratch (BCE, early stop on 15% val split)
  frozen  - deployed backbone frozen + new head
  meta_T* - adapt() ranking-loss from saved tier meta weights (T10/T20/T20cap200/T30/T40)
  N levels 10/25/50/100/200/400/all; meta: 5 seeds x 3 reps; single/frozen: 5 seeds x 1

Usage:
  python nm_v4clean.py build
  python nm_v4clean.py eval
"""
import os, sys, json, subprocess, copy
import numpy as np
import pandas as pd

BASE = "/root/autodl-tmp/pepscan"
MMSEQS = "/root/autodl-tmp/mmseqs/bin/mmseqs"
NEWFUNC_DIR = f"{BASE}/data/newfunc"
OUT_DIR = f"{BASE}/output/nm_v4clean"
WEIGHTS_DIR = f"{BASE}/web/weights"
TIERS = ["T10", "T20", "T20cap200", "T30", "T40", "T40c"]
SEEDS = [42, 7, 13, 15, 2026]
REPS = [1, 2, 3]
N_LEVELS = [10, 25, 50, 100, 200, 400, "all"]
TEST_FRAC = 0.2
FUNCS = ["Anti-Gram-neg", "Antifungal", "Hemolytic", "Antibiofilm", "DPPIV",
         "Antioxidant", "Immunoactive", "Antimalarial", "Insecticidal",
         "Thrombolytic", "SARSCoV2", "Osteogenic"]

sys.path.insert(0, BASE)


def load_positives(func):
    with open(os.path.join(NEWFUNC_DIR, f"clean_{func}.txt")) as f:
        return sorted({l.strip().upper() for l in f if l.strip()})


def uniprot_negatives(pos, seed=0, extra=0):
    rng = np.random.RandomState(seed)
    prots, cur = [], []
    with open(os.path.join(NEWFUNC_DIR, "uniprot_reviewed.fasta")) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    prots.append("".join(cur))
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        prots.append("".join(cur))
    VALID = set("ACDEFGHIKLMNPQRSTVWY")
    posset = set(pos)
    out, used = [], set()
    lens = [len(s) for s in pos] + [20] * extra
    for L in lens:
        for _ in range(60):
            p = prots[rng.randint(len(prots))]
            if len(p) < L:
                continue
            i = rng.randint(len(p) - L + 1)
            frag = p[i:i + L]
            if set(frag) <= VALID and frag not in posset and frag not in used:
                used.add(frag)
                out.append(frag)
                break
    return out


def mmseqs_cluster(seqs, workdir):
    os.makedirs(workdir, exist_ok=True)
    fa = os.path.join(workdir, "all.fa")
    with open(fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    out = os.path.join(workdir, "clu")
    subprocess.run([MMSEQS, "easy-cluster", fa, out, os.path.join(workdir, "tmp"),
                    "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1"],
                   check=True, capture_output=True)
    clu = {}
    with open(out + "_cluster.tsv") as f:
        for line in f:
            rep, mem = line.strip().split("\t")
            clu[mem] = rep
    return [clu[f"s{i}"] for i in range(len(seqs))]


def cluster_split(df, test_frac=TEST_FRAC, seed=0):
    test_clus = []
    for cls in [1, 0]:
        sub = df[df.label == cls]
        sizes = sub.groupby("clu").size()
        other = set(df[df.label != cls]["clu"])
        sizes = sizes[~sizes.index.isin(other)]
        sizes = sizes.sample(frac=1, random_state=seed).sort_values(ascending=False)
        target = max(2, int(round(len(sub) * test_frac)))
        n = 0
        for clu, k in sizes.items():
            if n >= target:
                break
            test_clus.append(clu)
            n += k
    is_test = df["clu"].isin(test_clus)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def purge_test(pool, test, workdir):
    """Drop test seqs with >=30% identity hit vs pool (strict disjoint)."""
    os.makedirs(workdir, exist_ok=True)
    with open(f"{workdir}/q.fa", "w") as f:
        for i, s in enumerate(test["seq"]):
            f.write(f">q{i}\n{s}\n")
    with open(f"{workdir}/t.fa", "w") as f:
        for i, s in enumerate(sorted(set(pool["seq"]))):
            f.write(f">t{i}\n{s}\n")
    subprocess.run([MMSEQS, "easy-search", f"{workdir}/q.fa", f"{workdir}/t.fa",
                    f"{workdir}/h.tsv", f"{workdir}/tmp",
                    "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1",
                    "--alignment-mode", "3", "-s", "7.5"],
                   check=True, capture_output=True)
    hits = set()
    if os.path.exists(f"{workdir}/h.tsv"):
        for line in open(f"{workdir}/h.tsv"):
            hits.add(line.split("\t")[0])
    keep = [i for i in range(len(test)) if f"q{i}" not in hits]
    return test.iloc[keep].reset_index(drop=True), len(hits)


def build():
    for func in FUNCS:
        fdir = os.path.join(OUT_DIR, func)
        if os.path.exists(f"{fdir}/test.csv"):
            print(f"[{func}] exists, skip", flush=True)
            continue
        pos = load_positives(func)
        neg = uniprot_negatives(pos, seed=0, extra=50)
        seqs = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)
        df = pd.DataFrame({"seq": seqs, "label": labels})
        df["clu"] = mmseqs_cluster(seqs, os.path.join(fdir, "mmseqs"))
        pool, test = cluster_split(df)
        test, npur = purge_test(pool, test, os.path.join(fdir, "purge"))
        os.makedirs(fdir, exist_ok=True)
        pool.to_csv(f"{fdir}/pool.csv", index=False)
        test.to_csv(f"{fdir}/test.csv", index=False)
        print(f"[{func}] total pos={len(pos)} -> pool pos={int(pool.label.sum())} "
              f"test pos={int(test.label.sum())} neg={int((1-test.label).sum())} "
              f"(purged {npur})", flush=True)
    print("BUILD_DONE", flush=True)


def evaluate():
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from pepscan.data import ESM2Encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K_EPOCHS, LR_IN, MARGIN = 10, 1e-3, 1.0

    class SingleMLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(
                nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1))
        def forward(s, x):
            return s.net(x).squeeze(-1)

    class DeployedD(nn.Module):
        def __init__(s, num_tasks=14):
            super().__init__()
            s.shared_backbone = nn.Sequential(
                nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
            s.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])
            s.new_head = nn.Linear(64, 1)
        def forward(s, x):
            return s.new_head(s.shared_backbone(x)).squeeze(-1)

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
        m = copy.deepcopy(theta).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=LR_IN, weight_decay=1e-4)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr, dtype=torch.float32, device=device)
        n, bs = len(ytr), min(32, len(ytr))
        g = torch.Generator(device=device).manual_seed(seed)
        for _ in range(K_EPOCHS):
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
                torch.relu(MARGIN - (sp[:, None] - sn[None, :])).mean().backward()
                opt.step()
        m.eval()
        return m

    def train_model(model, Xtr, ytr, Xva, yva, lr=1e-3, freeze_bn_eval=False):
        model.to(device)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr, dtype=torch.float32, device=device)
        Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
        best_auc, best_state, bad = -1, None, 0
        n, bs = len(ytr), min(32, len(ytr))
        for epoch in range(100):
            model.train()
            if freeze_bn_eval:
                for m_ in model.modules():
                    if isinstance(m_, nn.BatchNorm1d):
                        m_.eval()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                lossf(model(Xt[idx]), yt[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv)).cpu().numpy()
            auc = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
            if auc > best_auc:
                best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        return model

    # load meta weights per tier per seed once
    meta_w = {}
    for t in TIERS:
        for s in SEEDS:
            p = f"{BASE}/output/nm_meta_v4/{t}/meta_seed{s}.pt"
            if os.path.exists(p):
                mm = MetaModel().to(device)
                mm.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                mm.eval()
                meta_w[(t, s)] = mm
    print(f"loaded {len(meta_w)} meta weights", flush=True)

    for func in FUNCS:
        fdir = os.path.join(OUT_DIR, func)
        pool = pd.read_csv(f"{fdir}/pool.csv")
        test = pd.read_csv(f"{fdir}/test.csv")
        cache = f"{fdir}/emb.npz"
        if os.path.exists(cache):
            z = np.load(cache)
            emb_pool, emb_test = z["pool"], z["test"]
        else:
            enc = ESM2Encoder(device=device)
            emb_pool = enc.encode(pool["seq"].tolist(), batch_size=32)
            emb_test = enc.encode(test["seq"].tolist(), batch_size=32)
            np.savez(cache, pool=emb_pool, test=emb_test)
        y_test = test["label"].values
        Xte = torch.tensor(emb_test, dtype=torch.float32, device=device)
        emb_pos = emb_pool[(pool.label == 1).values]
        emb_neg = emb_pool[(pool.label == 0).values]
        n_pos, n_neg = len(emb_pos), len(emb_neg)

        rp = f"{fdir}/results.json"
        results = json.load(open(rp)) if os.path.exists(rp) else {}

        for N in N_LEVELS:
            n_use = n_pos if N == "all" else min(N, n_pos)
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pi = rng.choice(n_pos, n_use, replace=False)
                ni = rng.choice(n_neg, min(n_use, n_neg), replace=False)
                X = np.concatenate([emb_pos[pi], emb_neg[ni]])
                y = np.array([1] * len(pi) + [0] * len(ni))
                n_tr = int(len(y) * 0.85)
                g = rng.permutation(len(y))
                tr, va = g[:n_tr], g[n_tr:]
                if len(np.unique(y[va])) < 2:
                    tr = g[:n_tr - 4]; va = g[n_tr - 4:n_tr]
                Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

                key = f"N{N}_seed{seed}"
                if key not in results:
                    out = {}
                    m = train_model(SingleMLP(), Xtr, ytr, Xva, yva)
                    with torch.no_grad():
                        out["single"] = float(roc_auc_score(y_test, torch.sigmoid(m(Xte)).cpu().numpy()))
                    wp = f"{WEIGHTS_DIR}/seed{seed}.pt"
                    if os.path.exists(wp):
                        m2 = DeployedD().to(device)
                        m2.load_state_dict(torch.load(wp, map_location=device, weights_only=False), strict=False)
                        for p_ in m2.shared_backbone.parameters():
                            p_.requires_grad = False
                        m2 = train_model(m2, Xtr, ytr, Xva, yva, freeze_bn_eval=True)
                        with torch.no_grad():
                            out["frozen"] = float(roc_auc_score(y_test, torch.sigmoid(m2(Xte)).cpu().numpy()))
                    out["n_train"] = int(len(ytr))
                    results[key] = out
                    json.dump(results, open(rp, "w"), indent=1)
                    print(f"[{func}] {key} single={out['single']:.3f}", flush=True)

                # meta tiers, 3 reps
                for t in TIERS:
                    if (t, seed) not in meta_w:
                        continue
                    for rep in REPS:
                        mkey = f"{key}_rep{rep}_{t}"
                        if mkey in results:
                            continue
                        torch.manual_seed(seed * 1000 + rep * 17 + 3 * 7919)
                        mm = adapt(meta_w[(t, seed)], Xtr, ytr, seed * 5 + rep)
                        with torch.no_grad():
                            sc = torch.sigmoid(mm(Xte)).cpu().numpy()
                        results[mkey] = {"meta": float(roc_auc_score(y_test, sc))}
                        del mm
                    json.dump(results, open(rp, "w"), indent=1)
                print(f"[{func}] {key} meta tiers done", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if sys.argv[1] == "build":
        build()
    else:
        evaluate()
