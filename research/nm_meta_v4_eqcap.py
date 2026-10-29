#!/usr/bin/env python3
"""Reptile meta-learning V4: 40-task scaling ladder (T10/T20/T30/T40).

Same protocol as nm_meta_v3_rank.py (ranking-loss Reptile, definitive
meta_test), but the practice pool expands to 40 functions:
  9 big tasks + 10 mined (v3) + 21 new (SATPdb/DRAMP/GRAMPA, decontaminated
  vs the 18 validation functions at 30% identity).
GRAMPA species tasks use measured-inactive negatives (MIC >= 100 uM) topped
up with UniProt fragments to 1:1; SATPdb/DRAMP tasks use UniProt fragments.

Tiers are nested subsets by positive-pool size (descending).
META_EPISODES = 100 * n_practice_tasks (episode budget per task constant).

Env:
  TIER=T10|T20|T30|T40   (default T40)
  CAPN=400|200|133|100   cap EVERY practice task at CAPN pos+neg (equal-volume ladder)

Output: output/nm_meta_v4/{TIER}[cap200]/meta_seed{s}.pt, {func}/
"""
import os, sys, json, glob, copy, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

BASE = "/root/autodl-tmp/pepscan"
SPLIT_DIR = f"{BASE}/data/hybrid_split"
FOUND_DIR = f"{BASE}/output/nm_foundation"
GRID_DIR = f"{BASE}/output/newfunc_nm"
E2_CACHE = f"{BASE}/output/e2/emb_cache.npz"
NEG_EMB = f"{BASE}/output/abce_unineg/neg_emb.npz"
NEWFUNC_DIR = f"{BASE}/data/newfunc"
META_TASK_DIR = f"{BASE}/data/newfunc_meta"
V4_POS_DIR = f"{BASE}/output/v4_decon"          # decon_<name>.txt
V4_NEG_DIR = f"{BASE}/data/v4_candidates_neg"   # GRAMPA_<sp>_neg.txt
MMSEQS = "/root/autodl-tmp/mmseqs/bin/mmseqs"

TIER = os.environ.get("TIER", "T40")
CAPN = int(os.environ.get("CAPN", "0"))  # cap EVERY practice task at CAPN pos + CAPN neg (eqcap ladder)
DROP_TINY = os.environ.get("DROP_TINY", "0") == "1"  # drop practice tasks with <25 positives
N_TIER = int(TIER[1:])
OUT_DIR = f"{BASE}/output/nm_meta_v4/{TIER}{'eqcap' + str(CAPN) if CAPN else ''}{'c' if DROP_TINY else ''}"

SEEDS = [42, 7, 13, 15, 2026]
REPS = [1, 2, 3]
BIG_TASKS = ["AMP", "Neuropeptide", "Anti-inflammatory", "Toxicity",
             "Anticancer", "Antiviral", "ACE-inhibitory", "AntiHIV", "CPP"]
OLD_EXTRA = ["Hormone", "ProteaseInhibitor", "Bacteriocin", "Defensin",
             "Neurotoxin", "IonChannelToxin", "Toxin", "Antimicrobial",
             "SerineProteaseInhibitor", "Hemostasis"]
NEW_TASKS = [
    # SATPdb / DRAMP (uniprot negatives)
    "SATPdb_DrugDelivery", "SATPdb_Antihypertensive", "SATPdb_Antibacterial",
    "SATPdb_CellCellComm", "DRAMP_AntiGramPos",
    # GRAMPA species (measured-inactive + uniprot top-up negatives)
    "GRAMPA_Ecoli", "GRAMPA_Saureus", "GRAMPA_Paeruginosa", "GRAMPA_Bsubtilis",
    "GRAMPA_Calbicans", "GRAMPA_Sepidermidis", "GRAMPA_Styphimurium",
    "GRAMPA_Mluteus", "GRAMPA_Efaecalis", "GRAMPA_Kpneumoniae",
    "GRAMPA_Abaumannii", "GRAMPA_Senterica", "GRAMPA_Psyringae",
    "GRAMPA_Lmonocytogenes", "GRAMPA_Bmegaterium", "GRAMPA_Efaecium",
    "GRAMPA_Smutans", "GRAMPA_Bcereus",
]
N_LEVELS = [10, 25, 50, 100, 200, 400, "all"]
K_EPOCHS = 10
EPS = 0.5
LR_IN = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, BASE)

VAL_FILES = [
    "clean_DPPIV.txt", "clean_Antioxidant.txt", "clean_Immunoactive.txt",
    "clean_Thrombolytic.txt", "clean_Osteogenic.txt",
    "clean_Antimalarial.txt", "clean_Antiprotozoal.txt",
    "clean_Insecticidal.txt", "clean_Antifungal.txt", "clean_Hemolytic.txt",
    "clean_Anti-Gram-neg.txt", "clean_AntiLeishmania.txt",
    "clean_AntiTrypanosoma.txt",
    "clean_SARSCoV2.txt", "clean_Cytotoxic.txt",
    "clean_AntiTB.txt", "clean_Antibiofilm.txt",
]


def validation_seqs():
    out = set()
    for f in VAL_FILES:
        p = os.path.join(NEWFUNC_DIR, f)
        if os.path.exists(p):
            with open(p) as fh:
                out |= {l.strip().upper() for l in fh if l.strip()}
    return sorted(out)


def decontaminate(train_seqs, val_seqs, workdir):
    os.makedirs(workdir, exist_ok=True)
    tfa = os.path.join(workdir, "train.fa")
    vfa = os.path.join(workdir, "val.fa")
    with open(tfa, "w") as f:
        for i, s in enumerate(train_seqs):
            f.write(f">t{i}\n{s}\n")
    with open(vfa, "w") as f:
        for i, s in enumerate(val_seqs):
            f.write(f">v{i}\n{s}\n")
    out = os.path.join(workdir, "res")
    subprocess.run([MMSEQS, "easy-search", tfa, vfa, out + ".tsv",
                    os.path.join(workdir, "tmp"),
                    "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "1",
                    "--alignment-mode", "3", "-s", "7.5"],
                   check=True, capture_output=True)
    contaminated = set()
    if os.path.exists(out + ".tsv"):
        with open(out + ".tsv") as f:
            for line in f:
                contaminated.add(line.split("\t")[0])
    keep = [s for i, s in enumerate(train_seqs) if f"t{i}" not in contaminated]
    return keep, len(contaminated)


def uniprot_negatives(pos, seed=0, exclude=None, n_target=None):
    rng = np.random.RandomState(seed)
    up = os.path.join(NEWFUNC_DIR, "uniprot_reviewed.fasta")
    prots, cur = [], []
    with open(up) as f:
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
    posset = set(pos) | (set(exclude) if exclude else set())
    n_target = n_target if n_target is not None else len(pos)
    out, used = [], set()
    for L in [len(s) for s in pos]:
        if len(out) >= n_target:
            break
        for _ in range(50):
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


class MetaModel(nn.Module):
    def __init__(s):
        super().__init__()
        s.shared_backbone = nn.Sequential(
            nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        s.head = nn.Linear(64, 1)

    def forward(s, x):
        return s.head(s.shared_backbone(x)).squeeze(-1)


def adapt(theta_model, Xtr, ytr, seed):
    m = copy.deepcopy(theta_model).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR_IN, weight_decay=1e-4)
    MARGIN = 1.0
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    n = len(ytr)
    bs = min(32, n)
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(K_EPOCHS):
        m.train()
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            sc = m(Xtr_t[idx])
            yb = ytr_t[idx]
            sp = sc[yb > 0.5]
            sn = sc[yb < 0.5]
            if len(sp) == 0 or len(sn) == 0:
                continue
            opt.zero_grad()
            loss = torch.relu(MARGIN - (sp[:, None] - sn[None, :])).mean()
            loss.backward()
            opt.step()
    m.eval()
    return m


def load_big():
    z = np.load(E2_CACHE, allow_pickle=True)
    seqs = z["seqs"].tolist()
    emb = z["emb"]
    sidx = {s: i for i, s in enumerate(seqs)}
    del z
    zn = np.load(NEG_EMB, allow_pickle=True)
    base = emb.shape[0]
    emb = np.concatenate([emb, zn["emb"]], axis=0)
    for i, s in enumerate(zn["seqs"].tolist()):
        sidx[s] = base + i
    del zn
    E = torch.tensor(emb, dtype=torch.float32, device=device)
    del emb

    def stack(ss):
        return E[[sidx[s] for s in ss]]

    data = {}
    for t in BIG_TASKS:
        tr = pd.read_csv(f"{SPLIT_DIR}/{t}_train.csv")
        keep_path = f"{FOUND_DIR}/decon/{t}/keep.json"
        if os.path.exists(keep_path):
            keep = set(json.load(open(keep_path)))
            tr = tr[tr["sequence"].isin(keep)].reset_index(drop=True)
        n_pos_orig = int((tr["label"] == 1).sum())
        if CAPN:
            rng = np.random.RandomState(0)
            pi = np.where(tr["label"].values == 1)[0]
            ni = np.where(tr["label"].values == 0)[0]
            if len(pi) > CAPN:
                pi = np.sort(rng.choice(pi, CAPN, replace=False))
            if len(ni) > CAPN:
                ni = np.sort(rng.choice(ni, CAPN, replace=False))
            tr = tr.loc[sorted(np.concatenate([pi, ni]))].reset_index(drop=True)
        va = pd.read_csv(f"{SPLIT_DIR}/{t}_val.csv")
        data[t] = {
            "train": (stack(tr["sequence"].tolist()), tr["label"].values),
            "val": (stack(va["sequence"].tolist()), va["label"].values),
            "n_pos": int((tr["label"] == 1).sum()),
            "n_pos_orig": n_pos_orig,
        }
        print(f"big {t}: train {len(tr)} pos {data[t]['n_pos']} (orig {n_pos_orig})", flush=True)
    return data


def load_extra(data):
    """Add extra practice tasks (old 10 + new 21): decontaminated positives,
    negatives, ESM-2 embeddings."""
    from pepscan.data import ESM2Encoder
    cache = f"{BASE}/output/nm_meta_v4/extra_emb_v4.npz"
    vseqs = validation_seqs()
    task_seqs, task_labels = {}, {}

    # ---- old 10 mined tasks (same as v3) ----
    for t in OLD_EXTRA:
        p = os.path.join(META_TASK_DIR, f"clean_{t}.txt")
        if not os.path.exists(p):
            print(f"extra {t}: MISSING, skip", flush=True)
            continue
        with open(p) as fh:
            pos = sorted({l.strip().upper() for l in fh if l.strip()})
        keep_path = f"{BASE}/output/nm_meta_v3_rank/decon_{t}.json"
        if os.path.exists(keep_path):
            pos = json.load(open(keep_path))
        else:
            pos, ncont = decontaminate(pos, vseqs, f"{BASE}/output/nm_meta_v4/decon_{t}")
            print(f"extra {t}: decon dropped {ncont}", flush=True)
        neg = uniprot_negatives(pos, seed=0)
        task_seqs[t] = pos + neg
        task_labels[t] = [1] * len(pos) + [0] * len(neg)
        print(f"extra {t}: pos {len(pos)} neg {len(neg)}", flush=True)

    # ---- new 21 tasks ----
    for t in NEW_TASKS:
        p = os.path.join(V4_POS_DIR, f"decon_{t}.txt")
        if not os.path.exists(p):
            print(f"new {t}: MISSING decon file, skip", flush=True)
            continue
        with open(p) as fh:
            pos = sorted({l.strip().upper() for l in fh if l.strip()})
        if t.startswith("GRAMPA_"):
            np_path = os.path.join(V4_NEG_DIR, f"{t}_neg.txt")
            meas = []
            if os.path.exists(np_path):
                meas = [l.strip().upper() for l in open(np_path) if l.strip()]
            topup = uniprot_negatives(pos, seed=0, exclude=meas,
                                      n_target=len(pos) - len(meas))
            neg = meas + topup
        else:
            neg = uniprot_negatives(pos, seed=0)
        task_seqs[t] = pos + neg
        task_labels[t] = [1] * len(pos) + [0] * len(neg)
        print(f"new {t}: pos {len(pos)} neg {len(neg)}", flush=True)

    orig_npos = {t: int(sum(task_labels[t])) for t in task_seqs}
    if CAPN:
        for t in list(task_seqs):
            pos = [s for s, l in zip(task_seqs[t], task_labels[t]) if l == 1][:CAPN]
            neg = [s for s, l in zip(task_seqs[t], task_labels[t]) if l == 0][:CAPN]
            task_seqs[t] = pos + neg
            task_labels[t] = [1] * len(pos) + [0] * len(neg)
        print(f"CAPN={CAPN} applied (all extra/new tasks)", flush=True)

    allseqs = sorted({s for ss in task_seqs.values() for s in ss})
    need_encode = True
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        emb, sseqs = z["emb"], z["seqs"].tolist()
        if not (set(allseqs) - set(sseqs)):
            need_encode = False
        else:
            print("cache stale, re-encoding", flush=True)
    if need_encode:
        os.makedirs(f"{BASE}/output/nm_meta_v4", exist_ok=True)
        enc = ESM2Encoder(device=device)
        emb = enc.encode(allseqs, batch_size=32).astype(np.float32)
        sseqs = allseqs
        np.savez(cache, emb=emb, seqs=np.array(sseqs))
    sidx = {s: i for i, s in enumerate(sseqs)}
    E = torch.tensor(emb, dtype=torch.float32, device=device)
    for t in task_seqs:
        X = E[[sidx[s] for s in task_seqs[t]]]
        y = np.array(task_labels[t])
        n_pos = int((y == 1).sum())
        data[t] = {"train": (X, y), "val": (X, y), "n_pos": n_pos,
                   "n_pos_orig": orig_npos.get(t, n_pos)}
    return [t for t in (OLD_EXTRA + NEW_TASKS) if t in data]


def meta_train(seed, data, practice_tasks, meta_episodes):
    torch.manual_seed(seed)
    np.random.seed(seed)
    theta = MetaModel().to(device)
    sd = torch.load(f"{FOUND_DIR}/seed{seed}.pt", map_location=device, weights_only=False)
    bb = {k: v for k, v in sd.items() if k.startswith("shared_backbone")}
    theta.load_state_dict(bb, strict=False)

    rng = np.random.RandomState(seed)
    best_val, best_state = -1, None
    for ep in range(1, meta_episodes + 1):
        t = practice_tasks[rng.randint(len(practice_tasks))]
        N = [10, 25, 50, 100][rng.randint(4)]
        X, y = data[t]["train"]
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        pi = rng.choice(pos, min(N, len(pos)), replace=False)
        ni = rng.choice(neg, min(N, len(neg)), replace=False)
        Xs = torch.cat([X[pi], X[ni]]).cpu().numpy()
        ys = np.concatenate([y[pi], y[ni]]).astype(np.float32)
        adapted = adapt(theta, Xs, ys, seed * 100000 + ep)
        with torch.no_grad():
            for p, pa in zip(theta.parameters(), adapted.parameters()):
                p.add_(pa - p, alpha=EPS)
        del adapted

        if ep % 200 == 0:
            aucs = []
            for t in BIG_TASKS:
                X, y = data[t]["train"]
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                r2 = np.random.RandomState(seed * 77 + ep)
                pi = r2.choice(pos, 25, replace=False)
                ni = r2.choice(neg, 25, replace=False)
                Xs = torch.cat([X[pi], X[ni]]).cpu().numpy()
                ys = np.concatenate([y[pi], y[ni]]).astype(np.float32)
                m = adapt(theta, Xs, ys, seed * 999 + ep)
                Xv, yv = data[t]["val"]
                with torch.no_grad():
                    pv = torch.sigmoid(m(Xv)).cpu().numpy()
                aucs.append(roc_auc_score(yv, pv))
                del m
            mv = float(np.mean(aucs))
            if mv > best_val:
                best_val = mv
                best_state = {k: v.clone() for k, v in theta.state_dict().items()}
            print(f"seed{seed} ep{ep}/{meta_episodes} meta-val {mv:.4f} (best {best_val:.4f})", flush=True)
    if best_state:
        theta.load_state_dict(best_state)
    torch.save(theta.state_dict(), f"{OUT_DIR}/meta_seed{seed}.pt")
    return theta


def meta_test(seed, theta):
    funcs = sorted(os.path.basename(p) for p in glob.glob(f"{GRID_DIR}/*_uniprot"))
    for func in funcs:
        outdir = os.path.join(OUT_DIR, func)
        os.makedirs(outdir, exist_ok=True)
        rp = os.path.join(outdir, "results.json")
        sp = os.path.join(outdir, "scores.npz")
        results = json.load(open(rp)) if os.path.exists(rp) else {}
        scores = dict(np.load(sp)) if os.path.exists(sp) else {}
        pool = pd.read_csv(os.path.join(GRID_DIR, func, "pool.csv"))
        test = pd.read_csv(os.path.join(GRID_DIR, func, "test.csv"))
        zem = np.load(os.path.join(GRID_DIR, func, "emb.npz"))
        emb_pool, emb_test = zem["pool"], zem["test"]
        y_test = test["label"].values
        Xte = torch.tensor(emb_test, dtype=torch.float32, device=device)
        emb_pos = emb_pool[(pool.label == 1).values]
        emb_neg = emb_pool[(pool.label == 0).values]
        n_pos, n_neg = len(emb_pos), len(emb_neg)
        for N in N_LEVELS:
            n_use = n_pos if N == "all" else min(N, n_pos)
            rng = np.random.RandomState(seed)
            pi = rng.choice(n_pos, n_use, replace=False)
            ni = rng.choice(n_neg, min(n_use, n_neg), replace=False)
            X = np.concatenate([emb_pos[pi], emb_neg[ni]])
            y = np.array([1] * len(pi) + [0] * len(ni))
            n_tr = int(len(y) * 0.85)
            g = rng.permutation(len(y))
            tr, va = g[:n_tr], g[n_tr:]
            if len(np.unique(y[va])) < 2:
                tr = g[:n_tr - 4]
            Xtr, ytr = X[tr], y[tr]
            for rep in REPS:
                key = f"N{N}_seed{seed}_rep{rep}"
                if key in results:
                    continue
                torch.manual_seed(seed * 1000 + rep * 17 + 3 * 7919)
                np.random.seed(seed * 1000 + rep * 17 + 3 * 7919)
                m = adapt(theta, Xtr, ytr, seed * 5 + rep)
                with torch.no_grad():
                    sc = torch.sigmoid(m(Xte)).cpu().numpy()
                results[key] = {"meta": float(roc_auc_score(y_test, sc))}
                scores[f"{key}_meta"] = sc.astype(np.float32)
                del m
                print(f"[{func}] {key}: meta={results[key]['meta']:.3f}", flush=True)
                json.dump(results, open(rp, "w"))
                np.savez_compressed(sp, **scores)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_big()
    extra = load_extra(data)
    all_tasks = BIG_TASKS + extra
    # nested tiers by ORIGINAL (uncapped) positive-pool size, descending
    order = sorted(all_tasks, key=lambda t: -data[t].get("n_pos_orig", data[t]["n_pos"]))
    practice = order[:N_TIER]
    if DROP_TINY:
        practice = [t for t in practice if data[t]["n_pos"] >= 25]
    meta_episodes = 100 * len(practice)
    json.dump({"tier": TIER, "capn": CAPN, "practice": practice,
               "sizes": {t: data[t]["n_pos"] for t in practice},
               "sizes_orig": {t: data[t].get("n_pos_orig", data[t]["n_pos"]) for t in practice},
               "peptide_total": int(sum(data[t]["n_pos"] for t in practice)),
               "meta_episodes": meta_episodes},
              open(f"{OUT_DIR}/tier_config.json", "w"), indent=1)
    print(f"practice tasks ({len(practice)}): {practice}", flush=True)
    print(f"meta_episodes={meta_episodes}", flush=True)
    for seed in SEEDS:
        mp = f"{OUT_DIR}/meta_seed{seed}.pt"
        if os.path.exists(mp):
            theta = MetaModel().to(device)
            theta.load_state_dict(torch.load(mp, map_location=device, weights_only=False))
            theta.eval()
            print(f"seed{seed} meta model exists, loaded", flush=True)
        else:
            theta = meta_train(seed, data, practice, meta_episodes)
        meta_test(seed, theta)
    print("META_V4_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
