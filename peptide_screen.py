#!/usr/bin/env python3
"""peptide_screen.py — Few-shot peptide-agent scoring with built-in
composition-bias control and diversity-aware candidate selection.

For every input peptide and every agent the tool reports, in one CSV:
  <func>_score   ensemble mean sigmoid score (RANKING score, not a probability)
  <func>_pct     percentile of that score within YOUR input library (0-1)
  <func>_sd      between-seed SD (reproducibility across the 5 agent seeds)
  <func>_z       sequence-specificity z-score vs composition-preserved shuffles
  <func>_p       empirical p-value of the same control (small = above shuffles)

Why the shuffle control matters
  Few-shot models can reward amino-acid composition rather than sequence.
  Each peptide is therefore compared against N shuffled versions of ITSELF
  (identical composition, permuted order). z ~= 0 means the score is driven
  by composition alone; z >= 1.5 indicates genuine sequence-level signal.

Optional --select K additionally writes a diversity-aware shortlist:
candidates passing z/p/stability filters are clustered (3-mer Jaccard >= 0.5)
and only the best representative of each sequence family is kept.

Quick start (after downloading code + agent weights):
  pip install torch fair-esm scikit-learn pandas numpy
  python peptide_screen.py --agents agents/Osteogenic agents/DPPIV \
      --input peptides.fasta --output scores.csv --select 12

ESM-2 (150M) downloads automatically on first run (~560 MB).
GPU is used when available; CPU works for small libraries.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

SEEDS = [42, 7, 13, 15, 2026]


def read_seqs(path):
    if path.endswith((".fasta", ".fa")):
        names, seqs, cur, name = [], [], [], None
        for line in open(path):
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    names.append(name)
                    seqs.append("".join(cur))
                name, cur = line[1:], []
            elif line:
                cur.append(line.upper())
        if cur:
            names.append(name)
            seqs.append("".join(cur))
        return names, seqs
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        seqs = df["seq"].astype(str).str.upper().tolist()
        names = (df["name"].tolist() if "name" in df
                 else [f"pep{i+1}" for i in range(len(df))])
        return names, seqs
    seqs = [l.strip().upper() for l in open(path) if l.strip()]
    return [f"pep{i+1}" for i in range(len(seqs))], seqs


def kmer_set(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def jaccard(a, b):
    A, B = kmer_set(a), kmer_set(b)
    return len(A & B) / max(1, len(A | B))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", required=True,
                    help="agent folders containing agent_seed<seed>.pt")
    ap.add_argument("--input", required=True, help="fasta / txt / csv with a 'seq' column")
    ap.add_argument("--output", default="scores.csv")
    ap.add_argument("--shuffle", type=int, default=20,
                    help="shuffled controls per peptide (0 disables; default 20)")
    ap.add_argument("--select", type=int, default=0,
                    help="also write a diversity-aware shortlist of K candidates")
    ap.add_argument("--z-min", type=float, default=1.5)
    ap.add_argument("--p-max", type=float, default=1.0,
                    help="maximum empirical p allowed for every agent; "
                         "1.0 = filter on z only (default), 0.2 = strict")
    ap.add_argument("--sd-max", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from pepscan.data import ESM2Encoder
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

    names, seqs = read_seqs(args.input)
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    bad = [s for s in seqs if not set(s) <= valid]
    if bad:
        raise ValueError(f"{len(bad)} sequences contain non-standard residues")
    # de-duplicate sequences (first name kept)
    seen, unames, useqs = set(), [], []
    for n, s in zip(names, seqs):
        if s not in seen:
            seen.add(s)
            unames.append(n)
            useqs.append(s)
    if len(useqs) < len(seqs):
        print(f"de-duplicated {len(seqs)} -> {len(useqs)} unique sequences", flush=True)
    names, seqs = unames, useqs

    # build the full sequence list: originals + composition-preserved shuffles
    owner = list(range(len(seqs)))
    full = list(seqs)
    rng = np.random.RandomState(7)
    if args.shuffle > 0:
        for i, s in enumerate(seqs):
            for _ in range(args.shuffle):
                chars = list(s)
                rng.shuffle(chars)
                full.append("".join(chars))
                owner.append(i)
    owner = np.array(owner)
    print(f"{len(seqs)} peptides -> {len(full)} sequences with controls", flush=True)

    enc = ESM2Encoder(device=device)
    X = enc.encode(full, batch_size=args.batch_size)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    out = pd.DataFrame({"name": names, "seq": seqs})
    score_cols = []
    for adir in args.agents:
        func = os.path.basename(adir.rstrip("/"))
        per_seed = []
        for s in SEEDS:
            wp = os.path.join(adir, f"agent_seed{s}.pt")
            if not os.path.exists(wp):
                raise FileNotFoundError(wp)
            m = MetaModel().to(device)
            m.load_state_dict(torch.load(wp, map_location=device, weights_only=False))
            m.eval()
            with torch.no_grad():
                per_seed.append(torch.sigmoid(m(Xt)).cpu().numpy())
            del m
        per_seed = np.stack(per_seed)
        sc_all = per_seed.mean(axis=0)
        raw = sc_all[:len(seqs)]
        sd = per_seed[:, :len(seqs)].std(axis=0)
        out[f"{func}_score"] = raw
        out[f"{func}_pct"] = pd.Series(raw).rank(pct=True).values
        out[f"{func}_sd"] = sd
        score_cols.append(f"{func}_score")
        if args.shuffle > 0:
            z = np.zeros(len(seqs))
            p = np.zeros(len(seqs))
            for i in range(len(seqs)):
                sh = sc_all[owner == i][1:]
                mu, sg = sh.mean(), sh.std()
                if sg < 1e-4:
                    # degenerate case (e.g. homopolymers): shuffles are identical
                    # to the original, so sequence-specificity cannot be shown
                    z[i], p[i] = 0.0, 1.0
                else:
                    z[i] = (raw[i] - mu) / sg
                    p[i] = (1 + (sh >= raw[i]).sum()) / (1 + args.shuffle)
            out[f"{func}_z"] = z
            out[f"{func}_p"] = p
        print(f"{func} scored", flush=True)

    out["mean_score"] = out[score_cols].mean(axis=1)
    out["rank"] = out["mean_score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("rank")
    out.to_csv(args.output, index=False)
    print(f"scores written -> {args.output}", flush=True)

    if args.select > 0:
        if args.shuffle <= 0:
            raise ValueError("--select requires the shuffle control (--shuffle > 0)")
        funcs = [os.path.basename(a.rstrip("/")) for a in args.agents]
        out["z_mean"] = out[[f"{f}_z" for f in funcs]].mean(axis=1)
        out["p_max"] = out[[f"{f}_p" for f in funcs]].max(axis=1)
        out["sd_mean"] = out[[f"{f}_sd" for f in funcs]].mean(axis=1)
        ok = out[(out["z_mean"] >= args.z_min) & (out["p_max"] <= args.p_max)
                 & (out["sd_mean"] <= args.sd_max)].copy()
        ok = ok.sort_values(["z_mean", "mean_score"], ascending=False)
        picked, clusters = [], []
        for _, r in ok.iterrows():
            if any(jaccard(r["seq"], q) >= 0.5 for q in clusters):
                continue
            clusters.append(r["seq"])
            picked.append(r)
            if len(picked) >= args.select:
                break
        sel = pd.DataFrame(picked)
        short = args.output.replace(".csv", "") + f"_shortlist{args.select}.csv"
        sel.to_csv(short, index=False)
        print(f"{len(sel)} diverse, sequence-specific candidates -> {short}", flush=True)
        if len(sel) < args.select:
            print("note: fewer candidates passed the filters than requested; "
                  "consider relaxing --z-min / --p-max / --sd-max", flush=True)


if __name__ == "__main__":
    main()
