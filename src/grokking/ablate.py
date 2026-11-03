"""Prove the 4 key frequencies are the algorithm: delete things and see what breaks.

Run:  uv run python -m grokking.ablate --run main
Makes figures/<run>_ablations.png and figures/<run>_progress_measures.png, writes runs/<run>/ablate.json.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F_nn

from grokking.data import make_split
from grokking.fourier import P, all_inputs, fourier_basis, load, neuron_frequencies

X = all_inputs()                                    # all 12,769 inputs, row index = a*113 + b
Y = (X[:, 0] + X[:, 1]) % P


def score(logits, idx=None):
    """Loss and accuracy of [12769, 113] logits (optionally only on rows idx)."""
    if idx is not None:
        logits, y = logits[idx], Y[idx]
    else:
        y = Y
    logits = logits.double()
    return F_nn.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).double().mean().item()


def logits_of(model):
    with torch.no_grad():
        return model(X)


def with_embedding_freqs(model, Fb, keep):
    """Copy of the model whose number embeddings keep ONLY the Fourier rows where keep[row] is True."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        W = m.embed.weight[:P].double()
        coef = Fb @ W                                   # embedding in wave terms
        coef[~keep] = 0                                 # delete the unwanted waves
        m.embed.weight[:P] = (Fb.T @ coef).float()      # rotate back (F is orthonormal, so F^-1 = F^T)
    return m


def freq_mask(freqs, include_const):
    """Boolean mask over the 113 Fourier rows: True for const (optional) and the cos/sin rows of freqs."""
    keep = torch.zeros(P, dtype=torch.bool)
    keep[0] = include_const
    for k in freqs:
        keep[2 * k - 1] = keep[2 * k] = True            # rows 2k-1 = cos k, 2k = sin k
    return keep


def logit_directions(freqs):
    """For each frequency k: the two 'answer' patterns cos(w(a+b-c)) and sin(w(a+b-c)) over (a, b, c),
    each normalised. These are exactly what the clock algorithm should output."""
    a = torch.arange(P).view(P, 1, 1).double()
    b = torch.arange(P).view(1, P, 1).double()
    c = torch.arange(P).view(1, 1, P).double()
    dirs = []
    for k in freqs:
        ang = 2 * math.pi * k * (a + b - c) / P
        for d in (torch.cos(ang), torch.sin(ang)):
            dirs.append((d / d.norm()).reshape(P * P, P))
    return dirs


def restricted_and_excluded(logits, dirs):
    """Split logits into (part made of the key-frequency answer patterns, everything else)."""
    L = logits.double()
    restricted = torch.zeros_like(L)
    for d in dirs:
        restricted += (L * d).sum() * d                 # projection onto each (orthonormal) pattern
    return restricted, L - restricted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    run = Path("runs") / args.run
    four = json.loads((run / "fourier.json").read_text())
    key = four["key_freqs"]
    Fb, _ = fourier_basis()
    model = load(run / "final.pt")
    others = [k for k in range(1, 57) if k not in key]
    res = {"key_freqs": key}

    # ---- 1. Embedding ablations ----
    res["full"] = score(logits_of(model))
    res["embed_keep_key_only"] = score(logits_of(with_embedding_freqs(model, Fb, freq_mask(key, True))))
    res["embed_remove_key"] = score(logits_of(with_embedding_freqs(model, Fb, ~freq_mask(key, False))))
    rng = random.Random(0)
    ctrl = [score(logits_of(with_embedding_freqs(model, Fb, ~freq_mask(rng.sample(others, len(key)), False))))
            for _ in range(10)]
    res["embed_remove_random_4_x10"] = {"loss_mean": sum(c[0] for c in ctrl) / 10,
                                        "acc_min": min(c[1] for c in ctrl), "acc_mean": sum(c[1] for c in ctrl) / 10}
    res["embed_remove_one_key"] = {}
    for k in key:
        res["embed_remove_one_key"][k] = score(logits_of(with_embedding_freqs(model, Fb, ~freq_mask([k], False))))

    # ---- 2. Neuron-team ablations: silence every MLP neuron that works on frequency k ----
    _, best_k, _, _ = neuron_frequencies(model, Fb)
    res["neurons_remove_team"] = {}
    for k in key:
        m = copy.deepcopy(model)
        with torch.no_grad():
            m.mlp.W_out.weight[:, best_k == k] = 0      # that team's output is cut off
        res["neurons_remove_team"][k] = score(logits_of(m))
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.mlp.W_out.weight[:] = 0
    res["neurons_remove_all"] = score(logits_of(m))

    # ---- 3. Logit-level: is the OUTPUT just the 4 clock patterns? ----
    dirs = logit_directions(key)
    L = logits_of(model)
    r, e = restricted_and_excluded(L, dirs)
    res["logits_restricted"] = score(r)
    res["logits_excluded"] = score(e)
    res["logit_share_in_key_patterns"] = (r.pow(2).sum() / (L.double() - L.double().mean(-1, keepdim=True)).pow(2).sum()).item()

    for name, v in res.items():
        print(f"{name:28s} {v}")

    # ---- 4. Progress measures over training (Nanda et al.'s restricted / excluded loss) ----
    split = make_split("random", seed=0)                 # main run's split: train rows = these (a, b)
    tr = split["train"][0]
    train_idx = tr[:, 0] * P + tr[:, 1]
    h = {"step": [], "train": [], "test": [], "restricted": [], "excluded_train": []}
    test_mask = torch.ones(P * P, dtype=torch.bool)
    test_mask[train_idx] = False
    for c in sorted((run / "ckpts").glob("step*.pt")):
        L = logits_of(load(c))
        r, e = restricted_and_excluded(L, dirs)
        h["step"].append(int(c.stem[4:]))
        h["train"].append(score(L, train_idx)[0])
        h["test"].append(score(L, test_mask)[0])
        h["restricted"].append(score(r)[0])
        h["excluded_train"].append(score(e, train_idx)[0])
    res["progress"] = h
    (run / "ablate.json").write_text(json.dumps(res))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = [("full model", res["full"][1]),
            ("embedding: keep ONLY the 4 key freqs", res["embed_keep_key_only"][1]),
            ("embedding: remove the 4 key freqs", res["embed_remove_key"][1]),
            ("embedding: remove 4 random other freqs (worst of 10)", res["embed_remove_random_4_x10"]["acc_min"])]
    bars += [(f"embedding: remove only freq {k}", res["embed_remove_one_key"][k][1]) for k in key]
    bars += [(f"MLP: silence the freq-{k} neuron team", res["neurons_remove_team"][k][1]) for k in key]
    bars += [("logits: keep only the 8 clock patterns", res["logits_restricted"][1]),
             ("logits: remove the 8 clock patterns", res["logits_excluded"][1])]
    colors = (["tab:blue", "tab:green", "tab:red", "tab:gray"] + ["tab:purple"] * len(key)
              + ["tab:orange"] * len(key) + ["tab:green", "tab:red"])
    ypos = list(range(len(bars)))[::-1]
    ax.barh(ypos, [b[1] for b in bars], color=colors)
    for y_, (_, v) in zip(ypos, bars):
        ax.text(v + 0.01, y_, f"{v:.1%}", va="center", fontsize=8)
    ax.axvline(1 / P, color="black", ls=":", lw=1, label="chance (1/113)")
    ax.set_yticks(ypos, [b[0] for b in bars], fontsize=8.5)
    ax.set(xlabel="accuracy on all 12,769 pairs", xlim=(0, 1.12),
           title=f"Ablations: freqs {key} are necessary and sufficient")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"figures/{args.run}_ablations.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(h["step"], h["train"], label="train loss")
    ax.plot(h["step"], h["test"], label="test loss")
    ax.plot(h["step"], h["restricted"], label="restricted loss (only the 4 clock patterns)")
    ax.plot(h["step"], h["excluded_train"], label="excluded loss on TRAIN (clock patterns removed)")
    ax.axhline(math.log(P), color="gray", ls=":", lw=1, label="random guessing (ln 113)")
    ax.set(yscale="log", xlabel="step", ylabel="cross-entropy loss",
           title="Progress measures: the clock circuit grows, then memorization is removed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/{args.run}_progress_measures.png", dpi=150)
    print("saved figures")


if __name__ == "__main__":
    main()
