"""Look inside the trained model: which sine/cosine waves does it use?

Run:  uv run python -m grokking.fourier --run main
Makes figures/<run>_fourier_embed.png, <run>_fourier_over_training.png, <run>_neurons.png
and writes runs/<run>/fourier.json.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from grokking.model import Transformer

P = 113


def fourier_basis(p=P):
    """A p x p matrix whose rows are waves over the numbers 0..p-1, each scaled to length 1.

    Row 0 is a constant. Then for each frequency k = 1..(p-1)/2 there is a cos row and a sin row:
    cos(2*pi*k*x/p) and sin(2*pi*k*x/p). The rows are orthonormal, so multiplying by this matrix
    is just a rotation: it re-describes the same vectors "in terms of waves" without losing anything.
    """
    x = torch.arange(p, dtype=torch.float64)
    rows, names = [torch.ones(p, dtype=torch.float64)], ["const"]
    for k in range(1, (p - 1) // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * x / p), torch.sin(2 * math.pi * k * x / p)]
        names += [f"cos {k}", f"sin {k}"]
    F = torch.stack(rows)
    return F / F.norm(dim=1, keepdim=True), names


def energy_per_frequency(W, F):
    """W: [p, d] (one row per number). Returns a length-(1 + 56) tensor: the share of W's squared
    size that sits in the constant (index 0) and in each frequency k (cos and sin added together)."""
    c = (F @ W.double()).pow(2).sum(dim=1)           # squared size of each Fourier row
    per_k = torch.cat([c[:1], c[1::2] + c[2::2]])     # [const, k=1, k=2, ..., k=56]
    return per_k / per_k.sum()


def load(path):
    model = Transformer()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--n-key", type=int, default=0,
                    help="force this many key frequencies (0 = pick automatically)")
    args = ap.parse_args()
    run = Path("runs") / args.run
    F, _ = fourier_basis()

    # ---- 1. The final model: embedding (input side) and unembedding (output side) ----
    model = load(run / "final.pt")
    W_E = model.embed.weight[:P].detach()             # [113, 128]: drop the "=" token's row
    W_U = model.unembed.weight.detach()               # [113, 128]: one row per possible answer
    e_E, e_U = energy_per_frequency(W_E, F), energy_per_frequency(W_U, F)

    # Key frequencies: those holding more than 5x their "fair share" (1/56) of the embedding.
    freqs = torch.arange(1, 57)
    if args.n_key:
        key = sorted(freqs[e_E[1:].argsort(descending=True)[: args.n_key]].tolist())
    else:
        key = freqs[e_E[1:] > 5 / 56].tolist()
    share_E = e_E[key].sum().item()                   # e_E index k == frequency k (index 0 is const)
    share_U = e_U[key].sum().item()
    print(f"key frequencies: {key}")
    print(f"share of embedding in key freqs:   {share_E:.1%}")
    print(f"share of unembedding in key freqs: {share_U:.1%}")
    print("top 8 embedding frequencies:",
          [(int(k), round(e_E[k].item(), 3)) for k in e_E[1:].argsort(descending=True)[:8] + 1])

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, e, title in [(axes[0], e_E, "Embedding W_E (how the model reads a and b)"),
                         (axes[1], e_U, "Unembedding W_U (how the model writes the answer)")]:
        colors = ["tab:red" if k in key else "tab:gray" for k in range(57)]
        ax.bar(range(57), e.numpy(), color=colors)
        ax.axhline(1 / 56, color="black", ls=":", lw=1, label="fair share if spread evenly (1/56)")
        ax.set(ylabel="share of squared size", title=title)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3, axis="y")
    axes[1].set_xlabel("frequency k  (0 = constant)")
    axes[1].set_xticks(range(0, 57, 4))
    fig.suptitle(f"Fourier view of the trained model: key frequencies {key} in red", y=0.995)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{args.run}_fourier_embed.png", dpi=150)

    # ---- 2. Over training: when do the key frequencies take over? ----
    ckpts = sorted((run / "ckpts").glob("step*.pt"))
    steps, key_share, top_share = [], [], []
    for c in ckpts:
        W = load(c).embed.weight[:P].detach()
        e = energy_per_frequency(W, F)
        steps.append(int(c.stem[4:]))
        key_share.append(e[key].sum().item())
        # Same count of frequencies, but the biggest ones AT THAT MOMENT (not the final key set),
        # so this measures "how concentrated" the embedding is without assuming the end result.
        top_share.append(e[1:].sort(descending=True).values[: len(key)].sum().item())
    h = json.loads((run / "metrics.json").read_text())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, key_share, "o-", color="tab:red", ms=3, label=f"embedding share in final key freqs {key}")
    ax.plot(steps, top_share, "--", color="tab:orange", label=f"share in top-{len(key)} freqs at that step")
    ax.axhline(len(key) / 56, color="tab:red", ls=":", lw=1, label=f"random baseline ({len(key)}/56)")
    ax.set(xlabel="step", ylabel="share of embedding", ylim=(0, 1.02))
    ax2 = ax.twinx()
    ax2.plot(h["step"], h["test_acc"], color="tab:blue", alpha=0.6, label="test accuracy")
    ax2.plot(h["step"], h["train_acc"], color="tab:blue", ls="--", alpha=0.4, label="train accuracy")
    ax2.set(ylabel="accuracy", ylim=(0, 1.02))
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="center right", fontsize=8)
    ax.set_xlim(0, max(steps))
    ax.set_title("The embedding concentrates on a few waves as the model groks")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"figures/{args.run}_fourier_over_training.png", dpi=150)

    # ---- 3. MLP neurons: what does one neuron's activation look like over every (a, b)? ----
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    grid = torch.stack([a, b, torch.full_like(a, P)], dim=1)        # all 12,769 inputs
    with torch.no_grad():
        model(grid)
    acts = model.mlp.last_hidden[:, -1, :]                           # [12769, 512] at the "=" position
    # For each neuron, which single frequency best explains it? Project its (a, b) map onto 2-D waves.
    A = acts.double().reshape(P, P, -1)                              # [a, b, neuron]
    C = torch.einsum("ia,jb,abn->ijn", F, F, A).pow(2)               # 2-D Fourier energy per neuron
    C[0, 0] = 0                                                      # ignore the constant part
    per_k = torch.zeros(57, C.shape[-1], dtype=torch.float64)
    for i in range(P):
        for j in range(P):
            ki, kj = (i + 1) // 2, (j + 1) // 2                      # frequency of row i / column j
            if ki == kj or ki == 0 or kj == 0:                       # terms that use one frequency
                per_k[max(ki, kj)] += C[i, j]
    best_k = per_k.argmax(dim=0)
    best_frac = per_k.max(dim=0).values / C.sum(dim=(0, 1))
    alive = acts.abs().sum(dim=0) > 0
    counts = {int(k): int(((best_k == k) & alive).sum()) for k in key}
    clean = int(((best_frac > 0.5) & alive).sum())
    print(f"neurons alive: {int(alive.sum())}/512; mostly one frequency (>50% of energy): {clean}")
    print(f"neurons per key frequency: {counts}; in other frequencies: {int(alive.sum()) - sum(counts.values())}")

    fig, axes = plt.subplots(1, len(key), figsize=(3.2 * len(key), 3.4))
    axes = [axes] if len(key) == 1 else axes
    for ax, k in zip(axes, key):
        cands = ((best_k == k) & alive).nonzero().flatten()
        n = cands[best_frac[cands].argmax()].item() if len(cands) else 0
        ax.imshow(A[:, :, n].numpy(), cmap="viridis", origin="lower")
        ax.set(title=f"neuron {n} (freq {k})", xlabel="b", ylabel="a")
    fig.suptitle("One MLP neuron per key frequency: activation over all 12,769 inputs (a, b) repeats with that frequency")
    fig.tight_layout()
    fig.savefig(f"figures/{args.run}_neurons.png", dpi=150)

    (run / "fourier.json").write_text(json.dumps({
        "key_freqs": key, "share_embed": share_E, "share_unembed": share_U,
        "energy_embed": e_E.tolist(), "energy_unembed": e_U.tolist(),
        "steps": steps, "key_share": key_share, "top_share": top_share,
        "neurons_alive": int(alive.sum()), "neurons_one_freq": clean, "neurons_per_key": counts,
    }))
    print("saved figures/", args.run, "_fourier_embed.png, _fourier_over_training.png, _neurons.png", sep="")


if __name__ == "__main__":
    main()
