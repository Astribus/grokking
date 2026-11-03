"""Does the model learn the CLOCK, or just the TABLE? Analyse the structured-holdout runs.

Run:  uv run python -m grokking.ood
Reads runs/<split>_s<seed>/ for every split in SPLITS, plus random_s0 = runs/main.
Makes figures/ood_summary.png, figures/ood_curves.png, figures/ood_repair.png; writes runs/ood_summary.json.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from grokking.data import make_split
from grokking.fourier import P, energy_per_frequency, fourier_basis, load

SPLITS = ["random", "triangle", "holdout_a", "nowrap", "holdout_answer"]
NICE = {"random": "random 30%\n(normal)", "triangle": "only a ≤ b\n(test a > b)",
        "holdout_a": "20 numbers never\nin 1st slot", "nowrap": "only a+b < 113\n(test wraparound)",
        "holdout_answer": "one answer c₀\nnever a label"}
SEEDS = [0, 1, 2]


def run_dir(split, seed):
    return Path("runs") / ("main" if (split, seed) == ("random", 0) else f"{split}_s{seed}")


def first_step(h, key, thresh=0.99):
    for s, v in zip(h["step"], h[key]):
        if v >= thresh:
            return s
    return None


def accuracy(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(-1) == y).double().mean().item() if len(y) else float("nan")


def key_freqs(W, Fb):
    e = energy_per_frequency(W, Fb)
    key = [k for k in range(1, 57) if e[k] > 5 / 56]
    return key, e[key].sum().item() if key else 0.0, e


def repair_answer_row(model, c0, key):
    """Rebuild the unembedding row of the never-seen answer c0 from the OTHER 112 rows.

    If the readout uses the clock, each of W_U's 128 columns is a sum of the key waves over c.
    So fit those waves by least squares on the 112 known answers and read off the value at c0.
    """
    W = model.unembed.weight.detach().double()                    # [113 answers, 128]
    c = torch.arange(P, dtype=torch.float64)
    cols = [torch.ones(P, dtype=torch.float64)]
    for k in key:
        cols += [torch.cos(2 * math.pi * k * c / P), torch.sin(2 * math.pi * k * c / P)]
    B = torch.stack(cols, 1)                                      # [113, 1 + 2K] wave design matrix
    known = torch.arange(P) != c0
    coef = torch.linalg.lstsq(B[known], W[known]).solution        # [1 + 2K, 128]
    fitted = B @ coef
    with torch.no_grad():
        model.unembed.weight[c0] = fitted[c0].float()
    fit_r2 = 1 - (W[known] - fitted[known]).pow(2).sum() / (W[known] - W[known].mean(0)).pow(2).sum()
    return model, fit_r2.item()


def main():
    Fb, _ = fourier_basis()
    rows = []
    for split in SPLITS:
        for seed in SEEDS:
            d = run_dir(split, seed)
            if not (d / "final.pt").exists():
                print("missing", d)
                continue
            h = json.loads((d / "metrics.json").read_text())
            model = load(d / "final.pt")
            data = make_split(split, seed=seed)
            r = {"split": split, "seed": seed, "run": d.name, "info": h.get("split_info", {}),
                 "train_acc": accuracy(model, *data["train"]),
                 "test_id_acc": accuracy(model, *data["test_id"]),
                 "ood_acc": accuracy(model, *data["ood"]),
                 "grok_step_id": first_step(h, "test_acc"),
                 "ood_step": first_step(h, "ood_acc") if h.get("ood_acc") else None}
            key, share, e = key_freqs(model.embed.weight[:P].detach(), Fb)
            r["key_freqs"], r["key_share"] = key, share
            r["embed_energy"] = e.tolist()

            if split == "holdout_answer":
                c0 = r["info"]["held_out_answer"]
                x, y = data["ood"]
                with torch.no_grad():
                    lo = model(x)
                pred = lo.argmax(-1)
                rank = (lo > lo[torch.arange(len(y)), y].unsqueeze(1)).sum(1)   # 0 = top
                r["c0"] = c0
                r["ood_pred_offsets"] = ((pred - y) % P).tolist()                # how far off, on the clock
                r["c0_median_rank"] = rank.median().item()
                norms = model.unembed.weight.norm(dim=1)
                r["W_U_row_norm_c0_vs_mean"] = (norms[c0] / norms[torch.arange(P) != c0].mean()).item()
                fixed, r2 = repair_answer_row(load(d / "final.pt"), c0, key)
                r["repair_fit_r2"] = r2
                r["repair_ood_acc"] = accuracy(fixed, *data["ood"])
                r["repair_test_id_acc"] = accuracy(fixed, *data["test_id"])
                r["repair_train_acc"] = accuracy(fixed, *data["train"])

            if split == "triangle":
                x, _ = data["ood"]
                xs = x[:, [1, 0, 2]]                                   # swap a and b
                with torch.no_grad():
                    r["swap_agreement"] = (model(x).argmax(-1) == model(xs).argmax(-1)).double().mean().item()

            if split == "nowrap":
                x, y = data["ood"]
                with torch.no_grad():
                    pred = model(x).argmax(-1)
                s_ = x[:, 0] + x[:, 1]
                r["ood_pred_is_112"] = (pred == P - 1).double().mean().item()       # "clamp at the top"
                r["ood_pred_is_correct"] = (pred == y).double().mean().item()
                r["ood_pred_ge_100"] = (pred >= 100).double().mean().item()
                r["ood_mean_pred"] = pred.double().mean().item()
                r["ood_mean_true_sum"] = s_.double().mean().item()

            rows.append(r)
            short = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                     if k not in ("embed_energy", "ood_pred_offsets", "info")}
            print(short)
    Path("runs/ood_summary.json").write_text(json.dumps(rows))

    # ---- Figure 1: in-distribution vs OOD accuracy per split (dots = seeds) ----
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, split in enumerate(SPLITS):
        rs = [r for r in rows if r["split"] == split]
        for j, (key_, col, lab) in enumerate([("test_id_acc", "tab:blue", "held-out pairs, same kind as training"),
                                              ("ood_acc", "tab:red", "structured holdout (never-seen kind)")]):
            vals = [r[key_] for r in rs if not math.isnan(r[key_])]
            if not vals:
                continue
            xpos = i + (j - 0.5) * 0.36
            ax.bar(xpos, sum(vals) / len(vals), 0.34, color=col, alpha=0.75, label=lab if i == 1 else None)
            ax.scatter([xpos] * len(vals), vals, color="black", s=12, zorder=3)
            ax.text(xpos, sum(vals) / len(vals) + 0.03, f"{sum(vals) / len(vals):.0%}", ha="center", fontsize=8)
        if split == "holdout_answer" and rs:
            vals = [r["repair_ood_acc"] for r in rs]
            ax.scatter([i + 0.18] * len(vals), vals, marker="*", s=120, color="gold", edgecolor="black",
                       zorder=4, label="after rebuilding c₀'s readout row from the clock")
    ax.axhline(1 / P, color="gray", ls=":", lw=1)
    ax.set_xticks(range(len(SPLITS)), [NICE[s] for s in SPLITS], fontsize=9)
    ax.set(ylabel="accuracy (final, step 20k / 40k)", ylim=(0, 1.15),
           title="Does a grokked transformer learn the clock or the table? (3 seeds each)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/ood_summary.png", dpi=150)

    # ---- Figure 2: OOD accuracy over training ----
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, split in zip(axes, [s for s in SPLITS if s != "random"]):
        for seed in SEEDS:
            d = run_dir(split, seed)
            if not (d / "metrics.json").exists():
                continue
            h = json.loads((d / "metrics.json").read_text())
            ln, = ax.plot(h["step"], h["test_acc"], lw=1.2, label=f"seed {seed}: same-kind test")
            ax.plot(h["step"], h["ood_acc"], lw=1.2, ls="--", color=ln.get_color(), label=f"seed {seed}: OOD")
        ax.set(title=NICE[split].replace("\n", " "), xlabel="step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("Solid = held-out pairs of the same kind as training; dashed = the never-seen kind", y=1.0)
    fig.tight_layout()
    fig.savefig("figures/ood_curves.png", dpi=150)
    print("saved figures/ood_summary.png, figures/ood_curves.png")


if __name__ == "__main__":
    main()
