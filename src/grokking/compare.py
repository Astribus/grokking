"""Overlay test accuracy from several runs -> figures/compare_<name>.png

Run:  uv run python -m grokking.compare --runs main pytorch-init --labels "paper init" "PyTorch default init" --name init
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--name", default="runs")
    args = ap.parse_args()
    labels = args.labels or args.runs

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, label in zip(args.runs, labels):
        h = json.loads((Path("runs") / run / "metrics.json").read_text())
        steps = [max(s, 1) for s in h["step"]]
        line, = ax.plot(steps, h["test_acc"], label=f"{label}: test")
        ax.plot(steps, h["train_acc"], ls="--", color=line.get_color(), alpha=0.5, label=f"{label}: train")
    ax.set(xscale="log", xlabel="step", ylabel="accuracy", title="When does the model grok?")
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path("figures") / f"compare_{args.name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
