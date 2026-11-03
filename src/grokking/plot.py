"""Plot the grokking curve from runs/<run>/metrics.json -> figures/<run>_curve.png

Run:  uv run python -m grokking.plot --run main
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()

    h = json.loads((Path("runs") / args.run / "metrics.json").read_text())
    steps = [max(s, 1) for s in h["step"]]   # log-scale x-axis can't show step 0

    fig, (ax_acc, ax_loss, ax_norm) = plt.subplots(1, 3, figsize=(16, 4.5))

    ax_acc.plot(steps, h["train_acc"], label="train")
    ax_acc.plot(steps, h["test_acc"], label="test")
    ax_acc.set(xscale="log", xlabel="step", ylabel="accuracy", title="Accuracy: memorize, then grok")
    ax_acc.axhline(1 / 113, color="gray", ls=":", lw=1, label="chance (1/113)")

    ax_loss.plot(steps, h["train_loss"], label="train")
    ax_loss.plot(steps, h["test_loss"], label="test")
    ax_loss.set(xscale="log", yscale="log", xlabel="step", ylabel="cross-entropy loss", title="Loss")

    ax_norm.plot(steps, h["weight_norm"], color="tab:green")
    ax_norm.set(xscale="log", xlabel="step", ylabel="total weight norm", title="Weight norm (weight decay at work)")

    for ax in (ax_acc, ax_loss):
        ax.legend()
    for ax in (ax_acc, ax_loss, ax_norm):
        ax.grid(alpha=0.3)

    Path("figures").mkdir(exist_ok=True)
    out = Path("figures") / f"{args.run}_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
