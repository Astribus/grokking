"""Train the transformer on modular addition and log everything we need to see grokking.

Run:  uv run python -m grokking.train --steps 40000 --run main
Saves runs/<run>/metrics.json and a checkpoint every --ckpt-every steps.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grokking.data import make_dataset
from grokking.model import Transformer


def loss_and_acc(model, x, y):
    logits = model(x)
    # Loss in float64 when the device allows it (CPU). Tiny float32 rounding errors in log_softmax
    # cause sudden loss spikes once the loss gets very small (Nanda et al. hit this too).
    if logits.device.type == "cpu":
        logits = logits.double()
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(dim=-1) == y).float().mean()
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)          # weight decay (the key ingredient)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--run", default="main")
    ap.add_argument("--init", choices=["paper", "pytorch"], default="paper")  # starting-weight scheme
    args = ap.parse_args()

    torch.manual_seed(args.seed)                              # same starting weights every run
    dev = torch.device(args.device)
    train_x, train_y, test_x, test_y = (t.to(dev) for t in make_dataset(train_frac=args.train_frac, seed=args.seed))

    model = Transformer()
    if args.init == "paper":
        model.init_like_paper()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    # Warm up: ramp the learning rate from ~0 to lr over the first 10 steps, then hold it constant.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min((step + 1) / 10, 1.0))

    out = Path("runs") / args.run
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    history = {"args": vars(args), "step": [], "train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": [], "weight_norm": []}

    start = time.time()
    for step in range(args.steps + 1):
        # ---- one training step: full batch (all 3,830 training examples at once) ----
        model.train()
        loss, acc = loss_and_acc(model, train_x, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        # ---- logging ----
        if step % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = loss_and_acc(model, test_x, test_y)
                norm = torch.sqrt(sum((p ** 2).sum() for p in model.parameters()))
            for key, val in [("step", step), ("train_loss", loss.item()), ("test_loss", test_loss.item()),
                             ("train_acc", acc.item()), ("test_acc", test_acc.item()), ("weight_norm", norm.item())]:
                history[key].append(val)
            if step % (args.log_every * 10) == 0:
                print(f"step {step:6d} | train loss {loss.item():.4f} acc {acc.item():.3f} | "
                      f"test loss {test_loss.item():.4f} acc {test_acc.item():.3f} | "
                      f"|W| {norm.item():.1f} | {time.time() - start:.0f}s", flush=True)

        if step % args.ckpt_every == 0:
            torch.save(model.state_dict(), out / "ckpts" / f"step{step:06d}.pt")

    (out / "metrics.json").write_text(json.dumps(history))
    torch.save(model.state_dict(), out / "final.pt")
    print(f"done in {time.time() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
