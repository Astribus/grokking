"""Build the modular-addition dataset: every (a, b) pair, split into train and test.
"""
import torch

P = 113               # the modulus: answers are (a + b) mod 113
EQUALS_TOKEN = P      # token id 113 means "=" (ids 0..112 are the numbers themselves)


def make_dataset(p: int = P, train_frac: float = 0.3, seed: int = 0, n_train: int | None = None):
    """Return (train_x, train_y, test_x, test_y).

    x has shape [n, 3]: each row is the 3 input tokens [a, b, =].
    y has shape [n]:    each entry is the correct answer (a + b) mod p.
    """
    a = torch.arange(p).repeat_interleave(p)   # 0,0,0,...,1,1,1,...  (each a repeated p times)
    b = torch.arange(p).repeat(p)              # 0,1,2,...,0,1,2,...  (the whole range, p times)
    eq = torch.full_like(a, EQUALS_TOKEN)      # the "=" token in every row

    x = torch.stack([a, b, eq], dim=1)         # [p*p, 3]
    y = (a + b) % p                            # [p*p]

    gen = torch.Generator().manual_seed(seed)  # fixed seed -> same split every run
    order = torch.randperm(p * p, generator=gen)
    if n_train is None:
        n_train = int(train_frac * p * p)
    train_idx, test_idx = order[:n_train], order[n_train:]

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


SPLITS = ["random", "nowrap", "triangle", "holdout_answer", "holdout_a"]


def make_split(kind: str = "random", p: int = P, n_train: int = 3830, seed: int = 0):
    """Structured train/test splits for the out-of-distribution (OOD) experiments.

    Every split trains on exactly n_train pairs, so only the *shape* of the training set changes,
    not its size. Returns a dict of (x, y) pairs:
      "train"   : the training set, drawn at random from an allowed pool
      "test_id" : the rest of the allowed pool (in-distribution: same kind of pairs as training)
      "ood"     : the structured holdout, a kind of pair the model NEVER sees in training
      "info"    : what was held out (for the record)

    kinds:
      random         : the normal split (identical to make_dataset). No OOD set.
      nowrap         : train only on a + b < p (the clock never wraps). OOD = every wraparound pair.
      triangle       : train only on a <= b. OOD = every pair with a > b (tests commutativity).
      holdout_answer : one answer c0 never appears as a training label. OOD = the 113 pairs that sum to c0.
      holdout_a      : 20 numbers never appear in the FIRST slot (they still appear second).
                       OOD = every pair whose first number is one of them.
    """
    if kind == "random":
        tx, ty, vx, vy = make_dataset(p, seed=seed, n_train=n_train)
        empty = (tx[:0], ty[:0])
        return {"train": (tx, ty), "test_id": (vx, vy), "ood": empty, "info": {}}

    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, EQUALS_TOKEN)], dim=1)
    y = (a + b) % p
    gen = torch.Generator().manual_seed(seed)
    info = {}

    if kind == "nowrap":
        ood = a + b >= p
    elif kind == "triangle":
        ood = a > b
    elif kind == "holdout_answer":
        c0 = int(torch.randint(p, (1,), generator=gen))
        ood, info = y == c0, {"held_out_answer": c0}
    elif kind == "holdout_a":
        held = torch.randperm(p, generator=gen)[:20]
        ood, info = torch.isin(a, held), {"held_out_first_operand": sorted(held.tolist())}
    else:
        raise ValueError(f"unknown split {kind!r}; choose from {SPLITS}")

    pool = (~ood).nonzero().flatten()
    pool = pool[torch.randperm(len(pool), generator=gen)]
    tr, te = pool[:n_train], pool[n_train:]
    oo = ood.nonzero().flatten()
    return {"train": (x[tr], y[tr]), "test_id": (x[te], y[te]), "ood": (x[oo], y[oo]), "info": info}


if __name__ == "__main__":
    for k in SPLITS:
        d = make_split(k)
        print(f"{k:15s} train {len(d['train'][1]):5d}  test_id {len(d['test_id'][1]):5d}  "
              f"ood {len(d['ood'][1]):5d}  {d['info']}")
