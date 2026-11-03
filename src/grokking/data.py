"""Build the modular-addition dataset: every (a, b) pair, split into train and test.
"""
import torch

P = 113               # the modulus: answers are (a + b) mod 113
EQUALS_TOKEN = P      # token id 113 means "=" (ids 0..112 are the numbers themselves)


def make_dataset(p: int = P, train_frac: float = 0.3, seed: int = 0):
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
    n_train = int(train_frac * p * p)
    train_idx, test_idx = order[:n_train], order[n_train:]

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


if __name__ == "__main__":
    tx, ty, vx, vy = make_dataset()
    print("train:", tx.shape, "test:", vx.shape)
    print("first 3 training rows [a, b, =] -> answer:")
    for row, ans in zip(tx[:3].tolist(), ty[:3].tolist()):
        print(f"  {row} -> {ans}   check: ({row[0]} + {row[1]}) % 113 = {(row[0] + row[1]) % 113}")
