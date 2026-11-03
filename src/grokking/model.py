"""A 1-layer transformer, written from scratch so every line can be explained.

Shape names used in comments:
    batch = number of examples      pos = 3 token positions [a, b, =]
    d_model = 128                   heads = 4, d_head = 32 (4 * 32 = 128)
    d_mlp = 512                     vocab_in = 114 (0..112 and "=")   vocab_out = 113
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # One weight matrix each for queries, keys, values, and the output projection.
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                   # x: [batch, pos, d_model]
        batch, pos, d_model = x.shape

        def split_heads(t):                                 # [batch, pos, d_model] -> [batch, heads, pos, d_head]
            return t.view(batch, pos, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(self.W_Q(x)), split_heads(self.W_K(x)), split_heads(self.W_V(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)       # [batch, heads, pos, pos]
        # Causal mask: a position may only look at itself and earlier positions.
        mask = torch.triu(torch.ones(pos, pos, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = scores.softmax(dim=-1)                    # attention weights; each row sums to 1

        out = pattern @ v                                   # [batch, heads, pos, d_head]
        out = out.transpose(1, 2).reshape(batch, pos, d_model)  # glue the heads back together
        self.last_pattern = pattern.detach()                # saved so we can look at it later
        return self.W_O(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)               # 128 -> 512 (with bias)
        self.W_out = nn.Linear(d_mlp, d_model)              # 512 -> 128 (with bias)

    def forward(self, x):
        hidden = F.relu(self.W_in(x))                       # ReLU: negative numbers -> 0
        self.last_hidden = hidden.detach()                  # the 512 "neurons"; analysed in fourier.py
        return self.W_out(hidden)


class Transformer(nn.Module):
    def __init__(self, vocab_in=114, vocab_out=113, n_pos=3, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_in, d_model)        # token id -> 128 numbers
        self.pos_embed = nn.Embedding(n_pos, d_model)       # position 0/1/2 -> 128 numbers
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)
        self.unembed = nn.Linear(d_model, vocab_out, bias=False)  # 128 numbers -> 113 answer scores

    def init_like_paper(self):
        """Match Nanda et al.'s starting weights: random normal, scaled by 1/sqrt(fan-in), zero biases.

        PyTorch's defaults differ a lot: nn.Embedding starts at std 1.0 (~11x bigger than the paper's
        1/sqrt(128) ~ 0.088), and nn.Linear uses a smaller uniform range.
        """
        d_model = self.embed.embedding_dim
        with torch.no_grad():
            for emb in (self.embed, self.pos_embed):
                emb.weight.normal_(0, 1 / math.sqrt(d_model))
            for lin in (self.attn.W_Q, self.attn.W_K, self.attn.W_V, self.attn.W_O, self.mlp.W_in, self.mlp.W_out):
                lin.weight.normal_(0, 1 / math.sqrt(lin.in_features))
                if lin.bias is not None:
                    lin.bias.zero_()
            self.unembed.weight.normal_(0, 1 / math.sqrt(self.unembed.out_features))  # paper: 1/sqrt(d_vocab)
        return self

    def forward(self, tokens):                              # tokens: [batch, 3]
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        resid = self.embed(tokens) + self.pos_embed(positions)  # the "residual stream": [batch, pos, d_model]
        resid = resid + self.attn(resid)                    # attention ADDS its output to the stream
        resid = resid + self.mlp(resid)                     # so does the MLP
        logits = self.unembed(resid[:, -1, :])              # read the answer at the "=" position only
        return logits                                       # [batch, 113]


if __name__ == "__main__":
    model = Transformer()
    n = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n:,}")
    print("output shape:", model(torch.zeros(5, 3, dtype=torch.long)).shape)
