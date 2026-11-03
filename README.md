# Grokking Modular Addition: Clock or Table?

A 1-layer transformer trained on `(a + b) mod 113` memorizes its training set in 200 steps, looks stuck
for thousands more, then suddenly generalizes (**grokking**). This repo reproduces that from scratch,
reverse-engineers the algorithm the network learns (it adds numbers by rotating them around a circle), proves
the algorithm with ablations, and then runs a new experiment:

> **If the network really learned a clock, it should handle kinds of inputs it has never seen.**
> Train on structured splits (never a wraparound, never `a > b`, never a certain answer, never a certain
> number in the first slot) and test on exactly what was left out.

Parts 1–2 reproduce Nanda et al. 2023 ([arXiv 2301.05217](https://arxiv.org/abs/2301.05217)).
Part 3 is our own experiment: we searched for prior work on these structured holdouts and didn't find it
(see *Prior work* at the end of Part 3).

<!-- RESULTS-TLDR -->

---

## Part 1: Grokking, reproduced

![grokking curve](figures/main_curve.png)

| | Train acc ≥ 99% | Test acc ≥ 99% |
|---|---|---|
| Paper initialization | step 200 | **step 4,500** |
| PyTorch default initialization (embeddings ~11× larger) | step 400 | **step 39,100** |

- Setup: 1-layer transformer (d_model 128, 4 heads, MLP 512, no LayerNorm, 226,688 params), trained
  full-batch with AdamW (lr 1e-3, **weight decay 1.0**) on a random 30% of all 12,769 pairs.
- Starting weights ~11× larger delayed grokking ~9× (one seed each). This is consistent with weight
  decay needing longer to shrink a larger memorizing solution (Liu et al. 2022, *Omnigrok*).

![initialization comparison](figures/compare_init.png)

## Part 2: What the network learned, and proof

**It represents every number as 4 waves.** Projecting the embedding onto the Fourier basis
(cos/sin of 2πk·x/113), 4 frequencies, **k = 16, 19, 49, 56**, hold **94.8%** of the embedding and **98.1%**
of the unembedding. All 512 MLP neurons specialize to one of these 4 frequencies (467 cleanly).

![Fourier spectrum](figures/main_fourier_embed.png)

**The waves form long before the jump.** The embedding's share in those 4 frequencies rises smoothly
from step 0 (7.5% → 42% at step 4k → 94% at 10k). The final 4 frequencies are already the top 4 by step 1,000.

![Fourier over training](figures/main_fourier_over_training.png)

**Ablations: the 4 frequencies are necessary and sufficient.**

| Edit (scored on all 12,769 pairs) | Accuracy |
|---|---|
| Keep ONLY the 4 key frequencies in the embedding | 99.9% |
| Remove the 4 key frequencies | 1.0% (chance = 0.9%) |
| Control: remove 4 random other frequencies (10 tries) | 100% (every try) |
| Logits restricted to the 8 patterns cos/sin(w_k(a+b−c)) | 100% |
| Logits with those 8 patterns removed | 0% |

The 8 "clock" patterns explain 93.5% of the variance of the logits. So the output is literally
`logit(c) ≈ Σ_k α_k cos(w_k (a + b − c))`, which peaks exactly at `c = a + b mod 113`.

![ablations](figures/main_ablations.png)

**Progress measures.** The clock part of the logits alone already reaches loss 0.19 at step 2,000, while the
test loss is still ~17. The circuit exists but is buried under memorization. The non-clock part (the
memorized table) gets erased by weight decay around steps 3k–4.5k, and that's when test accuracy jumps.

![progress measures](figures/main_progress_measures.png)

<!-- PART3 -->

---

## Run it

```bash
uv sync                                                    # Python 3.12, torch 2.15
uv run python -m grokking.train --run main                 # 40k steps, ~13 min on CPU
uv run python -m grokking.plot --run main
uv run python -m grokking.fourier --run main               # key frequencies + figures
uv run python -m grokking.ablate --run main                # ablations + progress measures
bash scripts/run_batch.sh                                  # structured-split experiment (15 runs in parallel)
uv run python -m grokking.ood                              # analyse it
```

| File | What it does |
|---|---|
| `src/grokking/data.py` | All 12,769 pairs; random split + 4 structured splits |
| `src/grokking/model.py` | 1-layer transformer from scratch (no TransformerLens), paper initialization |
| `src/grokking/train.py` | Full-batch AdamW, float64 loss on CPU, metrics + checkpoints |
| `src/grokking/fourier.py` | Fourier basis, key frequencies, neuron frequency assignment |
| `src/grokking/ablate.py` | Frequency / neuron / logit ablations, restricted + excluded loss |
| `src/grokking/ood.py` | Structured-holdout analysis and the readout-row repair |

## Limitations

- One architecture, one modulus (113), one training-set size (3,830 pairs). The structured-split results use 3 seeds each.
- The init-scale comparison in Part 1 is one seed per setting.
- Neuron-team ablations zero neurons rather than mean-ablating them, so they can overstate each team's importance.

## References

- Power et al. 2022, *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (arXiv 2201.02177)
- Nanda et al. 2023, *Progress Measures for Grokking via Mechanistic Interpretability* (arXiv 2301.05217)
- Liu et al. 2022, *Omnigrok: Grokking Beyond Algorithmic Data* (arXiv 2210.01117)
- Lyu et al. 2023, *Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking* (arXiv 2311.18817)
- He et al. 2026, *On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking* (arXiv 2602.16849)
