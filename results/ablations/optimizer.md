# Ablation, optimizer

**Question.** Does routing hidden matmul weights to Muon beat sending everything to AdamW, at equal step budget and seed?

![optimizer](optimizer.png)

| variant | val loss | val ppl | steps → target | seconds → target | tok/s |
|---|---|---|---|---|---|
| AdamW only | 0.4039 | 1.498 | 105 | 33.51 | 6286.8 |
| Muon + AdamW | 0.3896 | 1.476 | 50 | 21.09 | 4331.1 |

## Findings

- **Muon + AdamW is 3.5% better** on final validation loss (0.3896 vs 0.4039). It also reaches the target loss in **2.10x fewer steps** (50 vs 105).
  <br/>*2D hidden matrices to Muon; embeddings, head, norms to AdamW.*

## How to read this

All arms share one seed, one data order, one schedule and a fixed step budget; they differ only in the fields named in the variant. Runs are `nano` tier on CPU.

**Steps-to-target is the trustworthy column; wall-clock is not.** These runs were executed sequentially on a shared laptop, so tokens/s is sensitive to whatever else the machine was doing. A per-step cost difference that is real (Muon adds five Newton-Schulz matmuls per 2D weight) is therefore mixed with measurement noise here. Treat the seconds columns as indicative and the step counts as the result.

**These are single-seed results on a ~5M-parameter model trained on a synthetic corpus.** They are directional confirmations (or non-confirmations) of published findings obtained at 100–1000× this scale, not independent evidence about them. Differences below 2% in final loss are reported as *no measurable difference*, because at one seed that is what they are. A lever that matters at scale can be invisible here; a small model in a narrow domain is exactly the regime where stability aids have little to stabilise.

Reproduce with: `python scripts/ablate.py --suite optimizer`
