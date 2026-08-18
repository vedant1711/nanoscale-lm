# Methodology

Every algorithm in NanoScale-LM, with its formula, its citation, and a pointer to the
test that checks it. This page is the seed of the paper.

The organising rule: **implement the math, test the math, then explain the math.** Where
a claim is measured, the measurement lives in [Results](results.md) and is produced by a
committed script. Where a claim is not measured, this page says so.

---

## 1. Tokenizer: byte-level BPE

**Reference.** Sennrich et al., *Neural Machine Translation of Rare Words with Subword
Units* (arXiv:1508.07909); Radford et al., GPT-2 (byte-level variant).

Initialise the vocabulary with the 256 byte values, so every UTF-8 string is
representable and `<unk>` cannot occur. Split the corpus with a pre-tokenization regex so
merges never cross a word/punctuation boundary. Then repeatedly mint a token for the most
frequent adjacent pair.

The implementation keeps the corpus as *unique* pre-token symbol sequences with
multiplicities, an incremental `Counter` of pair frequencies, and an inverted index
`pair → {word indices}`. A merge therefore touches only the words containing it. That is
the difference between training a 32k vocabulary in seconds and in an hour.

!!! note "A property that is false, and worth knowing"
    BPE token counts are **not subadditive**: with the committed `nano` vocabulary,
    `encode("eps")` is 1 token, `encode("ep")` is 1 token, and `encode("epsep")` is 3.
    Greedy rank-ordered merging is path-dependent and cannot backtrack. Practical
    consequence: you cannot cache a tokenization by concatenating cached pieces, unless
    the join lands on a pre-token boundary, where encoding *is* exactly concatenative.
    Both facts are pinned in `tests/property/test_tokenizer_properties.py`.

**Tests.** Exact round-trip over 15 scripts plus hypothesis over arbitrary Unicode;
vocabulary layout; merges never cross word boundaries, and a three-part `tiktoken`
comparison (exact pre-tokenization equality, in-domain length parity, bounded
out-of-domain degradation).

---

## 2. Model

### Attention

**Reference.** Vaswani et al. (arXiv:1706.03762); Ainslie et al., GQA (arXiv:2305.13245).

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V$$

**Grouped-query attention** gives each group of query heads one shared key/value head.
With `n_heads=8, n_kv_heads=4` the KV cache halves, and at decode time, where the
bottleneck is memory bandwidth rather than FLOPs, that is close to a 2× throughput win.

**QK-norm** RMS-normalizes queries and keys before the dot product, bounding the logits
entering the softmax regardless of activation scale. It is applied **before** RoPE: RoPE
is norm-preserving so the normalization commutes with it, but the *learned* QK gain does
not, so the order is a real choice and is pinned by a test that first asserts the two
orderings actually differ.

**Tests.** Our manual attention equals `F.scaled_dot_product_attention` in fp32 and fp64
across MHA/GQA/MQA, with and without QK-norm and padding masks, and equals a hand-written
`softmax(QKᵀ/√d + M)V` independently of SDPA.

### RoPE

**Reference.** Su et al., *RoFormer* (arXiv:2104.09864).

Treat consecutive coordinate pairs as points in the plane and rotate pair $i$ at
position $m$ by $m\theta_i$ with $\theta_i = \text{base}^{-2i/d}$:

$$\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Rotations are orthogonal, so position never changes a vector's magnitude, and because
rotations compose additively, $\langle \mathrm{RoPE}(q,m), \mathrm{RoPE}(k,n)\rangle$
depends only on $m-n$. That relative property is the entire point and is tested to
$10^{-12}$ rather than assumed.

This repo uses the paper's **interleaved-pair** convention, not LLaMA's rotate-half
pairing. The two are equivalent up to a permutation of head coordinates but are *not*
weight-compatible, so the convention is pinned by a test.

### Norm and MLP

**RMSNorm** (Zhang & Sennrich, arXiv:1910.07467) drops LayerNorm's mean-centering:

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac1d\sum_i x_i^2 + \varepsilon}} \odot g$$

**SwiGLU** (Shazeer, arXiv:2002.05202): $(\mathrm{SiLU}(xW_g) \odot xW_u)W_d$, with the
hidden width at $\tfrac83 d_{\text{model}}$ so the parameter count matches a classic
$4d$ two-matrix MLP and the comparison against ReLU² is fair.

### Initialisation

Zero-init every projection that writes into the residual stream: attention `o_proj`,
MLP `down_proj`, the LM head. The network then starts as the identity on the residual
stream and emits uniform logits, so the **initial loss is exactly $\ln V$**. A test
asserts that, which is a sharp check that nothing is silently non-zero.

This interacts with weight tying: a tied head *is* the embedding matrix, so zeroing it
would destroy the input embeddings. Tied models keep a random head and do not start at
$\ln V$: documented and tested rather than left as a surprise.

---

## 3. Optimizer, Muon

**Reference.** Jordan et al., Muon; the modded-nanoGPT speedrun lineage.

Adam normalises each *coordinate*. For a weight **matrix** that is the wrong geometry:
what matters is the matrix's action as a linear map, and a momentum buffer is typically
dominated by one or two singular directions. Muon replaces the momentum matrix with the
orthogonal factor of its polar decomposition,

$$M = U\Sigma V^\top \longrightarrow O = UV^\top,$$

so every singular value becomes 1 and the update moves equally in every direction the
momentum has support on.

An SVD per step would be far too slow, so Muon runs five **Newton–Schulz** iterations of

$$X \leftarrow aX + b(XX^\top)X + c(XX^\top)^2X, \qquad (a,b,c) = (3.4445, -4.7750, 2.0315)$$

on a spectrally-normalised $X$. Those coefficients are deliberately *not* the textbook
$(1.5, -0.5, 0)$: they overshoot near zero, which converges much faster for the small
singular values that dominate a real momentum matrix, at the cost of a fixed point near
$1 \pm 0.3$ rather than exactly 1. Only the direction is used, so that is the right
trade, and the tests measure the resulting singular values (observed range 0.68–1.13)
rather than assuming them.

**Scope.** 2D hidden matmul weights only. Embeddings are a lookup table, norm gains are
vectors of independent scalars; neither has the geometry the orthogonalisation exploits.

**Result.** Muon reaches the target loss in **50 steps against AdamW's 105** at `nano`
scale: see [Results](results.md#optimizer-ablation).

!!! warning "Where Muon does *not* help"
    On a convex single-matrix least-squares problem, AdamW beats Muon at every learning
    rate we tried. Adam's diagonal preconditioner is close to optimal on a
    well-conditioned quadratic and no spiky momentum spectrum accumulates. Muon's
    advantage needs **depth and stochasticity**. Both results are pinned as tests.

**Cautious weight decay.** Decay is applied only where it agrees with the optimizer's own
update: with the step written $p \leftarrow p - \text{lr}\,u$, the two point the same way
exactly when $u_i p_i > 0$. Elsewhere the decay is skipped rather than partially
cancelling the update.

---

## 4. Alignment

### DPO

**Reference.** Rafailov et al. (arXiv:2305.18290).

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

The bracketed quantities are the **implicit rewards**. Log-probabilities are *summed*
over response tokens, which is what the derivation gives, and is the origin of DPO's
length bias: a longer response has more terms to accumulate advantage over, so the loss
can be reduced by lengthening rather than improving.

### SimPO

**Reference.** Meng et al., SimPO.

$$\mathcal{L}_{SimPO} = -\mathbb{E}\left[\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

Length normalisation removes the mechanical incentive to lengthen; dropping the reference
model halves the memory; the target margin $\gamma$ asks for preference *by a margin*.

### The finding this project actually produced

All three preference arms reach 100% preference accuracy, but plain DPO's mean per-token
log-probability of the **chosen** response *falls* (−0.045) while the margin rises. DPO
optimises a *difference*, so it can satisfy itself by pushing both sides down. A rising
margin and a degrading model are entirely compatible.

Adding an RPO-style auxiliary NLL on the chosen response anchors the absolute likelihood
and flips that to +0.010, and `DPO+NLL` is the only arm that **wins** the scripted
head-to-head against the SFT model (3–0–37, versus plain DPO's 0–7–33). See
[Results](results.md#alignment).

### GRPO (optional track)

**Reference.** Shao et al., DeepSeekMath (arXiv:2402.03300).

PPO needs a learned critic. GRPO removes it by sampling a **group** of $G$ completions per
prompt and using the group as its own baseline:

$$A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G}) + \epsilon}$$

Rewards here are **programmatic**: arithmetic answers checked by evaluation, so there
is no reward model to hack. A unanimous group has exactly zero advantage and is skipped:
if all $G$ samples are right, or all wrong, the comparison says nothing.

**2026 successors to cite:** GSPO (sequence-level importance ratios) and DHPO (hybrid
token+sequence). Neither is implemented here.

---

## 5. Distillation

**References.** Hinton et al. (arXiv:1503.02531); Kim & Rush, SeqKD (arXiv:1606.07947);
Gu et al., MiniLLM (arXiv:2306.08543).

The three objectives differ in one choice, which direction of the KL, and what to sample
from, and that choice is mechanical:

- **Forward KL** minimises $KL(p\|q)$. The integrand $p\log(p/q)$ explodes wherever the
  teacher has mass and the student does not, forcing the student to **cover every mode**
  including the teacher's low-confidence tail.
- **Reverse KL** minimises $KL(q\|p)$. The integrand $q\log(q/p)$ only penalises mass the
  student *invents*, so it may ignore the tail and concentrate on modes it can represent
, the right trade for a strictly smaller student. The expectation is under $q_\theta$,
  so it requires **on-policy** rollouts and a policy-gradient estimator.
- **SeqKD** trains by MLE on teacher samples, sidestepping the asymmetry.

The $\tau^2$ factor on the forward-KL term is not cosmetic: dividing logits by $\tau$
scales KD gradients by $1/\tau^2$, so without it, tuning $\tau$ silently retunes $\alpha$.

**Two details that were not optional**, both established by measurement:

1. **A warm-start is required.** On-policy reverse KL from a randomly-initialised student
   samples noise the teacher finds uniformly unlikely; the reward carries no signal.
   Measured without it: student perplexity ~1000 against a teacher at ~3.
2. **The on-policy phase needs a smaller step.** A REINFORCE-style estimator is far
   noisier than a supervised one; without `onpolicy_lr_scale` the policy gradient undoes
   the warm-start (perplexity 41 rather than 2.5).

**Result.** Reverse KL reaches perplexity 2.50 against forward-KL's 2.05, *worse*: with
a repetition rate of 0.0000 against 0.0064 and SeqKD's 0.0372, and below the teacher's own
0.0383. That is MiniLLM's qualitative finding reproduced: judging distillation by
perplexity alone ranks these backwards.

**Documented next steps:** GKD (interpolating on/off-policy over divergences) and
DistiLLM.

---

## 6. Quantization

**References.** Frantar et al., GPTQ (arXiv:2210.17323); Lin et al., AWQ
(arXiv:2306.00978); Xiao et al., SmoothQuant (arXiv:2211.10438).

### What RTN gets wrong

Round-to-nearest minimises error **in the weights**. What matters is error in the layer's
**output**:

$$\arg\min_{\hat W} \|WX - \hat W X\|_2^2$$

Those coincide only if $X$ is white. Transformer activations are emphatically not.

### GPTQ

Expanding that objective gives a quadratic form with Hessian $H = 2XX^\top$. GPTQ
quantizes column by column and distributes each column's rounding error over the
*not-yet-quantized* columns:

$$\delta_j = \frac{w_j - \mathrm{quant}(w_j)}{[H^{-1}]_{jj}}, \qquad W_{:,j+1:} \mathrel{-}= \delta_j [H^{-1}]_{j,\,j+1:}$$

Implementation details that matter: Cholesky rather than an explicit inverse (a
near-singular Hessian does not invert stably); dampening (calibration Hessians are
routinely rank-deficient); lazy block updates (one matmul instead of a chain of rank-1
updates), and activation ordering.

!!! tip "GPTQ has the worst weight error and the best perplexity"
    At 2 bits GPTQ's mean relative *weight* error is 0.714 against RTN's 0.459, nearly
    double, yet it produces the better model (perplexity 1.500 vs 1.541). That is the
    method's thesis, not a contradiction: it will happily accept a larger perturbation on
    a weight multiplying a quiet channel to buy a smaller one on a loud channel. Ranking
    these methods by weight error ranks them backwards.

### AWQ

Scaling input channel $j$ by $s_j$ and the corresponding weight column by $1/s_j$ leaves
the output identical: $(X \oslash s)(W \odot s)^\top = XW^\top$. Quantizing $W \odot s$
therefore protects salient channels without any mixed-precision storage. AWQ
parameterises $s = \mathrm{mean}(|X_j|)^\alpha$ and grid-searches $\alpha$ on the true
output error.

### Effective bits

The frontier plots against **effective** bits, nominal width plus the amortised cost of
the stored scales. A "4-bit" model at group size 64 with fp16 scale and zero-point costs
4.5 bits per weight. Plotting against the nominal width lets a method buy accuracy with
smaller groups and appear to win for free.

---

## 7. Speculative decoding

**References.** Leviathan et al. (arXiv:2211.17192); Chen et al. (arXiv:2302.01318);
Cai et al., Medusa (arXiv:2401.10774).

### The rule, and why it is exact

Draw $x \sim q$; accept with probability $\min(1, p(x)/q(x))$; on rejection, resample from
the normalised residual $(p-q)_+ / \sum_v (p-q)_+$.

Fix a token $x$. The accepted branch contributes $q(x)\min(1, p(x)/q(x)) = \min(q(x),p(x))$.
Rejection occurs with probability $\beta = 1 - \sum_v \min(q,p)$, and since the residual's
normaliser is exactly $\beta$, the resampled branch contributes $\max(0, p(x)-q(x))$.
Summing:

$$\min(q(x),p(x)) + \max(0, p(x)-q(x)) = p(x)$$

for every $x$. The output distribution is $p$ **exactly**: not approximately: regardless
of draft quality. A bad draft only lowers the acceptance rate, i.e. the speedup. The
expected acceptance rate is $\sum_v \min(p,q) = 1 - TV(p,q)$.

### How it is verified

Three independent levels: algebraically (the branches sum to $p$ in fp64); statistically
(120k samples match direct sampling from $p$ within a total-variation bound, **and a
deliberately broken rule fails the same test**, so the tolerance demonstrably has teeth);
and end-to-end (greedy speculative decoding is token-for-token identical to greedy
autoregressive decoding).

### Medusa

Extra heads on the target's own final hidden state predict several tokens ahead, removing
the second model. Candidate paths are packed into one sequence with an **ancestor-only
attention mask**, so one forward pass evaluates every root-to-leaf path.

Positions matter as much as the mask: a node's index in the packed sequence is not its
position in the sequence it belongs to, so position IDs are derived from tree **depth**.
Feeding packing-order positions makes RoPE place sibling branches at the wrong distance
from the prefix: a bug the path-equivalence test caught.

Medusa's accept rule is exact-match, which is **not** distribution-preserving. Draft–target
is lossless by construction; Medusa trades a little fidelity for not needing a draft model.

**Documented next step:** EAGLE-2/EAGLE-3, which draft on the target's hidden features.

---

## 8. Reporting

**Reference.** Miller (Anthropic), *Adding Error Bars to Evals* (arXiv:2411.00640).

Perplexity is reported with the standard error of the mean per-token NLL. The tiny
benchmark reports a binomial standard error (~9 points at $n=28$ near 50%), which is what
makes it obvious that a 5-point difference on it is noise. The ablation harness refuses
to name a winner on a final-loss gap under 2% at a single seed.
