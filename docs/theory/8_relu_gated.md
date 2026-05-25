# ReLU vs. Gated Activations

Modern language models have moved away from the original ReLU non-linearity in their MLP blocks toward **gated activations** (e.g. SwiGLU). This page explains the structural difference.

---

## ReLU (Classical MLP Non-linearity)

**Formula:** `output = relu(x @ W)`

ReLU uses a **single branch**: apply a linear transformation, then threshold negatives to zero.

```
x  →  Linear (x @ W)  →  ReLU (max(0, z))  →  output
```

- `max(0, x)`: negatives become 0, positives pass through unchanged
- **All-or-nothing** per dimension: a feature is either killed completely or passed through linearly
- No in-between --> no way to say "pass 40% of this feature through"
- Used in the original 2017 Transformer paper

---

## Gated Activations (Modern LLMs, e.g. SwiGLU)

**Formula:** `output = swish(x @ W1) * (x @ W2)`

Gated activations use **two branches**, multiplied element-wise:

```
x  →  Linear (x @ W1)  →  Swish  ─┐
                                    × →  output
x  →  Linear (x @ W2)  ────────────┘
         ↑ content branch    ↑ gate branch
```

| Branch | Role |
|---|---|
| **Gate branch** `swish(x @ W1)` | A learned, input-dependent valve which controls how much of each feature passes through |
| **Content branch** `x @ W2` | The signal being modulated |

Both branches use the **same input** `x` but with **different learned weight matrices** (`W1` and `W2`).

---

## The Real Difference

> The key difference is **structural**, not about learnable parameters since both approaches have weight matrices.

| | ReLU | Gated (SwiGLU) |
|---|---|---|
| Branches | 1 | 2 (multiplied together) |
| Non-linearity | Fixed (hard threshold) | Continuous, input-dependent |
| Per-dimension control | Kill or pass (binary) | Dial from 0% to 100% |
| Parameters | `W` (one matrix) | `W1` + `W2` (two matrices) |
| Compute | Lower | ~2× higher |
| Quality | Good | Noticeably better at scale |

**Example:** A gated activation can say *"for this particular input, let dimension 5 pass through at 20%; for another input, let it through at 100%"*. ReLU can only kill a dimension or pass it linearly, there is no fine-grained control.

---

## Variants of Gated Activations

| Variant | Gate activation | Used in |
|---|---|---|
| **GLU** | Sigmoid | Original gating paper |
| **SwiGLU** | Swish | Llama, Mistral, PaLM |
| **GeGLU** | GELU | Various modern models |

SwiGLU is the most popular today. The extra parameter cost (two projection matrices instead of one) is consistently worth the quality gain in large models.
