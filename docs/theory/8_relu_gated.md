# ReLU vs Gated Activations

## ReLU (classical MLP nonlinearity)
- Formula: `output = relu(x @ W)`
- ONE branch: linear transformation, then apply ReLU
- ReLU itself: `max(0, x)` — negatives become 0, positives pass through unchanged
- All-or-nothing per dimension: a dimension is either killed or passed through linearly
- Used in the original 2017 transformer paper

## Gated (modern LLM nonlinearity, e.g., SwiGLU)
- Formula: `output = swish(x @ W1) * (x @ W2)`
- TWO branches, multiplied element-wise:
    - Gate branch: `swish(x @ W1)` — acts as a learned, input-dependent valve
    - Content branch: `x @ W2` — the signal being gated
- Both branches use the same input x but with different learned matrices (W1 and W2)
- Used in modern LLMs: Llama, Mistral, PaLM, etc.

## Key Difference (the actual one)
- NOT "ReLU has no learnable params, gated does" — both have learnable params (inside the W matrices)
- The real difference is structural:
    - ReLU = 1 branch → fixed nonlinearity → all-or-nothing per dim
    - Gated = 2 branches multiplied → continuous, input-dependent valve
- Example: gating can say "for this input, dial dimension 5 down to 20%; for that input, let it through at 100%"
- ReLU can only kill or pass linearly — no in-between

## Cost of Gating
- More parameters (two projection matrices W1, W2 instead of one W)
- More compute per forward pass
- Worth it in practice: noticeable quality gain in large models

## Variants of Gated
- GLU: original, uses sigmoid as the gate activation
- SwiGLU: uses swish — most popular in modern LLMs (Llama, Mistral)
- GeGLU: uses GELU as the gate activation