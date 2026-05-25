# Multilayer Perceptrons (MLP)

Also called a **Feed-Forward Network (FFN)**, the MLP is the second core sub-block inside a Transformer, applied after the attention mechanism. It introduces non-linearity and lets the model process each token's representation independently.

## Role in a Transformer Block

```
Input tokens
     ↓
[Multi-Head Self-Attention]
     ↓
[MLP / FFN]   ← applied independently to each token
     ↓
Output tokens
```

## Steps

For each token independently:

1. **Linear projection (up)** --> expand the token embedding into a larger hidden dimension (typically 4×)
2. **Activation (ReLU / GELU)** --> introduce non-linearity; ReLU zeros out negatives, GELU is a smoother variant
3. **Linear projection (down)** --> compress back to the original embedding dimension

```
token (d_model)
   → Linear: d_model     --> d_model * 4
   → ReLU / GELU
   → Linear: 4·d_model   --> d_model
```

## Key Points

| Property | Detail |
|---|---|
| Applied per token | Each token goes through the MLP independently, i.e. no cross-token mixing |
| Same weights | The same MLP is reused for every token position |
| Complements attention | Attention mixes information *across* tokens; MLP processes each token *within* |
| Parameter-heavy | The two projection matrices hold ~2/3 of a Transformer's total parameters |
