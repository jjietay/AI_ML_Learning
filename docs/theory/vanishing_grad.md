# Vanishing Gradient Problem

The vanishing gradient problem occurs in very deep networks during backpropagation, where gradients shrink exponentially as they travel back through layers which eventually becoming so small that early layers stop learning.

## What Happens

During `loss.backward()`, gradients are calculated and sent backwards through every layer. In a deep network:

1. The gradient reaches the earlier hidden layers having been **multiplied many times**
2. Each multiplication can shrink the value toward zero
3. The gradient effectively "vanishes" before it reaches the early layers

## Consequences

| Symptom | Effect |
|---|---|
| Early layers stop learning | Weights near the input never update |
| Only the last few layers update | The network learns very shallow features |
| Deep networks underperform shallow ones | Depth provides no benefit if gradients can't flow |

## Common Fixes

- **Residual connections** (ResNets) which adds skip connections so gradients have a direct path backward
- **Better activations** such as ReLU avoids the saturation that crushes sigmoid gradients
- **Batch normalisation** keeps activations in a healthy range at each layer
- **Gradient clipping** caps exploding gradients (the opposite problem)
