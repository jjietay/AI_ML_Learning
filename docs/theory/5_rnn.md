# Recurrent Neural Networks (RNN)

## 1. Core Concepts

| Concept | Explanation |
|---|---|
| **Hidden state** \(h_t\) | The network's "memory" which is updated at each timestep and passed forward |
| **Unrolling** | Visualising the RNN as a chain of identical networks, one per timestep |
| **Weight sharing** | The *same* weights \(W_{xh}\), \(W_{hh}\), \(W_{hy}\) are reused at every timestep, which enables variable-length sequences |

---

## 2. Main Formulas

The operations inside a standard RNN cell are typically broken down into two steps: updating the memory and calculating the prediction.

An RNN cell computes two things at each timestep:

**Hidden state update (memory):**

$$h_t = \alpha (W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

| Symbol | Meaning |
|------|-------|
| \(h_t\) | New hidden state at timestep \(t\) |
| \(h_{t-1}\) | Hidden state from the previous timestep |
| \(x_t\) | Current input at timestep \(t\) |
| \(W_{xh}\) | Weights applied to the input |
| \(W_{hh}\) | Weights applied to the previous hidden state |
| \(b_h\) | Bias term |
| \(\alpha\) | Activation function — typically \(\tanh\) or \(\text{ReLU}\) |

**Output calculation (prediction):**

$$y_t = W_{hy} h_t + b_y$$

---

## 3. Types of RNNs

| Type | Shape | Example use case |
|---|---|---|
| **Many-to-Many (Synced)** | Input at every step, output at every step | Stock price forecasting, video frame classification |
| **Many-to-One** | Full sequence in, single output | Sentiment analysis, spam detection |
| **One-to-Many** | Single input, sequence output | Image captioning |
| **Many-to-Many (Delayed)** | Encoder compresses sequence and then Decoder generates sequence | Machine translation |

---

## 4. Backpropagation Through Time (BPTT)

Training an RNN requires a time-aware variant of backpropagation called **BPTT**

### Process

1. **Forward pass** --> compute outputs from \(t=1\) to \(t=T\)
2. **Calculate loss** --> evaluate prediction error at relevant timesteps
3. **Backward pass** --> compute gradients from \(t=T\) back to \(t=1\)
4. **Sum gradients** --> because weights are shared, the total gradient for \(W_{xh}\) is the *sum* of its gradients across all timesteps
5. **Weight update** --> apply SGD / Adam to minimise loss

### The Fatal Flaw: Vanishing & Exploding Gradients

BPTT multiplies \(W_{hh}\) by itself once per timestep:

| Problem | Cause | Effect |
|---|---|---|
| **Vanishing gradient** | \(\|W_{hh}\| < 1\) → gradients shrink exponentially | Early timesteps stop learning, RNN forgets long-range dependencies |
| **Exploding gradient** | \(\|W_{hh}\| > 1\) → gradients grow exponentially | `NaN` losses, training crashes |

> **Fix:** LSTMs and GRUs were designed specifically to address vanishing gradients

---

## 5. Memory Compression Over Time

Treating a single RNN cell as a function \(f\), the hidden state nests recursively:

| Timestep | Hidden state |
|---|---|
| \(t=1\) | \(h_1 = f(x_1,\ h_0)\) |
| \(t=2\) | \(h_2 = f(x_2,\ f(x_1, h_0))\) |
| \(t=3\) | \(h_3 = f(x_3,\ f(x_2,\ f(x_1, h_0)))\) |

By timestep 3, the hidden state contains deeply nested, mathematically compressed information from every prior input. However early inputs become harder and harder to "un-compress", which is exactly why the vanishing gradient problem hurts RNNs so much.
