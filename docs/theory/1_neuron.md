# Fully-Connected Networks & Squeeze-and-Excitation Blocks

## What Is a Fully-Connected (FC) Layer?

A **Fully-Connected (FC) layer** is the most fundamental building block of neural networks. Every input connects to every neuron in the layer — no input is hidden from any neuron.

### One Neuron

A single neuron takes inputs, multiplies each by a learned **weight**, sums them, and passes the result through an **activation function**:

Given:
- Input: `x = [2, 3, 1]`
- Weights: `w = [0.5, -0.4, 0.8]`

The neuron computes:

$$z = (2 \times 0.5) + (3 \times -0.4) + (1 \times 0.8) = 0.6$$

After ReLU: output = **0.6**

### Many Neurons in Parallel = FC Layer

A FC layer is a collection of neurons, all receiving the same input but each with their own weights, producing one output per neuron:

```text
Input x₁ ──┬──► Neuron 1 ──► output 1
Input x₂ ──┼──► Neuron 2 ──► output 2
Input x₃ ──┴──► Neuron 3 ──► output 3
           └──► Neuron 4 ──► output 4
```

In matrix form:

$$\mathbf{h} = \text{activation}(W \cdot \mathbf{x} + \mathbf{b})$$

Where `W` is a weight matrix of shape `[outputs × inputs]` and `b` is a bias vector.

### A Fully-Connected Network = Multiple Layers Stacked

```text
Input → [FC Layer 1 + ReLU] → [FC Layer 2 + ReLU] → [FC Layer 3 + Sigmoid] → Output
```

Each layer builds higher-level abstractions on top of the previous layer's output.

---

### Why Activation Functions Matter

Without activation functions between layers, stacking multiple FC layers collapses to a single linear transformation:

$$W_2(W_1 \mathbf{x}) = (W_2 W_1)\mathbf{x} = W_{\text{combined}} \mathbf{x}$$

Two layers without activation = one layer. Activations like **ReLU** introduce non-linearity, allowing the network to learn complex, curved decision boundaries.

