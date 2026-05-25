# Dropout & Co-adaptation

## The Problem: Co-adaptation

**Co-adaptation** happens when neurons become overly reliant on each other. When one neuron fires, and another simply copies it rather than learning independently. This leads to overfitting where the network memorises the training set instead of generalising.

> **Analogy:** Imagine a smart classmate sitting next to us. Every test, we copy his answers. We never actually learn the material.

## The Fix: Dropout

**Dropout** is a regularisation technique that randomly *turns off* a fraction of neurons during each training step. Because any neuron might be absent, every neuron is forced to learn on its own — reducing co-adaptation.

> **Analogy:** Now our smart classmate might randomly be absent on a test day. We can no longer rely on copying him. Therefore we have to learn the material ourselves.

## How It Works

During **training**, each neuron is independently zeroed out with probability `p` (commonly `0.5`):

```
Layer output (no dropout):  [0.9,  0.4,  0.7,  0.2,  0.8]
Dropout mask (p = 0.5):     [  1,    0,    1,    0,    1 ]
Layer output (with dropout):[0.9,  0.0,  0.7,  0.0,  0.8]
```

During **inference**, dropout is turned off and all neurons are active.

## Benefits

| Benefit | Explanation |
|---|---|
| Reduces overfitting | Neurons learn robust, independent features |
| Reduces co-adaptation | No neuron can rely on any specific other neuron |
| Acts as ensemble learning | Each training step trains a slightly different sub-network |
