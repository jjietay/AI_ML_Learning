# Ensemble Methods

**Ensemble learning** combines multiple models to produce a better prediction than any single model alone. The key idea: diverse models make different mistakes, and averaging them out improves overall accuracy.

## Combining Predictions

Two common strategies for merging individual model outputs:

| Method | How it works | Best for |
|---|---|---|
| **Majority vote** | Choose the most common prediction across all models | Classification |
| **Summation** | Sum the probability scores for each class; pick the highest | Classification with probabilities |

> Note: Averaging is equivalent to summation for the purpose of picking the winner, so the extra division step is unnecessary.

---

# Bootstrap Aggregating (Bagging)

## Introduction

**Bagging** reduces overfitting by training each model on a *different random sample* of the data, rather than all models seeing the same training set.

## How It Works

1. From the original dataset, **randomly sample with replacement** to create a new dataset of the same size
2. Train one model on this new dataset
3. Repeat for every model in the ensemble where each sees a slightly different dataset
4. At inference time, aggregate all model predictions (majority vote or sum)

> **With replacement** means the same data point can appear multiple times in one dataset, and some points may not appear at all (these are called *out-of-bag* samples and can be used for validation).

## Effect

Because each model is trained on a different subset of the data, individual models overfit to different noise. When combined, the noise cancels out and the true signal is amplified.

---

# Boosting

## Introduction

**Boosting** improves performance on hard or rare examples by iteratively focusing each new model on the mistakes of the previous one. It is more powerful than bagging but slower to train.

## How It Works

1. Train a base model on the original dataset with **equal weights** on all samples
2. Identify which samples were **misclassified**
3. **Increase the weights** of misclassified samples (they matter more now)
4. **Decrease the weights** of correctly classified samples
5. Train the next model on the re-weighted dataset
6. Repeat the training where each model in the sequence focuses more on what previous ones got wrong
7. Final prediction is a **weighted sum** of all models:

$$
\text{prediction} = w_1 \cdot \text{out}_1 + w_2 \cdot \text{out}_2 + w_3 \cdot \text{out}_3
$$

where each weight \(w_i\) reflects that model's accuracy on its training data.

## Bagging vs. Boosting

| | Bagging | Boosting |
|---|---|---|
| Training | Parallel (models are independent) | Sequential (each depends on previous) |
| Focus | Reduce variance / overfitting | Reduce bias / improve hard examples |
| Speed | Faster | Slower |
| Example algorithms | Random Forest | AdaBoost, XGBoost, LightGBM |
