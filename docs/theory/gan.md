# Generative Adversarial Networks (GAN)

A GAN pits two neural networks against each other in a competitive training loop. Through this adversarial process, the **Generator** learns to produce outputs realistic enough to fool the **Discriminator**.

## The Two Networks

| Network | Role | Input | Output |
|---|---|---|---|
| **Generator** | Creates fake samples | Random noise vector `z` | Fake image / data |
| **Discriminator** | Judges real vs. fake | Image (real or generated) | Probability: real or fake |

## Training Loop

1. **Generator** produces a fake sample from random noise
2. **Discriminator** sees both real samples and the fake and then outputs a score for each
3. **Discriminator loss** will penalise discriminator when it mis-classifies real as fake (or vice versa)
4. **Generator loss** — penalised when the discriminator correctly identifies its output as fake
5. Both networks update their weights, then repeat

```
Noise z → [Generator] → Fake sample
                              ↓
Real data ────────→ [Discriminator] → Real / Fake score
```

## Convergence

Training converges when the Generator produces samples so realistic that the Discriminator can do no better than random guessing (50% accuracy). At this point, the Generator has effectively learned the distribution of real data.

## Common Challenges

- **Mode collapse**: Generator learns to produce only a few types of outputs
- **Training instability**: the two networks can diverge rather than converge
- **Evaluation difficulty**: hard to quantify output quality objectively (Fréchet Inception Distance or FID score is commonly used)

## Fréchet Inception Distance (FID)

FID measures how similar generated images are to real images by comparing them in **feature space**. Note: not pixel space.

### How It Works

1. Pass a batch of real images and a batch of generated images through a pretrained **Inception v3** network
2. Extract feature activations from an intermediate layer
3. Model each batch's features as a multivariate Gaussian and compute the **mean** ($\mu$) and **covariance** ($\Sigma$)
4. Compute the Fréchet distance between the two distributions:

$$
FID = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

### Key Points

- **Lower FID = better** which means generated distribution is closer to real
- Captures both **quality** (realism) and **diversity** (variety)
- A GAN suffering from **mode collapse** will score poorly on diversity even if individual samples look realistic