# StyleGAN Exploration: Style-Based GAN in TensorFlow/Keras on CIFAR-10

![TensorFlow Keras](https://img.shields.io/badge/framework-TensorFlow%2FKeras-orange)
![CIFAR-10 dataset](https://img.shields.io/badge/dataset-CIFAR--10-blue)
![Image generation](https://img.shields.io/badge/task-image%20generation-2f855a)
![Jupyter Notebook](https://img.shields.io/badge/interface-Jupyter%20Notebook-f37626)

A compact **StyleGAN implementation in TensorFlow/Keras** for **image generation on CIFAR-10**, written as a single Jupyter notebook. It shows how a **mapping network** and **AdaIN (adaptive instance normalization) style modulation** drive image synthesis, and demonstrates two **latent space** probes: **style mixing** and **latent interpolation**.

**Keywords:** StyleGAN, generative adversarial network (GAN), style-based generator, AdaIN, mapping network, latent space, style mixing, latent interpolation, CIFAR-10, TensorFlow, Keras, deep learning, image synthesis, Jupyter Notebook.

Everything lives in [`StyleGAN.ipynb`](StyleGAN.ipynb): data prep, model, training loop, and visualizations.

> **Scope.** Educational implementation, not a reproduction of StyleGAN / StyleGAN2 / StyleGAN3. No equalized learning rate, path-length regularization, progressive growing, or weight demodulation.

## Highlights

- Mapping network: 8-layer MLP, $z \rightarrow w$ (128-d each)
- Learned $4 \times 4$ constant input, 8 styled conv blocks with AdaIN and per-layer noise
- Nearest-neighbor upsampling to $32 \times 32$ RGB output
- Convolutional discriminator with non-saturating logistic (softplus) loss
- Style mixing and 9-step latent interpolation after training

## Architecture

**Generator**

1. Latent mapping network: normalized $z$ goes through 8 dense + leaky-ReLU blocks (He-normal init) to give $w$.
2. Learned constant: trainable $4 \times 4 \times 128$ tensor, broadcast over the batch.
3. Styled convolutions: 8 blocks, resolution $4 \rightarrow 8 \rightarrow 16 \rightarrow 32$.
4. Output: $1 \times 1$ conv with `tanh` to 3 channels.

AdaIN, with scale and bias produced by an affine map of $w$:

```text
AdaIN(x, w) = scale(w) * (x - mean(x)) / sqrt(var(x) + eps) + bias(w)
```

**Discriminator**

4 downsampling conv blocks (32, 64, 128, 128 channels), flatten, single logit output.

**Losses**

```text
L_D = mean(softplus(-D(real))) + mean(softplus(D(fake)))
L_G = mean(softplus(-D(fake)))        # fake = G(z)
```

## Configuration

| Parameter | Value |
|---|---:|
| Dataset | CIFAR-10 (first 20,000 train images, labels unused) |
| Resolution | $32 \times 32$ RGB, scaled to $[-1, 1]$ |
| Batch size | 64 |
| Latent dim $Z$ / style dim $W$ | 128 / 128 |
| Styled conv stages | 8 |
| Epochs | 10 |
| Optimizer | Adam, lr $2 \times 10^{-4}$, $\beta_1=0$, $\beta_2=0.99$ |
| Seed | 42 |

All values are set near the top of the notebook.

## Latent-space experiments

**Style mixing.** Take the first $k$ per-layer styles from $w_A$ and the rest from $w_B$, for $k \in \{2, 4, 6\}$:

$$
w_i =
\begin{cases}
w_{A,i}, & i < k \\
w_{B,i}, & i \geq k.
\end{cases}
$$

Early layers are expected to affect coarse structure and later layers finer detail. This is something to inspect visually, not a guaranteed result.

**Latent interpolation.** Linear path between two latents over 9 steps:

$$
z(t) = (1-t)z_1 + tz_2, \qquad t \in [0,1].
$$

## Quick start

Requirements: Python 3.9+, TensorFlow 2.x, NumPy, Matplotlib, Jupyter.

```bash
git clone https://github.com/jeyadev-jd/StyleGAN-exploration.git
cd StyleGAN-exploration
pip install tensorflow numpy matplotlib jupyter
jupyter notebook StyleGAN.ipynb
```

Run cells top to bottom. CIFAR-10 downloads automatically on first run.

## Outputs

- Grid of real CIFAR-10 samples
- Generator and discriminator parameter counts
- Per-epoch loss curves
- Grid of generated images
- 5-image style-mixing comparison
- 9-image interpolation strip

Losses alone do not measure image quality. For a proper evaluation, add FID / KID, precision/recall, multiple seeds, and longer training.

## Limitations and next steps

Small subset, 10 epochs, no checkpointing, no quantitative metrics. Ideas:

- Train on full CIFAR-10 for much longer
- Add checkpoint save/restore and experiment logging
- Add FID / KID and precision/recall
- Compare direct latent injection vs. mapped style injection
- Swap AdaIN for StyleGAN2 weight demodulation
- Add class conditioning

## Repository structure

```text
.
├── StyleGAN.ipynb   # Model, training loop, latent-space analysis
└── README.md
```

## References

- Goodfellow et al., *Generative Adversarial Nets*, 2014.
- Karras, Laine, Aila, *A Style-Based Generator Architecture for Generative Adversarial Networks*, 2019.
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 (CIFAR-10).

## Credits

Forked from [Naveenvaidani/StyleGAN-exploration](https://github.com/Naveenvaidani/StyleGAN-exploration). Notebook and original implementation by the upstream author; README revised by [@jeyadev-jd](https://github.com/jeyadev-jd).

## License

MIT License. See [LICENSE](LICENSE).
