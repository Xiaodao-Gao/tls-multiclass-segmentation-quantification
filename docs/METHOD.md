# Implemented method

## Architecture contract

`PatentSegmenter` uses an SMP 0.5.0 U-Net with EfficientNet-B3. For input `[B,3,224,224]`, the native encoder returns six tensors with channels `(3,40,32,48,136,384)`. Only list indices `[2,3,4,5]` feed the auxiliary fusion branch:

```text
Encoder stage 2: [B, 32,56,56] → 1×1 adapter → [B, 64,56,56]
Encoder stage 3: [B, 48,28,28] → 1×1 adapter → [B,128,28,28]
Encoder stage 4: [B,136,14,14] → 1×1 adapter → [B,256,14,14]
Encoder stage 5: [B,384, 7, 7] → 1×1 adapter → [B,512, 7, 7]
```

Here channels and spatial resolution are separate quantities. The native decoder receives the complete, unmodified encoder tensor list. No decoder feature supplies the fusion pyramid. The supplied fusion mathematics is preserved: the contextual block maps the deepest adapted feature from 7×7 to 4×4; the four APF stages align and fuse at 7×7, 14×14, 28×28 and 56×56, with output channels 256,128,64,32. FAB plus GFU stages provide global guidance at these levels, yielding `D` with shape `[B,32,56,56]`.

The class historically named `SynchronizedBatchNorm2d` in the supplied branch actually subclasses GroupNorm. It has no batch running statistics or distributed synchronization. The native SMP U-Net retains its own original normalization. Attention axes, interpolation, activations and dropout in the shared fusion and denoising components remain those of the supplied source; the refactor does not replace their mathematics with another attention implementation.

## Training noise and one reconstruction

The default schedule has **50 array indices, 0 through 49**. Define `beta[t] = linspace(0.0001,0.02,50)[t]`, `alpha[t]=1-beta[t]`, `alpha_bar[t]=product(alpha[0:t+1])`, `a[t]=sqrt(alpha_bar[t])` and `b[t]=sqrt(1-alpha_bar[t])`.

For each sample in a minibatch:

1. Compute its current fused feature `D` from the RGB image.
2. Independently sample `t` uniformly from `{0,…,49}` and a fresh tensor `epsilon ~ N(0,I)` with the same shape as `D`. A training iteration does not run every forward-diffusion index sequentially. Random sampling does not guarantee all indices occur in a single epoch.
3. Directly construct `D_t = a[t] D + b[t] epsilon`. The sampled noise is the aggregate noise used in this closed-form construction, not an earlier iteration's noise or only the last incremental forward-step noise.
4. Predict `epsilon_hat = denoiser(D_t,t)`. The MSE target is **the same epsilon sampled in step 2**.
5. Reconstruct once: `D_hat = (D_t - b[t] epsilon_hat) / (a[t] + 1e-8)`.
6. Project and inject `D_hat`, compute segmentation and noise objectives, and backpropagate jointly. Features are not detached from the encoder/fusion branch.

The denoiser uses a 128-dimensional sinusoidal time embedding, two 32-channel convolutional layers with time conditioning, 4×4 patch embedding to 128 dimensions, 196 tokens, two DiT blocks with four heads and MLP ratio 4, and a transpose-convolution unpatch layer back to 32 channels. AdaLN modulation and the unpatch output layer have their supplied zero initializations.

## Deterministic inference

Set the model to evaluation mode, compute the clean feature `D`, and make **one** denoiser call with fixed index `t=0`:

```text
epsilon_hat = denoiser(D,0)
D_hat = (D - b[0] epsilon_hat) / (a[0] + 1e-8)
```

No Gaussian tensor is sampled during inference. At default index zero, `a[0]≈0.99995` and `b[0]≈0.01`; it is **not** an exact clean endpoint. Training at index zero uses `a[0]D+0.01 epsilon`, while inference supplies clean `D`. This retains the supplied convention and introduces a small train/inference input difference that requires empirical assessment. The maximum index also does not reach pure noise (`alpha_bar[49]≈0.603`). This procedure is deterministic feature refinement, not a claim of one-step sampling from a full generative diffusion chain or a distilled sampler.

The evaluation-mode `noise_mse` tensor is an API placeholder equal to zero. Reports write it as `null`, since inference has no sampled target noise against which to measure it.

## One segmentation output path

```text
R = bilinear_upsample(Conv1×1_32→16(D_hat), size=(224,224), align_corners=False)
F_refined = F_native_final + tanh(g) * R
logits = native_segmentation_head(F_refined)
probabilities = softmax(logits, class_axis)
```

`F_native_final` has 16 channels. The trainable gate `g` has shape `[1,16,1,1]`, starts at exactly zero, and broadcasts over batch and space. Therefore initial evaluation logits equal the original native U-Net logits exactly. Gate gradients can open the residual path; subsequent segmentation gradients reach the projection and fusion modules. The noise objective already trains the noise-prediction branch. There is no alternative direct 32→4 segmentation head.

## Objectives and evaluation

The objective is mean valid-pixel cross entropy plus `1 - mean(soft_Dice[1:4])` plus `lambda_noise * noise_MSE`. Soft Dice pools over batch and space, uses smoothing `1e-6`, and includes all three foreground classes in the loss. Invalid pixels have mask 255. Entirely ignored minibatches are skipped. The default `lambda_noise` is 0.01.

Evaluation accumulates a global 4×4 pixel confusion matrix with ground-truth rows and prediction columns. Per-class Dice is `2TP/(GT+Pred)`, IoU is `TP/(GT+Pred-TP)`, precision is `TP/Pred`, and recall is `TP/GT`. A zero denominator produces null; foreground macro means exclude background and the undefined entries for that particular metric. This global aggregation is not the same as averaging tile-level Dice. Losses are sample-weighted means of batch objectives and are labeled separately from global metrics.

## Coordinate fusion and area

For each covered grid pixel and class, fuse probabilities using `sum(weight * probability)/sum(weight)` across tiles, then apply argmax. Default weights use a Hann window clipped away from zero at the boundary. Count every covered tissue pixel once. If no tissue mask is given, the caller has declared every grid pixel eligible; this choice must be described in the study's quantification protocol.

Complete metadata defines physical pixel area as `mpp_x * downsample_x * mpp_y * downsample_y / 1e6` square millimeters. The downsample factors map one **output grid pixel** back to basic-level pixels and must include reading-level scale and every later resizing operation. Area fractions use covered valid tissue as denominator, not the sum of overlapping tile areas. Uncovered pixels and non-tissue are excluded. Connected-component counts are not reported as independent TLS structure counts.
