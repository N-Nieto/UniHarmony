(deepharmony)=
# DeepHarmony

**DeepHarmony** [^1] harmonizes multi-contrast MR images across a change of
scanner or protocol. A U-Net is trained on an **overlap cohort**, subjects scanned
with both protocols, to map the images of one protocol (the *source*) to the
contrasts of the other (the *target*). Volumes segmented from harmonized images
are then consistent across protocols, which lets long-term studies change scanner
hardware or protocol without invalidating previously acquired data.

**Paper**

Dewey, B. E., Zhao, C., Reinhold, J. C., Carass, A., Fitzgerald, K. C., Sotirchos, E. S., Saidha, S., Oh, J.,
Pham, D. L., Calabresi, P. A., van Zijl, P. C. M., & Prince, J. L. (2019). DeepHarmony: A deep learning approach
to contrast harmonization across scanner changes. *Magnetic Resonance Imaging*, 64, 160-170.
<https://doi.org/10.1016/j.mri.2019.05.041>

**Original code**

- <https://gitlab.com/iacl/synmi> (Keras / TensorFlow). UniHarmony re-implements the method in PyTorch.

## Method

| Component | Detail (as in the paper) |
|-----------|--------------------------|
| **Network** | 2D U-Net with 16, 32, 64, 128 feature maps per level and 256 at the bottleneck |
| **Down / upsampling** | 4×4 convolutions with stride 2 / 4×4 transposed convolutions with stride 1/2 (no pooling) |
| **Blocks** | Convolution → ReLU → batch normalization |
| **Output** | Input contrasts concatenated to the last feature map, then 1×1 convolution + ReLU (no normalization) |
| **Multi-contrast** | All input contrasts predict all output contrasts at once (e.g., T1w, FLAIR, PD, T2 → T1w, FLAIR, PD, T2) |
| **Training data** | 128×128 patches centered on random non-zero voxels, sampled with replacement |
| **Training** | Batch size 8, 250 batches per epoch, mean absolute error loss, Adam (learning rate 0.001), no regularization or dropout. The paper used the models after 120 epochs. |
| **2.5D prediction** | One network per orientation (axial, coronal, sagittal); the three predicted volumes are combined with a voxel-wise median |
| **Harmonizing the target protocol** | Optional secondary networks take target-protocol images as input and the *harmonized* source images as targets, so images of both protocols go through the same synthesis and share its (lower) noise level |

In the paper, an overlap cohort of only 8 training subjects was enough. In that
cohort, volumes segmented from harmonized images showed no significant bias between
protocols for cortical grey matter, white matter, thalamus, lateral ventricles,
intracranial volume and lesions. In a 10-year longitudinal cohort, the protocol effect
was no longer significant for most volumes; cortical grey matter kept a small but
significant effect (−0.64%, compared with 5.08% without harmonization).

### Differences from the original implementation

- **Framework**: PyTorch instead of Keras/TensorFlow. Keras defaults are kept where
  the paper relies on them: Glorot-uniform weight initialization with zero biases,
  batch normalization momentum 0.99 and epsilon 1e-3, and Adam epsilon 1e-7.
- **Batch normalization recalibration** (`bn_recalibration_batches`, default 100):
  after training, the batch normalization statistics used for prediction are
  re-estimated exactly on training patches, with the weights frozen ("precise BN").
  The running averages kept during training lag behind the activations and only
  become accurate after many updates; the paper trained for 30,000. Without
  recalibration, shorter trainings behave differently at prediction time. Set
  `bn_recalibration_batches=0` to reproduce the original behaviour.
- **Prediction**: whole slices (zero-padded to a multiple of 16) are passed through
  the fully convolutional network instead of patches. This is approximately
  equivalent: the network's receptive field (about 156 pixels) is larger than a
  128×128 patch, so whole slices give each voxel slightly more context.

## Required preprocessing

DeepHarmony expects preprocessed volumes; UniHarmony does not do these steps (see
Section 2.2 of the paper):

1. Inhomogeneity correction (e.g., N4).
2. Resampling of all contrasts to a common grid (the paper super-resolved 2D
   acquisitions with SMORE).
3. Rigid registration of all contrasts of both protocols of a subject to a common
   reference (the paper used the target-protocol T1-weighted image).
4. Intensity gain correction (the paper scaled each image linearly so that its
   white matter peak was at the same intensity).
5. A common orientation: spatial axes 0, 1 and 2 are sliced for the sagittal,
   coronal and axial networks (e.g., RAS-oriented arrays).

Intensities must be non-negative (the output layer is a ReLU). Voxels that are zero
in every input contrast are treated as background and never used as patch centers.

## Example

```python
import nibabel as nib
import numpy as np

from uniharmony.dl import DeepHarmony


def load(paths):
    """Stack co-registered contrasts of one subject: (n_contrasts, X, Y, Z)."""
    return np.stack([nib.load(p).get_fdata(dtype=np.float32) for p in paths])


# Overlap cohort: the same subjects scanned with both protocols (same contrast order)
X_source = [load(paths) for paths in source_paths]  # protocol #1
X_target = [load(paths) for paths in target_paths]  # protocol #2

model = DeepHarmony(harmonize_target=True, random_state=0)
model.fit(X_source, X_target)

# Harmonize new data acquired with either protocol
harmonized_old = model.transform(new_source_volumes)                   # protocol #1 -> harmonized
harmonized_new = model.transform(new_target_volumes, domain="target")  # protocol #2 -> harmonized
```

With the paper's settings, each network trains for 120 epochs of 250 batches. The
paper reports 160 minutes for 200 epochs on an NVIDIA K80 GPU, so about 100 minutes
per network. There are 3 networks, or 6 with `harmonize_target=True`. Training on a
CPU is possible but slow.

To monitor training, pass held-out paired subjects as
`fit(..., validation_data=(X_val, y_val))`. The validation error is then stored in
`model.history_`.

[^1]: Dewey, B. E., et al. (2019). DeepHarmony: A deep learning approach to contrast harmonization across scanner changes. *Magnetic Resonance Imaging*, 64, 160-170. https://doi.org/10.1016/j.mri.2019.05.041
