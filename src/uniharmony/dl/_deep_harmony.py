"""Provide DeepHarmony, a deep learning contrast harmonization method."""

# Implementation of:
# Dewey, B. E., Zhao, C., Reinhold, J. C., Carass, A., Fitzgerald, K. C., Sotirchos, E. S., Saidha, S.,
# Oh, J., Pham, D. L., Calabresi, P. A., van Zijl, P. C. M., & Prince, J. L. (2019).
# DeepHarmony: A deep learning approach to contrast harmonization across scanner changes.
# Magnetic Resonance Imaging, 64, 160-170. https://doi.org/10.1016/j.mri.2019.05.041

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from uniharmony.dl._deep_harmony_unet import SIZE_MULTIPLE, DeepHarmonyUNet
from uniharmony.dl._torch import torch


__all__ = ["DeepHarmony"]

logger = structlog.get_logger()
logger = logger.bind(src="DeepHarmony")

#: Spatial axis of a (n_contrasts, X, Y, Z) volume that is sliced for each orientation.
#: Assumes volumes in a canonical orientation (e.g., RAS: X = left-right,
#: Y = posterior-anterior, Z = inferior-superior).
ORIENTATION_AXES = {"sagittal": 0, "coronal": 1, "axial": 2}

# Below this fraction of foreground voxels, foreground indices are stored
# explicitly instead of using rejection sampling.
_MIN_FOREGROUND_FRACTION_FOR_REJECTION = 0.05

Volumes = npt.NDArray | Sequence[npt.ArrayLike]


def _check_volumes(volumes: Volumes, name: str) -> tuple[list[npt.NDArray], bool]:
    """Validate volumes and return them as a list of float32 arrays.

    Parameters
    ----------
    volumes : array-like or sequence of array-like
        One volume of shape (n_contrasts, X, Y, Z), an array of shape
        (n_subjects, n_contrasts, X, Y, Z), or a sequence of volumes of shape
        (n_contrasts, X, Y, Z) (subjects may differ in X, Y, Z).
    name : str
        Name used in error messages.

    Returns
    -------
    list of ndarray
        The volumes, each of shape (n_contrasts, X, Y, Z) and dtype float32.
    bool
        Whether a single volume was given.

    Raises
    ------
    ValueError
        If the volumes are empty, have the wrong number of dimensions, contain
        non-finite values or differ in their number of contrasts.

    """
    single = False
    if isinstance(volumes, np.ndarray):
        if volumes.ndim == 4:
            volumes, single = [volumes], True
        elif volumes.ndim == 5:
            volumes = list(volumes)
        else:
            raise ValueError(
                f"{name} must be a volume of shape (n_contrasts, X, Y, Z), an array of shape "
                f"(n_subjects, n_contrasts, X, Y, Z) or a list of volumes; got an array with shape {volumes.shape}"
            )
    volumes = list(volumes)
    if not volumes:
        raise ValueError(f"{name} is empty")

    checked = []
    for i, volume in enumerate(volumes):
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 4:
            raise ValueError(f"{name}[{i}] must have shape (n_contrasts, X, Y, Z), got {volume.shape}")
        if not np.all(np.isfinite(volume)):
            raise ValueError(f"{name}[{i}] contains NaN or infinite values")
        checked.append(volume)

    n_contrasts = {volume.shape[0] for volume in checked}
    if len(n_contrasts) > 1:
        raise ValueError(f"All volumes in {name} must have the same number of contrasts, got {sorted(n_contrasts)}")
    return checked, single


def _check_paired(inputs: list[npt.NDArray], targets: list[npt.NDArray], input_name: str, target_name: str) -> None:
    """Check that input and target volumes are paired (same subjects, co-registered)."""
    if len(inputs) != len(targets):
        raise ValueError(
            f"{input_name} and {target_name} must contain the same subjects (paired data), "
            f"got {len(inputs)} and {len(targets)} volumes"
        )
    for i, (x, y) in enumerate(zip(inputs, targets, strict=True)):
        if x.shape[1:] != y.shape[1:]:
            raise ValueError(
                f"{input_name}[{i}] and {target_name}[{i}] must be co-registered with the same spatial shape, "
                f"got {x.shape[1:]} and {y.shape[1:]}"
            )


class _PatchSampler:
    """Sample 2D training patches centered on random non-zero voxels.

    Following the paper, patch centers are drawn uniformly (with replacement)
    from all voxels that are non-zero in any input contrast, over all training
    subjects. The patch is the ``patch_size`` x ``patch_size`` window of the
    slice through the center voxel in the requested orientation, zero-padded at
    the volume borders.

    Parameters
    ----------
    inputs : list of ndarray, shape (n_input_contrasts, X, Y, Z)
        Input volumes.
    targets : list of ndarray, shape (n_output_contrasts, X, Y, Z)
        Target volumes, co-registered with ``inputs``.
    patch_size : int
        Patch height and width.

    """

    def __init__(self, inputs: list[npt.NDArray], targets: list[npt.NDArray], patch_size: int) -> None:
        self.patch_size = patch_size
        self.half = patch_size // 2
        # References, not copies: patches are zero-padded on extraction
        self.inputs = inputs
        self.targets = targets

        self.shapes = [volume.shape[1:] for volume in inputs]
        self.masks = []
        self.foreground_indices: list[npt.NDArray | None] = []
        counts = []
        for volume in inputs:
            mask = np.any(volume != 0, axis=0)
            count = int(mask.sum())
            counts.append(count)
            self.masks.append(mask)
            # Rejection sampling is memory-free; store indices only for sparse foregrounds
            if 0 < count < _MIN_FOREGROUND_FRACTION_FOR_REJECTION * mask.size:
                self.foreground_indices.append(np.flatnonzero(mask))
            else:
                self.foreground_indices.append(None)
        counts = np.asarray(counts, dtype=float)
        if counts.sum() == 0:
            raise ValueError("The input volumes have no non-zero voxels to sample training patches from")
        # Uniform over all foreground voxels = subjects weighted by their foreground size
        self.subject_probabilities = counts / counts.sum()

    def _random_center(self, subject: int, rng: np.random.Generator) -> tuple[int, int, int]:
        """Draw a uniformly random foreground voxel of a subject."""
        shape = self.shapes[subject]
        indices = self.foreground_indices[subject]
        if indices is not None:
            return tuple(int(i) for i in np.unravel_index(indices[rng.integers(len(indices))], shape))
        mask = self.masks[subject]
        while True:
            center = tuple(int(rng.integers(size)) for size in shape)
            if mask[center]:
                return center

    def _extract(
        self,
        volume: npt.NDArray,
        center: tuple[int, int, int],
        axis: int,
        out: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Extract the 2D patch around ``center``, zero-padded outside the volume.

        The patch lies in the plane orthogonal to ``axis`` (remaining axes in
        increasing order) and ``center`` is at index ``(half, half)``. If
        ``out`` (zero-filled, shape (n_contrasts, patch_size, patch_size)) is
        given, the patch is written into it.
        """
        patch = np.zeros((volume.shape[0], self.patch_size, self.patch_size), dtype=volume.dtype) if out is None else out
        source: list[int | slice] = [slice(None)]
        target: list[slice] = [slice(None)]
        for dim in range(3):
            if dim == axis:
                source.append(center[dim])
                continue
            start = center[dim] - self.half
            stop = start + self.patch_size
            clipped_start, clipped_stop = max(start, 0), min(stop, volume.shape[dim + 1])
            source.append(slice(clipped_start, clipped_stop))
            target.append(slice(clipped_start - start, clipped_stop - start))
        patch[tuple(target)] = volume[tuple(source)]
        return patch

    def sample(self, batch_size: int, axis: int, rng: np.random.Generator) -> tuple[npt.NDArray, npt.NDArray]:
        """Sample a batch of input and target patches.

        Returns
        -------
        ndarray, shape (batch_size, n_input_contrasts, patch_size, patch_size)
            Input patches.
        ndarray, shape (batch_size, n_output_contrasts, patch_size, patch_size)
            Target patches.

        """
        subjects = rng.choice(len(self.inputs), size=batch_size, p=self.subject_probabilities)
        size = (self.patch_size, self.patch_size)
        x_batch = np.zeros((batch_size, self.inputs[0].shape[0], *size), dtype=np.float32)
        y_batch = np.zeros((batch_size, self.targets[0].shape[0], *size), dtype=np.float32)
        for i, subject in enumerate(subjects):
            center = self._random_center(subject, rng)
            self._extract(self.inputs[subject], center, axis, out=x_batch[i])
            self._extract(self.targets[subject], center, axis, out=y_batch[i])
        return x_batch, y_batch


class DeepHarmony(TransformerMixin, BaseEstimator):
    """DeepHarmony: deep learning contrast harmonization across scanner changes.

    DeepHarmony [1]_ learns to map multi-contrast MR images acquired with one
    protocol (or scanner), the *source*, to the contrasts of another protocol,
    the *target*. It is trained on an *overlap cohort*: subjects scanned with
    both protocols, whose images are co-registered.

    The method, as described in the paper:

    * A 2D U-Net (:class:`~uniharmony.dl.DeepHarmonyUNet`) maps all input
      contrasts to all output contrasts at once (multi-contrast training).
    * Training uses 128 x 128 patches centered on random non-zero voxels,
      sampled with replacement, the mean absolute error (L1) loss and Adam
      (learning rate 0.001), without regularization or dropout.
    * 2.5D prediction: one network is trained per orientation (axial, coronal
      and sagittal) and their predicted volumes are combined with a voxel-wise
      median.
    * Optionally (``harmonize_target=True``), a secondary set of networks is
      trained to pass target-protocol images through the same synthesis
      process: target images are the input and the *harmonized* source images
      are the targets. This gives harmonized images of both protocols the same
      (synthetic) noise characteristics.

    Parameters
    ----------
    orientations : sequence of {"axial", "coronal", "sagittal"}, optional \
            (default ("axial", "coronal", "sagittal"))
        Orientations to train one network for. The paper's 2.5D model uses all
        three; ``("axial",)`` gives its single-orientation variant.
    patch_size : int, optional (default 128)
        Height and width of the training patches. Must be a multiple of 16.
    batch_size : int, optional (default 8)
        Number of patches per training batch.
    batches_per_epoch : int, optional (default 250)
        Number of batches per epoch.
    n_epochs : int, optional (default 120)
        Number of training epochs per network. The paper trained for 200 epochs
        and selected the models at epoch 120 using validation data.
    learning_rate : float, optional (default 0.001)
        Learning rate of the Adam optimizer.
    harmonize_target : bool, optional (default False)
        Whether to also train the secondary networks that harmonize
        target-protocol images (see ``transform(..., domain="target")``).
    bn_recalibration_batches : int, optional (default 100)
        After training each network, re-estimate the batch normalization
        statistics used at prediction time as exact averages over this many
        training batches, with the weights frozen ("precise BN"). The running
        averages kept during training (Keras-like momentum of 0.99) lag behind
        the evolving activations and only become accurate after many updates
        (the paper used 30,000); without recalibration, shorter trainings give
        a mismatch between training and prediction. Set to 0 to keep the
        running averages, as in the original implementation.
    inference_batch_size : int, optional (default 16)
        Number of slices per forward pass at prediction time.
    validation_frequency : int, optional (default 5)
        If validation data is passed to ``fit``, compute the validation error
        every ``validation_frequency`` epochs (the paper saved the weights
        every 5 epochs for post-hoc validation).
    device : str or None, optional (default None)
        PyTorch device (e.g., "cpu", "cuda", "cuda:1"). If None, use CUDA when
        available and the CPU otherwise.
    random_state : int, RandomState instance or None, optional (default None)
        Seed controlling weight initialization and patch sampling. Pass an int
        for reproducible training (exactly reproducible on CPU; GPU kernels may
        be non-deterministic).

    Attributes
    ----------
    networks_ : dict of str to DeepHarmonyUNet
        Trained source-to-target network for each orientation.
    target_networks_ : dict of str to DeepHarmonyUNet
        Trained secondary (target-protocol) network for each orientation.
        Empty if ``harmonize_target=False``.
    n_input_contrasts_ : int
        Number of contrasts of the source images.
    n_output_contrasts_ : int
        Number of contrasts of the target images.
    history_ : dict of str to dict
        Training history per network, keyed ``"source/<orientation>"`` or
        ``"target/<orientation>"``. Each entry has ``"loss"`` (mean training
        L1 loss per epoch) and ``"val_mae"`` (list of ``(epoch, mae)`` pairs,
        empty without validation data). The validation error is the mean
        absolute error of that orientation's network alone (not of the 2.5D
        median), within the voxels that are non-zero in any input contrast.

    Notes
    -----
    Input volumes must be preprocessed as in the paper before using this
    estimator; DeepHarmony does not do it:

    * Inhomogeneity (bias field) correction, e.g., N4.
    * Resampling of all contrasts to a common grid (the paper super-resolved
      2D acquisitions with SMORE).
    * Rigid co-registration of all contrasts of both protocols of each subject
      to a common reference image (the paper used the target-protocol
      T1-weighted image).
    * Intensity gain correction, e.g., linear scaling so that the white matter
      peak is at the same intensity for all images.
    * A common orientation (see ``ORIENTATION_AXES``): axis 0, 1 and 2 of the
      spatial dimensions are sliced for the sagittal, coronal and axial
      networks, respectively.

    The background (voxels that are zero in every input contrast) is not used
    as patch center. The final ReLU makes all outputs non-negative, so target
    intensities should be non-negative.

    References
    ----------
    .. [1] Dewey, B. E., Zhao, C., Reinhold, J. C., Carass, A., Fitzgerald, K. C.,
           Sotirchos, E. S., Saidha, S., Oh, J., Pham, D. L., Calabresi, P. A.,
           van Zijl, P. C. M., & Prince, J. L. (2019).
           "DeepHarmony: A deep learning approach to contrast harmonization
           across scanner changes." Magnetic Resonance Imaging, 64, 160-170.
           https://doi.org/10.1016/j.mri.2019.05.041

    Examples
    --------
    >>> # X_source, X_target: lists of co-registered volumes, shape (n_contrasts, X, Y, Z)
    >>> model = DeepHarmony(harmonize_target=True, random_state=0)
    >>> model.fit(X_source_train, X_target_train)  # doctest: +SKIP
    >>> X_source_harmonized = model.transform(X_source_new)  # doctest: +SKIP
    >>> X_target_harmonized = model.transform(X_target_new, domain="target")  # doctest: +SKIP

    """

    def __init__(
        self,
        orientations: Sequence[Literal["axial", "coronal", "sagittal"]] = ("axial", "coronal", "sagittal"),
        patch_size: int = 128,
        batch_size: int = 8,
        batches_per_epoch: int = 250,
        n_epochs: int = 120,
        learning_rate: float = 1e-3,
        harmonize_target: bool = False,
        bn_recalibration_batches: int = 100,
        inference_batch_size: int = 16,
        validation_frequency: int = 5,
        device: str | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.orientations = orientations
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.harmonize_target = harmonize_target
        self.bn_recalibration_batches = bn_recalibration_batches
        self.inference_batch_size = inference_batch_size
        self.validation_frequency = validation_frequency
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: Volumes,
        y: Volumes | None = None,
        validation_data: tuple[Volumes, Volumes] | None = None,
    ) -> "DeepHarmony":
        """Train DeepHarmony on an overlap cohort.

        Parameters
        ----------
        X : array-like or sequence of array-like
            Source-protocol volumes of the overlap cohort, each of shape
            (n_input_contrasts, X, Y, Z). A sequence (one volume per subject),
            an array of shape (n_subjects, n_input_contrasts, X, Y, Z) or a
            single volume.
        y : array-like or sequence of array-like
            Target-protocol volumes of the same subjects, in the same order,
            each of shape (n_output_contrasts, X, Y, Z) and co-registered with
            the corresponding volume in ``X``.
        validation_data : tuple of (X_val, y_val) or None, optional (default None)
            Paired held-out volumes. If given, the mean absolute error within
            the foreground of ``X_val`` is recorded in ``history_`` every
            ``validation_frequency`` epochs.

        Returns
        -------
        self : DeepHarmony
            The fitted estimator.

        """
        self._validate_params()
        if y is None:
            raise ValueError("DeepHarmony is supervised: y (the target-protocol volumes of the same subjects) is required")
        X, _ = _check_volumes(X, "X")
        y, _ = _check_volumes(y, "y")
        _check_paired(X, y, "X", "y")
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val, _ = _check_volumes(X_val, "X_val")
            y_val, _ = _check_volumes(y_val, "y_val")
            _check_paired(X_val, y_val, "X_val", "y_val")
            if X_val[0].shape[0] != X[0].shape[0] or y_val[0].shape[0] != y[0].shape[0]:
                raise ValueError("validation_data must have the same number of contrasts as X and y")
        if any(np.any(volume < 0) for volume in y):
            logger.warning(
                "The target volumes contain negative intensities, which DeepHarmony cannot reproduce "
                "(its output layer is a ReLU). Rescale the intensities to be non-negative."
            )

        self.n_input_contrasts_ = X[0].shape[0]
        self.n_output_contrasts_ = y[0].shape[0]
        self._device = self._resolve_device()
        orientations = list(self.orientations)
        seeds = check_random_state(self.random_state).randint(0, 2**31 - 1, size=2 * len(orientations))
        self.history_ = {}

        logger.info(
            f"Training DeepHarmony on {len(X)} subjects: {self.n_input_contrasts_} -> {self.n_output_contrasts_} "
            f"contrasts, orientations {orientations}, device {self._device}"
        )
        self.networks_ = self._train_networks(
            X,
            y,
            seeds[: len(orientations)],
            domain="source",
            validation_data=None if validation_data is None else (X_val, y_val),
        )

        self.target_networks_ = {}
        if self.harmonize_target:
            # Secondary networks: target images as input, harmonized source images as targets
            logger.info("Training secondary networks to harmonize target-protocol images")
            X_harmonized = self._predict(self.networks_, X)
            target_validation = None
            if validation_data is not None:
                target_validation = (y_val, self._predict(self.networks_, X_val))
            self.target_networks_ = self._train_networks(
                y,
                X_harmonized,
                seeds[len(orientations) :],
                domain="target",
                validation_data=target_validation,
            )
        return self

    def transform(self, X: Volumes, domain: Literal["source", "target"] = "source") -> npt.NDArray | list[npt.NDArray]:
        """Harmonize volumes.

        Parameters
        ----------
        X : array-like or sequence of array-like
            Volumes to harmonize, each of shape (n_contrasts, X, Y, Z). The
            contrasts must be those of the source protocol (``domain="source"``)
            or of the target protocol (``domain="target"``), in the training order.
        domain : {"source", "target"}, optional (default "source")
            Protocol the volumes were acquired with. ``"target"`` requires
            ``harmonize_target=True`` at fit time.

        Returns
        -------
        ndarray or list of ndarray
            Harmonized volumes, each of shape (n_output_contrasts, X, Y, Z),
            in the target-protocol contrasts. A single array if a single volume
            was given, otherwise a list.

        Raises
        ------
        ValueError
            If ``domain`` is invalid, the secondary networks were not trained,
            or the number of contrasts does not match.

        """
        check_is_fitted(self)
        if domain == "source":
            networks, n_contrasts = self.networks_, self.n_input_contrasts_
        elif domain == "target":
            if not self.target_networks_:
                raise ValueError('transform(..., domain="target") requires fitting with harmonize_target=True')
            networks, n_contrasts = self.target_networks_, self.n_output_contrasts_
        else:
            raise ValueError(f'domain must be "source" or "target", got {domain!r}')

        volumes, single = _check_volumes(X, "X")
        if volumes[0].shape[0] != n_contrasts:
            raise ValueError(f"X has {volumes[0].shape[0]} contrasts, but the {domain} networks expect {n_contrasts} contrasts")
        self._device = self._resolve_device()
        harmonized = self._predict(networks, volumes)
        return harmonized[0] if single else harmonized

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "networks_")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _validate_params(self) -> None:
        """Validate the constructor parameters."""
        orientations = list(self.orientations)
        if not orientations:
            raise ValueError("orientations must contain at least one orientation")
        unknown = [o for o in orientations if o not in ORIENTATION_AXES]
        if unknown:
            raise ValueError(f"Unknown orientations {unknown}; choose from {list(ORIENTATION_AXES)}")
        if len(set(orientations)) != len(orientations):
            raise ValueError(f"orientations must not contain duplicates, got {orientations}")
        if not isinstance(self.patch_size, (int, np.integer)) or self.patch_size <= 0 or self.patch_size % SIZE_MULTIPLE:
            raise ValueError(f"patch_size must be a positive multiple of {SIZE_MULTIPLE}, got {self.patch_size}")
        for name in ("batch_size", "batches_per_epoch", "n_epochs", "inference_batch_size", "validation_frequency"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if not isinstance(self.bn_recalibration_batches, (int, np.integer)) or self.bn_recalibration_batches < 0:
            raise ValueError(f"bn_recalibration_batches must be a non-negative integer, got {self.bn_recalibration_batches}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    def _resolve_device(self) -> "torch.device":
        """Return the PyTorch device to use."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_networks(
        self,
        inputs: list[npt.NDArray],
        targets: list[npt.NDArray],
        seeds: npt.NDArray,
        domain: str,
        validation_data: tuple[list[npt.NDArray], list[npt.NDArray]] | None,
    ) -> dict[str, DeepHarmonyUNet]:
        """Train one network per orientation."""
        sampler = _PatchSampler(inputs, targets, self.patch_size)
        networks = {}
        for orientation, seed in zip(self.orientations, seeds, strict=True):
            key = f"{domain}/{orientation}"
            logger.info(f"Training {key} network")
            networks[orientation] = self._train_network(sampler, ORIENTATION_AXES[orientation], int(seed), key, validation_data)
        return networks

    def _train_network(
        self,
        sampler: _PatchSampler,
        axis: int,
        seed: int,
        key: str,
        validation_data: tuple[list[npt.NDArray], list[npt.NDArray]] | None,
    ) -> DeepHarmonyUNet:
        """Train the network of one orientation."""
        rng = np.random.default_rng(seed)
        n_inputs = sampler.inputs[0].shape[0]
        n_outputs = sampler.targets[0].shape[0]
        # Seed weight initialization without touching the global PyTorch RNG state
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = DeepHarmonyUNet(n_inputs, n_outputs)
        network.to(self._device)
        # Keras' Adam epsilon (the original implementation) instead of PyTorch's 1e-8
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, eps=1e-7)
        loss_function = torch.nn.L1Loss()  # mean absolute error

        history: dict[str, list[Any]] = {"loss": [], "val_mae": []}
        for epoch in range(1, self.n_epochs + 1):
            network.train()
            epoch_loss = 0.0
            for _ in range(self.batches_per_epoch):
                x_batch, y_batch = sampler.sample(self.batch_size, axis, rng)
                x_batch = torch.from_numpy(x_batch).to(self._device)
                y_batch = torch.from_numpy(y_batch).to(self._device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(network(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            history["loss"].append(epoch_loss / self.batches_per_epoch)

            message = f"[{key}] epoch {epoch}/{self.n_epochs} - loss {history['loss'][-1]:.6f}"
            if validation_data is not None and (epoch % self.validation_frequency == 0 or epoch == self.n_epochs):
                val_mae = self._validation_mae(network, axis, *validation_data)
                history["val_mae"].append((epoch, val_mae))
                message += f" - val_mae {val_mae:.6f}"
            logger.debug(message)

        if self.bn_recalibration_batches > 0:
            self._recalibrate_batch_norm(network, sampler, axis, rng)
            if validation_data is not None:
                # Record the validation error of the final (recalibrated) network
                history["val_mae"][-1] = (self.n_epochs, self._validation_mae(network, axis, *validation_data))

        network.train(False)
        network.to("cpu")
        self.history_[key] = history
        return network

    def _recalibrate_batch_norm(
        self,
        network: DeepHarmonyUNet,
        sampler: _PatchSampler,
        axis: int,
        rng: np.random.Generator,
    ) -> None:
        """Re-estimate batch normalization statistics on training patches ("precise BN").

        The running mean and variance of every batch normalization layer are
        reset and recomputed as cumulative averages over
        ``bn_recalibration_batches`` training batches, without updating any
        weight.
        """
        batch_norms = [module for module in network.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        momenta = [module.momentum for module in batch_norms]
        for module in batch_norms:
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average = exact mean over the batches
        network.train()
        with torch.no_grad():
            for _ in range(self.bn_recalibration_batches):
                x_batch, _ = sampler.sample(self.batch_size, axis, rng)
                network(torch.from_numpy(x_batch).to(self._device))
        for module, momentum in zip(batch_norms, momenta, strict=True):
            module.momentum = momentum
        network.train(False)

    def _validation_mae(
        self,
        network: DeepHarmonyUNet,
        axis: int,
        inputs: list[npt.NDArray],
        targets: list[npt.NDArray],
    ) -> float:
        """Mean absolute error within the input foreground of the validation volumes."""
        errors, n_voxels = 0.0, 0
        for x, y in zip(inputs, targets, strict=True):
            prediction = self._predict_orientation(network, x, axis)
            mask = np.any(x != 0, axis=0)
            errors += float(np.abs(prediction - y)[:, mask].sum())
            n_voxels += int(mask.sum()) * y.shape[0]
        return errors / max(n_voxels, 1)

    def _predict(self, networks: dict[str, DeepHarmonyUNet], volumes: list[npt.NDArray]) -> list[npt.NDArray]:
        """2.5D prediction: voxel-wise median of the per-orientation predictions."""
        for network in networks.values():
            network.to(self._device)
        try:
            harmonized = []
            for volume in volumes:
                predictions = [
                    self._predict_orientation(network, volume, ORIENTATION_AXES[orientation])
                    for orientation, network in networks.items()
                ]
                if len(predictions) == 1:
                    harmonized.append(predictions[0])
                else:
                    harmonized.append(np.median(np.stack(predictions), axis=0).astype(np.float32))
        finally:
            # Fitted networks are kept on the CPU (e.g., for pickling)
            for network in networks.values():
                network.to("cpu")
        return harmonized

    def _predict_orientation(self, network: DeepHarmonyUNet, volume: npt.NDArray, axis: int) -> npt.NDArray:
        """Predict a volume slice by slice along ``axis``.

        Slices are zero-padded to a multiple of 16 (the network is fully
        convolutional), predicted in batches and cropped back. The network is
        used on the device it is on.
        """
        was_training = network.training
        device = next(network.parameters()).device
        network.train(False)
        slices = np.moveaxis(volume, axis + 1, 0)  # (n_slices, n_contrasts, H, W)
        n_slices, n_contrasts, height, width = slices.shape
        padded_height = -(-height // SIZE_MULTIPLE) * SIZE_MULTIPLE
        padded_width = -(-width // SIZE_MULTIPLE) * SIZE_MULTIPLE

        output = np.empty((n_slices, network.n_output_contrasts, height, width), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n_slices, self.inference_batch_size):
                stop = min(start + self.inference_batch_size, n_slices)
                batch = np.zeros((stop - start, n_contrasts, padded_height, padded_width), dtype=np.float32)
                batch[:, :, :height, :width] = slices[start:stop]
                prediction = network(torch.from_numpy(batch).to(device))
                output[start:stop] = prediction[:, :, :height, :width].cpu().numpy()
        network.train(was_training)
        return np.moveaxis(output, 0, axis + 1)
