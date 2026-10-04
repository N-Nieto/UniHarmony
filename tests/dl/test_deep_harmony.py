"""Tests for DeepHarmony (Dewey et al., 2019)."""

import pickle

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


torch = pytest.importorskip("torch")

from uniharmony.dl import DeepHarmony, DeepHarmonyUNet  # noqa: E402
from uniharmony.dl._deep_harmony import ORIENTATION_AXES, _PatchSampler  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic overlap cohort
# ---------------------------------------------------------------------------

# Intensity of each tissue (background, CSF, GM, WM) for two contrasts in each protocol
SOURCE_INTENSITIES = np.array([[0.0, 0.3, 0.6, 1.0], [0.0, 1.0, 0.7, 0.4]])
TARGET_INTENSITIES = np.array([[0.0, 0.15, 0.75, 1.0], [0.0, 1.0, 0.55, 0.35]])

# Small, fast training configuration for the tests
FAST = {"patch_size": 16, "batch_size": 4, "batches_per_epoch": 3, "n_epochs": 2, "bn_recalibration_batches": 2, "device": "cpu"}


def make_subject(seed: int, shape: tuple[int, int, int] = (20, 18, 12)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one subject scanned with two protocols.

    Returns source volume (2, *shape), target volume (2, *shape) and head mask.
    The spatial shape is deliberately not a multiple of 16.
    """
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=shape), 2)
    field = (field - field.mean()) / field.std()
    grid = np.indices(shape)
    center = np.array(shape)[:, None, None, None] / 2
    radius = 0.45 * np.array(shape)[:, None, None, None]
    head = (((grid - center) / radius) ** 2).sum(axis=0) < 1
    tissue = np.digitize(field, [-0.5, 0.5]) + 1
    tissue[~head] = 0
    source = SOURCE_INTENSITIES[:, tissue] + rng.normal(0, 0.03, size=(2, *shape)) * head
    target = TARGET_INTENSITIES[:, tissue] + rng.normal(0, 0.03, size=(2, *shape)) * head
    return np.clip(source, 0, None).astype(np.float32), np.clip(target, 0, None).astype(np.float32), head


@pytest.fixture(scope="module")
def cohort() -> dict:
    """Training (4 subjects) and test (2 subjects) data of a synthetic overlap cohort."""
    train = [make_subject(seed) for seed in range(4)]
    test = [make_subject(100 + seed) for seed in range(2)]
    return {
        "X": [s for s, _, _ in train],
        "y": [t for _, t, _ in train],
        "X_test": [s for s, _, _ in test],
        "y_test": [t for _, t, _ in test],
        "heads_test": [h for _, _, h in test],
    }


@pytest.fixture(scope="module")
def fitted(cohort) -> DeepHarmony:
    """DeepHarmony with the secondary (target) networks, trained with the fast configuration."""
    return DeepHarmony(harmonize_target=True, random_state=0, **FAST).fit(cohort["X"], cohort["y"])


# ---------------------------------------------------------------------------
# U-Net architecture (Fig. 2 of the paper)
# ---------------------------------------------------------------------------


def _n_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_unet_feature_maps_match_paper() -> None:
    """Every stage has the spatial size and number of feature maps of Fig. 2."""
    network = DeepHarmonyUNet(n_input_contrasts=4, n_output_contrasts=4)
    expected = {
        "enc0": (16, 128, 128),
        "down1": (16, 64, 64),
        "enc1": (32, 64, 64),
        "down2": (32, 32, 32),
        "enc2": (64, 32, 32),
        "down3": (64, 16, 16),
        "enc3": (128, 16, 16),
        "down4": (128, 8, 8),
        "bottleneck": (256, 8, 8),
        "up3": (128, 16, 16),
        "dec3": (128, 16, 16),
        "up2": (64, 32, 32),
        "dec2": (64, 32, 32),
        "up1": (32, 64, 64),
        "dec1": (32, 64, 64),
        "up0": (16, 128, 128),
        "dec0": (16, 128, 128),
        "out": (4, 128, 128),
    }
    shapes = {}
    for name in expected:
        getattr(network, name).register_forward_hook(
            lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:]))
        )
    network(torch.rand(2, 4, 128, 128))
    assert shapes == expected


def test_unet_multi_contrast_parameter_overhead_matches_paper() -> None:
    """M2O has "only about 500 additional parameters compared to the O2O network" (Section 2.3.3).

    With 3 extra input contrasts: 3 x 3x3 x 16 weights in the first convolution
    plus 3 weights in the final 1x1 convolution (inputs are concatenated before it).
    """
    m2o = DeepHarmonyUNet(n_input_contrasts=4, n_output_contrasts=1)
    o2o = DeepHarmonyUNet(n_input_contrasts=1, n_output_contrasts=1)
    assert _n_parameters(m2o) - _n_parameters(o2o) == 3 * 3 * 3 * 16 + 3


def test_unet_layer_types() -> None:
    """Strided 4x4 (de)convolutions, Conv-ReLU-BN blocks and a final 1x1 conv with ReLU and no BN."""
    network = DeepHarmonyUNet(2, 2)
    assert not any(isinstance(m, (torch.nn.MaxPool2d, torch.nn.Upsample, torch.nn.Dropout)) for m in network.modules())
    for name in ("down1", "down2", "down3", "down4"):
        conv = getattr(network, name)[0]
        assert conv.kernel_size == (4, 4) and conv.stride == (2, 2)
    for name in ("up0", "up1", "up2", "up3"):
        conv = getattr(network, name)[0]
        assert isinstance(conv, torch.nn.ConvTranspose2d) and conv.kernel_size == (4, 4) and conv.stride == (2, 2)
    for name in ("enc0", "dec0", "bottleneck"):
        block = getattr(network, name)
        assert [type(m) for m in block] == [torch.nn.Conv2d, torch.nn.ReLU, torch.nn.BatchNorm2d]
        assert block[0].kernel_size == (3, 3)
    out = network.out
    assert [type(m) for m in out] == [torch.nn.Conv2d, torch.nn.ReLU]
    assert out[0].kernel_size == (1, 1) and out[0].in_channels == 16 + 2


def test_unet_keras_initialization() -> None:
    """Biases start at zero (Keras default, the framework of the original implementation)."""
    network = DeepHarmonyUNet(2, 2)
    for module in network.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            assert torch.all(module.bias == 0)


def test_unet_output_non_negative_and_shape() -> None:
    """The final ReLU makes outputs non-negative; any size multiple of 16 works."""
    network = DeepHarmonyUNet(3, 2).train(False)
    out = network(torch.randn(1, 3, 48, 32))
    assert out.shape == (1, 2, 48, 32)
    assert torch.all(out >= 0)


@pytest.mark.parametrize(
    ("shape", "match"),
    [((1, 3, 40, 32), "multiples of 16"), ((1, 2, 32, 32), r"\(batch, 3, H, W\)"), ((3, 32, 32), r"\(batch, 3, H, W\)")],
)
def test_unet_rejects_invalid_input(shape: tuple, match: str) -> None:
    """Wrong number of contrasts or sizes that are not multiples of 16 are rejected."""
    with pytest.raises(ValueError, match=match):
        DeepHarmonyUNet(3, 2)(torch.rand(*shape))


def test_unet_rejects_invalid_contrast_numbers() -> None:
    """At least one input and one output contrast are required."""
    with pytest.raises(ValueError, match=">= 1"):
        DeepHarmonyUNet(0, 1)


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_patches_are_centered_on_foreground_voxels(cohort, axis: int) -> None:
    """Patches are centered on non-zero voxels and taken from the slice of the requested orientation."""
    X, y = cohort["X"][:2], cohort["y"][:2]
    sampler = _PatchSampler(X, y, patch_size=16)
    rng = np.random.default_rng(0)
    for _ in range(50):
        subject = int(rng.integers(2))
        center = sampler._random_center(subject, rng)
        assert sampler.masks[subject][center]
        patch_x = sampler._extract(sampler.inputs[subject], center, axis)
        patch_y = sampler._extract(sampler.targets[subject], center, axis)
        assert patch_x.shape == (2, 16, 16)
        np.testing.assert_array_equal(patch_x[:, 8, 8], X[subject][(slice(None), *center)])
        np.testing.assert_array_equal(patch_y[:, 8, 8], y[subject][(slice(None), *center)])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_border_patches_are_zero_padded(axis: int) -> None:
    """Patches near the volume border match a zero-padded slice of the volume."""
    rng = np.random.default_rng(0)
    volume = rng.random((2, 9, 7, 5)).astype(np.float32) + 0.1
    sampler = _PatchSampler([volume], [volume], patch_size=16)
    padded = np.pad(volume, ((0, 0), (8, 8), (8, 8), (8, 8)))
    for center in [(0, 0, 0), (8, 6, 4), (4, 3, 2), (0, 6, 2)]:
        patch = sampler._extract(volume, center, axis)
        window = [slice(None)]
        for dim in range(3):
            window.append(center[dim] + 8 if dim == axis else slice(center[dim], center[dim] + 16))
        np.testing.assert_array_equal(patch, padded[tuple(window)])


def test_patch_sampler_does_not_copy_volumes(cohort) -> None:
    """The sampler keeps references to the volumes (no padded copies), to bound memory use."""
    sampler = _PatchSampler(cohort["X"], cohort["y"], patch_size=128)
    assert all(a is b for a, b in zip(sampler.inputs, cohort["X"], strict=True))
    assert all(a is b for a, b in zip(sampler.targets, cohort["y"], strict=True))


def test_patch_sampler_batches() -> None:
    """Batches have the expected shapes, also for sparse foregrounds (stored indices)."""
    volume = np.zeros((2, 30, 30, 30), dtype=np.float32)
    volume[:, 15, 15, 15] = 1.0  # a single foreground voxel: sparse path
    sampler = _PatchSampler([volume], [volume[:1]], patch_size=16)
    assert sampler.foreground_indices[0] is not None
    x_batch, y_batch = sampler.sample(5, axis=2, rng=np.random.default_rng(0))
    assert x_batch.shape == (5, 2, 16, 16) and y_batch.shape == (5, 1, 16, 16)
    assert np.all(x_batch[:, :, 8, 8] == 1.0)


def test_patch_sampler_empty_foreground_raises() -> None:
    """Training needs non-zero input voxels."""
    volume = np.zeros((1, 8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="no non-zero voxels"):
        _PatchSampler([volume], [volume], patch_size=16)


# ---------------------------------------------------------------------------
# Estimator: fit / transform
# ---------------------------------------------------------------------------


def test_fit_attributes(fitted) -> None:
    """One network per orientation for each domain, with training history."""
    assert set(fitted.networks_) == set(ORIENTATION_AXES)
    assert set(fitted.target_networks_) == set(ORIENTATION_AXES)
    assert fitted.n_input_contrasts_ == 2 and fitted.n_output_contrasts_ == 2
    assert set(fitted.history_) == {f"{d}/{o}" for d in ("source", "target") for o in ORIENTATION_AXES}
    assert all(len(h["loss"]) == FAST["n_epochs"] for h in fitted.history_.values())


def test_transform_shapes_and_return_types(fitted, cohort) -> None:
    """A list returns a list; a single volume returns an array; outputs are non-negative."""
    harmonized = fitted.transform(cohort["X_test"])
    assert isinstance(harmonized, list) and len(harmonized) == 2
    for out, volume in zip(harmonized, cohort["X_test"], strict=True):
        assert out.shape == volume.shape and out.dtype == np.float32
        assert np.all(out >= 0)
    single = fitted.transform(cohort["X_test"][0])
    assert isinstance(single, np.ndarray)
    np.testing.assert_allclose(single, harmonized[0])


def test_transform_accepts_5d_array(fitted, cohort) -> None:
    """Subjects of equal size can be passed as one (n_subjects, n_contrasts, X, Y, Z) array."""
    stacked = np.stack(cohort["X_test"])
    out = fitted.transform(stacked)
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_allclose(out[1], fitted.transform(cohort["X_test"][1]))


def test_transform_target_domain(fitted, cohort) -> None:
    """Target-protocol images are harmonized by the secondary networks."""
    out = fitted.transform(cohort["y_test"], domain="target")
    assert len(out) == 2 and out[0].shape == cohort["y_test"][0].shape


def test_2_5d_prediction_is_voxelwise_median(fitted, cohort) -> None:
    """The final volume is the voxel-wise median of the three orientation predictions."""
    volume = cohort["X_test"][0]
    per_orientation = [fitted._predict_orientation(fitted.networks_[o], volume, ORIENTATION_AXES[o]) for o in fitted.networks_]
    np.testing.assert_allclose(fitted.transform(volume), np.median(np.stack(per_orientation), axis=0), rtol=1e-6)


def test_single_orientation(cohort) -> None:
    """A single orientation (the paper's axial-only variant) is supported."""
    model = DeepHarmony(orientations=("axial",), random_state=0, **FAST).fit(cohort["X"], cohort["y"])
    assert list(model.networks_) == ["axial"]
    out = model.transform(cohort["X_test"][0])
    np.testing.assert_allclose(out, model._predict_orientation(model.networks_["axial"], cohort["X_test"][0], 2))


def test_different_number_of_input_and_output_contrasts(cohort) -> None:
    """Source and target protocols may have different contrasts."""
    X3 = [np.concatenate([x, x[:1]]) for x in cohort["X"]]
    y1 = [y[:1] for y in cohort["y"]]
    model = DeepHarmony(orientations=("axial",), harmonize_target=True, random_state=0, **FAST).fit(X3, y1)
    assert model.n_input_contrasts_ == 3 and model.n_output_contrasts_ == 1
    assert model.transform(X3[0]).shape == (1, *X3[0].shape[1:])
    assert model.transform(y1[0], domain="target").shape == y1[0].shape


def test_learns_contrast_mapping(cohort) -> None:
    """After training, harmonized source images are closer to the target protocol than acquired ones."""
    model = DeepHarmony(
        orientations=("axial",), patch_size=32, batch_size=8, batches_per_epoch=40, n_epochs=6, device="cpu", random_state=0
    ).fit(cohort["X"], cohort["y"])
    history = model.history_["source/axial"]["loss"]
    assert history[-1] < 0.5 * history[0]
    for source, target, head in zip(cohort["X_test"], cohort["y_test"], cohort["heads_test"], strict=True):
        acquired_error = np.abs(source - target)[:, head].mean()
        harmonized_error = np.abs(model.transform(source) - target)[:, head].mean()
        assert harmonized_error < 0.8 * acquired_error


def _training_batch_l1(network: DeepHarmonyUNet, cohort: dict, train_mode: bool) -> float:
    """L1 loss of a network on a fixed batch of training patches, with batch or running BN statistics."""
    x_batch, y_batch = _PatchSampler(cohort["X"], cohort["y"], 32).sample(64, axis=2, rng=np.random.default_rng(1))
    network.train(train_mode)
    with torch.no_grad():
        loss = float((network(torch.from_numpy(x_batch)) - torch.from_numpy(y_batch)).abs().mean())
    network.train(False)
    return loss


def test_batch_norm_recalibration_matches_training_statistics(cohort) -> None:
    """With precise BN, prediction-time (running) statistics reproduce the training-time behaviour.

    After a short training, the running averages lag behind the activations
    (momentum 0.99): the network behaves differently at prediction time unless
    the statistics are recalibrated.
    """
    params = {"orientations": ("axial",), "patch_size": 32, "batch_size": 8, "batches_per_epoch": 40, "n_epochs": 6}
    recalibrated = DeepHarmony(random_state=0, device="cpu", **params).fit(cohort["X"], cohort["y"])
    running = DeepHarmony(bn_recalibration_batches=0, random_state=0, device="cpu", **params).fit(cohort["X"], cohort["y"])

    net = recalibrated.networks_["axial"]
    assert _training_batch_l1(net, cohort, train_mode=False) < 1.25 * _training_batch_l1(net, cohort, train_mode=True)
    net = running.networks_["axial"]
    assert _training_batch_l1(net, cohort, train_mode=False) > 1.5 * _training_batch_l1(net, cohort, train_mode=True)


def test_batch_norm_recalibration_counts_batches(cohort) -> None:
    """Recalibration recomputes the statistics over exactly bn_recalibration_batches batches."""
    params = {**FAST, "orientations": ("axial",), "random_state": 0}
    model = DeepHarmony(**{**params, "bn_recalibration_batches": 7}).fit(cohort["X"], cohort["y"])
    model_without = DeepHarmony(**{**params, "bn_recalibration_batches": 0}).fit(cohort["X"], cohort["y"])
    batch_norms = [m for m in model.networks_["axial"].modules() if isinstance(m, torch.nn.BatchNorm2d)]
    batch_norms_without = [m for m in model_without.networks_["axial"].modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(m.num_batches_tracked == 7 for m in batch_norms)
    assert all(m.num_batches_tracked == FAST["n_epochs"] * FAST["batches_per_epoch"] for m in batch_norms_without)
    assert all(m.momentum == 0.01 for m in batch_norms)  # momentum restored after recalibration


def test_validation_history(cohort) -> None:
    """Validation MAE is recorded every validation_frequency epochs and at the last epoch."""
    model = DeepHarmony(
        orientations=("axial",), harmonize_target=True, validation_frequency=2, random_state=0, **{**FAST, "n_epochs": 3}
    ).fit(cohort["X"], cohort["y"], validation_data=(cohort["X_test"], cohort["y_test"]))
    for key in ("source/axial", "target/axial"):
        epochs = [epoch for epoch, _ in model.history_[key]["val_mae"]]
        assert epochs == [2, 3]
        assert all(np.isfinite(mae) and mae >= 0 for _, mae in model.history_[key]["val_mae"])


# ---------------------------------------------------------------------------
# Reproducibility, persistence, sklearn API
# ---------------------------------------------------------------------------


def test_reproducible_with_random_state(cohort) -> None:
    """The same random_state gives identical networks and outputs on CPU."""
    params = {"orientations": ("axial",), "random_state": 3, **FAST}
    out_1 = DeepHarmony(**params).fit(cohort["X"], cohort["y"]).transform(cohort["X_test"][0])
    out_2 = DeepHarmony(**params).fit(cohort["X"], cohort["y"]).transform(cohort["X_test"][0])
    np.testing.assert_array_equal(out_1, out_2)


def test_fit_does_not_change_global_torch_rng(cohort) -> None:
    """Seeding is local: the global PyTorch random state is left untouched."""
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    DeepHarmony(orientations=("axial",), random_state=0, **FAST).fit(cohort["X"][:1], cohort["y"][:1])
    np.testing.assert_array_equal(torch.rand(3).numpy(), expected.numpy())


def test_pickle_roundtrip(fitted, cohort) -> None:
    """A fitted model can be pickled and gives the same output."""
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.transform(cohort["X_test"][0]), fitted.transform(cohort["X_test"][0]))


def test_fit_transform_harmonizes_source(cohort) -> None:
    """fit_transform(X, y) returns the harmonized source volumes."""
    params = {"orientations": ("axial",), "random_state": 0, **FAST}
    out = DeepHarmony(**params).fit_transform(cohort["X"], cohort["y"])
    model = DeepHarmony(**params).fit(cohort["X"], cohort["y"])
    np.testing.assert_array_equal(out[0], model.transform(cohort["X"][0]))


def test_clone_and_get_params() -> None:
    """Hyperparameters follow the sklearn conventions; defaults match the paper."""
    model = DeepHarmony()
    params = model.get_params()
    assert params["patch_size"] == 128
    assert params["batch_size"] == 8
    assert params["batches_per_epoch"] == 250
    assert params["learning_rate"] == 1e-3
    assert tuple(params["orientations"]) == ("axial", "coronal", "sagittal")
    assert clone(model).get_params() == params


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises(cohort) -> None:
    """Transform requires a fitted model."""
    with pytest.raises(NotFittedError):
        DeepHarmony().transform(cohort["X_test"])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"orientations": ()}, "at least one orientation"),
        ({"orientations": ("axial", "oblique")}, "Unknown orientations"),
        ({"orientations": ("axial", "axial")}, "duplicates"),
        ({"patch_size": 20}, "multiple of 16"),
        ({"batch_size": 0}, "batch_size"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"bn_recalibration_batches": -1}, "bn_recalibration_batches"),
    ],
)
def test_invalid_parameters_raise(cohort, params: dict, match: str) -> None:
    """Invalid hyperparameters are rejected at fit time."""
    with pytest.raises(ValueError, match=match):
        DeepHarmony(**{**FAST, **params}).fit(cohort["X"], cohort["y"])


def test_missing_targets_raise(cohort) -> None:
    """DeepHarmony is supervised: fit without y fails with a clear message."""
    with pytest.raises(ValueError, match=r"y .* is required"):
        DeepHarmony(**FAST).fit(cohort["X"])


def test_unpaired_data_raises(cohort) -> None:
    """X and y must contain the same subjects."""
    with pytest.raises(ValueError, match="same subjects"):
        DeepHarmony(**FAST).fit(cohort["X"], cohort["y"][:-1])


def test_spatial_mismatch_raises(cohort) -> None:
    """Paired volumes must be co-registered (same spatial shape)."""
    y = list(cohort["y"])
    y[0] = y[0][:, :-1]
    with pytest.raises(ValueError, match="co-registered"):
        DeepHarmony(**FAST).fit(cohort["X"], y)


def test_non_finite_values_raise(cohort) -> None:
    """NaN or infinite intensities are rejected."""
    X = [x.copy() for x in cohort["X"]]
    X[1][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        DeepHarmony(**FAST).fit(X, cohort["y"])


def test_inconsistent_contrasts_raise(cohort) -> None:
    """All subjects need the same number of contrasts."""
    X = list(cohort["X"])
    X[0] = X[0][:1]
    with pytest.raises(ValueError, match="same number of contrasts"):
        DeepHarmony(**FAST).fit(X, cohort["y"])


def test_wrong_dimensions_raise() -> None:
    """Volumes must be 4D (n_contrasts, X, Y, Z)."""
    with pytest.raises(ValueError, match=r"\(n_contrasts, X, Y, Z\)"):
        DeepHarmony(**FAST).fit(np.zeros((4, 4, 4)), np.zeros((4, 4, 4)))


def test_transform_wrong_number_of_contrasts_raises(fitted, cohort) -> None:
    """The contrasts must match those seen during fit."""
    with pytest.raises(ValueError, match="expect 2 contrasts"):
        fitted.transform(cohort["X_test"][0][:1])


def test_transform_invalid_domain_raises(fitted, cohort) -> None:
    """Only "source" and "target" domains exist."""
    with pytest.raises(ValueError, match="domain must be"):
        fitted.transform(cohort["X_test"], domain="other")


def test_transform_target_requires_secondary_networks(cohort) -> None:
    """domain="target" needs harmonize_target=True."""
    model = DeepHarmony(orientations=("axial",), random_state=0, **FAST).fit(cohort["X"], cohort["y"])
    with pytest.raises(ValueError, match="harmonize_target=True"):
        model.transform(cohort["y_test"], domain="target")


def test_validation_data_contrast_mismatch_raises(cohort) -> None:
    """Validation data must have the same contrasts as the training data."""
    X_val = [x[:1] for x in cohort["X_test"]]
    with pytest.raises(ValueError, match="same number of contrasts as X and y"):
        DeepHarmony(**FAST).fit(cohort["X"], cohort["y"], validation_data=(X_val, cohort["y_test"]))
