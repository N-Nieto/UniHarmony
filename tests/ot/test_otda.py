"""Tests test suite for OptimalTransportDomainAdaptation."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from ot.da import SinkhornTransport
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation
from uniharmony.ot import OptimalTransportDomainAdaptation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_data():
    """Create simple 2D data with 2 sites for basic tests."""
    X, y, sites = make_multisite_classification()
    return X, y, sites


@pytest.fixture
def multi_site_data():
    """Create data with multiple sites."""
    X, y, sites = make_multisite_classification(n_sites=4)
    return X, y, sites


@pytest.fixture
def str_sites_data():
    """Create data with string site labels."""
    rng = np.random.RandomState(42)
    X = np.random.randn(100, 5)
    sites = np.array(["site_A"] * 50 + ["site_B"] * 50)
    y = rng.randint(0, 2, 100)
    return X, y, sites


def test_default_init():
    """Test default initialization."""
    otda = OptimalTransportDomainAdaptation()
    assert otda.ot_method == "emd"
    assert otda.metric == "euclidean"
    assert otda.reg == 1.0
    assert otda.eta == 0.1


def test_init_with_string_method():
    """Test initialization with string method."""
    otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.5)
    assert otda.ot_method == "sinkhorn"
    assert otda.reg == 0.5


def test_init_with_ot_instance():
    """Test initialization with pre-configured OT instance."""
    ot_solver = SinkhornTransport(reg_e=0.5, metric="sqeuclidean")
    otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)
    assert otda.ot_method is ot_solver


def test_init_with_wrong_ot_instance(simple_data):
    """Test initialization with pre-configured OT instance."""
    ot_solver = IntraSiteInterpolation()
    otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)
    X, y, sites = simple_data
    with pytest.raises(TypeError):
        otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    assert otda.ot_method is ot_solver


def test_init_with_all_params():
    """Test initialization with all parameters."""
    otda = OptimalTransportDomainAdaptation(
        ot_method="emd_laplace",
        metric="cityblock",
        reg=0.5,
        eta=0.2,
        max_iter=50,
        cost_norm="median",
        limit_max=20,
    )
    assert otda.metric == "cityblock"
    assert otda.cost_norm == "median"


# =============================================================================
# Test OptimalTransportDomainAdaptation _validate_sites
# =============================================================================


def test_single_string_site(str_sites_data):
    """Test validation with single string reference site."""
    X, _, sites = str_sites_data
    otda = OptimalTransportDomainAdaptation()
    X_ref, X_harm, _, _ = otda._split_ref_harm_data(X, sites, "site_A")

    assert np.shape(X_ref) == (50, 5)
    assert np.shape(X_harm) == (50, 5)


def test_single_integer_site(simple_data):
    """Test validation with integer reference site."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    X_ref, X_harm, _, _ = otda._split_ref_harm_data(X, sites, 0)

    assert np.shape(X_ref) == (500, 10)
    assert np.shape(X_harm) == (500, 10)


def test_list_of_sites(multi_site_data):
    """Test validation with list of reference sites."""
    X, _, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()
    X_ref, X_harm, _, _ = otda._split_ref_harm_data(X, sites, [1, 2])

    assert np.shape(X_ref) == (500, 10)  # A + B
    assert np.shape(X_harm) == (500, 10)  # C + D


def test_list_of_integer_sites(multi_site_data):
    """Test validation with list of integer reference sites."""
    X, _, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()
    _, _, _, _ = otda._split_ref_harm_data(X, sites, [0, 1])


def test_missing_site_error(simple_data):
    """Test error when reference site doesn't exist."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(ValueError):
        otda._split_ref_harm_data(X, sites, "nonexistent_site")


def test_missing_site_in_list_error(multi_site_data):
    """Test error when one reference site in list doesn't exist."""
    X, _, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(ValueError):
        otda._split_ref_harm_data(X, sites, ["A", "Z"])  # Z doesn't exist


def test_no_reference_samples_error(simple_data):
    """Test error when no reference samples found."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(ValueError):
        otda._split_ref_harm_data(X, sites, "Z")


# =============================================================================
# Test OptimalTransportDomainAdaptation Fit


def test_basic_fit(simple_data):
    """Test basic fit operation."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    result = otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    assert result is otda  # Returns self
    assert hasattr(otda, "ot_obj_")
    assert hasattr(otda, "coupling_")
    check_is_fitted(otda)


def test_fit_with_integer_sites(simple_data):
    """Test fit with integer site labels."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.set_params(ref_site=0, use_labels=True).fit(X, y, sites=sites)
    check_is_fitted(otda)


def test_fit_with_list_ref_site(multi_site_data):
    """Test fit with multiple reference sites."""
    X, y, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation(ref_site=[0, 1], use_labels=True)
    otda.fit(X, y, sites=sites)
    check_is_fitted(otda)


def test_fit_without_labels(simple_data):
    """Test unsupervised fit without labels."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.set_params(ref_site=1).fit(X, sites=sites)
    check_is_fitted(otda)


def test_fit_stores_attributes(simple_data):
    """Test that fit stores expected attributes."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation(ot_method="emd")

    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)

    assert otda.ref_site_ == 1
    assert otda.ot_obj_ is not None
    assert otda.coupling_ is not None
    assert otda.coupling_.shape == (500, 500)  # harm x ref


def test_fit_all_ot_methods(simple_data):
    """Test fit with all supported OT methods."""
    X, y, sites = simple_data

    methods = ["emd", "sinkhorn", "s", "sinkhorn_gl"]
    for method in methods:
        otda = OptimalTransportDomainAdaptation(ot_method=method, reg=0.1, eta=0.01)
        otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)


def test_fit_with_preconfigured_ot(simple_data):
    """Test fit with pre-configured OT instance."""
    X, y, sites = simple_data
    ot_solver = SinkhornTransport(reg_e=0.5, max_iter=100)
    otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)

    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    assert otda.ot_obj_ is ot_solver
    check_is_fitted(otda)


# =============================================================================
# Test OptimalTransportDomainAdaptation Transform
# =============================================================================
def test_transform_not_fitted_error(simple_data):
    """Test error when transforming before fitting."""
    X, _, _ = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(NotFittedError, match="not fitted yet"):
        otda.transform(X)


def test_basic_transform(simple_data):
    """Test basic transform."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)

    # Transform harmonization data
    X_transformed = otda.transform(X, sites=sites)

    assert X_transformed.shape == X.shape


def test_transform_batch_size(simple_data):
    """Test transform with different batch sizes."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    reference = otda.transform(X=X, sites=sites)

    # Test different batch sizes (a constructor parameter): same result
    for batch_size in [16, 32, 64, 128]:
        X_trans = otda.set_params(batch_size=batch_size).transform(X=X, sites=sites)
        assert X_trans.shape == X.shape
        assert_allclose(X_trans, reference)


# =============================================================================
# Test OptimalTransportDomainAdaptation Fit-Transform
# =============================================================================


def test_fit_transform(simple_data):
    """Test fit_transform in one call."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    X_transformed = otda.set_params(ref_site=1, use_labels=True).fit_transform(X, y, sites=sites)

    check_is_fitted(otda)
    assert X_transformed.shape == X.shape


def test_fit_transform_equivalence(simple_data):
    """Test that fit_transform equals fit then transform."""
    X, y, sites = simple_data
    np.random.seed(42)
    otda1 = OptimalTransportDomainAdaptation(ot_method="emd")

    X_ft = otda1.set_params(ref_site=1, use_labels=True).fit_transform(X, y, sites=sites)

    np.random.seed(42)
    otda2 = OptimalTransportDomainAdaptation(ot_method="emd")
    otda2.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    X_f_t = otda2.transform(X, sites=sites)
    assert_allclose(X_ft, X_f_t)


# =============================================================================
# Test OptimalTransportDomainAdaptation Inverse Transform
# =============================================================================
def test_inverse_transform_not_fitted_error(simple_data):
    """Test error when inverse transforming before fitting."""
    X, _, _ = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(
        NotFittedError,
        match="not fitted yet",
    ):
        otda.inverse_transform(X)


def test_inverse_transform(simple_data):
    """Test basic inverse transform."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)

    X_ref = X[sites == 1]
    X_inv = otda.inverse_transform(X_ref)

    assert X_inv.shape == X_ref.shape


def test_inverse_transform_with_sites(simple_data):
    """Test basic inverse transform."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)

    X_inv = otda.inverse_transform(X, sites=sites)

    assert X_inv.shape == X.shape


# =============================================================================
# Test OptimalTransportDomainAdaptation Properties
# =============================================================================


def test_coupling_property(simple_data):
    """Test coupling property access."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    assert otda.coupling_ is not None


def test_cost_matrix_property(simple_data):
    """Test cost_matrix property access."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    # Cost might be None depending on OT method
    if hasattr(otda, "cost_"):
        assert otda.cost_ is not None


# =============================================================================
# Integration Tests
# =============================================================================
def test_full_workflow_emd():
    """Test complete workflow with EMD."""
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2)
    sites = np.array([1] * 100 + [2] * 100)

    otda = OptimalTransportDomainAdaptation(ot_method="emd")
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    X_aligned = otda.transform(X[sites == 2])

    assert X_aligned.shape == (100, 10)


def test_full_workflow_sinkhorn(simple_data):
    """Test complete workflow with Sinkhorn."""
    X, y, sites = simple_data

    otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.1, metric="sqeuclidean")
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    X_aligned = otda.set_params(ref_site=1, use_labels=True).fit_transform(X, y, sites=sites)

    assert X_aligned.shape == (1000, 10)


def test_multi_site_harmonization(simple_data):
    """Test harmonizing multiple sites to one reference."""
    X, y, sites = simple_data

    otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.1)
    otda.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)

    # Transform each site separately
    X_transformed = otda.transform(X, sites=sites)
    X_site0 = X_transformed[sites == 0]
    X_site1 = X_transformed[sites == 0]

    assert X_site0.shape == (500, 10)
    assert X_site1.shape == (500, 10)


def test_consistency_across_runs(simple_data):
    """Test that same seed gives consistent results."""
    X, y, sites = simple_data
    otda1 = OptimalTransportDomainAdaptation(ot_method="emd")
    otda1.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    trans1 = otda1.transform(X, sites=sites)

    otda2 = OptimalTransportDomainAdaptation(ot_method="emd")
    otda2.set_params(ref_site=1, use_labels=True).fit(X, y, sites=sites)
    trans2 = otda2.transform(X, sites=sites)

    assert_allclose(trans1, trans2)


# =============================================================================
# Transform and inverse transform of new data (audit #4 and #5)
# =============================================================================


@pytest.fixture
def three_site_ot_data():
    """Three sites (``a`` is the reference) with shifted distributions, and a fitted model."""
    rng = np.random.default_rng(0)
    sites = np.repeat(["a", "b", "c"], 100)
    X = rng.normal(size=(300, 3)) + np.select([sites == "b", sites == "c"], [2.0, -1.0], 0.0)[:, None]
    otda = OptimalTransportDomainAdaptation(ref_site="a", ot_method="emd").fit(X, sites=sites)
    return X, sites, otda, rng


def test_transform_new_data_any_subset_of_sites(three_site_ot_data):
    """New data from a single non-reference site (no reference samples) can be transformed.

    Out-of-sample transport is row-wise, so a sample gets the same result in any batch.
    """
    _, _, otda, rng = three_site_ot_data
    X_new = rng.normal(size=(20, 3))
    sites_new = np.repeat(["a", "b", "c", "b"], 5)
    X_mixed = otda.transform(X_new, sites=sites_new)
    only_b = sites_new == "b"
    assert_allclose(otda.transform(X_new[only_b], sites=sites_new[only_b]), X_mixed[only_b])
    # Reference samples are returned unchanged; the others are transported
    assert_allclose(X_mixed[sites_new == "a"], X_new[sites_new == "a"])
    assert not np.allclose(X_mixed[~(sites_new == "a")], X_new[~(sites_new == "a")])


def test_transform_only_reference_samples_unchanged(three_site_ot_data):
    """A batch with only reference samples is returned unchanged."""
    _, _, otda, rng = three_site_ot_data
    X_new = rng.normal(size=(5, 3))
    assert_allclose(otda.transform(X_new, sites=np.repeat("a", 5)), X_new)
    assert_allclose(otda.inverse_transform(X_new, sites=np.repeat("a", 5)), X_new)


def test_transform_duplicate_rows_across_sites(three_site_ot_data):
    """Identical rows from different sites are handled by site, not by value."""
    _, _, otda, rng = three_site_ot_data
    row = rng.normal(size=(1, 3))
    X_new = np.vstack([row, row])
    X_transformed = otda.transform(X_new, sites=np.array(["a", "b"]))
    assert_allclose(X_transformed[0], row[0])
    assert_allclose(X_transformed[1], otda.transform(row, sites=np.array(["b"]))[0])


def test_transform_dataframe_input(three_site_ot_data):
    """DataFrames give the same result as arrays."""
    pd = pytest.importorskip("pandas")
    _, _, otda, rng = three_site_ot_data
    X_new = rng.normal(size=(10, 3))
    sites_new = np.repeat(["a", "b"], 5)
    assert_allclose(otda.transform(pd.DataFrame(X_new), sites=sites_new), otda.transform(X_new, sites=sites_new))
    assert_allclose(otda.inverse_transform(pd.DataFrame(X_new), sites=sites_new), otda.inverse_transform(X_new, sites=sites_new))


def test_transform_unseen_site_raises(three_site_ot_data):
    """Sites not seen during fit raise a clear error."""
    _, _, otda, rng = three_site_ot_data
    with pytest.raises(ValueError, match="not seen during fit"):
        otda.transform(rng.normal(size=(3, 3)), sites=np.array(["b", "z", "z"]))
    with pytest.raises(ValueError, match="not seen during fit"):
        otda.inverse_transform(rng.normal(size=(3, 3)), sites=np.array(["b", "z", "z"]))


def test_inverse_transform_undoes_transform():
    """With EMD between two equally sized sites the transport is a permutation, so the inverse is exact."""
    rng = np.random.default_rng(1)
    sites = np.repeat(["a", "b"], 100)
    X = rng.normal(size=(200, 3)) + (sites == "b")[:, None] * 2.0
    otda = OptimalTransportDomainAdaptation(ref_site="a", ot_method="emd").fit(X, sites=sites)
    X_transformed = otda.transform(X, sites=sites)
    assert_allclose(X_transformed[sites == "a"], X[sites == "a"])
    assert_allclose(otda.inverse_transform(X_transformed, sites=sites), X, atol=1e-10)


def test_inverse_transform_maps_non_reference_rows(three_site_ot_data):
    """inverse_transform maps the non-reference rows (moved by transform) and keeps the reference rows."""
    X, sites, otda, _ = three_site_ot_data
    X_transformed = otda.transform(X, sites=sites)
    X_inverse = otda.inverse_transform(X_transformed, sites=sites)
    assert X_inverse.shape == X.shape
    assert_allclose(X_inverse[sites == "a"], X[sites == "a"])
    # Mapped back towards the pooled non-reference distribution, away from the reference one
    non_ref = sites != "a"
    assert np.linalg.norm(X_inverse[non_ref].mean(0) - X[non_ref].mean(0)) < 0.1
    assert np.linalg.norm(X_transformed[non_ref].mean(0) - X[non_ref].mean(0)) > 0.4


def test_sklearn_tags():
    """Test initialization with pre-configured OT instance."""
    otda = OptimalTransportDomainAdaptation()
    otda.__sklearn_tags__()


# =============================================================================
# Input validation and tags (audit #13, #14)
# =============================================================================


def test_transform_with_other_number_of_features_raises(simple_data):
    """A different number of features at transform gives a clear error."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation().set_params(ref_site=sites[0]).fit(X, sites=sites)
    assert otda.n_features_in_ == X.shape[1]
    with pytest.raises(ValueError, match=f"X has {X.shape[1] - 1} features"):
        otda.transform(X[:, :-1], sites=sites)


def test_tags_describe_sites_and_finite_input():
    """The target is optional (only used with use_labels=True); missing values are rejected."""
    tags = OptimalTransportDomainAdaptation().__sklearn_tags__()
    assert not tags.target_tags.required
    assert not tags.target_tags.two_d_labels
    assert not tags.target_tags.positive_only
    assert tags.input_tags.allow_nan is False


def test_every_listed_metric_is_supported():
    """Every metric of ``MetricType`` is accepted by POT (removed SciPy metrics are not listed)."""
    from typing import get_args

    import ot

    from uniharmony.ot._otda import MetricType

    rng = np.random.default_rng(0)
    a, b = rng.random((4, 3)), rng.random((5, 3))
    for metric in get_args(MetricType):
        assert ot.dist(a, b, metric=metric).shape == (4, 5), metric


def test_ot_is_exported_at_top_level():
    """``uniharmony.ot`` is reachable from the top-level package (audit #16)."""
    import uniharmony

    assert "ot" in uniharmony.__all__
    assert uniharmony.ot.OptimalTransportDomainAdaptation is OptimalTransportDomainAdaptation


# =============================================================================
# Labels, sites as a column, scikit-learn API
# =============================================================================


def test_ref_site_is_required(simple_data):
    """ref_site must be set in the constructor."""
    X, _, sites = simple_data
    with pytest.raises(ValueError, match="ref_site must be set"):
        OptimalTransportDomainAdaptation().fit(X, sites=sites)


def test_labels_are_used_only_with_use_labels(simple_data):
    """The target is ignored unless use_labels=True, then it is required and changes the transport."""
    X, y, sites = simple_data
    unsupervised = OptimalTransportDomainAdaptation(ref_site=1).fit(X, sites=sites).transform(X, sites=sites)
    ignored = OptimalTransportDomainAdaptation(ref_site=1).fit(X, y, sites=sites).transform(X, sites=sites)
    assert_allclose(ignored, unsupervised)
    supervised = OptimalTransportDomainAdaptation(ref_site=1, use_labels=True).fit(X, y, sites=sites).transform(X, sites=sites)
    assert not np.allclose(supervised, unsupervised)
    with pytest.raises(ValueError, match="use_labels=True needs y"):
        OptimalTransportDomainAdaptation(ref_site=1, use_labels=True).fit(X, sites=sites)


def test_site_column_equals_keyword(three_site_ot_data):
    """Reading the sites from a column gives the keyword result, without the site column."""
    X, sites, *_ = three_site_ot_data
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df.insert(1, "site", sites)
    by_keyword = OptimalTransportDomainAdaptation(ref_site="a").fit(X, sites=sites).transform(X, sites=sites)
    otda = OptimalTransportDomainAdaptation(ref_site="a", site_column="site").set_output(transform="pandas")
    by_column = otda.fit_transform(df)
    assert list(by_column.columns) == [c for c in df.columns if c != "site"]
    assert_allclose(by_column.to_numpy(), by_keyword)
    # inverse_transform takes the harmonized features (no site column); the sites as keyword
    assert_allclose(
        otda.inverse_transform(by_column.to_numpy(), sites=sites),
        OptimalTransportDomainAdaptation(ref_site="a").fit(X, sites=sites).inverse_transform(by_keyword, sites=sites),
    )


def test_sites_given_twice_raises(three_site_ot_data):
    """The sites come from one place only."""
    X, sites, *_ = three_site_ot_data
    df = pd.DataFrame(X)
    df["site"] = sites
    with pytest.raises(ValueError, match="given twice"):
        OptimalTransportDomainAdaptation(ref_site="a", site_column="site").fit(df, sites=sites)


class _CheckedOTDA(OptimalTransportDomainAdaptation):
    """Test adapter: derive two sites ("a", "b") from the rows of X for scikit-learn's generic checks."""

    def _validate_harmonization_input(self, X, reset, required=(), copy=False, allow_nan_metadata=False, **values):
        try:
            rows = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        except (TypeError, ValueError):
            rows = None
        if rows is not None and rows.ndim == 2 and values.get("sites") is None:
            h = np.abs(np.sin(rows @ (np.arange(rows.shape[1]) + 1.0) * 12.9898) * 43758.5453) % 1.0
            values["sites"] = np.where(h < 0.5, "a", "b")
        return super()._validate_harmonization_input(X, reset, required, copy, allow_nan_metadata, **values)


@parametrize_with_checks(
    [_CheckedOTDA(ref_site="a")],
    expected_failed_checks=lambda _: {"check_transformer_n_iter": "max_iter is passed to POT, which does not report n_iter_"},
)
def test_otda_compat_sklearn(estimator, check):
    """Test sklearn compatibility (sites derived from the rows, see ``_CheckedOTDA``)."""
    check(estimator)
