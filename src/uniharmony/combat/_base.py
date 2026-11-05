"""Provide BaseComBat."""

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from uniharmony._data import (
    HarmonizationData,
    HarmonizationInputMixin,
    HarmonizationOutputMixin,
    MetadataSpec,
    TransformOutput,
    check_sites_seen,
)
from uniharmony._utils import validate_sites

from ._design_matrix_mixin import DesignMatrixMixin
from ._ls_mixin import LocationAndScaleMixin
from ._standardization_mixin import StandardizationMixin


__all__ = ["BaseComBat"]

#: Sites: one label of any type per sample
SITES = MetadataSpec("sites", "site_column", single=True, description="the site of every sample")
#: Categorical covariates: any dtype, one-hot encoded
CATEGORICAL = MetadataSpec("categorical_covariates", "categorical_columns")
#: Continuous covariates: numeric, linear effect
CONTINUOUS = MetadataSpec("continuous_covariates", "continuous_columns", numeric=True)
#: Smooth covariates: numeric, non-linear (spline) effect
SMOOTH = MetadataSpec("smooth_covariates", "smooth_columns", numeric=True, description="e.g., age")


class BaseComBat(
    HarmonizationInputMixin,
    HarmonizationOutputMixin,
    DesignMatrixMixin,
    StandardizationMixin,
    LocationAndScaleMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Base class for ComBat-based methods.

    The sites and covariates are read from columns of ``X`` (named in the
    constructor) or passed as keywords to ``fit`` and ``transform``; see
    :mod:`uniharmony._data`.
    """

    _metadata_specs: ClassVar[tuple[MetadataSpec, ...]] = (SITES, CATEGORICAL, CONTINUOUS)

    def _set_reference_sites(self) -> None:
        """Resolve ``ref_site`` against the fitted sites (``sites_``).

        Sets ``ref_sites_`` (the reference sites, or all sites) and
        ``_ref_site_idx`` (their positions in ``sites_``, or None for the
        pooled target), which ``fit_standardize`` and ``fit_ls_model`` use.

        Raises
        ------
        ValueError
            If a reference site was not seen during ``fit`` or is listed twice.

        """
        ref_site = getattr(self, "ref_site", None)
        if ref_site is None:
            self._ref_site_idx = None
            self.ref_sites_ = self.sites_
            return
        requested = [ref_site] if isinstance(ref_site, (str, int, np.integer)) or np.ndim(ref_site) == 0 else list(ref_site)
        if len(requested) == 0:
            raise ValueError("ref_site must name at least one site, or be None for the pooled target.")
        fitted = np.asarray(self.sites_)
        fitted_str = fitted.astype(str)
        positions = []
        for site in requested:
            match = np.flatnonzero(fitted == site) if np.isin(site, fitted) else np.flatnonzero(fitted_str == str(site))
            if match.size == 0:
                raise ValueError(f"ref_site {site!r} is not one of the sites seen during fit: {fitted.tolist()}.")
            positions.append(int(match[0]))
        if len(set(positions)) != len(positions):
            raise ValueError(f"ref_site lists a site twice: {requested}.")
        self._ref_site_idx = np.sort(np.asarray(positions))
        self.ref_sites_ = fitted[self._ref_site_idx]

    def _keep_reference_samples(self, X_harmonized: npt.NDArray, X: npt.NDArray, sites: npt.NDArray) -> npt.NDArray:
        """Return the input data for the samples of a single reference site, which ComBat leaves unchanged.

        Their harmonized values equal the input up to rounding; the input is
        returned exactly, as R ``sva::ComBat`` does with ``ref.batch``.
        """
        reference = getattr(self, "_ref_site_idx", None)
        if reference is None or len(reference) != 1:
            return X_harmonized
        in_reference = sites == self.sites_[reference[0]]
        if np.any(in_reference):
            X_harmonized = np.array(X_harmonized, copy=True)
            X_harmonized[in_reference] = X[in_reference]
        return X_harmonized

    def _validate_combat_input(
        self, X: npt.ArrayLike, reset: bool, required: tuple[str, ...] = (), **values: Any
    ) -> HarmonizationData:
        """Validate the input of ``fit`` (``reset=True``) or ``transform``.

        The sites are always required. In ``fit`` at least two sites are
        needed; in ``transform`` every site must have been seen in ``fit`` and
        the same covariates as in ``fit`` must be given.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The input.
        reset : bool
            True in ``fit``.
        required : tuple of str, optional (default ())
            Other required kinds of metadata.
        **values : array-like or None
            The keyword data.

        Returns
        -------
        HarmonizationData
            The validated input.

        """
        data = self._validate_harmonization_input(X, reset=reset, required=("sites", *required), **values)
        if reset:
            validate_sites(data.sites)
        else:
            check_sites_seen(data.sites, self.sites_, type(self).__name__)
            self._check_metadata_as_in_fit(data, [spec.keyword for spec in self._metadata_specs if spec.keyword != "sites"])
        return data

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **params: Any) -> TransformOutput:
        """Fit to data, then harmonize it.

        Equivalent to ``fit(X, y, **params).transform(X, **params)``: the
        sites and covariates are used by both.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The training input samples.
        y : None
            Ignored.
        **params : array-like
            The data keywords of :meth:`fit` (sites, covariates).

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized data.

        """
        return self.fit(X, y, **params).transform(X, **params)

    # Overridden for check_is_fitted() usage
    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "_gamma_star") and hasattr(self, "_delta_star")

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = False
        tags.input_tags.two_d_array = True
        # Site and categorical columns can be strings
        tags.input_tags.string = True
        # Missing and infinite values are rejected by the input validation
        tags.input_tags.allow_nan = False
        return tags
