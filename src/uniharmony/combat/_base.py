"""Provide BaseComBat."""

import numpy.typing as npt
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
)

from ._design_matrix_mixin import DesignMatrixMixin
from ._ls_mixin import LocationAndScaleMixin
from ._standardization_mixin import StandardizationMixin


sklearn.set_config(enable_metadata_routing=True)

__all__ = ["BaseComBat"]


class BaseComBat(DesignMatrixMixin, StandardizationMixin, LocationAndScaleMixin, TransformerMixin, BaseEstimator):
    """Base class for ComBat-based methods."""

    def _check_X_sites(  # noqa: N802
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        copy: bool = False,
        estimator: type["BaseComBat"] | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Check X and sites.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        sites : array-like, shape (n_samples,)
            Sites.
        copy : bool, optional (default False)
            Whether to copy objects when doing `check_array`.
        estimator : estimator instance, optional (default None)
            If passed, include the name of the estimator in warning messages.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            The converted and validated X.
        ndarray, shape (n_samples,)
            The converted and validated sites.

        """
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES, estimator=estimator)
        sites = check_array(sites, copy=copy, dtype=None, ensure_2d=False, estimator=estimator)
        check_consistent_length(X, sites)
        return X, sites

    # Overridden to allow sites
    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        **fit_params,
    ) -> npt.NDArray:
        """Fit to data, then transform it.

        Fits transformer to `X` and `sites` with optional parameters
        `fit_params` and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        sites : array-like, shape (n_samples, 1)
            Sites.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.

        """
        sites_routed = fit_params.pop("sites")
        assert sites == sites_routed
        return self.fit(X, sites, **fit_params).transform(X, sites, **fit_params)

    # Overridden for check_is_fitted() usage
    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "_gamma_star") and hasattr(self, "_delta_star")

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = True
        tags.target_tags.two_d_labels = True
        tags.target_tags.positive_only = True
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = True
        return tags
