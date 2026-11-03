"""Provide BaseComBat."""

import inspect
from typing import Any

import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
    validate_data,
)

from ._design_matrix_mixin import DesignMatrixMixin
from ._ls_mixin import LocationAndScaleMixin
from ._standardization_mixin import StandardizationMixin


__all__ = ["BaseComBat"]


class BaseComBat(DesignMatrixMixin, StandardizationMixin, LocationAndScaleMixin, TransformerMixin, BaseEstimator):
    """Base class for ComBat-based methods."""

    def _check_X_sites(  # noqa: N802
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        copy: bool = False,
        reset: bool = True,
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
        reset : bool, optional (default True)
            True in ``fit``: store the number of features (``n_features_in_``)
            and, for DataFrames, their names (``feature_names_in_``). False in
            ``transform``: check X against them.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            The converted and validated X.
        ndarray, shape (n_samples,)
            The converted and validated sites.

        Raises
        ------
        ValueError
            If X has missing or infinite values, or (``reset=False``) a
            different number of features than seen during ``fit``.

        """
        X = validate_data(self, X, reset=reset, copy=copy, dtype=FLOAT_DTYPES)
        sites = check_array(sites, copy=copy, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, sites)
        return X, sites

    def _check_covariates_as_in_fit(self, **covariates: npt.ArrayLike | None) -> None:
        """Check that ``transform`` receives the same kinds of covariates as ``fit``.

        Parameters
        ----------
        **covariates : array-like or None
            Covariate arguments of ``transform`` by name (e.g.,
            ``continuous_covariates=...``). ``fit`` records whether each was
            given in ``_<name>_used``.

        Raises
        ------
        ValueError
            If a covariate given to ``fit`` is missing, or a covariate not
            given to ``fit`` is passed.

        """
        for name, value in covariates.items():
            used_at_fit = getattr(self, f"_{name}_used", False)
            if used_at_fit and value is None:
                raise ValueError(
                    f"{name} were given to fit, so they must also be given to transform "
                    "(their effect is preserved for every sample)."
                )
            if not used_at_fit and value is not None:
                raise ValueError(
                    f"{name} were not given to fit, so transform cannot use them. Fit again with {name} to preserve their effect."
                )

    def _check_categorical_covariates(
        self,
        X: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike,
        copy: bool = False,
        estimator: type["BaseComBat"] | None = None,
    ) -> npt.NDArray:
        """Check categorical covariates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates)
            Categorical covariates.
        copy : bool, optional (default False)
            Whether to copy objects when doing `check_array`.
        estimator : estimator instance, optional (default None)
            If passed, include the name of the estimator in warning messages.

        Returns
        -------
        ndarray, shape (n_samples, n_categorical_covariates)
            The converted and validated categorical covariates.

        """
        categorical_covariates = check_array(categorical_covariates, copy=copy, dtype=None, ensure_2d=False, estimator=estimator)
        check_consistent_length(X, categorical_covariates)
        return categorical_covariates

    def _check_continuous_covariates(
        self,
        X: npt.ArrayLike,
        continuous_covariates: npt.ArrayLike,
        copy: bool = False,
        estimator: type["BaseComBat"] | None = None,
    ) -> npt.NDArray:
        """Check continuous covariates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates)
            Continuous covariates.
        copy : bool, optional (default False)
            Whether to copy objects when doing `check_array`.
        estimator : estimator instance, optional (default None)
            If passed, include the name of the estimator in warning messages.

        Returns
        -------
        ndarray, shape (n_samples, n_continuous_covariates)
            The converted and validated continuous covariates.

        """
        continuous_covariates = check_array(
            continuous_covariates, copy=copy, dtype=FLOAT_DTYPES, ensure_2d=False, estimator=estimator
        )
        check_consistent_length(X, continuous_covariates)
        return continuous_covariates

    # Overridden to allow sites. Subclasses override it again with their explicit, typed signature
    # (mirroring ``fit``) and call ``_fit_then_transform``.
    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        **fit_params: Any,
    ) -> npt.NDArray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        sites : array-like, shape (n_samples,)
            Sites.
        **fit_params : dict
            Additional arguments of :meth:`fit`. Those that :meth:`transform`
            also accepts (e.g., covariates) are passed to it too.

        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.

        """
        return self._fit_then_transform(X=X, sites=sites, **fit_params)

    def _fit_then_transform(self, **arguments: Any) -> npt.NDArray:
        """Call ``fit`` with all ``arguments`` and ``transform`` with those it accepts.

        Data shared by ``fit`` and ``transform`` (``X``, ``sites``, covariates)
        go to both; fit-only options (e.g., ``var_epsilon``, ``max_iter``,
        ``df``) only to ``fit``.

        Parameters
        ----------
        **arguments : dict
            Keyword arguments of :meth:`fit`.

        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.

        """
        transform_params = inspect.signature(self.transform).parameters
        transform_arguments = {name: value for name, value in arguments.items() if name in transform_params}
        return self.fit(**arguments).transform(**transform_arguments)

    # Overridden for check_is_fitted() usage
    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "_gamma_star") and hasattr(self, "_delta_star")

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        # ``sites`` takes the place of ``y``: required, one label per sample (1D), any values
        tags.target_tags.required = True
        tags.input_tags.two_d_array = True
        # Missing and infinite values are rejected by the input validation
        tags.input_tags.allow_nan = False
        return tags
