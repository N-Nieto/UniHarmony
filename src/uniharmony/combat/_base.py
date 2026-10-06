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

    # Overridden to allow sites and to route arguments between fit and transform
    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Fit to data, then transform it.

        Arguments are matched to the parameters of :meth:`fit` and
        :meth:`transform` by name, so the same call works for every ComBat
        variant:

        * data shared by both (e.g., covariates) are passed to both,
        * fit-only options (e.g., ``var_epsilon``, ``max_iter``, ``df``) only to :meth:`fit`,
        * transform-only options only to :meth:`transform`.

        Positional arguments after ``sites`` follow the order of :meth:`fit`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        sites : array-like, shape (n_samples,)
            Sites.
        *args : tuple
            Further positional arguments of :meth:`fit`.
        **kwargs : dict
            Keyword arguments of :meth:`fit` and/or :meth:`transform`.

        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.

        Raises
        ------
        TypeError
            If an argument is accepted by neither :meth:`fit` nor :meth:`transform`.

        """
        fit_params = inspect.signature(self.fit).parameters
        transform_params = inspect.signature(self.transform).parameters

        unknown = [name for name in kwargs if name not in fit_params and name not in transform_params]
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.fit_transform() got unexpected keyword argument(s) {unknown}; "
                f"valid arguments are {sorted(set(fit_params) | set(transform_params))}."
            )
        fit_kwargs = {name: value for name, value in kwargs.items() if name in fit_params}
        transform_only = {name: value for name, value in kwargs.items() if name not in fit_params}

        # Resolve every fit argument by name (raises TypeError like a direct fit call would)
        bound = inspect.signature(self.fit).bind(X, sites, *args, **fit_kwargs)
        transform_kwargs = {name: value for name, value in bound.arguments.items() if name in transform_params}
        transform_kwargs.update(transform_only)

        return self.fit(*bound.args, **bound.kwargs).transform(**transform_kwargs)

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
