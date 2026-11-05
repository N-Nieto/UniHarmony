"""Provide the harmonization step of PrettYharmonize for any ComBat-based model."""

# Adapted from:
# https://github.com/juaml/PrettYharmonize
# licensed under AGPL-3.0.

from collections.abc import Iterable, Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from uniharmony._verbose import verbosity_context
from uniharmony.combat._available_models import ComBatModels, CovariateArguments


__all__ = ["TargetHarmonizer", "target_argument"]


def target_argument(arguments: CovariateArguments, categorical: bool, covariates: dict[str, Any]) -> str:
    """Return the covariate argument of a ComBat-based model that receives the target.

    Parameters
    ----------
    arguments : CovariateArguments
        Covariate arguments of the model.
    categorical : bool
        Whether the target is categorical (classification).
    covariates : dict
        Additional covariates given by the user.

    Returns
    -------
    str
        For a categorical target, the categorical argument if the model has
        one, else the continuous argument (the target is then one-hot
        encoded). For a continuous target, the smooth argument (non-linear
        effect) if the model has one and the user does not use it, else the
        continuous argument (linear effect).

    """
    if categorical and arguments.categorical is not None:
        return arguments.categorical
    if not categorical and arguments.smooth is not None and arguments.smooth not in covariates:
        return arguments.smooth
    if arguments.continuous is None:  # pragma: no cover - every registered model has one
        raise ValueError("The harmonization model has no covariate argument that can receive the target.")
    return arguments.continuous


def _as_columns(values: npt.ArrayLike) -> npt.NDArray:
    values = np.asarray(values)
    return values.reshape(-1, 1) if values.ndim == 1 else values


class TargetHarmonizer:
    """Harmonize with a ComBat-based model that preserves the target.

    The target is added (as first column) to the covariate argument
    ``argument`` of ``harm_model``. Training data are harmonized with their
    true target; new data are harmonized once per *pretended* target.

    Parameters
    ----------
    harm_model : ComBat-based model
        Unfitted harmonization model. It is cloned.
    argument : str
        Covariate argument of ``harm_model`` that receives the target.
    n_classes : int or None
        Number of classes of a categorical target (encoded as ``0..n_classes-1``),
        None for a continuous target.
    categorical_argument : bool
        Whether ``argument`` takes categorical values. If False, a categorical
        target is one-hot encoded (first class dropped).

    """

    def __init__(self, harm_model: ComBatModels, argument: str, n_classes: int | None, categorical_argument: bool) -> None:
        self.harm_model = cast("ComBatModels", clone(harm_model))
        self.argument = argument
        self.n_classes = n_classes
        self.categorical_argument = categorical_argument

    def _encode(self, y: npt.NDArray) -> npt.NDArray:
        """Return the target column(s) passed to ``harm_model``."""
        if self.n_classes is None:
            return np.asarray(y, dtype=float).reshape(-1, 1)
        if self.categorical_argument:
            return np.asarray(y).reshape(-1, 1)
        # Numeric argument: one-hot encoding without the first class (the
        # dropped column is collinear with the site indicators).
        return (np.asarray(y, dtype=int).reshape(-1, 1) == np.arange(1, self.n_classes)).astype(float)

    def _arguments(self, y: npt.NDArray, covariates: dict[str, Any]) -> dict[str, Any]:
        """Return the covariates of ``harm_model`` with the target prepended to its argument."""
        arguments = dict(covariates)
        target = self._encode(y)
        user = arguments.get(self.argument)
        if user is not None:
            user = _as_columns(user)
            if self.categorical_argument:
                target, user = target.astype(object), user.astype(object)
            target = np.hstack([target, user])
        arguments[self.argument] = target
        return arguments

    def fit(
        self, X: npt.NDArray, y: npt.NDArray, sites: npt.NDArray, covariates: dict[str, Any], options: dict[str, Any]
    ) -> None:
        """Fit ``harm_model`` preserving the true target.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Encoded target.
        sites : array, shape (n_samples,)
            Sites.
        covariates : dict
            Additional covariates of ``harm_model``.
        options : dict
            Other arguments of ``harm_model.fit`` (e.g., spline bounds).

        """
        # The ComBat models warn that preserving the ML target leaks it. Here the
        # target of the data to predict is never used, so the warning does not apply.
        with verbosity_context("error"):
            self.harm_model.fit(X, sites=sites, **self._arguments(y, covariates), **options)

    def transform(self, X: npt.NDArray, y: npt.NDArray, sites: npt.NDArray, covariates: dict[str, Any]) -> npt.NDArray:
        """Harmonize data with a given (true or pretended) target.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data.
        y : array, shape (n_samples,)
            Encoded target.
        sites : array, shape (n_samples,)
            Sites.
        covariates : dict
            Additional covariates of ``harm_model``.

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized data.

        Notes
        -----
        With :class:`~uniharmony.combat.ComBatGAM`, smooth covariates (and a
        smooth target) outside the range seen during fit are clipped to it, so
        their effect is extrapolated as a constant, and a warning is logged.

        """
        return self.harm_model.transform(X, sites=sites, **self._arguments(y, covariates))

    def iter_pretend(
        self, X: npt.NDArray, sites: npt.NDArray, targets: Iterable[Any], covariates: dict[str, Any]
    ) -> Iterator[npt.NDArray]:
        """Harmonize ``X`` once per pretended target.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data.
        sites : array, shape (n_samples,)
            Sites.
        targets : iterable
            Pretended (encoded) targets: scalars or arrays of shape (n_samples,).
        covariates : dict
            Additional covariates of ``harm_model``.

        Yields
        ------
        array, shape (n_samples, n_features)
            ``X`` harmonized as if its target was the pretended one.

        """
        for target in targets:
            yield self.transform(X, np.broadcast_to(np.asarray(target), (X.shape[0],)), sites, covariates)
