"""Registry of the ComBat-based harmonization models.

Other modules (e.g., :mod:`uniharmony.prettyharmonize`) use this registry to
type and validate a ComBat-based model and to know which covariate arguments
its ``fit`` and ``transform`` methods accept. A new ComBat-based model must be
registered in :data:`COMBAT_MODELS`; a test checks that every model exported
by :mod:`uniharmony.combat` is.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ._combat_gam import ComBatGAM
from ._covbat import CovBat
from ._neuro_combat import NeuroComBat


__all__ = ["COMBAT_MODELS", "ComBatModels", "CovariateArguments", "get_covariate_arguments"]

#: Any of the registered ComBat-based harmonization models (union of their types).
ComBatModels: TypeAlias = NeuroComBat | CovBat | ComBatGAM


@dataclass(frozen=True)
class CovariateArguments:
    """Covariate arguments accepted by the ``fit`` and ``transform`` methods of a ComBat-based model.

    Each field is the name of the argument receiving one kind of covariate, or
    None if the model does not support that kind.

    Parameters
    ----------
    categorical : str or None
        Argument for categorical covariates (any dtype, one-hot encoded by the model).
    continuous : str or None
        Argument for numeric covariates with a linear effect.
    smooth : str or None
        Argument for numeric covariates with a non-linear (spline) effect.
    required : tuple of str
        Arguments that must always be given.
    smooth_bounds : str or None
        Option of ``fit`` setting the range ``(min, max)`` covered by the
        splines of a single smooth covariate.

    """

    categorical: str | None = None
    continuous: str | None = None
    smooth: str | None = None
    required: tuple[str, ...] = ()
    smooth_bounds: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Names of all the accepted covariate arguments."""
        return tuple(name for name in (self.categorical, self.continuous, self.smooth) if name is not None)


#: Registered ComBat-based models and their covariate arguments.
COMBAT_MODELS: dict[type, CovariateArguments] = {
    NeuroComBat: CovariateArguments(categorical="categorical_covariates", continuous="continuous_covariates"),
    CovBat: CovariateArguments(categorical="categorical_covariates", continuous="continuous_covariates"),
    ComBatGAM: CovariateArguments(
        continuous="continuous_covariates",
        smooth="smooth_covariates",
        required=("smooth_covariates",),
        smooth_bounds="smooth_covariates_bounds",
    ),
}


def get_covariate_arguments(model: object) -> CovariateArguments:
    """Return the covariate arguments of a registered ComBat-based model.

    Subclasses of a registered model inherit its entry.

    Parameters
    ----------
    model : object
        A ComBat-based model instance.

    Returns
    -------
    CovariateArguments
        Its covariate arguments.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of a registered ComBat-based model.

    """
    for cls in type(model).__mro__:
        if cls in COMBAT_MODELS:
            return COMBAT_MODELS[cls]
    available = ", ".join(cls.__name__ for cls in COMBAT_MODELS)
    raise TypeError(f"Expected a ComBat-based model ({available}), got {type(model).__name__}.")
