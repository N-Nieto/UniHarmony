"""Utility functions for OT-based harmonization methods."""

import numpy.typing as ntp
import ot
from ot.da import EMDLaplaceTransport, EMDTransport, SinkhornL1l2Transport, SinkhornTransport


def create_ot_object(
    name: str, **kwargs
) -> type[EMDTransport] | type[SinkhornTransport] | type[SinkhornL1l2Transport] | type[EMDLaplaceTransport]:
    """Create an OT object based on a string name.

    Parameters
    ----------
    name : str
        Name of OT object.
    **kwargs : dict
        Extra keyword arguments for the OT object.

    Returns
    -------
    object
        Initialised OT object instance.

    Raises
    ------
    ValueError
        If ``name`` is invalid.

    """
    mapping = {
        "emd": ot.da.EMDTransport,
        "sinkhorn": ot.da.SinkhornTransport,
        "sinkhorn_gl": ot.da.SinkhornL1l2Transport,
        "emd_laplace": ot.da.EMDLaplaceTransport,
    }

    name = name.lower()
    if name not in mapping:
        raise ValueError(f"Unsupported OT method: {name}")

    return mapping[name](**kwargs)


def data_consistency_check(
    X_source: ntp.ArrayLike,
    X_target: ntp.ArrayLike,
    y_source: ntp.ArrayLike | None = None,
    y_target: ntp.ArrayLike | None = None,
):
    """Check data dimensions.

    Args:
        X_source (ArrayLike): Source data [Samples x Features]
        X_target (ArrayLike): Target data [Samples x Features]
        y_source (ArrayLike optional): Source Targets [Samples]. Defaults to None.
        y_target (ArrayLike, optional): Target Targets [Sample]. Defaults to None.

    """
    # Check consistensy if ys is provided
    if y_source is not None:
        # Data sanity check
        if X_source.shape[0] != y_source.shape[0]:
            raise RuntimeError("Mismatch in source samples")

    # Check consistensy if yt is provided
    if y_target is not None:
        if X_target.shape[0] != y_target.shape[0]:
            raise RuntimeError("Mismatch in target samples")
