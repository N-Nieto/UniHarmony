"""Generate ``neurocombat_0.2.12_reference.npz``, the frozen neuroCombat output used by the conformance tests.

Run it only to rebuild the reference (it needs ``neuroCombat==0.2.12``)::

    python tests/combat/make_neurocombat_reference.py

The designs are simulated from the ComBat model (location and scale site
effects, covariate effects). The data and the continuous covariates are
rounded to values that are exact in float32, because neuroCombat converts both
to float32: with such inputs its conversion loses nothing, and the frozen
output can be compared with UniHarmony at a tight tolerance.

Keys are ``"<design>__<name>"`` with names ``X``, ``sites``, ``categorical``
(absent without categorical covariates), ``continuous`` (absent without
continuous covariates), ``mean`` (neuroCombat's ``stand.mean + mod.mean``) and
one output per mode: ``eb_parametric``, ``eb_non_parametric``, ``no_eb`` and
``mean_only``, plus ``eb_parametric_converged``.

``eb_parametric_converged`` is the parametric empirical Bayes output with the
iterative solver (``it_sol``) run to convergence (relative change < 1e-13).
neuroCombat's stopping rule, ``max(abs(g_new - g_old) / g_old, ...)``, divides
by the signed previous estimate, so features with a negative location effect
never block the stop and the default output (``eb_parametric``) is not
converged to its own tolerance. The converged output is a fixed point of the
same equations, independent of any stopping rule.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat


OUTPUT = Path(__file__).parent / "neurocombat_0.2.12_reference.npz"

#: Simulated designs: subjects per site, features, categorical covariate levels, continuous covariates
DESIGNS = {
    "one_categorical_one_continuous": {"n_per_site": [6, 8, 10], "n_features": 8, "levels": [2], "n_continuous": 1},
    "several_covariates": {"n_per_site": [12, 15, 18, 20], "n_features": 10, "levels": [3, 2], "n_continuous": 2},
    "no_covariates": {"n_per_site": [10, 15, 20], "n_features": 10, "levels": [], "n_continuous": 0},
    "imbalanced_sites": {"n_per_site": [5, 15, 60], "n_features": 10, "levels": [2], "n_continuous": 1},
    "large_scale_covariates": {
        "n_per_site": [15, 20, 25],
        "n_features": 10,
        "levels": [2],
        "n_continuous": 2,
        "continuous_scale": 1e4,
    },
    "covariate_confounded_with_site": {
        "n_per_site": [15, 20, 25],
        "n_features": 10,
        "levels": [],
        "n_continuous": 1,
        "site_shift": 2.0,
    },
}

#: neuroCombat options of each mode
MODES = {
    "eb_parametric": {},
    "eb_non_parametric": {"parametric": False},
    "no_eb": {"eb": False},
    "mean_only": {"mean_only": True},
}


def _float32_exact(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32).astype(np.float64)


def simulate(
    rng: np.random.Generator,
    n_per_site: list[int],
    n_features: int,
    levels: list[int],
    n_continuous: int,
    continuous_scale: float = 1.0,
    site_shift: float = 0.0,
) -> dict[str, np.ndarray]:
    """Simulate one design from the ComBat model."""
    n_sites = len(n_per_site)
    site_index = np.repeat(np.arange(n_sites), n_per_site)
    n = len(site_index)
    data: dict[str, np.ndarray] = {"sites": np.array([f"site{k}" for k in site_index])}
    # y = alpha + covariate effects + sigma * (gamma_site + delta_site * eps)
    sigma = rng.uniform(0.5, 2, n_features)
    X = rng.normal(5, 1, n_features) + np.zeros((n, n_features))
    if levels:
        # Every level appears at every site (balanced within site), so the design has full rank
        categorical = np.column_stack(
            [np.concatenate([rng.permutation(np.arange(n_i) % n_levels) for n_i in n_per_site]) for n_levels in levels]
        )
        data["categorical"] = categorical
        for j, n_levels in enumerate(levels):
            X += rng.normal(0, 0.5, (n_levels, n_features))[categorical[:, j]]
    if n_continuous:
        continuous = rng.normal(size=(n, n_continuous)) + site_shift * site_index[:, None]
        continuous = _float32_exact(continuous_scale * (continuous + 3))
        data["continuous"] = continuous
        X += (continuous / continuous_scale) @ rng.normal(0, 0.5, (n_continuous, n_features))
    location = rng.normal(0, 0.5, n_sites)[:, None] + 0.3 * rng.standard_normal((n_sites, n_features))
    scale = np.sqrt(9.0 / rng.gamma(10.0, 1.0, (n_sites, n_features)))
    X += sigma * (location[site_index] + scale[site_index] * rng.standard_normal((n, n_features)))
    data["X"] = _float32_exact(X)
    return data


def run_neurocombat(data: dict[str, np.ndarray], **options) -> tuple[np.ndarray, np.ndarray]:
    """Harmonize with neuroCombat; return the output (n_samples, n_features) and its standardized mean."""
    covars = {"site": data["sites"]}
    categorical_cols = [f"categorical{j}" for j in range(data["categorical"].shape[1])] if "categorical" in data else []
    continuous_cols = [f"continuous{j}" for j in range(data["continuous"].shape[1])] if "continuous" in data else []
    for j, name in enumerate(categorical_cols):
        covars[name] = data["categorical"][:, j]
    for j, name in enumerate(continuous_cols):
        covars[name] = data["continuous"][:, j]
    with contextlib.redirect_stdout(io.StringIO()):
        out = neuroCombat(
            dat=data["X"].T,
            covars=pd.DataFrame(covars),
            batch_col="site",
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            **options,
        )
    return out["data"].T.astype(np.float64), (out["estimates"]["stand.mean"] + out["estimates"]["mod.mean"]).T


@contextlib.contextmanager
def converged_solver():
    """Run neuroCombat's iterative solver to convergence (tolerance 1e-13) inside the context."""
    module = sys.modules[neuroCombat.__module__]
    it_sol = module.it_sol

    def it_sol_converged(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-13):
        return it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-13)

    module.it_sol = it_sol_converged
    try:
        yield
    finally:
        module.it_sol = it_sol


def main() -> None:
    """Simulate every design, run neuroCombat in every mode and save the result."""
    rng = np.random.default_rng(20261103)
    arrays = {}
    for design, spec in DESIGNS.items():
        data = simulate(rng, **spec)
        for name, values in data.items():
            arrays[f"{design}__{name}"] = values
        for mode, options in MODES.items():
            output, mean = run_neurocombat(data, **options)
            arrays[f"{design}__{mode}"] = output
        with converged_solver():
            arrays[f"{design}__eb_parametric_converged"] = run_neurocombat(data)[0]
        arrays[f"{design}__mean"] = mean
    np.savez_compressed(OUTPUT, **arrays)
    print(f"Saved {len(DESIGNS)} designs x {len(MODES) + 1} outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
