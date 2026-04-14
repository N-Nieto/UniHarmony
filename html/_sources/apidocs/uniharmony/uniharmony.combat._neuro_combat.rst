:py:mod:`uniharmony.combat._neuro_combat`
=========================================

.. py:module:: uniharmony.combat._neuro_combat

.. autodoc2-docstring:: uniharmony.combat._neuro_combat
   :parser: rst
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NeuroComBat <uniharmony.combat._neuro_combat.NeuroComBat>`
     - .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.combat._neuro_combat.__all__>`
     - .. autodoc2-docstring:: uniharmony.combat._neuro_combat.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.combat._neuro_combat.logger>`
     - .. autodoc2-docstring:: uniharmony.combat._neuro_combat.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.combat._neuro_combat.__all__
   :value: ['NeuroComBat']

   .. autodoc2-docstring:: uniharmony.combat._neuro_combat.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.combat._neuro_combat.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.combat._neuro_combat.logger
      :parser: rst

.. py:class:: NeuroComBat(empirical_bayes: bool = True, parametric_adjustments: bool = True, mean_only: bool = False, copy: bool = True)
   :canonical: uniharmony.combat._neuro_combat.NeuroComBat

   Bases: :py:obj:`sklearn.base.TransformerMixin`, :py:obj:`sklearn.base.BaseEstimator`

   .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat
      :parser: rst

   .. rubric:: Initialization

   .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat.__init__
      :parser: rst

   .. py:method:: fit(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, categorical_covariates: numpy.typing.ArrayLike | None = None, continuous_covariates: numpy.typing.ArrayLike | None = None, var_epsilon: float = 1e-08, delta_epsilon: float = 1e-08, tau_2_epsilon: float = 1e-10, max_iter: int = 1000) -> uniharmony.combat._neuro_combat.NeuroComBat
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat.fit

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat.fit
         :parser: rst

   .. py:method:: transform(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, categorical_covariates: numpy.typing.ArrayLike | None = None, continuous_covariates: numpy.typing.ArrayLike | None = None) -> numpy.typing.NDArray
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat.transform

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat.transform
         :parser: rst

   .. py:method:: fit_transform(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, **fit_params) -> numpy.typing.NDArray
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat.fit_transform

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat.fit_transform
         :parser: rst

   .. py:method:: _make_design_matrix(sites: numpy.typing.NDArray, categorical_covariates: numpy.typing.NDArray | None, continuous_covariates: numpy.typing.NDArray | None, fitting: bool = False) -> numpy.typing.NDArray
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._make_design_matrix

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._make_design_matrix
         :parser: rst

   .. py:method:: _standardize_across_features(X: numpy.typing.ArrayLike, design: numpy.typing.NDArray, n_samples: int, n_samples_per_site: numpy.typing.NDArray, fitting: bool = False, epsilon: float = 1e-08) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._standardize_across_features

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._standardize_across_features
         :parser: rst

   .. py:method:: _fit_ls_model(standardized_data: numpy.typing.NDArray, design: numpy.typing.NDArray, idx_per_site: list[list[int]], epsilon: float = 1e-08) -> tuple[numpy.typing.NDArray, list]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._fit_ls_model

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._fit_ls_model
         :parser: rst

   .. py:method:: _find_priors(gamma_hat: numpy.typing.NDArray, delta_hat: list[numpy.typing.NDArray], delta_epsilon: float = 1e-08, tau_2_epsilon: float = 1e-10) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, list[numpy.typing.NDArray], list[numpy.typing.NDArray]]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._find_priors

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._find_priors
         :parser: rst

   .. py:method:: _find_parametric_adjustments(standardized_data: numpy.typing.NDArray, idx_per_site: list[list[int]], gamma_hat: numpy.typing.NDArray, delta_hat: list[numpy.typing.NDArray], gamma_bar: numpy.typing.NDArray, tau_2: numpy.typing.NDArray, a_prior: list[numpy.typing.NDArray], b_prior: list[numpy.typing.NDArray], max_iter: int = 1000) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._find_parametric_adjustments

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._find_parametric_adjustments
         :parser: rst

   .. py:method:: _iteration_solver(standardized_data: numpy.typing.NDArray, gamma_hat: numpy.typing.NDArray, delta_hat: numpy.typing.NDArray, gamma_bar: numpy.typing.NDArray, tau_2: numpy.typing.NDArray, a_prior: numpy.typing.NDArray, b_prior: numpy.typing.NDArray, convergence: float = 0.0001, max_iter: int = 1000) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._iteration_solver

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._iteration_solver
         :parser: rst

   .. py:method:: _find_non_parametric_adjustments(standardized_data: numpy.typing.NDArray, idx_per_site: list[list[int]], gamma_hat: numpy.typing.NDArray, delta_hat: list[numpy.typing.NDArray]) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._find_non_parametric_adjustments

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._find_non_parametric_adjustments
         :parser: rst

   .. py:method:: _int_eprior(standardized_data: numpy.typing.NDArray, gamma_hat: numpy.typing.NDArray, delta_hat: numpy.typing.NDArray) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._int_eprior

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._int_eprior
         :parser: rst

   .. py:method:: _adjust_data_final(standardized_data: numpy.typing.NDArray, standardized_mean: numpy.typing.NDArray, idx_per_site: list[list[int]], epsilon: float = 1e-08)
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._adjust_data_final

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._adjust_data_final
         :parser: rst

   .. py:method:: _compute_inverse_gamma_priors(delta_hat: numpy.typing.NDArray) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._compute_inverse_gamma_priors

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._compute_inverse_gamma_priors
         :parser: rst

   .. py:method:: _postmean(gamma_hat: numpy.typing.ArrayLike, gamma_bar: numpy.typing.ArrayLike, n: int | numpy.typing.NDArray, delta_star: float | numpy.typing.ArrayLike, tau_2: numpy.typing.ArrayLike) -> numpy.typing.NDArray
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._postmean

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._postmean
         :parser: rst

   .. py:method:: _postvar(sum_2: float | numpy.typing.ArrayLike, n: int | numpy.typing.ArrayLike, a_prior: numpy.typing.ArrayLike, b_prior: numpy.typing.ArrayLike) -> numpy.typing.NDArray
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat._postvar

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat._postvar
         :parser: rst

   .. py:method:: __sklearn_is_fitted__() -> bool
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat.__sklearn_is_fitted__

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat.__sklearn_is_fitted__
         :parser: rst

   .. py:method:: __sklearn_tags__() -> sklearn.utils.Tags
      :canonical: uniharmony.combat._neuro_combat.NeuroComBat.__sklearn_tags__

      .. autodoc2-docstring:: uniharmony.combat._neuro_combat.NeuroComBat.__sklearn_tags__
         :parser: rst
