:py:mod:`uniharmony.interpolation._inter_site_matched`
======================================================

.. py:module:: uniharmony.interpolation._inter_site_matched

.. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched
   :parser: rst
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InterSiteMatchedInterpolation <uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation>`
     - .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation
          :parser: rst
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_find_matches <uniharmony.interpolation._inter_site_matched._find_matches>`
     - .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched._find_matches
          :parser: rst
          :summary:
   * - :py:obj:`_reverse_matches <uniharmony.interpolation._inter_site_matched._reverse_matches>`
     - .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched._reverse_matches
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.interpolation._inter_site_matched.__all__>`
     - .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.interpolation._inter_site_matched.logger>`
     - .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.interpolation._inter_site_matched.__all__
   :value: ['InterSiteMatchedInterpolation']

   .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.interpolation._inter_site_matched.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.logger
      :parser: rst

.. py:class:: InterSiteMatchedInterpolation(alpha: float | tuple[float, float] | list[float] = 0.3, target_tolerance: float | None = None, covariate_tolerance: numpy.typing.ArrayLike | None = None, k: int | typing.Literal[max, average] = 1, mode: typing.Literal[pairwise, base_to_others] = 'pairwise', *, concatenate: bool = True, random_state: int | numpy.random.RandomState | None = None)
   :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation

   Bases: :py:obj:`imblearn.base.SamplerMixin`, :py:obj:`sklearn.base.BaseEstimator`

   .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation
      :parser: rst

   .. rubric:: Initialization

   .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation.__init__
      :parser: rst

   .. py:method:: _validate_init_params() -> None
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._validate_init_params

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._validate_init_params
         :parser: rst

   .. py:method:: fit_resample(X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, *, categorical_covariate: numpy.typing.ArrayLike | None = None, continuous_covariate: numpy.typing.ArrayLike | None = None, allow_nan=False) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[typing.Any]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation.fit_resample

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation.fit_resample
         :parser: rst

   .. py:method:: _validate_inputs(X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike, categorical_covariate: numpy.typing.ArrayLike | None, continuous_covariate: numpy.typing.ArrayLike | None, allow_nan: bool = False) -> tuple[numpy.typing.NDArray[typing.Any] | None, numpy.typing.NDArray[numpy.float64] | None]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._validate_inputs

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._validate_inputs
         :parser: rst

   .. py:method:: _log_configuration(cat_cov: numpy.typing.NDArray[typing.Any] | None, cont_cov: numpy.typing.NDArray[numpy.float64] | None) -> None
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._log_configuration

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._log_configuration
         :parser: rst

   .. py:method:: _log_completion(y_orig: numpy.typing.NDArray[typing.Any], synthetic_y: list[numpy.typing.NDArray[typing.Any]]) -> None
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._log_completion

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._log_completion
         :parser: rst

   .. py:method:: _generate_samples(X: numpy.typing.NDArray[numpy.float64], y: numpy.typing.NDArray[typing.Any], sites: numpy.typing.NDArray[typing.Any], cat_cov: numpy.typing.NDArray[typing.Any] | None, cont_cov: numpy.typing.NDArray[numpy.float64] | None) -> tuple[list[numpy.typing.NDArray[numpy.float64]], list[numpy.typing.NDArray[typing.Any]], list[numpy.typing.NDArray[typing.Any]]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._generate_samples

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._generate_samples
         :parser: rst

   .. py:method:: _generate_pairwise(X: numpy.typing.NDArray[numpy.float64], y: numpy.typing.NDArray[typing.Any], sites: numpy.typing.NDArray[typing.Any], cat_cov: numpy.typing.NDArray[typing.Any] | None, cont_cov: numpy.typing.NDArray[numpy.float64] | None) -> tuple[list[numpy.typing.NDArray[numpy.float64]], list[numpy.typing.NDArray[typing.Any]], list[numpy.typing.NDArray[typing.Any]]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._generate_pairwise

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._generate_pairwise
         :parser: rst

   .. py:method:: _generate_base_to_others(X: numpy.typing.NDArray[numpy.float64], y: numpy.typing.NDArray[typing.Any], sites: numpy.typing.NDArray[typing.Any], cat_cov: numpy.typing.NDArray[typing.Any] | None, cont_cov: numpy.typing.NDArray[numpy.float64] | None) -> tuple[list[numpy.typing.NDArray[numpy.float64]], list[numpy.typing.NDArray[typing.Any]], list[numpy.typing.NDArray[typing.Any]]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._generate_base_to_others

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._generate_base_to_others
         :parser: rst

   .. py:method:: _interpolate(X_src: numpy.typing.NDArray[numpy.float64], y_src: numpy.typing.NDArray[typing.Any], X_dst: numpy.typing.NDArray[numpy.float64], y_dst: numpy.typing.NDArray[typing.Any], matches: list[list[int]], alpha_min: float, alpha_max: float) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[typing.Any], int]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interpolate

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interpolate
         :parser: rst

   .. py:method:: _interp_average(X_src: numpy.typing.NDArray[numpy.float64], y_src: numpy.typing.NDArray[typing.Any], X_dst: numpy.typing.NDArray[numpy.float64], y_dst: numpy.typing.NDArray[typing.Any], matches: list[list[int]], matched_idx: numpy.typing.NDArray[numpy.int64], alphas: numpy.typing.NDArray[numpy.float64]) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[typing.Any]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interp_average

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interp_average
         :parser: rst

   .. py:method:: _interp_max(X_src: numpy.typing.NDArray[numpy.float64], y_src: numpy.typing.NDArray[typing.Any], X_dst: numpy.typing.NDArray[numpy.float64], y_dst: numpy.typing.NDArray[typing.Any], matches: list[list[int]], matched_idx: numpy.typing.NDArray[numpy.int64], alphas: numpy.typing.NDArray[numpy.float64]) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[typing.Any]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interp_max

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interp_max
         :parser: rst

   .. py:method:: _interp_k(X_src: numpy.typing.NDArray[numpy.float64], y_src: numpy.typing.NDArray[typing.Any], X_dst: numpy.typing.NDArray[numpy.float64], y_dst: numpy.typing.NDArray[typing.Any], matches: list[list[int]], matched_idx: numpy.typing.NDArray[numpy.int64], alphas: numpy.typing.NDArray[numpy.float64], k: int) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[typing.Any]]
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interp_k

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._interp_k
         :parser: rst

   .. py:method:: _fit_resample(X: numpy.typing.ArrayLike, y: numpy.typing.ArrayLike, **params: typing.Any) -> None
      :canonical: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._fit_resample

      .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched.InterSiteMatchedInterpolation._fit_resample
         :parser: rst

.. py:function:: _find_matches(y_src: numpy.typing.NDArray[typing.Any], y_dst: numpy.typing.NDArray[typing.Any], cat_src: numpy.typing.NDArray[typing.Any] | None, cat_dst: numpy.typing.NDArray[typing.Any] | None, cont_src: numpy.typing.NDArray[numpy.float64] | None, cont_dst: numpy.typing.NDArray[numpy.float64] | None, target_tol: float | None, cov_tol: numpy.typing.NDArray[numpy.float64] | None) -> list[list[int]]
   :canonical: uniharmony.interpolation._inter_site_matched._find_matches

   .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched._find_matches
      :parser: rst

.. py:function:: _reverse_matches(matches_fwd: list[list[int]], n_dst: int) -> list[list[int]]
   :canonical: uniharmony.interpolation._inter_site_matched._reverse_matches

   .. autodoc2-docstring:: uniharmony.interpolation._inter_site_matched._reverse_matches
      :parser: rst
