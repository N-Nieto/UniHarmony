:py:mod:`uniharmony._utils`
===========================

.. py:module:: uniharmony._utils

.. autodoc2-docstring:: uniharmony._utils
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`solve_ordinary_least_squares <uniharmony._utils.solve_ordinary_least_squares>`
     - .. autodoc2-docstring:: uniharmony._utils.solve_ordinary_least_squares
          :parser: rst
          :summary:
   * - :py:obj:`handle_near_zero_values <uniharmony._utils.handle_near_zero_values>`
     - .. autodoc2-docstring:: uniharmony._utils.handle_near_zero_values
          :parser: rst
          :summary:
   * - :py:obj:`handle_negative_variance <uniharmony._utils.handle_negative_variance>`
     - .. autodoc2-docstring:: uniharmony._utils.handle_negative_variance
          :parser: rst
          :summary:
   * - :py:obj:`minimum_samples_warning <uniharmony._utils.minimum_samples_warning>`
     - .. autodoc2-docstring:: uniharmony._utils.minimum_samples_warning
          :parser: rst
          :summary:
   * - :py:obj:`validate_covariates <uniharmony._utils.validate_covariates>`
     - .. autodoc2-docstring:: uniharmony._utils.validate_covariates
          :parser: rst
          :summary:
   * - :py:obj:`validate_sites <uniharmony._utils.validate_sites>`
     - .. autodoc2-docstring:: uniharmony._utils.validate_sites
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony._utils.__all__>`
     - .. autodoc2-docstring:: uniharmony._utils.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony._utils.logger>`
     - .. autodoc2-docstring:: uniharmony._utils.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony._utils.__all__
   :value: ['handle_near_zero_values', 'handle_negative_variance', 'minimum_samples_warning', 'solve_ordinary_l...

   .. autodoc2-docstring:: uniharmony._utils.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony._utils.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony._utils.logger
      :parser: rst

.. py:function:: solve_ordinary_least_squares(gram_matrix: numpy.typing.NDArray, X: numpy.typing.ArrayLike, design: numpy.typing.NDArray) -> numpy.typing.NDArray
   :canonical: uniharmony._utils.solve_ordinary_least_squares

   .. autodoc2-docstring:: uniharmony._utils.solve_ordinary_least_squares
      :parser: rst

.. py:function:: handle_near_zero_values(values: numpy.typing.ArrayLike, epsilon: float = 1e-08, context: str = 'features') -> numpy.typing.NDArray
   :canonical: uniharmony._utils.handle_near_zero_values

   .. autodoc2-docstring:: uniharmony._utils.handle_near_zero_values
      :parser: rst

.. py:function:: handle_negative_variance(variance: numpy.typing.ArrayLike) -> numpy.typing.NDArray
   :canonical: uniharmony._utils.handle_negative_variance

   .. autodoc2-docstring:: uniharmony._utils.handle_negative_variance
      :parser: rst

.. py:function:: minimum_samples_warning(n_samples_per_site: list[list[int]] | numpy.typing.NDArray, min_samples_limit: int = 16) -> None
   :canonical: uniharmony._utils.minimum_samples_warning

   .. autodoc2-docstring:: uniharmony._utils.minimum_samples_warning
      :parser: rst

.. py:function:: validate_covariates(covariates: numpy.typing.NDArray | None, n_samples: int, name: str) -> numpy.typing.NDArray | None
   :canonical: uniharmony._utils.validate_covariates

   .. autodoc2-docstring:: uniharmony._utils.validate_covariates
      :parser: rst

.. py:function:: validate_sites(sites: numpy.typing.NDArray) -> None
   :canonical: uniharmony._utils.validate_sites

   .. autodoc2-docstring:: uniharmony._utils.validate_sites
      :parser: rst
