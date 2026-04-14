:py:mod:`uniharmony.interpolation._utils`
=========================================

.. py:module:: uniharmony.interpolation._utils

.. autodoc2-docstring:: uniharmony.interpolation._utils
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`create_interpolator <uniharmony.interpolation._utils.create_interpolator>`
     - .. autodoc2-docstring:: uniharmony.interpolation._utils.create_interpolator
          :parser: rst
          :summary:
   * - :py:obj:`validate_class_representation <uniharmony.interpolation._utils.validate_class_representation>`
     - .. autodoc2-docstring:: uniharmony.interpolation._utils.validate_class_representation
          :parser: rst
          :summary:
   * - :py:obj:`validate_covariates <uniharmony.interpolation._utils.validate_covariates>`
     - .. autodoc2-docstring:: uniharmony.interpolation._utils.validate_covariates
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.interpolation._utils.__all__>`
     - .. autodoc2-docstring:: uniharmony.interpolation._utils.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.interpolation._utils.logger>`
     - .. autodoc2-docstring:: uniharmony.interpolation._utils.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.interpolation._utils.__all__
   :value: ['create_interpolator', 'validate_class_representation', 'validate_covariates']

   .. autodoc2-docstring:: uniharmony.interpolation._utils.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.interpolation._utils.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.interpolation._utils.logger
      :parser: rst

.. py:function:: create_interpolator(name: str, random_state: int | numpy.random.RandomState = 23, **kwargs) -> type[imblearn.over_sampling.SMOTE] | type[imblearn.over_sampling.BorderlineSMOTE] | type[imblearn.over_sampling.SVMSMOTE] | type[imblearn.over_sampling.ADASYN] | type[imblearn.over_sampling.KMeansSMOTE] | type[imblearn.over_sampling.RandomOverSampler]
   :canonical: uniharmony.interpolation._utils.create_interpolator

   .. autodoc2-docstring:: uniharmony.interpolation._utils.create_interpolator
      :parser: rst

.. py:function:: validate_class_representation(y: numpy.typing.NDArray, sites: numpy.typing.NDArray) -> None
   :canonical: uniharmony.interpolation._utils.validate_class_representation

   .. autodoc2-docstring:: uniharmony.interpolation._utils.validate_class_representation
      :parser: rst

.. py:function:: validate_covariates(n_samples: int, categorical_covariate: numpy.typing.ArrayLike | None, continuous_covariate: numpy.typing.ArrayLike | None, covariate_tolerance: numpy.typing.ArrayLike | None, *, allow_nan: bool = False) -> tuple[numpy.typing.NDArray[typing.Any] | None, numpy.typing.NDArray[numpy.float64] | None, numpy.typing.NDArray[numpy.float64] | None]
   :canonical: uniharmony.interpolation._utils.validate_covariates

   .. autodoc2-docstring:: uniharmony.interpolation._utils.validate_covariates
      :parser: rst
