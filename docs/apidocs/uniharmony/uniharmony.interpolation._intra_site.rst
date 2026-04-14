:py:mod:`uniharmony.interpolation._intra_site`
==============================================

.. py:module:: uniharmony.interpolation._intra_site

.. autodoc2-docstring:: uniharmony.interpolation._intra_site
   :parser: rst
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`IntraSiteInterpolation <uniharmony.interpolation._intra_site.IntraSiteInterpolation>`
     - .. autodoc2-docstring:: uniharmony.interpolation._intra_site.IntraSiteInterpolation
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.interpolation._intra_site.__all__>`
     - .. autodoc2-docstring:: uniharmony.interpolation._intra_site.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.interpolation._intra_site.logger>`
     - .. autodoc2-docstring:: uniharmony.interpolation._intra_site.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.interpolation._intra_site.__all__
   :value: ['IntraSiteInterpolation']

   .. autodoc2-docstring:: uniharmony.interpolation._intra_site.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.interpolation._intra_site.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.interpolation._intra_site.logger
      :parser: rst

.. py:class:: IntraSiteInterpolation(interpolator: str | imblearn.base.SamplerMixin = 'smote', interpolator_kwargs: dict | None = None, random_state: int | numpy.random.RandomState | None = None)
   :canonical: uniharmony.interpolation._intra_site.IntraSiteInterpolation

   Bases: :py:obj:`imblearn.base.SamplerMixin`, :py:obj:`sklearn.base.BaseEstimator`

   .. autodoc2-docstring:: uniharmony.interpolation._intra_site.IntraSiteInterpolation
      :parser: rst

   .. rubric:: Initialization

   .. autodoc2-docstring:: uniharmony.interpolation._intra_site.IntraSiteInterpolation.__init__
      :parser: rst

   .. py:method:: fit_resample(X: numpy.ndarray, y: numpy.ndarray, sites: numpy.ndarray)
      :canonical: uniharmony.interpolation._intra_site.IntraSiteInterpolation.fit_resample

      .. autodoc2-docstring:: uniharmony.interpolation._intra_site.IntraSiteInterpolation.fit_resample
         :parser: rst

   .. py:method:: _fit_resample(X, y, **params)
      :canonical: uniharmony.interpolation._intra_site.IntraSiteInterpolation._fit_resample

      .. autodoc2-docstring:: uniharmony.interpolation._intra_site.IntraSiteInterpolation._fit_resample
         :parser: rst

   .. py:method:: __sklearn_tags__() -> sklearn.utils.Tags
      :canonical: uniharmony.interpolation._intra_site.IntraSiteInterpolation.__sklearn_tags__

      .. autodoc2-docstring:: uniharmony.interpolation._intra_site.IntraSiteInterpolation.__sklearn_tags__
         :parser: rst
