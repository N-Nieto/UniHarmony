:py:mod:`uniharmony.ot._otda`
=============================

.. py:module:: uniharmony.ot._otda

.. autodoc2-docstring:: uniharmony.ot._otda
   :parser: rst
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`OptimalTransportDomainAdaptation <uniharmony.ot._otda.OptimalTransportDomainAdaptation>`
     - .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.ot._otda.__all__>`
     - .. autodoc2-docstring:: uniharmony.ot._otda.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.ot._otda.logger>`
     - .. autodoc2-docstring:: uniharmony.ot._otda.logger
          :parser: rst
          :summary:
   * - :py:obj:`OTMethodType <uniharmony.ot._otda.OTMethodType>`
     - .. autodoc2-docstring:: uniharmony.ot._otda.OTMethodType
          :parser: rst
          :summary:
   * - :py:obj:`MetricType <uniharmony.ot._otda.MetricType>`
     - .. autodoc2-docstring:: uniharmony.ot._otda.MetricType
          :parser: rst
          :summary:
   * - :py:obj:`NormalizationType <uniharmony.ot._otda.NormalizationType>`
     - .. autodoc2-docstring:: uniharmony.ot._otda.NormalizationType
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.ot._otda.__all__
   :value: ['OptimalTransportDomainAdaptation']

   .. autodoc2-docstring:: uniharmony.ot._otda.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.ot._otda.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.ot._otda.logger
      :parser: rst

.. py:data:: OTMethodType
   :canonical: uniharmony.ot._otda.OTMethodType
   :value: None

   .. autodoc2-docstring:: uniharmony.ot._otda.OTMethodType
      :parser: rst

.. py:data:: MetricType
   :canonical: uniharmony.ot._otda.MetricType
   :value: None

   .. autodoc2-docstring:: uniharmony.ot._otda.MetricType
      :parser: rst

.. py:data:: NormalizationType
   :canonical: uniharmony.ot._otda.NormalizationType
   :value: None

   .. autodoc2-docstring:: uniharmony.ot._otda.NormalizationType
      :parser: rst

.. py:class:: OptimalTransportDomainAdaptation(ot_method: str | ot.da.BaseTransport = 'emd', metric: uniharmony.ot._otda.MetricType = 'euclidean', reg: float | None = 1.0, eta: float | None = 0.1, max_iter: int | None = 10, cost_norm: uniharmony.ot._otda.NormalizationType = None, limit_max: int | None = 10, copy: bool = True)
   :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation

   Bases: :py:obj:`ot.da.BaseTransport`, :py:obj:`sklearn.base.TransformerMixin`, :py:obj:`sklearn.base.BaseEstimator`

   .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation
      :parser: rst

   .. rubric:: Initialization

   .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation.__init__
      :parser: rst

   .. py:method:: fit(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, ref_site: str | list[str] | int | list[int], y: numpy.typing.ArrayLike | None = None) -> uniharmony.ot._otda.OptimalTransportDomainAdaptation
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation.fit

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation.fit
         :parser: rst

   .. py:method:: transform(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike | None = None, y: numpy.typing.ArrayLike | None = None, batch_size: int = 128) -> numpy.typing.NDArray
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation.transform

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation.transform
         :parser: rst

   .. py:method:: fit_transform(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, ref_site: str | list[str] | int | list[int], y: numpy.typing.ArrayLike | None = None, batch_size: int = 128) -> numpy.typing.NDArray
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation.fit_transform

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation.fit_transform
         :parser: rst

   .. py:method:: inverse_transform(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike | None = None, y: numpy.typing.ArrayLike | None = None, batch_size: int = 128) -> numpy.typing.NDArray
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation.inverse_transform

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation.inverse_transform
         :parser: rst

   .. py:method:: _resolve_ot_method(**kwargs: typing.Any) -> ot.da.BaseTransport
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation._resolve_ot_method

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation._resolve_ot_method
         :parser: rst

   .. py:method:: _split_ref_harm_data(X: numpy.typing.ArrayLike, sites: numpy.typing.ArrayLike, ref_site: str | int | list[str] | list[int], y: numpy.typing.ArrayLike | None = None) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray | None, numpy.typing.NDArray | None]
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation._split_ref_harm_data

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation._split_ref_harm_data
         :parser: rst

   .. py:method:: __sklearn_tags__() -> sklearn.utils.Tags
      :canonical: uniharmony.ot._otda.OptimalTransportDomainAdaptation.__sklearn_tags__

      .. autodoc2-docstring:: uniharmony.ot._otda.OptimalTransportDomainAdaptation.__sklearn_tags__
         :parser: rst
