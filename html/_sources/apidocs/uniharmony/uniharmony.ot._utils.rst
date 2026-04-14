:py:mod:`uniharmony.ot._utils`
==============================

.. py:module:: uniharmony.ot._utils

.. autodoc2-docstring:: uniharmony.ot._utils
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`create_ot_object <uniharmony.ot._utils.create_ot_object>`
     - .. autodoc2-docstring:: uniharmony.ot._utils.create_ot_object
          :parser: rst
          :summary:
   * - :py:obj:`data_consistency_check <uniharmony.ot._utils.data_consistency_check>`
     - .. autodoc2-docstring:: uniharmony.ot._utils.data_consistency_check
          :parser: rst
          :summary:

API
~~~

.. py:function:: create_ot_object(name: str, **kwargs) -> type[ot.da.EMDTransport] | type[ot.da.SinkhornTransport] | type[ot.da.SinkhornL1l2Transport] | type[ot.da.EMDLaplaceTransport]
   :canonical: uniharmony.ot._utils.create_ot_object

   .. autodoc2-docstring:: uniharmony.ot._utils.create_ot_object
      :parser: rst

.. py:function:: data_consistency_check(X_source: numpy.typing.ArrayLike, X_target: numpy.typing.ArrayLike, y_source: numpy.typing.ArrayLike | None = None, y_target: numpy.typing.ArrayLike | None = None) -> None
   :canonical: uniharmony.ot._utils.data_consistency_check

   .. autodoc2-docstring:: uniharmony.ot._utils.data_consistency_check
      :parser: rst
