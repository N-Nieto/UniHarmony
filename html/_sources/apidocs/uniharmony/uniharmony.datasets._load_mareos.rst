:py:mod:`uniharmony.datasets._load_mareos`
==========================================

.. py:module:: uniharmony.datasets._load_mareos

.. autodoc2-docstring:: uniharmony.datasets._load_mareos
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`load_MAREoS <uniharmony.datasets._load_mareos.load_MAREoS>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.load_MAREoS
          :parser: rst
          :summary:
   * - :py:obj:`_load_mareos_single_dataset <uniharmony.datasets._load_mareos._load_mareos_single_dataset>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos._load_mareos_single_dataset
          :parser: rst
          :summary:
   * - :py:obj:`_ensure_mareos_data <uniharmony.datasets._load_mareos._ensure_mareos_data>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos._ensure_mareos_data
          :parser: rst
          :summary:
   * - :py:obj:`_validate_mareos_parameters <uniharmony.datasets._load_mareos._validate_mareos_parameters>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos._validate_mareos_parameters
          :parser: rst
          :summary:
   * - :py:obj:`_create_target_dir <uniharmony.datasets._load_mareos._create_target_dir>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos._create_target_dir
          :parser: rst
          :summary:
   * - :py:obj:`_process_effect_param <uniharmony.datasets._load_mareos._process_effect_param>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos._process_effect_param
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.datasets._load_mareos.__all__>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.datasets._load_mareos.logger>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.logger
          :parser: rst
          :summary:
   * - :py:obj:`MAREOS_ZIP_URL <uniharmony.datasets._load_mareos.MAREOS_ZIP_URL>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.MAREOS_ZIP_URL
          :parser: rst
          :summary:
   * - :py:obj:`VALID_EFFECTS <uniharmony.datasets._load_mareos.VALID_EFFECTS>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.VALID_EFFECTS
          :parser: rst
          :summary:
   * - :py:obj:`VALID_EFFECT_TYPES <uniharmony.datasets._load_mareos.VALID_EFFECT_TYPES>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.VALID_EFFECT_TYPES
          :parser: rst
          :summary:
   * - :py:obj:`VALID_EFFECT_EXAMPLES <uniharmony.datasets._load_mareos.VALID_EFFECT_EXAMPLES>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.VALID_EFFECT_EXAMPLES
          :parser: rst
          :summary:
   * - :py:obj:`DATASET_NAMES <uniharmony.datasets._load_mareos.DATASET_NAMES>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.DATASET_NAMES
          :parser: rst
          :summary:
   * - :py:obj:`mareos_pooch <uniharmony.datasets._load_mareos.mareos_pooch>`
     - .. autodoc2-docstring:: uniharmony.datasets._load_mareos.mareos_pooch
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.datasets._load_mareos.__all__
   :value: ['_ensure_mareos_data', 'load_MAREoS']

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.datasets._load_mareos.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.logger
      :parser: rst

.. py:data:: MAREOS_ZIP_URL
   :canonical: uniharmony.datasets._load_mareos.MAREOS_ZIP_URL
   :value: 'https://www.imardgroup.com/mareos-benchmark/public_datasets.zip'

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.MAREOS_ZIP_URL
      :parser: rst

.. py:data:: VALID_EFFECTS
   :canonical: uniharmony.datasets._load_mareos.VALID_EFFECTS
   :value: ['eos', 'true']

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.VALID_EFFECTS
      :parser: rst

.. py:data:: VALID_EFFECT_TYPES
   :canonical: uniharmony.datasets._load_mareos.VALID_EFFECT_TYPES
   :value: ['simple', 'interaction']

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.VALID_EFFECT_TYPES
      :parser: rst

.. py:data:: VALID_EFFECT_EXAMPLES
   :canonical: uniharmony.datasets._load_mareos.VALID_EFFECT_EXAMPLES
   :value: ['1', '2']

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.VALID_EFFECT_EXAMPLES
      :parser: rst

.. py:data:: DATASET_NAMES
   :canonical: uniharmony.datasets._load_mareos.DATASET_NAMES
   :value: ['eos_simple1', 'eos_simple2', 'eos_interaction1', 'eos_interaction2', 'true_simple1', 'true_simple2...

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.DATASET_NAMES
      :parser: rst

.. py:data:: mareos_pooch
   :canonical: uniharmony.datasets._load_mareos.mareos_pooch
   :value: 'create(...)'

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.mareos_pooch
      :parser: rst

.. py:function:: load_MAREoS(effects: list[str] | str | None = None, effect_types: list[str] | str | None = None, effect_examples: list[str] | str | None = None, as_numpy: bool = True, data_dir: pathlib.Path | None = None, force_download: bool = False) -> dict[str, dict[str, pandas.DataFrame | numpy.ndarray]]
   :canonical: uniharmony.datasets._load_mareos.load_MAREoS

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos.load_MAREoS
      :parser: rst

.. py:function:: _load_mareos_single_dataset(data_dir: pathlib.Path, effect: str, effect_type: str, effect_example: str, as_numpy: bool = True) -> tuple[pandas.DataFrame | numpy.ndarray, ...]
   :canonical: uniharmony.datasets._load_mareos._load_mareos_single_dataset

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos._load_mareos_single_dataset
      :parser: rst

.. py:function:: _ensure_mareos_data(data_dir: pathlib.Path | str | None = None, force_download: bool = False) -> pathlib.Path
   :canonical: uniharmony.datasets._load_mareos._ensure_mareos_data

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos._ensure_mareos_data
      :parser: rst

.. py:function:: _validate_mareos_parameters(effects: str | list[str] | None = None, effect_types: str | list[str] | None = None, effect_examples: str | list[str] | None = None) -> tuple[list[str], list[str], list[str]]
   :canonical: uniharmony.datasets._load_mareos._validate_mareos_parameters

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos._validate_mareos_parameters
      :parser: rst

.. py:function:: _create_target_dir(data_dir: pathlib.Path | str | None) -> pathlib.Path
   :canonical: uniharmony.datasets._load_mareos._create_target_dir

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos._create_target_dir
      :parser: rst

.. py:function:: _process_effect_param(param: str | list[str] | None, default_values: list[str], param_name: str) -> list[str]
   :canonical: uniharmony.datasets._load_mareos._process_effect_param

   .. autodoc2-docstring:: uniharmony.datasets._load_mareos._process_effect_param
      :parser: rst
