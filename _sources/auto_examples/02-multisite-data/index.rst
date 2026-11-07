

.. _sphx_glr_auto_examples_02-multisite-data:

Multisite Data
==============


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="   This example downloads files from the ON-Harmony dataset with DataLad    (which requires git-annex), so it is not executed when building the    documentation. Run it locally.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_04-download_on_harmony_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/04-download_on_harmony`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Download ON-Harmony dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="make_multisite_classification can also simulate covariates, such as age and sex, whose effect on the features is the same in every site. The presets &quot;age&quot; and &quot;sex&quot; give a ready-made example.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_05-plot_simulate_multisite_data_with_covariates_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/05-plot_simulate_multisite_data_with_covariates`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multisite data generation with covariates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="   This example downloads the MAREoS datasets (about 3 MB), so it is not    executed when building the documentation. Run it locally to see the outputs;    the expected outputs are shown as comments.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_02-load_mareos_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/02-load_mareos`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Load MAREoS dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a traveling-subjects study, the same people are measured in every site. Their true values do not change, so the differences between the measurements of a subject are measurement noise and site effects only. This makes it the most direct check of a harmonization: afterwards, a subject&#x27;s measurements should agree across sites about as well as repeated measurements in one site. make_traveling_subjects simulates such a study.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_08-plot_traveling_subjects_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/08-plot_traveling_subjects`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Traveling subjects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Covariate sets the distribution of a covariate in every site and its effect on the features. Here the two sites recruit very different populations, so part of the difference between the sites is biology (age and sex), not a site effect: the covariates are confounded with the site.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_06-plot_simulate_multisite_data_with_costume_covariates_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/06-plot_simulate_multisite_data_with_costume_covariates`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multisite data generation with custom covariates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="make_multisite_regression simulates multi-site data with a continuous target, e.g. age in a brain-age study. Sites often recruit different ranges of the target (a young and an old cohort): the sites then differ in their features because of the target, not only because of the site effect. Here the two sites overlap in the middle of the target range.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_07-plot_generate_multisite_regression_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/07-plot_generate_multisite_regression`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multisite regression data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to generate an unbalanced multisite dataset using the balance_per_site parameter of the make_multisite_classification function.">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_03-plot_generate_imbalance_multisite_data_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/03-plot_generate_imbalance_multisite_data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Generate imbalance multisite data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Multisite data generation">

.. only:: html

  .. image:: /auto_examples/02-multisite-data/images/thumb/sphx_glr_01-plot_generate_multisite_data_thumb.png
    :alt:

  :doc:`/auto_examples/02-multisite-data/01-plot_generate_multisite_data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multisite data generation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/02-multisite-data/04-download_on_harmony
   /auto_examples/02-multisite-data/05-plot_simulate_multisite_data_with_covariates
   /auto_examples/02-multisite-data/02-load_mareos
   /auto_examples/02-multisite-data/08-plot_traveling_subjects
   /auto_examples/02-multisite-data/06-plot_simulate_multisite_data_with_costume_covariates
   /auto_examples/02-multisite-data/07-plot_generate_multisite_regression
   /auto_examples/02-multisite-data/03-plot_generate_imbalance_multisite_data
   /auto_examples/02-multisite-data/01-plot_generate_multisite_data

