Quickstart
==========

Installation
------------

.. code-block:: powershell

	pip install -e .

If you plan to build the docs, add the ``docs`` extra:

.. code-block:: powershell

	pip install -e ".[docs]"

Create a drug object
--------------------

.. code-block:: python

	from drugs import Drug, PUBCHEM_MINIMAL_STABLE

	# Start from any identifier
	aspirin = Drug.from_pubchem_cid(2244)
	# alternatives:
	# Drug.from_chembl_id("CHEMBL25")
	# Drug.from_inchikey("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")

	print(aspirin.map_ids())

Fetch properties and text
-------------------------

.. code-block:: python

	props = aspirin.fetch_pubchem_properties()
	text = aspirin.fetch_pubchem_text(PUBCHEM_MINIMAL_STABLE)

Structure and fingerprints
--------------------------

.. code-block:: python

	smiles = aspirin.smiles()
	selfies = aspirin.selfies()
	fp = aspirin.molecular_fingerprint(method="morgan")

Similarity and batch workflows
------------------------------

.. code-block:: python

	ibuprofen = Drug.from_chembl_id("CHEMBL521")
	sim = aspirin.similarity_to(ibuprofen)

	cohort = Drug.from_batch([2244, "CHEMBL521", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"])
	mat = Drug.batch_similarity_matrix(cohort)

Bioactivity and safety
----------------------

.. code-block:: python

	acts = aspirin.fetch_chembl_bioactivities(min_pchembl=6.0, assay_types=["B", "F"])
	ddis = aspirin.fetch_drug_interactions()

RDKit property calculators
--------------------------

.. code-block:: python

	props = aspirin.molecular_properties()
	print(props["qed"], props["tpsa"], props["lipinski_violations"])

Mechanisms and targets
----------------------

.. code-block:: python

	mechs = aspirin.fetch_chembl_mechanisms()
	accessions = aspirin.target_accessions()
	genes = aspirin.target_gene_symbols()

Embeddings
----------

Plug in any embedding function. A trivial example:

.. code-block:: python

	vec = aspirin.text_embedding(lambda s: s.upper())

Report generation
-----------------

.. code-block:: python

	aspirin.write_drug_markdown(output_path="aspirin.md")
	# yields artifacts/embeddings when using cached embedding helpers
