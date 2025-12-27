Tutorial: from CID to embeddings
================================

This short walkthrough demonstrates how to start from a PubChem CID, fetch core
metadata, and compute embeddings.

Prerequisites
-------------

.. code-block:: powershell

	pip install -e .

If you want to run the optional embedding helpers, install extras as needed:

.. code-block:: powershell

	# voyage example
	pip install langchain-voyageai
	# or OpenAI
	pip install openai
	# or sentence-transformers
	pip install sentence-transformers

Step 1: create a drug object
----------------------------

.. code-block:: python

	from drugs import Drug

	aspirin = Drug.from_pubchem_cid(2244)
	print(aspirin.map_ids())

Step 2: inspect properties and text
-----------------------------------

.. code-block:: python

	props = aspirin.fetch_pubchem_properties()
	text = aspirin.fetch_pubchem_text()

	print(props.get("IUPACName"))
	print(list(text))  # headings fetched

Step 3: mechanisms and targets
------------------------------

.. code-block:: python

	mechs = aspirin.fetch_chembl_mechanisms()
	print(mechs[:1])

	print(aspirin.target_accessions())
	print(aspirin.target_gene_symbols())

Step 4: generate embeddings (optional)
--------------------------------------

.. code-block:: python

	# Dummy embedding function; replace with your model
	vec = aspirin.text_embedding(lambda text: text[:128])
	print(vec)

Step 5: write a markdown report
--------------------------------

.. code-block:: python

	path = aspirin.write_drug_markdown()
	print(f"Report written to {path}")

Tips
----

- Use ``drugs.core.list_pubchem_text_headings(cid)`` to see available headings.
- The caching helpers ``protein_embedding_cached`` and ``text_embedding_cached``
  store artifacts under ``artifacts/embeddings`` by default.
