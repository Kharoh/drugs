import numpy as np

from drugs.core import Drug


def test_identifier_resolution_and_maps(monkeypatch):
	drug = Drug.from_chembl_id("CHEMBL25")

	monkeypatch.setattr("drugs.core.chembl_to_pubchem_cid", lambda chembl_id: 2244)
	monkeypatch.setattr("drugs.core.pubchem_cid_to_chembl", lambda cid: "CHEMBL25")
	monkeypatch.setattr("drugs.core.pubchem_cid_to_inchikey", lambda cid: "ABC-INCHI")

	monkeypatch.setattr("drugs.core.pubchem_properties", lambda cid: {"InChIKey": "ABC-INCHI", "IUPACName": "Test"})

	ids = drug.map_ids()
	assert ids["pubchem_cid"] == "2244"
	assert ids["chembl_id"] == "CHEMBL25"
	assert ids["inchikey"] == "ABC-INCHI"


def test_target_accessions_and_gene_symbols(monkeypatch):
	drug = Drug.from_pubchem_cid(2244)

	monkeypatch.setattr("drugs.core.pubchem_cid_to_chembl", lambda cid: "CHEMBL25")
	monkeypatch.setattr("drugs.core.chembl_mechanisms", lambda chembl_id, limit=50: [
		{"target_chembl_id": "CHEMBL_T1", "molecule_chembl_id": chembl_id, "target_pref_name": "Target A"},
		{"target_chembl_id": "CHEMBL_T1", "molecule_chembl_id": chembl_id, "target_pref_name": "Target A"},
	])

	def fake_target_detail(target_id):
		return {
			"target_components": [
				{
					"accession": "P12345",
					"target_component_synonyms": [
						{"syn_type": "GENE_SYMBOL", "component_synonym": "GENE1"},
						{"syn_type": "GENE_SYMBOL", "component_synonym": "GENE1"},
					],
				}
			]
		}

	monkeypatch.setattr("drugs.core.chembl_target_detail", fake_target_detail)

	accessions = drug.target_accessions()
	symbols = drug.target_gene_symbols()

	assert accessions == ["P12345"]
	assert symbols == ["GENE1"]


def test_text_embedding_cached(tmp_path, monkeypatch):
	drug = Drug.from_pubchem_cid(2244)

	monkeypatch.setattr(
		"drugs.core.pubchem_text_snippets",
		lambda cid, headings: {"Record Description": {"RecordTitle": "Title", "Strings": ["hello"]}},
	)
	monkeypatch.setattr("drugs.core.pubchem_properties", lambda cid: {"IUPACName": "Test"})
	monkeypatch.setattr("drugs.core.pubchem_cid_to_chembl", lambda cid: "CHEMBL25")

	path = tmp_path / "text.npy"
	called = {"count": 0}

	def embed_fn(text: str):
		called["count"] += 1
		return np.array([1.0, float(len(text))])

	# first call should compute and store
	arr1 = drug.text_embedding_cached(embed_fn, path=path, force=False, load_if_exists=True)
	assert called["count"] == 1
	assert path.exists()

	# second call should load from disk without recomputing
	arr2 = drug.text_embedding_cached(embed_fn, path=path, force=False, load_if_exists=True)
	assert called["count"] == 1
	assert np.allclose(arr1, arr2)


def test_smiles_selfies_and_fingerprints(monkeypatch):
	drug = Drug.from_pubchem_cid(1)
	# Seed properties cache to avoid network
	drug._pubchem_properties_cache = {"CanonicalSMILES": "CCO", "IUPACName": "ethanol"}

	import types, sys
	fake_selfies = types.SimpleNamespace(encoder=lambda s: f"SELFIES({s})")
	monkeypatch.setitem(sys.modules, "selfies", fake_selfies)

	assert drug.smiles() == "CCO"
	assert drug.selfies() == "SELFIES(CCO)"

	fp = drug.molecular_fingerprint(method="morgan", n_bits=128)
	assert fp.shape == (128,)

	drug2 = Drug.from_pubchem_cid(2)
	drug2._pubchem_properties_cache = {"CanonicalSMILES": "CCO"}
	sim = drug.similarity_to(drug2, fingerprint_method="morgan", n_bits=128)
	assert sim == 1.0


def test_molecular_properties(monkeypatch):
	drug = Drug.from_pubchem_cid(1)
	drug._pubchem_properties_cache = {"CanonicalSMILES": "CCO"}
	props = drug.molecular_properties()
	assert "lipinski_pass" in props
	if props["qed"] is not None:
		assert 0 <= props["qed"] <= 1
	if props["tpsa"] is not None:
		assert props["tpsa"] >= 0


def test_batch_and_similarity_matrix(monkeypatch):
	ids = [1, "CHEMBL25", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]
	drugs = Drug.from_batch(ids, prefetch_properties=False, max_workers=2)
	assert len(drugs) == 3
	for d, expected_type in zip(drugs, [int, str, str]):
		if expected_type is int:
			assert d.pubchem_cid == 1
		elif expected_type is str and d._chembl_id:
			assert d._chembl_id == "CHEMBL25"

	# Provide SMILES to avoid network and compute matrix
	for d in drugs:
		d._pubchem_properties_cache = {"CanonicalSMILES": "CCO"}
	mat = Drug.batch_similarity_matrix(drugs, n_bits=64)
	assert mat.shape == (3, 3)
	assert np.allclose(np.diag(mat), 1.0)


def test_fetch_chembl_bioactivities(monkeypatch):
	drug = Drug.from_chembl_id("CHEMBL25")

	monkeypatch.setattr(
		"drugs.core.chembl_bioactivities",
		lambda chembl_id, min_pchembl, assay_types, limit: [
			{"activity_id": 1, "pchembl_value": 7.0, "assay_type": "B"},
		],
	)

	rows = drug.fetch_chembl_bioactivities(min_pchembl=6.0, assay_types=["B"], limit=10)
	assert rows[0]["pchembl_value"] == 7.0


def test_fetch_drug_interactions(monkeypatch):
	drug = Drug.from_pubchem_cid(2244)
	drug._pubchem_properties_cache = {"IUPACName": "aspirin"}

	payload = {
		"interactionTypeGroup": [
			{
				"interactionType": [
					{
						"sourceDisclaimer": "Micromedex",
						"interactionPair": [
							{
								"description": "Increased bleeding risk",
								"interactionConcept": [
									{"minConceptItem": {"name": "aspirin"}},
									{"minConceptItem": {"name": "warfarin"}},
								],
							}
						],
					}
				]
			}
		]
	}

	monkeypatch.setattr("drugs.core.rxnav_interactions", lambda drug_name: payload)
	interactions = drug.fetch_drug_interactions()
	assert interactions[0]["source"] == "Micromedex"
	assert "warfarin" in interactions[0]["interactants"]
