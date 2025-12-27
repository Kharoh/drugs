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
