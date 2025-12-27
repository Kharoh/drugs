# drugs

Lightweight Python utilities to work with small-molecule identifiers and metadata across PubChem and ChEMBL. The library exposes a single `Drug` class that lazily resolves identifiers (PubChem CID, ChEMBL ID, InChIKey), fetches PubChem properties/text, pulls ChEMBL mechanisms, and provides hooks for plugging in your own text or protein embedding functions with optional on-disk caching.

## Highlights

- Lazy identifier translation between PubChem CID, ChEMBL ID, and InChIKey (via UniChem and PUG-REST)
- PubChem properties and PUG-View text retrieval with curated heading presets
- Structure representations: canonical SMILES + SELFIES
- Fingerprints (Morgan/MACCS/Daylight) with Tanimoto/Dice similarity + batch similarity matrices
- ChEMBL mechanisms, target details, and bioactivity rows (pChEMBL/IC50/EC50 filters)
- Drug-drug interactions via RxNav
- RDKit molecular property panel (QED, TPSA, Lipinski violations, synthetic accessibility)
- Embedding hooks for text and protein/sequence features, with simple caching helpers
- Markdown report generation for a drug snapshot

## Installation

Python 3.9+ is required.

```powershell
pip install -e .
```

For development (linting/tests/docs):

```powershell
pip install -e ".[dev]"
```

## Quick start

```python
from drugs import Drug, PUBCHEM_MINIMAL_STABLE

# Start from any identifier
aspirin = Drug.from_pubchem_cid(2244)
# or: Drug.from_chembl_id("CHEMBL25") / Drug.from_inchikey("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")

print(aspirin.map_ids())

props = aspirin.fetch_pubchem_properties()
text = aspirin.fetch_pubchem_text(PUBCHEM_MINIMAL_STABLE)
mechs = aspirin.fetch_chembl_mechanisms()
targets = aspirin.target_accessions()

# Structural views
print(aspirin.smiles())
print(aspirin.selfies())

# Fingerprints + similarity
fp = aspirin.molecular_fingerprint(method="morgan")
ibuprofen = Drug.from_chembl_id("CHEMBL521")
sim = aspirin.similarity_to(ibuprofen)

# Bioactivities and DDIs
acts = aspirin.fetch_chembl_bioactivities(min_pchembl=6.0, assay_types=["B", "F"])
ddis = aspirin.fetch_drug_interactions()

# Batch helpers
batch = Drug.from_batch([2244, "CHEMBL521", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"])
sim_matrix = Drug.batch_similarity_matrix(batch)

# RDKit property panel
print(aspirin.molecular_properties())

# Plug in your own embedding functions
vec = aspirin.text_embedding(lambda s: s.upper())  # replace with your model

# Write a markdown report
aspirin.write_drug_markdown(output_path="aspirin.md")
```

### Caching

API responses (PubChem/ChEMBL/RxNav) are cached to ``artifacts/cache/api_cache.json`` by default with a 24h TTL.
Configure via environment variables:

- ``DRUGS_CACHE_PATH`` – override cache path
- ``DRUGS_CACHE_TTL_SECONDS`` – TTL in seconds
- ``DRUGS_CACHE_DISABLED=1`` – disable disk caching

## API surface

- `Drug.pubchem_cid`, `Drug.chembl_id`, `Drug.inchikey`: resolved identifiers
- `Drug.fetch_pubchem_properties()`: dict of core PubChem properties
- `Drug.fetch_pubchem_text(headings)`: filtered PUG-View text sections
- Structure: `Drug.smiles()`, `Drug.selfies()`, `Drug.molecular_fingerprint()`, `Drug.similarity_to()`
- Bioactivity/targets: `Drug.fetch_chembl_mechanisms()`, `Drug.fetch_chembl_bioactivities()`, `Drug.fetch_target_details()`, `Drug.target_accessions()`, `Drug.target_gene_symbols()`
- Safety: `Drug.fetch_drug_interactions()`
- RDKit properties: `Drug.molecular_properties()`
- Batch helpers: `Drug.from_batch()`, `Drug.batch_similarity_matrix()`
- Embedding helpers: `text_embedding`, `text_embedding_cached`, `protein_embedding`, `protein_embedding_cached`
- Reporting: `write_drug_markdown`

### Heading presets
Curated heading sets live in `drugs.constants` (e.g., `PUBCHEM_MINIMAL_STABLE`, `PUBCHEM_ADME_PK`, `PUBCHEM_MEANING`, etc.). Use `drugs.core.list_pubchem_text_headings(cid)` to inspect available headings for a given CID.

## Tests and quality

```powershell
make test   # runs pytest
make lint   # ruff + mypy
make format # black + autofix lint
```

## Documentation

Build and view the Sphinx docs locally:

```powershell
pip install -e ".[docs]"
cd docs
make html  # or: python -m sphinx -b html . _build/html
```

Then open `_build/html/index.html` in your browser.

### Publishing to GitHub Pages

A GitHub Actions workflow (`.github/workflows/docs.yml`) builds the Sphinx HTML
docs on every push to `main` and publishes them to GitHub Pages.

One-time repo setup:
- In GitHub, go to **Settings → Pages** and set **Source** to **GitHub Actions**.

Manual trigger: use **Actions → docs → Run workflow** to publish immediately.

## Publishing

This project uses Hatchling. To build and publish (requires valid PyPI credentials):

```powershell
pip install hatch
hatch build
hatch publish
```

## Notes

- Network access is required for live API calls to PubChem, ChEMBL, and UniChem.
- Protein embedding cache utilities expect `torch` if you use `protein_embedding_cached`; otherwise no heavy dependencies are required.
