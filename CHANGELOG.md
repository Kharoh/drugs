# Changelog

## [0.1.2] - 2025-12-27
### Added
- SMILES and SELFIES accessors on `Drug`
- Fingerprints (Morgan/MACCS/Daylight), Tanimoto/Dice similarity, and batch similarity matrix
- RDKit molecular property calculators (QED, TPSA, Lipinski violations, synthetic accessibility)
- ChEMBL bioactivity retrieval with pChEMBL and assay-type filters
- Drug-drug interaction lookup via RxNav
- Batch constructor `Drug.from_batch` for concurrent instance creation
- API response caching with configurable TTL and on-disk persistence

### Documentation
- Expanded README and quickstart with structural, bioactivity, safety, and batch examples
- Added API docs for new chemistry and cache modules

## [0.1.1] - 2025-12-27
### Added
- Initial release
- Core functionality
