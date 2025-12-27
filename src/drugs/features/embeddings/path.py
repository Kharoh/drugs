"""Feature-level helpers for embeddings and caching paths."""

from __future__ import annotations

from pathlib import Path


def default_embedding_path(identifier: str, kind: str, suffix: str) -> Path:
	"""Build a default filesystem path for an embedding artifact.

	Parameters
	----------
	identifier : str
		Unique identifier for the entity (e.g., CID, ChEMBL ID).
	kind : str
		Embedding type label (e.g., ``"text"`` or ``"protein"``).
	suffix : str
		File suffix/extension without the leading dot.

	Returns
	-------
	Path
		Path pointing to ``artifacts/embeddings/<kind>_<identifier>.<suffix>`` with
		slashes replaced to keep it filesystem-safe.
	"""

	safe_id = identifier.replace("/", "_") if identifier else "unknown"
	return Path("artifacts/embeddings") / f"{kind}_{safe_id}.{suffix}"


__all__ = ["default_embedding_path"]
