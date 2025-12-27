"""Feature subpackage: embedding helpers and paths."""

from .embeddings import (
	TextEmbedConfig,
	fetch_uniprot_sequence,
	make_esm_embed_fn,
	make_text_embed_fn,
    default_embedding_path
)

__all__ = [
	"default_embedding_path",
	"TextEmbedConfig",
	"make_text_embed_fn",
	"fetch_uniprot_sequence",
	"make_esm_embed_fn",
]
