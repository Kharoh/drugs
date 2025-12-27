from .text import TextEmbedConfig, make_text_embed_fn
from .esm import fetch_uniprot_sequence, make_esm_embed_fn
from .path import default_embedding_path

__all__ = [
    "TextEmbedConfig",
    "make_text_embed_fn",
    "fetch_uniprot_sequence",
    "make_esm_embed_fn",
]
