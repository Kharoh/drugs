"""ESM protein embedding helper using fair-esm models.

Optional dependency: ``pip install fair-esm``. Sequences are fetched from UniProt.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import requests
import torch

try:  # Lazy import guard for environments without fair-esm
    import esm  # type: ignore[import]
except ImportError as e:  # pragma: no cover - import-time optional dependency
    raise ImportError("Missing dependency. Install ESM with: pip install fair-esm") from e


def _parse_fasta(text: str) -> str:
    """Extract the sequence string from a FASTA payload.

    Parameters
    ----------
    text : str
        Raw FASTA text containing header lines and sequence lines.

    Returns
    -------
    str
        Concatenated sequence string with headers removed.

    Raises
    ------
    ValueError
        If no sequence lines are present.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seq = "".join(ln for ln in lines if not ln.startswith(">"))
    if not seq:
        raise ValueError("Empty FASTA / no sequence lines found.")
    return seq


@lru_cache(maxsize=10_000)
def fetch_uniprot_sequence(accession: str, *, timeout_s: int = 30) -> str:
    """Fetch a protein sequence from UniProt's REST API in FASTA format.

    Parameters
    ----------
    accession : str
        UniProt accession to fetch.
    timeout_s : int, default=30
        Request timeout in seconds.

    Returns
    -------
    str
        Amino-acid sequence string.

    Raises
    ------
    HTTPError
        If the response status is not successful.
    ValueError
        If the FASTA payload contains no sequence lines.
    """

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return _parse_fasta(r.text)


def make_esm_embed_fn(
    *,
    model_name: str = "esm2_t12_35M_UR50D",
    repr_layer: Optional[int] = None,
    device: Optional[str] = None,
) -> Callable[[List[str]], torch.Tensor]:
    """Factory that builds an ESM embedding function for UniProt accessions.

    Parameters
    ----------
    model_name : str, default="esm2_t12_35M_UR50D"
        Name of the pretrained ESM model (``esm.pretrained.<model_name>()``).
    repr_layer : int, optional
        Layer index to extract representations from. Defaults to the last layer.
    device : str, optional
        Torch device (``"cuda"`` or ``"cpu"``). Auto-selects GPU if available.

    Returns
    -------
    Callable[[List[str]], torch.Tensor]
        Function that accepts a list of UniProt accessions and returns a stacked
        tensor of per-sequence embeddings (mean pooled token representations).

    Raises
    ------
    ValueError
        If the model name is unknown.
    ImportError
        If ``fair-esm`` is not installed.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = getattr(esm.pretrained, model_name, None)
    if loader is None or not callable(loader):
        raise ValueError(f"Unknown ESM pretrained model: {model_name}")

    model, alphabet = loader()
    model.eval()
    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()

    if repr_layer is None:
        repr_layer = model.num_layers

    @torch.no_grad()
    def embed_fn(accessions: List[str]) -> torch.Tensor:
        """Compute embeddings for UniProt accessions using the configured ESM model.

        Parameters
        ----------
        accessions : list[str]
            UniProt accessions to embed.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(len(accessions), hidden_dim)`` on CPU.
        """
        data: List[Tuple[str, str]] = [(acc, fetch_uniprot_sequence(acc)) for acc in accessions]

        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        token_reps = out["representations"][repr_layer]

        lens = (tokens != alphabet.padding_idx).sum(1)

        seq_reps = []
        for i, L in enumerate(lens.tolist()):
            seq_reps.append(token_reps[i, 1 : L - 1].mean(0).detach().cpu())

        return torch.stack(seq_reps, dim=0)

    return embed_fn


__all__ = ["fetch_uniprot_sequence", "make_esm_embed_fn"]
