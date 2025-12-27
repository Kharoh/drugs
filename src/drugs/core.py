"""High-level Drug object that lazily resolves identifiers and fetches data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .constants import PUBCHEM_MINIMAL_STABLE
from .data_sources import (
    chembl_mechanisms,
    chembl_molecules_by_inchikey,
    chembl_target_detail,
    list_pubchem_text_headings,
    pubchem_properties,
    pubchem_text_snippets,
)
from .id_mapping import (
    chembl_to_pubchem_cid,
    pubchem_cid_from_inchikey,
    pubchem_cid_to_chembl,
    pubchem_cid_to_inchikey,
)
from .utils import dedupe_preserve_order


@dataclass
class Drug:
    """
    Represent a drug and lazily translate between PubChem CID, ChEMBL ID, and InChIKey.

    Network calls are performed on demand and results are cached on the instance to
    avoid repeated HTTP calls. The class is intentionally model-agnostic: callers can
    plug in any embedding function via ``text_embedding``/``protein_embedding``.
    """

    _pubchem_cid: Optional[int] = field(default=None, repr=False)
    _chembl_id: Optional[str] = field(default=None, repr=False)
    _inchikey: Optional[str] = field(default=None, repr=False)

    synonyms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _pubchem_properties_cache: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _pubchem_text_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _chembl_mechanism_cache: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)
    _target_detail_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # Identifier resolution -------------------------------------------------
    @property
    def pubchem_cid(self) -> Optional[int]:
        if self._pubchem_cid is not None:
            return self._pubchem_cid

        cid: Optional[int] = None
        if self._chembl_id:
            mapped = chembl_to_pubchem_cid(self._chembl_id)
            if isinstance(mapped, int):
                cid = mapped
            elif isinstance(mapped, list) and mapped:
                cid = int(mapped[0])

        if cid is None and self._inchikey:
            mapped = pubchem_cid_from_inchikey(self._inchikey)
            if isinstance(mapped, int):
                cid = mapped
            elif isinstance(mapped, list) and mapped:
                cid = int(mapped[0])

        if cid is not None:
            self._pubchem_cid = cid
        return self._pubchem_cid

    @property
    def chembl_id(self) -> Optional[str]:
        if self._chembl_id is not None:
            return self._chembl_id

        chembl: Optional[str] = None
        if self._pubchem_cid:
            mapped = pubchem_cid_to_chembl(self._pubchem_cid)
            if isinstance(mapped, str):
                chembl = mapped
            elif isinstance(mapped, list) and mapped:
                chembl = mapped[0]

        if chembl is None and self._inchikey:
            candidates = chembl_molecules_by_inchikey(self._inchikey, limit=5)
            chembl = next((c.get("molecule_chembl_id") for c in candidates if c.get("molecule_chembl_id")), None)

        if chembl:
            self._chembl_id = chembl
        return self._chembl_id

    @property
    def inchikey(self) -> Optional[str]:
        if self._inchikey:
            return self._inchikey

        if self._pubchem_properties_cache and self._pubchem_properties_cache.get("InChIKey"):
            self._inchikey = self._pubchem_properties_cache["InChIKey"]
        elif self._pubchem_cid:
            self._inchikey = pubchem_cid_to_inchikey(self._pubchem_cid)

        return self._inchikey

    # Fetchers -------------------------------------------------------------
    def fetch_pubchem_properties(self) -> Dict[str, Any]:
        """Retrieve and cache core PubChem properties.

        Returns
        -------
        dict
            Dictionary of properties for the resolved PubChem CID.

        Raises
        ------
        ValueError
            If no resolvable PubChem CID is available.
        """
        if self._pubchem_properties_cache is not None:
            return self._pubchem_properties_cache

        cid = self.pubchem_cid
        if cid is None:
            raise ValueError("Cannot fetch PubChem properties without a CID or a resolvable identifier.")

        props = pubchem_properties(cid)
        self._pubchem_properties_cache = props
        if not self._inchikey and props.get("InChIKey"):
            self._inchikey = props["InChIKey"]
        return props

    def fetch_pubchem_text(self, headings: Iterable[str] = PUBCHEM_MINIMAL_STABLE) -> Dict[str, Any]:
        """Fetch and cache selected PubChem PUG-View text sections.

        Parameters
        ----------
        headings : Iterable[str], optional
            Headings to request. Defaults to ``PUBCHEM_MINIMAL_STABLE``.

        Returns
        -------
        dict
            Mapping of heading -> section metadata and extracted strings.

        Raises
        ------
        ValueError
            If no resolvable PubChem CID is available.
        """
        key = tuple(headings)
        if key in self._pubchem_text_cache:
            return self._pubchem_text_cache[key]

        cid = self.pubchem_cid
        if cid is None:
            raise ValueError("Cannot fetch PubChem text without a CID or a resolvable identifier.")

        data = pubchem_text_snippets(cid, headings)
        self._pubchem_text_cache[key] = data
        return data

    def fetch_chembl_mechanisms(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch mechanisms of action for the drug's ChEMBL ID.

        Parameters
        ----------
        limit : int, default=50
            Maximum number of mechanism records to fetch.

        Returns
        -------
        list[dict]
            Mechanism entries from the ChEMBL API.

        Raises
        ------
        ValueError
            If no resolvable ChEMBL ID is available.
        """
        if self._chembl_mechanism_cache is not None:
            return self._chembl_mechanism_cache

        chembl_id = self.chembl_id
        if chembl_id is None:
            raise ValueError("Cannot fetch ChEMBL mechanisms without a ChEMBL ID or a resolvable identifier.")

        mechs = chembl_mechanisms(chembl_id, limit=limit)
        self._chembl_mechanism_cache = mechs
        return mechs

    def fetch_target_details(self, target_chembl_id: str) -> Dict[str, Any]:
        """Fetch and cache target details for a ChEMBL target ID.

        Parameters
        ----------
        target_chembl_id : str
            ChEMBL target identifier.

        Returns
        -------
        dict
            Target detail payload including components and synonyms.
        """
        if target_chembl_id in self._target_detail_cache:
            return self._target_detail_cache[target_chembl_id]

        detail = chembl_target_detail(target_chembl_id)
        self._target_detail_cache[target_chembl_id] = detail
        return detail

    # Derived views --------------------------------------------------------
    def target_accessions(self) -> List[str]:
        """Return UniProt accessions for all targets linked to the drug."""
        accessions: List[str] = []
        for mech in self.fetch_chembl_mechanisms() or []:
            tid = mech.get("target_chembl_id")
            if not tid:
                continue
            detail = self.fetch_target_details(tid)
            for comp in detail.get("target_components", []) or []:
                acc = comp.get("accession")
                if acc:
                    accessions.append(acc)
        return dedupe_preserve_order(accessions)

    def target_gene_symbols(self) -> List[str]:
        """Return gene symbols for all targets linked to the drug."""
        symbols: List[str] = []
        for mech in self.fetch_chembl_mechanisms() or []:
            tid = mech.get("target_chembl_id")
            if not tid:
                continue
            detail = self.fetch_target_details(tid)
            for comp in detail.get("target_components", []) or []:
                syns = comp.get("target_component_synonyms", []) or []
                for syn in syns:
                    if syn.get("syn_type") == "GENE_SYMBOL" and syn.get("component_synonym"):
                        symbols.append(syn["component_synonym"])
        return dedupe_preserve_order(symbols)

    def text_corpus(self, headings: Iterable[str] = PUBCHEM_MINIMAL_STABLE) -> str:
        """Concatenate PubChem text snippets into a markdown-ish corpus.

        Parameters
        ----------
        headings : Iterable[str], optional
            Headings to include. Defaults to ``PUBCHEM_MINIMAL_STABLE``.

        Returns
        -------
        str
            Formatted corpus with heading markers and snippets.
        """
        text = self.fetch_pubchem_text(headings)
        blocks: List[str] = []
        for heading, content in text.items():
            title = content.get("RecordTitle") or heading
            blocks.append(f"## {title}\n")
            for snippet in content.get("Strings", []):
                blocks.append(snippet)
        return "\n\n".join(blocks)

    # Embedding hooks ------------------------------------------------------
    def text_embedding(self, embed_fn: Callable[[str], Any], headings: Iterable[str] = PUBCHEM_MINIMAL_STABLE) -> Any:
        """Compute a text embedding over the PubChem corpus.

        Parameters
        ----------
        embed_fn : Callable[[str], Any]
            User-provided embedding function accepting a text corpus.
        headings : Iterable[str], optional
            Headings to include when building the corpus.

        Returns
        -------
        Any
            Embedding output as returned by ``embed_fn``.
        """
        corpus = self.text_corpus(headings)
        return embed_fn(corpus)

    def esm_inputs(self) -> List[str]:
        """Return UniProt accessions to feed into a protein embedding model."""
        return self.target_accessions()

    def protein_embedding(self, embed_fn: Callable[[List[str]], Any]) -> Any:
        """Compute a protein embedding over target accessions.

        Parameters
        ----------
        embed_fn : Callable[[List[str]], Any]
            Embedding function that consumes a list of UniProt accessions.

        Returns
        -------
        Any
            Embedding output as returned by ``embed_fn``.
        """
        accessions = self.esm_inputs()
        return embed_fn(accessions)

    def _default_embedding_path(self, kind: str, suffix: str) -> Path:
        """Construct a default embedding path based on available identifiers.

        Parameters
        ----------
        kind : str
            Embedding type label (e.g., ``"text"`` or ``"protein"``).
        suffix : str
            File extension without dot (e.g., ``"npy"``).

        Returns
        -------
        Path
            Filesystem path under ``artifacts/embeddings``.
        """
        identifier: Optional[str] = self.chembl_id
        if not identifier and self.pubchem_cid is not None:
            identifier = f"CID{self.pubchem_cid}"
        if not identifier and self.inchikey:
            identifier = self.inchikey
        if not identifier:
            identifier = "unknown"
        safe_id = str(identifier).replace("/", "_")
        return Path("artifacts/embeddings") / f"{kind}_{safe_id}.{suffix}"

    def protein_embedding_cached(
        self,
        embed_fn: Callable[[List[str]], Any],
        *,
        path: Optional[Path] = None,
        load_if_exists: bool = True,
        force: bool = False,
    ) -> Any:
        """Compute or load a cached protein embedding.

        Parameters
        ----------
        embed_fn : Callable[[List[str]], Any]
            Embedding function consuming accessions.
        path : pathlib.Path, optional
            Custom output path. Defaults to an auto-generated path.
        load_if_exists : bool, default=True
            Return the cached embedding if the file already exists.
        force : bool, default=False
            Recompute even if the cache exists.

        Returns
        -------
        Any
            Embedding output returned by ``embed_fn`` or loaded from disk.
        """
        import torch

        target_path = Path(path) if path else self._default_embedding_path("protein", "pt")

        if load_if_exists and target_path.exists() and not force:
            return torch.load(target_path, map_location="cpu")

        emb = self.protein_embedding(embed_fn)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, target_path)
        return emb

    def text_embedding_cached(
        self,
        embed_fn: Callable[[str], Any],
        *,
        headings: Iterable[str] = PUBCHEM_MINIMAL_STABLE,
        path: Optional[Path] = None,
        load_if_exists: bool = True,
        force: bool = False,
    ) -> Any:
        """Compute or load a cached text embedding.

        Parameters
        ----------
        embed_fn : Callable[[str], Any]
            Embedding function consuming a corpus string.
        headings : Iterable[str], optional
            Headings to include when building the corpus.
        path : pathlib.Path, optional
            Custom output path. Defaults to an auto-generated path.
        load_if_exists : bool, default=True
            Return the cached embedding if the file already exists.
        force : bool, default=False
            Recompute even if the cache exists.

        Returns
        -------
        Any
            Embedding output returned by ``embed_fn`` or loaded from disk.
        """
        target_path = Path(path) if path else self._default_embedding_path("text", "npy")

        if load_if_exists and target_path.exists() and not force:
            return np.load(target_path)

        emb = self.text_embedding(embed_fn, headings=headings)
        emb_arr = np.asarray(emb)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, emb_arr)
        return emb_arr

    def map_ids(self) -> Dict[str, Optional[str]]:
        """Return a dictionary with the best-effort resolved identifiers."""
        cid = self.pubchem_cid
        chembl = self.chembl_id
        inchikey = self.inchikey
        return {
            "pubchem_cid": str(cid) if cid is not None else None,
            "chembl_id": chembl,
            "inchikey": inchikey,
        }

    # Constructors ---------------------------------------------------------
    @classmethod
    def from_pubchem_cid(cls, cid: int) -> "Drug":
        return cls(_pubchem_cid=int(cid))

    @classmethod
    def from_chembl_id(cls, chembl_id: str) -> "Drug":
        return cls(_chembl_id=str(chembl_id))

    @classmethod
    def from_inchikey(cls, inchikey: str) -> "Drug":
        return cls(_inchikey=inchikey)

    def write_drug_markdown(
        self,
        *,
        headings: Iterable[str] = PUBCHEM_MINIMAL_STABLE,
        output_path: Path = Path("drug_report.md"),
        include_mechanisms: bool = True,
        include_targets: bool = True,
    ) -> Path:
        """Fetch drug information and persist a Markdown report.

        Parameters
        ----------
        headings : Iterable[str], optional
            PUG-View headings to include in the text corpus.
        output_path : pathlib.Path, default="drug_report.md"
            Destination path for the report.
        include_mechanisms : bool, default=True
            Include ChEMBL mechanism-of-action entries.
        include_targets : bool, default=True
            Include UniProt accessions and gene symbols.

        Returns
        -------
        pathlib.Path
            Path to the written Markdown file.
        """

        ids = self.map_ids()
        props = self.fetch_pubchem_properties()
        text = self.fetch_pubchem_text(headings)

        mechanisms = self.fetch_chembl_mechanisms() if include_mechanisms else []
        accessions = self.target_accessions() if include_targets else []
        genes = self.target_gene_symbols() if include_targets else []

        lines: List[str] = []

        title = props.get("IUPACName")
        if not title:
            title = text.get("Drug and Medication Information", {}).get("RecordTitle") if text else None
        if not title:
            title = ids.get("chembl_id") or ids.get("pubchem_cid") or "Drug report"
        lines.append(f"# {title}\n")

        lines.append("## Identifiers")
        for key, value in ids.items():
            lines.append(f"- **{key}**: {value or 'N/A'}")
        lines.append("")

        lines.append("## PubChem Properties")
        if props:
            for k, v in props.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- (no properties fetched)")
        lines.append("")

        lines.append("## PubChem Text")
        if text:
            for heading, content in text.items():
                lines.append(f"### {heading}")
                strings = content.get("Strings", [])
                if not strings:
                    lines.append("(no text snippets)")
                else:
                    for s in strings:
                        lines.append(f"- {s}")
                lines.append("")
        else:
            lines.append("(no text fetched)")
            lines.append("")

        if include_mechanisms:
            lines.append("## ChEMBL Mechanisms")
            if mechanisms:
                for mech in mechanisms:
                    mid = mech.get("molecule_chembl_id")
                    moa = mech.get("mechanism_of_action") or mech.get("mechanism_comment")
                    target = mech.get("target_pref_name") or mech.get("target_chembl_id")
                    lines.append(f"- **{mid or 'N/A'}** -> {target or 'N/A'}: {moa or 'N/A'}")
            else:
                lines.append("(no mechanisms)")
            lines.append("")

        if include_targets:
            lines.append("## Targets")
            lines.append("### UniProt Accessions")
            if accessions:
                for acc in accessions:
                    lines.append(f"- {acc}")
            else:
                lines.append("- (none)")
            lines.append("")

            lines.append("### Gene Symbols")
            if genes:
                for g in genes:
                    lines.append(f"- {g}")
            else:
                lines.append("- (none)")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


__all__ = ["Drug", "list_pubchem_text_headings"]

