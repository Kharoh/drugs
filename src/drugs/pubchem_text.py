"""Helpers for fetching and extracting text sections from PubChem PUG-View."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .constants import PUBCHEM_PUG_VIEW
from .utils import dedupe_preserve_order, get_json


def _walk_extract_strings(obj: Any, max_chars: int = 20_000) -> List[str]:
    """Recursively collect human-readable strings from a PUG-View tree.

    Parameters
    ----------
    obj : Any
        JSON-like object returned by PUG-View.
    max_chars : int, default=20000
        Skip strings longer than this threshold to avoid oversized chunks.

    Returns
    -------
    list[str]
        Deduplicated list of readable string fragments.
    """

    out: List[str] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            string_val = x.get("String")
            if isinstance(string_val, str) and len(string_val) <= max_chars:
                out.append(string_val)
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(obj)
    return dedupe_preserve_order(out)


def pubchem_pug_view_index(cid: int) -> Dict[str, Any]:
    """Fetch the PUG-View index JSON for a compound.

    Parameters
    ----------
    cid : int
        PubChem compound identifier.

    Returns
    -------
    dict
        Index payload listing available headings for the compound.
    """

    url = f"{PUBCHEM_PUG_VIEW}/data/compound/{cid}/JSON"
    return get_json(url, params={"response_type": "index"})


def pubchem_pug_view_heading(cid: int, heading: str) -> Dict[str, Any]:
    """Fetch a single heading from the PUG-View record.

    Parameters
    ----------
    cid : int
        PubChem compound identifier.
    heading : str
        Heading label to retrieve (must exist for the compound).

    Returns
    -------
    dict
        PUG-View payload scoped to the requested heading.
    """

    url = f"{PUBCHEM_PUG_VIEW}/data/compound/{cid}/JSON"
    return get_json(url, params={"heading": heading, "response_type": "display"})


def _collect_toc_headings(pug_view_json: Any) -> List[str]:
    """Collect ``TOCHeading`` labels from a PUG-View JSON tree.

    Parameters
    ----------
    pug_view_json : Any
        JSON-like PUG-View structure.

    Returns
    -------
    list[str]
        Unique heading labels in first-seen order.
    """

    headings: List[str] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            toc_heading = x.get("TOCHeading")
            if isinstance(toc_heading, str) and toc_heading.strip():
                headings.append(toc_heading.strip())
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(pug_view_json)
    return dedupe_preserve_order(headings)


def pubchem_text_snippets(cid: int, headings: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch text snippets for selected PUG-View headings.

    Invalid headings for the specific compound are skipped silently to avoid
    PUG-View errors.

    Parameters
    ----------
    cid : int
        PubChem compound identifier.
    headings : Iterable[str]
        Heading labels to fetch for the compound.

    Returns
    -------
    dict[str, dict]
        Mapping of heading -> payload containing the record title, section tree, and
        list of readable string snippets.
    """

    index_json = pubchem_pug_view_index(cid)
    available = set(_collect_toc_headings(index_json))

    results: Dict[str, Dict[str, Any]] = {}
    for heading in headings:
        if heading not in available:
            # Skip invalid headings to avoid PUG-View errors for this CID.
            continue
        data = pubchem_pug_view_heading(cid, heading)
        record = data.get("Record", {}) if isinstance(data, dict) else {}
        results[heading] = {
            "RecordTitle": record.get("RecordTitle"),
            "Section": record.get("Section", []),
            "Strings": _walk_extract_strings(record),
        }
    return results


def list_pubchem_text_headings(cid: int) -> List[str]:
    """List PUG-View headings available for a compound.

    Parameters
    ----------
    cid : int
        PubChem compound identifier.

    Returns
    -------
    list[str]
        Unique heading labels in first-seen order.
    """

    idx = pubchem_pug_view_index(cid)
    return list(_collect_toc_headings(idx))


__all__ = [
    "pubchem_text_snippets",
    "list_pubchem_text_headings",
    "pubchem_pug_view_index",
    "pubchem_pug_view_heading",
]
