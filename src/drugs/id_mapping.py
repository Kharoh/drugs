"""Identifier mapping helpers across UniChem, PubChem, and ChEMBL."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import requests

from .constants import PUBCHEM_PUG_REST, UNICHEM_V2
from .utils import get_json


def _get_json_allow_404(url: str, *, timeout: float = 30.0) -> Optional[dict]:
    """HTTP GET helper that returns ``None`` on 404 responses.

    Parameters
    ----------
    url : str
        Endpoint to request.
    timeout : float, default=30.0
        Request timeout in seconds.

    Returns
    -------
    dict or None
        Parsed JSON body, or ``None`` when the server returns 404.

    Raises
    ------
    HTTPError
        If the status code is non-404 and not successful.
    RuntimeError
        If the response is not JSON.
    """
    response = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    ct = (response.headers.get("content-type") or "").lower()
    if "json" not in ct:
        raise RuntimeError(f"Expected JSON from {url}, got content-type={ct}: {response.text[:200]}")
    return response.json()


@lru_cache(maxsize=1)
def unichem_sources() -> List[Dict[str, Any]]:
    """Retrieve and cache available UniChem sources.

    Returns
    -------
    list[dict]
        List of UniChem source metadata dictionaries. Empty if the API responds
        with an unexpected shape.
    """
    data = _get_json_allow_404(f"{UNICHEM_V2}/sources/") or {}
    if isinstance(data, dict):
        return data.get("sources") or []
    if isinstance(data, list):
        return data
    return []


@lru_cache(maxsize=256)
def unichem_src_id(name: str) -> int:
    """Resolve a UniChem source ID by fuzzy name matching.

    Parameters
    ----------
    name : str
        Canonical or alternative name for a UniChem source (case-insensitive).

    Returns
    -------
    int
        Matching UniChem source identifier.

    Raises
    ------
    ValueError
        If no source matches the provided name.
    """
    target = name.strip().lower()
    for src in unichem_sources():
        sid = src.get("sourceID") or src.get("sourceId") or src.get("src_id") or src.get("srcId")
        candidates = {
            str(src.get("name", "")).lower(),
            str(src.get("name_long", "")).lower(),
            str(src.get("nameLong", "")).lower(),
            str(src.get("name_label", "")).lower(),
            str(src.get("nameLabel", "")).lower(),
            str(src.get("longName", "")).lower(),
            str(src.get("shortName", "")).lower(),
        }
        if sid is not None and target in candidates:
            return int(sid)
    raise ValueError(f"Could not find UniChem source ID for '{name}'")


def _map_ids(
    from_id: str,
    from_source: str,
    to_source: str,
    *,
    include_obsolete: bool = False,
    timeout: float = 30.0,
) -> List[str]:
    """Translate identifiers between UniChem sources.

    Parameters
    ----------
    from_id : str
        Identifier to translate.
    from_source : str
        Source system name for ``from_id`` (e.g., ``"ChEMBL"``).
    to_source : str
        Destination source system name (e.g., ``"PubChem"``).
    include_obsolete : bool, default=False
        Whether to include obsolete mappings when provided by the API.
    timeout : float, default=30.0
        Request timeout in seconds.

    Returns
    -------
    list[str]
        Identifiers mapped to the destination source. Duplicates are removed but
        ordering is preserved based on API response.
    """
    from_sid = unichem_src_id(from_source)
    to_sid = unichem_src_id(to_source)

    url = f"{UNICHEM_V2}/compounds"
    payload = {"compound": from_id, "sourceID": from_sid, "type": "sourceID"}
    response = requests.post(
        url,
        json=payload,
        timeout=timeout,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    data: Any = response.json()

    compounds: Any = []
    if isinstance(data, dict):
        compounds = data.get("compounds") or data.get("results") or data.get("data") or []
    elif isinstance(data, list):
        compounds = data

    out: List[str] = []
    if isinstance(compounds, list):
        for comp in compounds:
            if not isinstance(comp, dict):
                continue
            sources = comp.get("sources") or []
            if not isinstance(sources, list):
                continue
            for src_entry in sources:
                if not isinstance(src_entry, dict):
                    continue
                sid = src_entry.get("id") or src_entry.get("sourceID") or src_entry.get("sourceId")
                if sid is None or int(sid) != to_sid:
                    continue
                cid = src_entry.get("compoundId") or src_entry.get("src_compound_id") or src_entry.get("compound")
                if cid:
                    out.append(str(cid))

    uniq: List[str] = []
    seen = set()
    for val in out:
        if val not in seen:
            uniq.append(val)
            seen.add(val)
    return uniq


def chembl_to_pubchem_cid(
    chembl_id: str,
    *,
    first: bool = True,
    include_obsolete: bool = False,
) -> Optional[Union[int, List[int]]]:
    """Map a ChEMBL ID to one or more PubChem CIDs via UniChem.

    Parameters
    ----------
    chembl_id : str
        ChEMBL molecule identifier to translate.
    first : bool, default=True
        Return only the first match when ``True``; otherwise return all.
    include_obsolete : bool, default=False
        Include obsolete mappings when available.

    Returns
    -------
    int or list[int] or None
        First CID (default), a list of CIDs, or ``None`` if no mapping is found.
    """
    hits = _map_ids(
        chembl_id,
        from_source="ChEMBL",
        to_source="PubChem",
        include_obsolete=include_obsolete,
    )
    cids = [int(h) for h in hits if str(h).isdigit()]
    if not cids:
        return None
    return cids[0] if first else cids


def pubchem_cid_to_chembl(
    cid: Union[int, str],
    *,
    first: bool = True,
    include_obsolete: bool = False,
) -> Optional[Union[str, List[str]]]:
    """Map a PubChem CID to one or more ChEMBL IDs via UniChem.

    Parameters
    ----------
    cid : int or str
        PubChem compound identifier.
    first : bool, default=True
        Return only the first match when ``True``; otherwise return all.
    include_obsolete : bool, default=False
        Include obsolete mappings when available.

    Returns
    -------
    str or list[str] or None
        First ChEMBL ID (default), list of IDs, or ``None`` if unmapped.
    """
    cid_str = str(cid)
    hits = _map_ids(
        cid_str,
        from_source="PubChem",
        to_source="ChEMBL",
        include_obsolete=include_obsolete,
    )
    chembl_ids = [h for h in hits if str(h).upper().startswith("CHEMBL")]
    if not chembl_ids:
        return None
    return chembl_ids[0] if first else chembl_ids


def pubchem_cid_to_inchikey(cid: Union[int, str]) -> Optional[str]:
    """Fetch the InChIKey for a PubChem CID using PUG-REST.

    Parameters
    ----------
    cid : int or str
        PubChem compound identifier.

    Returns
    -------
    str or None
        InChIKey string when available, otherwise ``None``.
    """
    url = f"{PUBCHEM_PUG_REST}/compound/cid/{cid}/property/InChIKey/JSON"
    data = get_json(url)
    props = data.get("PropertyTable", {}).get("Properties", [])
    if props:
        return props[0].get("InChIKey")
    return None


def pubchem_cid_from_inchikey(inchikey: str, *, first: bool = True) -> Optional[Union[int, List[int]]]:
    """Resolve PubChem CIDs from an InChIKey.

    Parameters
    ----------
    inchikey : str
        InChIKey to search for.
    first : bool, default=True
        Return only the first match when ``True``; otherwise return all.

    Returns
    -------
    int or list[int] or None
        First CID (default), list of CIDs, or ``None`` if no match.
    """
    url = f"{PUBCHEM_PUG_REST}/compound/inchikey/{inchikey}/cids/JSON"
    data = get_json(url)
    cids = data.get("IdentifierList", {}).get("CID", []) if data else []
    if not cids:
        return None
    return cids[0] if first else cids


__all__ = [
    "chembl_to_pubchem_cid",
    "pubchem_cid_from_inchikey",
    "pubchem_cid_to_chembl",
    "pubchem_cid_to_inchikey",
    "unichem_sources",
    "unichem_src_id",
]
