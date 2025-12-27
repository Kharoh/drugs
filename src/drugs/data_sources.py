"""Thin wrappers over PubChem and ChEMBL endpoints."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .constants import CHEMBL_API, PUBCHEM_PUG_REST, PUBCHEM_PUG_VIEW
from .pubchem_text import list_pubchem_text_headings, pubchem_text_snippets
from .utils import get_json


def pubchem_properties(
    cid: int,
    properties: Iterable[str] = (
        "IUPACName",
        "InChIKey",
        "CanonicalSMILES",
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
    ),
) -> Dict[str, Any]:
    """Fetch core compound properties from PubChem PUG-REST.

    Parameters
    ----------
    cid : int
        PubChem compound identifier.
    properties : Iterable[str], optional
        Property names to request. Defaults to a curated set including IUPAC name,
        InChIKey, SMILES, formula, and physicochemical descriptors.

    Returns
    -------
    dict
        First property record returned by PUG-REST, or an empty dict when missing.
    """

    prop_str = ",".join(properties)
    url = f"{PUBCHEM_PUG_REST}/compound/cid/{cid}/property/{prop_str}/JSON"
    data = get_json(url)
    props = data.get("PropertyTable", {}).get("Properties", [])
    return props[0] if props else {}


def pubchem_pug_view_record(cid: int) -> Dict[str, Any]:
    """Fetch the full PUG-View record for a compound.

    Parameters
    ----------
    cid : int
        PubChem compound identifier.

    Returns
    -------
    dict
        Complete PUG-View record payload in display mode.
    """

    url = f"{PUBCHEM_PUG_VIEW}/data/compound/{cid}/JSON"
    return get_json(url, params={"response_type": "display"})


def chembl_molecules_by_inchikey(inchikey: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Query ChEMBL for molecules by InChIKey.

    Parameters
    ----------
    inchikey : str
        Standard InChIKey to search for.
    limit : int, default=10
        Maximum number of molecules to return.

    Returns
    -------
    list[dict]
        Sequence of molecule records returned by the ChEMBL API.
    """

    url = f"{CHEMBL_API}/molecule.json"
    data = get_json(url, params={"molecule_structures__standard_inchi_key": inchikey, "limit": limit})
    return data.get("molecules", []) or data.get("objects", []) or []


def chembl_mechanisms(molecule_chembl_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch mechanisms of action for a ChEMBL molecule.

    Parameters
    ----------
    molecule_chembl_id : str
        ChEMBL molecule identifier.
    limit : int, default=50
        Maximum number of mechanism records to fetch.

    Returns
    -------
    list[dict]
        Mechanism-of-action entries returned by ChEMBL.
    """

    url = f"{CHEMBL_API}/mechanism.json"
    data = get_json(url, params={"molecule_chembl_id": molecule_chembl_id, "limit": limit})
    return data.get("mechanisms", []) or data.get("objects", []) or []


def chembl_target_detail(target_chembl_id: str) -> Dict[str, Any]:
    """Fetch target details from ChEMBL, including components.

    Parameters
    ----------
    target_chembl_id : str
        ChEMBL target identifier.

    Returns
    -------
    dict
        Target detail payload containing components, accessions, and synonyms.
    """

    url = f"{CHEMBL_API}/target/{target_chembl_id}.json"
    return get_json(url)


__all__ = [
    "pubchem_properties",
    "pubchem_pug_view_record",
    "pubchem_text_snippets",
    "list_pubchem_text_headings",
    "chembl_molecules_by_inchikey",
    "chembl_mechanisms",
    "chembl_target_detail",
]
