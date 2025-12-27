"""RDKit-backed molecular helpers (SMILES, SELFIES, fingerprints, properties)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Import lazily to allow the module to be importable without heavy deps at runtime.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys, QED, rdMolDescriptors, Crippen
    try:
        from rdkit.Chem import SA_Score  # type: ignore
    except Exception:  # pragma: no cover - optional
        SA_Score = None
except Exception as exc:  # pragma: no cover - import-time guard
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore
    AllChem = None  # type: ignore
    Descriptors = None  # type: ignore
    Lipinski = None  # type: ignore
    MACCSkeys = None  # type: ignore
    QED = None  # type: ignore
    rdMolDescriptors = None  # type: ignore
    Crippen = None  # type: ignore
    SA_Score = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MolType = Any


class RDKitNotAvailable(RuntimeError):
    """Raised when RDKit-dependent functionality is requested but unavailable."""


class SELFIESNotAvailable(RuntimeError):
    """Raised when SELFIES dependency is missing."""


def require_rdkit() -> None:
    if _IMPORT_ERROR or Chem is None:
        raise RDKitNotAvailable(
            "RDKit is required for this operation. Install the optional dependency 'rdkit-pypi' via the 'chem' extra."
        ) from _IMPORT_ERROR


def mol_from_smiles(smiles: str, *, sanitize: bool = True) -> Optional[MolType]:
    """Parse a SMILES string into an RDKit molecule, handling errors gracefully.

    When RDKit is unavailable, returns the SMILES string so downstream helpers can
    fall back to hashed fingerprints and lightweight heuristics.
    """
    if not smiles:
        return None
    if Chem is None:
        return smiles  # type: ignore[return-value]
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        return mol
    except Exception:
        return smiles  # type: ignore[return-value]


def to_selfies(smiles: str) -> Optional[str]:
    """Convert SMILES to SELFIES, returning ``None`` when conversion fails."""
    try:
        import selfies  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise SELFIESNotAvailable(
            "SELFIES is required for this operation. Install the optional dependency 'selfies'."
        ) from exc

    if not smiles:
        return None
    try:
        return selfies.encoder(smiles)
    except Exception:
        return None


def _bitvect_to_array(bitvect: Any) -> np.ndarray:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def molecular_fingerprint(
    mol: MolType,
    *,
    method: str = "morgan",
    n_bits: int = 2048,
    radius: int = 2,
    use_features: bool = False,
) -> np.ndarray:
    """Generate a molecular fingerprint as a numpy array of bits.

    If RDKit is unavailable, a deterministic hashed fingerprint derived from SMILES
    is returned instead.
    """
    if mol is None:
        raise ValueError("A molecule (or SMILES string) is required to compute fingerprints.")
    method = method.lower()
    if Chem is None or DataStructs is None:
        return _hash_fingerprint(str(mol), n_bits=n_bits)

    if method == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=use_features)
    elif method in {"maccs", "maccskeys"}:
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif method in {"rdkit", "daylight"}:
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    else:
        raise ValueError(f"Unsupported fingerprint method: {method}")
    return _bitvect_to_array(fp)


def fingerprint_similarity(fp_a: np.ndarray, fp_b: np.ndarray, *, metric: str = "tanimoto") -> float:
    """Compute similarity between two bit vectors using common coefficients."""
    if fp_a.shape != fp_b.shape:
        # Allow different lengths by truncating to min length for robustness
        min_len = min(fp_a.shape[0], fp_b.shape[0])
        fp_a = fp_a[:min_len]
        fp_b = fp_b[:min_len]

    a = fp_a.astype(bool)
    b = fp_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    if metric.lower() in {"tanimoto", "jaccard"}:
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union else 0.0
    elif metric.lower() == "dice":
        denom = a.sum() + b.sum()
        return float((2 * inter) / denom) if denom else 0.0
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


def lipinski_violations(mol: MolType) -> List[str]:
    """Return a list of violated Lipinski rules for the molecule."""
    if mol is None:
        return ["invalid_molecule"]
    if Chem is None or Descriptors is None:
        return []

    violations: List[str] = []
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)

    if mw > 500:
        violations.append("mw>500")
    if logp > 5:
        violations.append("logp>5")
    if hbd > 5:
        violations.append("hbd>5")
    if hba > 10:
        violations.append("hba>10")
    if rotb > 10:
        violations.append("rotb>10")
    return violations


def synthetic_accessibility(mol: MolType) -> Optional[float]:
    """Estimate synthetic accessibility score (lower is easier)."""
    if mol is None:
        return None
    if Chem is None:
        return None
    if SA_Score:
        try:
            return float(SA_Score.sascorer.calculateScore(mol))  # type: ignore[attr-defined]
        except Exception:
            return None
    try:
        # Fallback heuristic: heavier molecules and more rings → higher SA
        rings = Chem.GetSSSR(mol)
        mw = Descriptors.MolWt(mol)
        return float(0.01 * mw + 0.1 * rings)
    except Exception:
        return None


def rdkit_properties(mol: MolType) -> Dict[str, Any]:
    """Compute common RDKit-derived molecular properties.

    When RDKit is missing, returns a placeholder structure with ``None`` values and
    Lipinski checks skipped. This allows downstream code to continue running while
    reporting the absence of cheminformatics features.
    """
    if mol is None:
        return {
            "logp": None,
            "tpsa": None,
            "qed": None,
            "sa_score": None,
            "lipinski_violations": ["invalid_molecule"],
            "lipinski_pass": False,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
        }

    if Chem is None or Descriptors is None:
        return {
            "logp": None,
            "tpsa": None,
            "qed": None,
            "sa_score": None,
            "lipinski_violations": [],
            "lipinski_pass": True,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
        }

    props = {
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": float(QED.qed(mol)),
        "sa_score": synthetic_accessibility(mol),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
    }
    props["lipinski_violations"] = lipinski_violations(mol)
    props["lipinski_pass"] = len(props["lipinski_violations"]) == 0
    return props


def _hash_fingerprint(text: str, *, n_bits: int = 2048) -> np.ndarray:
    """Fallback fingerprint using a SHA256 hash of input text."""
    import hashlib

    bit_count = max(16, n_bits)
    buf = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    repeats = (bit_count + len(buf) - 1) // len(buf)
    data = (buf * repeats)[:bit_count]
    arr = np.frombuffer(data, dtype=np.uint8)
    # Expand bytes to bits
    bit_arr = np.unpackbits(arr, bitorder="big")[:bit_count]
    return bit_arr.astype(np.int8)


__all__ = [
    "RDKitNotAvailable",
    "SELFIESNotAvailable",
    "mol_from_smiles",
    "to_selfies",
    "molecular_fingerprint",
    "fingerprint_similarity",
    "rdkit_properties",
    "lipinski_violations",
    "synthetic_accessibility",
    "require_rdkit",
]
