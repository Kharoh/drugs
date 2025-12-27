"""Shared constants for endpoints and curated PubChem heading sets."""

PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
UNICHEM_V2 = "https://www.ebi.ac.uk/unichem/api/v1"

# Curated heading sets – carried over from the previous experimental library
PUBCHEM_MEANING = [
    "Drug Indication",
    "Mechanism of Action",
    "Pharmacology and Biochemistry",
    "Therapeutic Uses",
    "Drug Classes",
    "FDA Pharmacological Classification",
    "FDA Pharm Classes",
    "WHO ATC Classification System",
    "ATC Code / KEGG: ATC",
]

PUBCHEM_MOA_TARGET = [
    "Mechanism of Action",
    "Chemical-Target Interactions",
    "Pharmacology and Biochemistry",
    "Pathways",
    "Interactions and Pathways",
    "IUPHAR/BPS Guide to PHARMACOLOGY Target Classification",
    "ChEMBL ID",
]

PUBCHEM_INTERACTION = [
    "Drug-Drug Interactions",
    "Drug-Food Interactions",
    "Interactions",
    "Interactions and Pathways",
    "Drug Warnings",
    "Adverse Effects",
]

PUBCHEM_SAFETY = [
    "Drug Warnings",
    "Adverse Effects",
    "Toxicity Summary",
    "Toxicological Information",
    "Toxicity",
    "Toxicity Data",
    "Drug Induced Liver Injury",
    "Hepatotoxicity",
    "LiverTox Summary",
    "Populations at Special Risk",
]

PUBCHEM_ADME_PK = [
    "Absorption, Distribution and Excretion",
    "Metabolism/Metabolites",
    "Metabolite Pathways",
    "Biological Half-Life",
    "Protein Binding",
    "Maximum Drug Dose",
]

PUBCHEM_MINIMAL_STABLE = [
    "Record Description",
    "Drug and Medication Information",
    "Names and Identifiers",
    "Synonyms",
]

__all__ = [
    "PUBCHEM_PUG_REST",
    "PUBCHEM_PUG_VIEW",
    "CHEMBL_API",
    "UNICHEM_V2",
    "PUBCHEM_MEANING",
    "PUBCHEM_MOA_TARGET",
    "PUBCHEM_INTERACTION",
    "PUBCHEM_SAFETY",
    "PUBCHEM_ADME_PK",
    "PUBCHEM_MINIMAL_STABLE",
]
