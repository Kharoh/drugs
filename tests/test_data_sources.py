import pytest

from drugs import constants
from drugs.data_sources import (
    chembl_bioactivities,
    chembl_mechanisms,
    pubchem_properties,
    rxnav_interactions,
)
from drugs.utils import get_json


class _FakeResponse:
    def __init__(self, url, expected_url, params=None, expected_params=None, json_data=None, status_code=200, content_type="application/json"):
        self.url = url
        self._json_data = json_data or {}
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = ""
        self._expected_url = expected_url
        self._params = params or {}
        self._expected_params = expected_params or {}

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"status {self.status_code}")

    def json(self):
        assert self.url == self._expected_url
        if self._expected_params:
            assert self._params == self._expected_params
        return self._json_data


def test_pubchem_properties_requests_correct_url(monkeypatch):
    cid = 2244
    expected_url = f"{constants.PUBCHEM_PUG_REST}/compound/cid/{cid}/property/IUPACName,InChIKey,CanonicalSMILES,MolecularFormula,MolecularWeight,XLogP,TPSA/JSON"

    def fake_get(url, params=None, timeout=30.0, headers=None):
        return _FakeResponse(
            url,
            expected_url,
            params=params,
            expected_params=None,
            json_data={"PropertyTable": {"Properties": [{"IUPACName": "Aspirin", "CanonicalSMILES": "CCO"}]},},
        )

    monkeypatch.setattr("drugs.utils.requests.get", fake_get)
    props = pubchem_properties(cid)
    assert props["IUPACName"] == "Aspirin"
    assert "CanonicalSMILES" in props


def test_chembl_mechanisms_requests_correct_url(monkeypatch):
    chembl_id = "CHEMBL25"
    expected_url = f"{constants.CHEMBL_API}/mechanism.json"
    expected_params = {"molecule_chembl_id": chembl_id, "limit": 50}

    def fake_get(url, params=None, timeout=30.0, headers=None):
        return _FakeResponse(
            url,
            expected_url,
            params=params,
            expected_params=expected_params,
            json_data={"mechanisms": [{"molecule_chembl_id": chembl_id, "target_pref_name": "Target"}]},
        )

    monkeypatch.setattr("drugs.utils.requests.get", fake_get)
    mech = chembl_mechanisms(chembl_id)
    assert mech[0]["molecule_chembl_id"] == chembl_id
    assert mech[0]["target_pref_name"] == "Target"


def test_chembl_bioactivities_requests_correct_params(monkeypatch):
    chembl_id = "CHEMBL25"
    expected_url = f"{constants.CHEMBL_API}/activity.json"
    expected_params = {
        "molecule_chembl_id": chembl_id,
        "limit": 10,
        "pchembl_value__gte": 6.0,
        "assay_type__in": "B,F",
    }

    def fake_get(url, params=None, timeout=30.0, headers=None):
        return _FakeResponse(
            url,
            expected_url,
            params=params,
            expected_params=expected_params,
            json_data={"activities": [{"activity_id": 1, "pchembl_value": 7.1, "assay_type": "B"}]},
        )

    monkeypatch.setattr("drugs.utils.requests.get", fake_get)
    rows = chembl_bioactivities(chembl_id, min_pchembl=6.0, assay_types=("B", "F"), limit=10)
    assert rows[0]["activity_id"] == 1
    assert rows[0]["pchembl_value"] >= 6.0


def test_rxnav_interactions_requests_correct_url(monkeypatch):
    name = "aspirin"
    expected_url = "https://rxnav.nlm.nih.gov/REST/interaction/interaction.json"
    expected_params = {"iname": name}

    def fake_get(url, params=None, timeout=30.0, headers=None):
        return _FakeResponse(
            url,
            expected_url,
            params=params,
            expected_params=expected_params,
            json_data={"interactionTypeGroup": [{"interactionType": [{"sourceName": "Micromedex", "interactionPair": []}]}]},
        )

    monkeypatch.setattr("drugs.utils.requests.get", fake_get)
    data = rxnav_interactions(name)
    assert "interactionTypeGroup" in data
    assert data["interactionTypeGroup"][0]["interactionType"][0]["sourceName"] == "Micromedex"


def test_get_json_raises_on_non_json(monkeypatch):
    class FakeBadResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html>not json</html>"

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    def fake_get(url, params=None, timeout=30.0, headers=None):
        return FakeBadResponse()

    monkeypatch.setattr("drugs.utils.requests.get", fake_get)
    with pytest.raises(RuntimeError):
        get_json("http://example.com")
