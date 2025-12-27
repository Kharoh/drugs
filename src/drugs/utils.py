"""Utility helpers for HTTP requests and small collection helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, TypeVar

import requests

T = TypeVar("T")


def get_json(url: str, *, params: Optional[Dict[str, object]] = None, timeout: float = 30.0) -> dict:
	"""Fetch JSON content from an HTTP endpoint.

	Parameters
	----------
	url : str
		Endpoint to request.
	params : dict, optional
		Query parameters forwarded to ``requests.get``.
	timeout : float, default=30.0
		Request timeout in seconds.

	Returns
	-------
	dict
		Parsed JSON body from the response.

	Raises
	------
	HTTPError
		If the response status is not 2xx.
	RuntimeError
		If the content type is not JSON.
	"""

	response = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
	response.raise_for_status()
	content_type = (response.headers.get("content-type") or "").lower()
	if "json" not in content_type:
		snippet = response.text[:200]
		raise RuntimeError(f"Expected JSON from {url}, got content-type={content_type}: {snippet}")
	return response.json()


def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
	"""Remove duplicate values while preserving input order.

	Parameters
	----------
	items : Iterable[T]
		Items to deduplicate.

	Returns
	-------
	list[T]
		List of unique items in their first-seen order.
	"""

	seen = set()
	result: List[T] = []
	for item in items:
		if item in seen:
			continue
		seen.add(item)
		result.append(item)
	return result

