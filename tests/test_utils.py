from drugs.utils import dedupe_preserve_order

def test_dedupe_preserve_order():
	data = ["a", "b", "a", "c", "b"]
	assert dedupe_preserve_order(data) == ["a", "b", "c"]
