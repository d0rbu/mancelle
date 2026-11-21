"""Independent regression for the typed soft-prompt receipt boundary."""

from __future__ import annotations

import json

import pytest

from steering.eval.axbench_adv.errors import AxBenchAdvError
from tests.axbench_adv.unit.test_soft_prompt_receipts import (
    _receipt,
    _rewrite,
    _validate,
)
from tests.axbench_adv.unit.test_soft_prompt_receipts import (
    receipt_tree as receipt_tree,
)


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize(
    ("field", "value"),
    [("foreach", 0), ("fused", 0), ("amsgrad", 0), ("weight_decay", False)],
)
def test_receipt_only_optimizer_type_alias_is_rejected_after_rehash(
    receipt_tree, count: int, field: str, value: object
) -> None:
    root, manifest = receipt_tree
    _validate(root, manifest)
    document = json.loads(_receipt(root, count).read_text())
    document["effective_config"]["optimizer"][field] = value
    _rewrite(root, manifest, document, count)
    with pytest.raises(AxBenchAdvError, match="receipt identity"):
        _validate(root, manifest)
