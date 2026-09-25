"""The ground-truth fence holds for i3X and the assistant (ADR-0012, ADR-0016, ADR-0017).

Whatever i3X serves, an LLM may see. So the façade must never read ground truth, and
the assistant must have no way to the plant except i3X.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ("FaultInjection", "HAS_INJECTION", "fault_labels", "_sim/", "SUB_SIM")
STORE_CLIENTS = {"neo4j", "psycopg", "paho", "graph", "historian", "common.mqtt", "simulator"}


def sources(package: str) -> list[Path]:
    return sorted((ROOT / package).glob("*.py"))


@pytest.mark.parametrize("path", sources("i3x") + sources("assistant"), ids=lambda p: p.name)
def test_no_ground_truth_names(path):
    text = path.read_text()
    assert not [word for word in GROUND_TRUTH if word in text]


@pytest.mark.parametrize("path", sources("assistant"), ids=lambda p: p.name)
def test_assistant_imports_no_store_or_broker_client(path):
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    bad = {i for i in imported for s in STORE_CLIENTS if i == s or i.startswith(s + ".")}
    assert not bad
