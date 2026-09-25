"""The Mosquitto ACL enforces branch ownership (CLAUDE.md) and the `_sim` fence (ADR-0012)."""

import re
from pathlib import Path

import pytest
from paho.mqtt.client import topic_matches_sub

from common import uns
from common.uns import SimCommand, TopicClass
from tests.samples import UNIT

ROOT = Path(__file__).resolve().parents[1]


def load_acl() -> dict[str, list[tuple[str, str]]]:
    acl: dict[str, list[tuple[str, str]]] = {}
    user = None
    for raw in (ROOT / "mosquitto/acl").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split()
        if words[0] == "user":
            user = words[1]
            acl[user] = []
        elif words[0] == "topic":
            assert user, "topic before user"
            access, pattern = (words[1], words[2]) if len(words) == 3 else ("readwrite", words[1])
            acl[user].append((access, pattern))
        else:
            raise AssertionError(f"unexpected ACL line: {line}")
    return acl


ACL = load_acl()


def can(user: str, action: str, topic: str) -> bool:
    return any(
        (access == action or access == "readwrite") and topic_matches_sub(pattern, topic)
        for access, pattern in ACL[user]
    )


SERVICES = [
    "simulator",
    "edge-adapter",
    "historian",
    "graph-sync",
    "anomaly",
    "yield",
    "dashboard",
    "i3x",
]

OWNED = {
    "simulator": [
        uns.edge_raw("BR101", "BR101.AIC-102.PV"),
        uns.lab(UNIT, "titer"),
        uns.state_batch(UNIT),
        uns.state_operation(UNIT),
        uns.state_phase(UNIT, "PH_CTRL"),
        uns.events(UNIT, "batch"),
        uns.events(UNIT, "operator"),
        uns.sim_clock(),
        uns.sim_faults("BR-101"),
    ],
    "edge-adapter": [
        uns.pv(UNIT, "ph"),
        uns.sp(UNIT, "temperature"),
        uns.edge_unmapped(),
        uns.meta_tag("BR-101", TopicClass.PV, "ph"),
    ],
    "anomaly": [uns.ai_score(UNIT), uns.ai_alert(UNIT, "stats-co2_flow")],
    "yield": [uns.ai_prediction(UNIT), uns.ai_recommendation(UNIT)],
    "dashboard": [uns.sim_cmd(c) for c in SimCommand],
    "historian": [],
    "graph-sync": [],
    "i3x": [],
}
for _svc in SERVICES:
    OWNED[_svc].append(uns.meta_status(_svc))

ALL_OWNED = [(svc, t) for svc, topics in OWNED.items() for t in topics]


@pytest.mark.parametrize(("owner", "topic"), ALL_OWNED)
def test_exactly_one_writer_per_branch(owner, topic):
    writers = {u for u in ACL if can(u, "write", topic)}
    assert writers == {owner}


@pytest.mark.parametrize("user", ["anomaly", "yield", "i3x"])  # i3x feeds the LLM (ADR-0017)
@pytest.mark.parametrize(
    "topic", [uns.sim_faults("BR-101"), uns.sim_clock(), *[uns.sim_cmd(c) for c in SimCommand]]
)
def test_ai_services_cannot_see_sim(user, topic):
    assert not can(user, "read", topic)


@pytest.mark.parametrize(
    ("user", "topic"),
    [
        ("simulator", uns.sim_cmd(SimCommand.CLOCK)),
        ("simulator", uns.state_batch(UNIT)),  # monotonic clock start (ADR-0009)
        ("simulator", uns.sim_clock()),  # batch sequence survives a restart
        ("edge-adapter", uns.edge_raw("BR101", "BR101.AIC-102.PV")),
        ("edge-adapter", uns.state_batch(UNIT)),
        ("historian", uns.pv(UNIT, "ph")),
        ("historian", uns.sim_faults("BR-101")),
        ("graph-sync", uns.events(UNIT, "batch")),
        ("graph-sync", uns.sim_faults("BR-101")),
        ("anomaly", uns.pv(UNIT, "do")),
        ("yield", uns.lab(UNIT, "titer")),
        ("dashboard", uns.edge_unmapped()),
        ("dashboard", uns.sim_clock()),
        ("i3x", uns.pv(UNIT, "ph")),
        ("i3x", uns.ai_prediction(UNIT)),
    ],
)
def test_required_reads(user, topic):
    assert can(user, "read", topic)


def test_edge_adapter_cannot_read_state_beyond_batch():
    assert not can("edge-adapter", "read", uns.state_operation(UNIT))


def test_users_match_entrypoint_and_env_example():
    script = (ROOT / "mosquitto/entrypoint.sh").read_text()
    entry_users = set(re.search(r'USERS="([^"]+)"', script).group(1).split())
    env_users = {
        m.group(1).lower().replace("_", "-")
        for m in re.finditer(r"^MQTT_PASSWORD_(\w+)=", (ROOT / ".env.example").read_text(), re.M)
    }
    assert set(ACL) == entry_users == env_users
    assert set(SERVICES) <= set(ACL)
