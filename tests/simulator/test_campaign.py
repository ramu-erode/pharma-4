import itertools
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from common.models import Campaign, FaultType
from simulator import campaign, recipes
from simulator.recipes import LEVER_NAMES

END = datetime(2026, 9, 23, 13, 0, tzinfo=UTC)
PLAN = campaign.plan(200, END, seed=42)


def test_counts_and_ids():
    specs = PLAN.specs
    assert len(specs) == 200
    assert [int(s.batch_id.split("-")[1]) for s in specs] == list(range(200))
    assert all(s.batch_id == f"B{s.start.year}-{i:04d}" for i, s in enumerate(specs))
    assert specs[-1].batch_id.startswith("B2026-") and specs[0].batch_id.startswith("B2022-")
    assert len(PLAN.by_campaign(Campaign.PC)) == 50


def test_start_order_no_overlap_and_ends_before_first_boot():
    specs = PLAN.specs
    assert [s.start for s in specs] == sorted(s.start for s in specs)
    for cell in campaign.CELLS:
        mine = [s for s in specs if s.cell == cell]
        for a, b in itertools.pairwise(mine):
            assert b.start - a.start >= timedelta(days=14.5)
    last_end = max(s.start for s in specs) + timedelta(days=14.75)
    assert last_end < END - timedelta(hours=12)


def test_recipe_versions_follow_the_calendar():
    for s in PLAN.specs:
        assert s.recipe_id == recipes.current(s.start.date()).id
    assert {s.recipe_id for s in PLAN.specs} == {"v1", "v2", "v3"}


def test_levers_stay_inside_pars():
    for s in PLAN.specs:
        r = recipes.get(s.recipe_id)
        assert r.clip(s.levers) == s.levers


@pytest.mark.parametrize("lever", LEVER_NAMES)
def test_pc_design_covers_every_stratum(lever):
    pc = PLAN.by_campaign(Campaign.PC)
    lo, hi = recipes.get("v3").par[lever]
    u = np.array([(getattr(s.levers, lever) - lo) / (hi - lo) for s in pc])
    strata = np.floor(u * 10).astype(int)
    assert set(strata) == set(range(10))  # 50 points spread over all ten deciles


def test_manufacturing_levers_are_near_nominal():
    for s in PLAN.by_campaign(Campaign.MFG):
        nominal = recipes.get(s.recipe_id).nominal
        assert abs(s.levers.shift_day - nominal.shift_day) < 1.0


def test_fault_mix():
    faulty = [s for s in PLAN.specs if s.faults]
    assert len(faulty) == 30
    kinds = Counter(s.faults[0].kind for s in faulty)
    assert set(kinds) == set(FaultType)
    assert kinds[FaultType.CONTAMINATION] <= 3
    for s in faulty:
        f = s.faults[0]
        assert 2.0 <= f.onset_day <= 11.0
        if f.kind is FaultType.STUCK_SENSOR:
            assert f.params["tag"] in campaign.STUCK_TAGS


def test_plan_is_deterministic():
    assert campaign.plan(200, END, seed=42) == PLAN
    assert campaign.plan(200, END, seed=43) != PLAN


def test_small_plans_work():
    p = campaign.plan(5, END, seed=1)
    assert len(p.specs) == 5
