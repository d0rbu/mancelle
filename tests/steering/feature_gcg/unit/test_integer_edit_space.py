from __future__ import annotations

import random

from tests.steering.feature_gcg.unit.integer_edit_space import IntegerEditSpace, IntGroup


def test_integer_edit_space_round_trips_a_step():
    space = IntegerEditSpace()
    group = IntGroup(key_str="g0", members=[5, 7], is_control=False)
    validated = space.validate(group)
    assert validated.key_str == "g0"
    assert not space.is_trivial(validated)
    target = space.init_target(validated)               # starts at the first member
    assert isinstance(space.digest(target), str)
    proposals = space.rank_candidates(agg_grad=None, current=target, cfg=None)
    cands = space.sample_candidates(
        proposals, target, batch_size=4, temperature=0.1,
        rng=random.Random(0), visited={space.digest(target)},
    )
    assert all(isinstance(c, int) for c in cands)
    # model_item just pairs (member, candidate) so the core's objective_batch sees both.
    assert space.model_item(member=5, target=3) == (5, 3)
