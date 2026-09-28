from t7_routing.costs import routing_decision_cost


def test_all_arms_have_cost_entries() -> None:
    for arm in ("lr", "lr_goal", "hr1", "hr2", "pg", "pg_inferred"):
        cost = routing_decision_cost(arm)
        assert "C_route_definition" in cost
