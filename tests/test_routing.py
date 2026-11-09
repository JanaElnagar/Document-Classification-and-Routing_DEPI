from docrouter.routing import Router

ROUTER = Router(routes={"Invoice": "Finance", "Contract": "Legal"}, threshold=0.7)


def test_confident_prediction_is_routed():
    r = ROUTER.route({"Invoice": 0.9, "Contract": 0.1})
    assert r["category"] == "Invoice" and r["department"] == "Finance" and not r["needs_review"]


def test_low_confidence_goes_to_review():
    r = ROUTER.route({"Invoice": 0.55, "Contract": 0.45})
    assert r["needs_review"] and r["department"] == "Manual Review"


def test_top_k_sorted():
    r = ROUTER.route({"Invoice": 0.2, "Contract": 0.8})
    assert [t["label"] for t in r["top_k"]] == ["Contract", "Invoice"]


def test_config_file_loads():
    router = Router.from_yaml("configs/routing.yaml")
    assert len(router.routes) == 6 and 0 < router.threshold < 1
