import json

import pytest
from gradio_client import Client

from popper_upgrade.tests.e2e.conftest import ADMIN_PW, DESIGNER_LOG, USER_PW

pytestmark = pytest.mark.e2e


def _statuses(client):
    table, _ = client.predict("all", None, None, api_name="/list_skills")
    return sorted(row[table["headers"].index("status")] for row in table["data"])


def test_client_lists_skills(live_app):
    client = Client(live_app.url, auth=("admin", ADMIN_PW), verbose=False)
    table, badge = client.predict("all", None, None, api_name="/list_skills")
    assert len(table["data"]) == 3
    assert badge == "Pending: 1 | Flagged: 0"
    pending, _ = client.predict("pending", None, None, api_name="/list_skills")
    assert [row[0] for row in pending["data"]] == [live_app.skills["pending"]]


def test_login_is_required(live_app):
    with pytest.raises(ValueError):
        Client(live_app.url, auth=("admin", "wrong-password-1"), verbose=False)
    with pytest.raises(ValueError):
        Client(live_app.url, auth=("nobody", ADMIN_PW), verbose=False)


def test_user_cannot_change_library_over_the_api(live_app):
    admin = Client(live_app.url, auth=("admin", ADMIN_PW), verbose=False)
    user = Client(live_app.url, auth=("alice", USER_PW), verbose=False)
    before = _statuses(admin)
    user.predict(live_app.skills["approved"], "reject", "nope", api_name="/set_library_approval")
    user.predict(live_app.skills["approved"], api_name="/delete_skill")
    assert _statuses(admin) == before
    assert admin.predict(live_app.skills["approved"], api_name="/skill_detail")["status"] == "approved"


def test_single_test_run_detail_charts_its_point(live_app):
    client = Client(live_app.url, auth=("alice", USER_PW), verbose=False)
    chart = client.predict(live_app.single_run_id, api_name="/run_detail")[3]
    points = [row for row in chart["data"] if row[2] == "e-value"]
    assert chart["mark"] == "point"
    assert [(row[0], round(row[1], 3)) for row in points] == [(1.0, 2.887)]
    assert {row[1] for row in chart["data"] if row[2] == "threshold"} == {10.0}
    assert len(chart["data"]) > 2


def test_run_detail_api(live_app):
    client = Client(live_app.url, auth=("alice", USER_PW), verbose=False)
    run, skills, reviews, chart, designer, *_ = client.predict(live_app.run_id, api_name="/run_detail")
    assert run["id"] == live_app.run_id
    assert [r["reason"] for r in reviews] == ["looks fine"]
    assert {"e-value", "threshold"} <= set(json.dumps(chart).replace('"', " ").split())
    assert DESIGNER_LOG in json.dumps(designer)
