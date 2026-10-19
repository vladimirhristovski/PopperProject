import re

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

from popper_upgrade.tests.e2e.conftest import (  # noqa: E402
    ADMIN_PW,
    APPROVED_TEXT,
    DESIGNER_LOG,
    USER_PW,
)

pytestmark = pytest.mark.e2e

expect.set_options(timeout=20000)


def _login(page, live_app, username, password):
    page.goto(live_app.url)
    page.get_by_label("Username").fill(username)
    page.get_by_label("Password").fill(password)
    page.get_by_role("button", name="Login").click()
    expect(page.get_by_role("tab", name="Skills")).to_be_visible()


def _select_skill(page, skill_id):
    page.get_by_label("Skill ID", exact=True).fill(skill_id)


def test_user_login_and_flag_skill(page, live_app):
    _login(page, live_app, "alice", USER_PW)
    _select_skill(page, live_app.skills["pending"])
    page.get_by_label("Flag reason").fill("unclear applicability")
    page.get_by_role("button", name="Flag skill").click()
    expect(page.get_by_text("Skill flagged")).to_be_visible()
    page.get_by_role("button", name="Refresh skills").click()
    expect(page.locator("#skills_badge")).to_contain_text("Flagged: 1")


def test_admin_rejects_skill_and_it_leaves_retrieval_preview(page, live_app):
    approved_id = live_app.skills["approved"]
    _login(page, live_app, "admin", ADMIN_PW)
    page.get_by_role("button", name="Retrieval preview").click()
    page.get_by_label("Hypothesis", exact=True).fill(APPROVED_TEXT)
    page.get_by_role("button", name="Preview retrieval").click()
    expect(page.locator("#retrieval_preview")).to_contain_text(approved_id)

    _select_skill(page, approved_id)
    page.get_by_label("reject", exact=True).check()
    page.get_by_label("Decision reason").fill("too generic")
    page.get_by_role("button", name="Record decision").click()
    expect(page.get_by_text("Decision recorded")).to_be_visible()

    page.get_by_role("button", name="Preview retrieval").click()
    expect(page.locator("#retrieval_preview")).not_to_contain_text(approved_id)

    page.get_by_role("button", name="Restore").click()
    expect(page.get_by_text("Skill restored")).to_be_visible()
    page.get_by_role("button", name="Preview retrieval").click()
    expect(page.locator("#retrieval_preview")).to_contain_text(approved_id)


def test_user_sees_no_admin_tab(page, live_app):
    _login(page, live_app, "alice", USER_PW)
    expect(page.get_by_role("tab", name="Run detail")).to_be_visible()
    expect(page.get_by_role("tab", name="Admin")).to_have_count(0)


def test_admin_sees_admin_tab(page, live_app):
    _login(page, live_app, "admin", ADMIN_PW)
    expect(page.get_by_role("tab", name="Admin")).to_have_count(1)


def test_run_detail_renders_chart_and_logs(page, live_app):
    _login(page, live_app, "alice", USER_PW)
    page.get_by_role("tab", name="Run detail").click()
    page.get_by_label("Run ID").fill(live_app.run_id)
    page.get_by_role("button", name="Load run").click()
    expect(page.locator("#evalue_chart svg, #evalue_chart canvas").first).to_be_visible()
    expect(page.get_by_text(DESIGNER_LOG)).to_be_visible()
    expect(page.get_by_text("executor ran code")).to_be_visible()
    expect(page.get_by_text("round 1 attempt 1: approved - looks fine")).to_be_visible()


def test_curation_override_flow(page, live_app):
    _login(page, live_app, "alice", USER_PW)
    page.get_by_role("tab", name="Curation").click()
    expect(page.locator("#curation_hypotheses input:checked")).to_have_count(5)
    page.get_by_role("button", name="Run curator").click()
    expect(page.locator("#curation_token").get_by_role("textbox")).to_have_value(re.compile(r".+"))
    flips = page.locator("#curation_flips")
    expect(flips).to_contain_text("approved")
    flips.get_by_label(APPROVED_TEXT, exact=False).check()
    page.get_by_role("button", name="Apply overrides").click()
    expect(page.locator("#curation_applied")).to_contain_text("Overrides applied: 1")


def _archive(live_app, tmp_path):
    import json
    import tarfile

    from popper_upgrade.db.bundle import BundleSink
    from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
    from popper_upgrade.tests.fakes import FakeEmbedder, run_config

    snapshot_path = tmp_path / "snapshot.json"
    with live_app.ctx.conn() as conn:
        export_snapshot(conn, snapshot_path)
    snapshot = SnapshotSkillSource(snapshot_path, FakeEmbedder())
    bundle = tmp_path / "bundle"
    sink = BundleSink(bundle, snapshot)
    run_id = sink.create_run(run_config(environment="cluster", hypothesis="archived upload"))
    (bundle / "trajectories").mkdir()
    (bundle / "trajectories" / f"{run_id}.json").write_text(json.dumps({"tracked_stat": [0.1]}), encoding="utf-8")
    sink.finish_run(run_id, {"res": True, "res_stat": 2.0, "trajectory_path": f"/cluster/{run_id}.json"})
    archive = tmp_path / "popper_results_demo_20261019T000000Z.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=archive.name.removesuffix(".tar.gz"))
    return archive, run_id


def test_admin_uploads_a_run_archive(page, live_app, tmp_path):
    archive, run_id = _archive(live_app, tmp_path)
    _login(page, live_app, "admin", ADMIN_PW)
    page.get_by_role("tab", name="Admin").click()
    page.get_by_text("Import run bundle").click()
    page.locator("#archive_file input[type=file]").set_input_files(str(archive))
    page.get_by_role("button", name="Import archive").click()
    expect(page.get_by_text("design_reviews")).to_be_visible()
    with live_app.ctx.conn() as conn:
        row = conn.execute("select hypothesis, environment from runs.runs where id = %s", (run_id,)).fetchone()
    assert row == {"hypothesis": "archived upload", "environment": "cluster"}
