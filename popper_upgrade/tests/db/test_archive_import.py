import io
import json
import tarfile
import uuid

import pytest

from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.skills.embedding import DEFAULT_EMBEDDING_MODEL
from popper_upgrade.tests.fakes import FakeEmbedder, run_config

pytestmark = pytest.mark.db


@pytest.fixture
def bundle(db, tmp_path):
    export_snapshot(db, tmp_path / "snapshot.json")
    snapshot = SnapshotSkillSource(tmp_path / "snapshot.json", FakeEmbedder(model_name=DEFAULT_EMBEDDING_MODEL))
    sink = BundleSink(tmp_path / "bundle", snapshot)
    experiment_id = str(uuid.uuid4())
    sink.create_experiment({"id": experiment_id, "name": "archived", "spec": {}, "status": "open"})
    run_id = sink.create_run(run_config(environment="cluster", experiment_id=experiment_id, task="t", replicate=0))
    trajectories = tmp_path / "bundle" / "trajectories"
    trajectories.mkdir()
    (trajectories / f"{run_id}.json").write_text(json.dumps({"tracked_stat": [0.1]}), encoding="utf-8")
    sink.finish_run(run_id, {"res": True, "res_stat": 3.0, "trajectory_path": f"/cluster/work/{run_id}.json"})
    return tmp_path / "bundle", run_id


def _pack(directory, archive, top=None):
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(directory, arcname=top if top is not None else ".")
    return archive


def _count_runs(db):
    return db.execute("select count(*) as n from runs.runs").fetchone()["n"]


@pytest.mark.parametrize("top", ["popper_results_demo_20261019T000000Z", None])
def test_archive_import_is_idempotent(db, tmp_path, bundle, top):
    directory, run_id = bundle
    archive = _pack(directory, tmp_path / "results.tar.gz", top)
    before = _count_runs(db)

    first = import_bundle(db, archive)
    second = import_bundle(db, archive)

    assert first["runs"] == 1
    assert second["runs"] == 0
    assert _count_runs(db) == before + 1
    assert db.execute("select status, environment from runs.runs where id = %s", (run_id,)).fetchone() == {
        "status": "completed",
        "environment": "cluster",
    }


def test_trajectory_is_copied_next_to_the_app_and_path_rewritten(db, tmp_path, bundle):
    directory, run_id = bundle
    archive = _pack(directory, tmp_path / "results.tar.gz", "top")
    target = tmp_path / "app_trajectories"

    import_bundle(db, archive, target)

    stored = db.execute("select trajectory_path from runs.runs where id = %s", (run_id,)).fetchone()
    assert stored["trajectory_path"] == str(target / f"{run_id}.json")
    assert json.loads((target / f"{run_id}.json").read_text(encoding="utf-8")) == {"tracked_stat": [0.1]}


def test_directory_import_still_works(db, tmp_path, bundle):
    directory, _ = bundle

    assert import_bundle(db, directory)["runs"] == 1


def _archive_with(tmp_path, entries):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            else:
                data = b"{}"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return archive


@pytest.mark.parametrize(
    "entries",
    [
        [("top/manifest.json", "file"), ("../escaped.txt", "file")],
        [("top/manifest.json", "file"), ("/tmp/absolute.txt", "file")],
        [("top/manifest.json", "file"), ("top/link", "symlink")],
    ],
)
def test_unsafe_archive_entries_are_rejected_before_extraction(db, tmp_path, entries):
    archive = _archive_with(tmp_path, entries)
    before = _count_runs(db)

    with pytest.raises(ValueError, match="unsafe entry"):
        import_bundle(db, archive)

    assert not (tmp_path.parent / "escaped.txt").exists()
    assert _count_runs(db) == before


def test_archive_without_a_bundle_is_rejected(db, tmp_path):
    archive = _archive_with(tmp_path, [("top/readme.txt", "file")])

    with pytest.raises(ValueError, match="does not contain a bundle"):
        import_bundle(db, archive)
