from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from popper_upgrade.tools.fetch_base import CUTOFF, choose_revision


def _commit(commit_id, *stamp):
    return SimpleNamespace(commit_id=commit_id, created_at=datetime(*stamp, tzinfo=timezone.utc))


COMMITS = [
    _commit("newest", 2026, 10, 1),
    _commit("overwritten", 2026, 9, 20, 8),
    _commit("proven", 2026, 9, 18, 18, 59, 59),
    _commit("older", 2026, 9, 10),
]


def test_auto_picks_latest_commit_before_cutoff():
    assert choose_revision(COMMITS) == "proven"


def test_cutoff_is_exclusive():
    commits = [SimpleNamespace(commit_id="at", created_at=CUTOFF), COMMITS[-1]]

    assert choose_revision(commits) == "older"


def test_explicit_revision_wins():
    assert choose_revision(COMMITS, "abc123") == "abc123"


def test_no_eligible_commit_is_an_error():
    with pytest.raises(ValueError, match="older than"):
        choose_revision(COMMITS[:2])
