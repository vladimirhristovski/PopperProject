import pytest

from popper_upgrade.app.auth import (
    authenticate,
    create_user,
    hash_password,
    set_active,
    set_role,
    user_by_name,
    verify_password,
)

pytestmark = pytest.mark.db


def _audit_actions(db, target_id):
    rows = db.execute("select action, user_id::text as user_id from audit.events where target_id = %s", (target_id,))
    return rows.fetchall()


def test_hash_uses_cost_12():
    hashed = hash_password("a-long-enough-password")
    assert hashed.startswith("$2b$12$")
    assert verify_password("a-long-enough-password", hashed)
    assert not verify_password("another-password-1", hashed)


def test_authenticate_ok_and_wrong_password(db, plain_user):
    assert authenticate(db, "alice", "alice-password-1") == plain_user
    assert authenticate(db, "alice", "alice-password-2") is None
    assert authenticate(db, "nobody", "alice-password-1") is None


def test_inactive_user_cannot_login(db, admin_user, plain_user):
    set_active(db, plain_user.id, False, admin_user)
    assert authenticate(db, "alice", "alice-password-1") is None
    set_active(db, plain_user.id, True, admin_user)
    assert authenticate(db, "alice", "alice-password-1") == plain_user


def test_short_password_rejected(db):
    with pytest.raises(ValueError):
        create_user(db, "bob", "short", "user", None)
    assert user_by_name(db, "bob") is None


def test_overlong_password_rejected(db):
    with pytest.raises(ValueError):
        create_user(db, "bob", "x" * 73, "user", None)
    assert user_by_name(db, "bob") is None


def test_duplicate_username_and_unknown_role_rejected(db, plain_user):
    with pytest.raises(ValueError):
        create_user(db, "alice", "another-password-1", "user", None)
    with pytest.raises(ValueError):
        create_user(db, "carol", "carol-password-1", "root", None)


def test_user_cannot_create_users(db, plain_user):
    with pytest.raises(PermissionError):
        create_user(db, "carol", "carol-password-1", "user", plain_user)
    assert user_by_name(db, "carol") is None


def test_role_change_requires_admin_and_is_audited(db, admin_user, plain_user):
    with pytest.raises(PermissionError):
        set_role(db, plain_user.id, "admin", plain_user)
    assert user_by_name(db, "alice").role == "user"
    assert _audit_actions(db, plain_user.id) == [{"action": "user.create", "user_id": None}]

    set_role(db, plain_user.id, "admin", admin_user)
    assert user_by_name(db, "alice").role == "admin"
    events = db.execute(
        "select user_id::text as user_id, before, after from audit.events where action = 'user.set_role' "
        "and target_id = %s",
        (plain_user.id,),
    ).fetchall()
    assert events == [{"user_id": admin_user.id, "before": {"role": "user"}, "after": {"role": "admin"}}]
