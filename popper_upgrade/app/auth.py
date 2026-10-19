import uuid
from dataclasses import dataclass

import bcrypt

from popper_upgrade.db.audit import log_event

MIN_PASSWORD_LENGTH = 10
BCRYPT_COST = 12
ROLES = ("user", "admin")


@dataclass(frozen=True)
class User:
    id: str
    username: str
    role: str


def hash_password(pw):
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(BCRYPT_COST)).decode("ascii")


def verify_password(pw, hashed):
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("ascii"))
    except ValueError:
        return False


def require_admin(user):
    if user is None or user.role != "admin":
        raise PermissionError("admin role required")


def _check_role(role):
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")


def _row_to_user(row):
    return User(row["id"], row["username"], row["role"])


def user_by_name(conn, username):
    row = conn.execute(
        "select id::text as id, username, role from auth.users where username = %s", (username,)
    ).fetchone()
    return _row_to_user(row) if row else None


def user_by_id(conn, user_id):
    row = conn.execute("select id::text as id, username, role from auth.users where id = %s", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def list_users(conn):
    return conn.execute(
        "select id::text as id, username, role, active, created_at from auth.users order by created_at, username"
    ).fetchall()


def create_user(conn, username, password, role, actor):
    if actor is not None:
        require_admin(actor)
    _check_role(role)
    username = username.strip()
    if not username:
        raise ValueError("username is required")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if user_by_name(conn, username) is not None:
        raise ValueError(f"username {username} is taken")
    user_id = str(uuid.uuid4())
    conn.execute(
        "insert into auth.users (id, username, password_hash, role) values (%s, %s, %s, %s)",
        (user_id, username, hash_password(password), role),
    )
    log_event(
        conn,
        actor.id if actor else None,
        "user.create",
        "user",
        user_id,
        after={"username": username, "role": role},
    )
    return User(user_id, username, role)


def _current(conn, user_id):
    row = conn.execute("select role, active from auth.users where id = %s", (user_id,)).fetchone()
    if row is None:
        raise KeyError(user_id)
    return row


def set_active(conn, user_id, active, actor):
    require_admin(actor)
    before = _current(conn, user_id)
    conn.execute("update auth.users set active = %s where id = %s", (active, user_id))
    log_event(
        conn,
        actor.id,
        "user.set_active",
        "user",
        user_id,
        before={"active": before["active"]},
        after={"active": active},
    )


def set_role(conn, user_id, role, actor):
    require_admin(actor)
    _check_role(role)
    before = _current(conn, user_id)
    conn.execute("update auth.users set role = %s where id = %s", (role, user_id))
    log_event(conn, actor.id, "user.set_role", "user", user_id, before={"role": before["role"]}, after={"role": role})


def authenticate(conn, username, password):
    row = conn.execute(
        "select id::text as id, username, role, password_hash, active from auth.users where username = %s",
        (username,),
    ).fetchone()
    if row is None or not row["active"] or not verify_password(password, row["password_hash"]):
        return None
    return _row_to_user(row)
