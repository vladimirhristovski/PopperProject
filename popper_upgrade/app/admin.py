import argparse
import getpass
import sys

from popper_upgrade.app.auth import create_user
from popper_upgrade.db.connection import connect
from popper_upgrade.db.migrate import migrate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.app.admin")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user")
    create.add_argument("--username", required=True)
    create.add_argument("--role", choices=("user", "admin"), default="user")
    args = parser.parse_args(argv)
    password = getpass.getpass("Password: ")
    with connect() as conn:
        migrate(conn)
        try:
            user = create_user(conn, args.username, password, args.role, None)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    print(f"created {user.role} {user.username}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
