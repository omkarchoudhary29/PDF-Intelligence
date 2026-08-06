"""Enterprise Knowledge Navigator — Administrative CLI.

Commands:
- create-owner: Provision the initial enterprise workspace owner account safely.
- assign-owner: Assign an existing user as owner of unowned or specified workspaces.
"""

import argparse
import getpass
import secrets
import sys

from backend.database.connection import get_db_session
from backend.services.auth_service import auth_service


def cmd_generate_secret() -> None:
    """Generate a high-entropy secret key suitable for JWT signing."""
    new_secret = secrets.token_hex(32)
    print("\nGenerated Cryptographic Secret Key (64 characters / 256 bits):")
    print("-" * 64)
    print(new_secret)
    print("-" * 64)
    print("Set this in your .env or environment as:")
    print(f"JWT_SECRET_KEY={new_secret}\n")


def cmd_create_owner(args: argparse.Namespace) -> None:
    """Create the initial owner account."""
    name = args.name
    if not name:
        name = input("Enter Owner Full Name: ").strip()

    email = args.email
    if not email:
        email = input("Enter Owner Email Address: ").strip()

    password = args.password
    if not password:
        password = getpass.getpass("Enter Owner Password: ")
        confirm_password = getpass.getpass("Confirm Owner Password: ")
        if password != confirm_password:
            print("Error: Passwords do not match.", file=sys.stderr)
            sys.exit(1)

    assign_workspaces = args.assign_workspaces
    if not args.non_interactive and not assign_workspaces:
        ans = input("Assign all existing unowned workspaces to this owner? [y/N]: ").strip().lower()
        assign_workspaces = ans in ("y", "yes")

    with get_db_session() as db:
        try:
            owner_user, assigned_count = auth_service.create_owner_user(
                db=db,
                full_name=name,
                email=email,
                password=password,
                assign_existing_workspaces=assign_workspaces,
            )
            print("\nOwner account provisioned successfully!")
            print(f"User ID:  {owner_user.id}")
            print(f"Name:     {owner_user.full_name}")
            print(f"Email:    {owner_user.email}")
            if assigned_count > 0:
                print(f"Assigned: {assigned_count} existing workspace(s) as Owner.")
        except Exception as exc:
            print(f"Error provisioning owner: {exc}", file=sys.stderr)
            sys.exit(1)


def cmd_assign_owner(args: argparse.Namespace) -> None:
    """Assign an existing user as owner of specified or all workspaces."""
    email = args.email
    if not email:
        email = input("Enter User Email: ").strip()

    from backend.database.models import User, Workspace, WorkspaceMembership

    with get_db_session() as db:
        user = db.query(User).filter(User.email == email.lower().strip()).first()
        if not user:
            print(f"Error: User with email '{email}' was not found.", file=sys.stderr)
            sys.exit(1)

        ws_query = db.query(Workspace)
        if args.workspace_id:
            ws_query = ws_query.filter(Workspace.id == args.workspace_id)

        workspaces = ws_query.all()
        if not workspaces:
            print("No matching workspaces found.")
            return

        assigned = 0
        for ws in workspaces:
            mem = (
                db.query(WorkspaceMembership)
                .filter(
                    WorkspaceMembership.workspace_id == ws.id,
                    WorkspaceMembership.user_id == user.id,
                )
                .first()
            )
            if mem:
                mem.role = "owner"
                mem.membership_status = "active"
            else:
                mem = WorkspaceMembership(
                    workspace_id=ws.id,
                    user_id=user.id,
                    role="owner",
                    membership_status="active",
                )
                db.add(mem)
            assigned += 1

        db.commit()
        print(f"Successfully assigned '{user.email}' as Owner of {assigned} workspace(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="EKN Administrative CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-owner command
    p_create = subparsers.add_parser("create-owner", help="Provision the initial workspace owner")
    p_create.add_argument("--name", help="Owner full name")
    p_create.add_argument("--email", help="Owner email address")
    p_create.add_argument("--password", help="Owner password")
    p_create.add_argument("--assign-workspaces", action="store_true", help="Assign existing unowned workspaces")
    p_create.add_argument("--non-interactive", action="store_true", help="Do not prompt interactively")

    # assign-owner command
    p_assign = subparsers.add_parser("assign-owner", help="Assign an existing user as owner")
    p_assign.add_argument("--email", help="User email address")
    p_assign.add_argument("--workspace-id", help="Target workspace ID (default: all)")

    # generate-secret command
    subparsers.add_parser("generate-secret", help="Generate a secure random JWT secret key")

    args = parser.parse_args()
    if args.command == "create-owner":
        cmd_create_owner(args)
    elif args.command == "assign-owner":
        cmd_assign_owner(args)
    elif args.command == "generate-secret":
        cmd_generate_secret()


if __name__ == "__main__":
    main()
