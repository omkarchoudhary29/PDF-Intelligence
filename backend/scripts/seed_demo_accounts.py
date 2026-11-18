"""Idempotent Development Seeding Script for EKN Demonstration Accounts.

Provisions two dedicated interview demonstration accounts in the existing
Aerospace & Engineering workspace:
1. Administrator: admin.demo@example.com (role: admin)
2. Employee: employee.demo@example.com (role: employee)

Credentials are generated securely using Argon2id password hashing. Plaintext
credentials are saved strictly in the git-ignored local .env file.
"""

from pathlib import Path
import re
import secrets
import string
from typing import Any, Dict, Optional, Tuple

from backend.core.security import hash_password, verify_password
from backend.database.connection import get_db_session
from backend.database.models import User, Workspace, WorkspaceMembership
from backend.services.auth_service import auth_service
from config.settings import get_settings


def generate_secure_password(length: int = 16) -> str:
    """Generate a strong enterprise-compliant password (upper, lower, digits, symbols)."""
    upper = string.ascii_uppercase
    lower = string.ascii_lowercase
    digits = string.digits
    symbols = "!@#$%^&*()-_=+"

    # Guarantee at least 2 of each required class
    chars = [
        secrets.choice(upper),
        secrets.choice(upper),
        secrets.choice(lower),
        secrets.choice(lower),
        secrets.choice(digits),
        secrets.choice(digits),
        secrets.choice(symbols),
        secrets.choice(symbols),
    ]
    all_chars = upper + lower + digits + symbols
    chars += [secrets.choice(all_chars) for _ in range(max(0, length - len(chars)))]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def update_env_variable(env_path: Path, key: str, value: str) -> None:
    """Safely append or update a key=value line in the local .env file."""
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    content = env_path.read_text(encoding="utf-8")
    pattern = rf"^{key}=.*$"
    if re.search(pattern, content, flags=re.MULTILINE):
        new_content = re.sub(pattern, f"{key}={value}", content, flags=re.MULTILINE)
    else:
        new_content = content.rstrip() + f"\n{key}={value}\n"
    env_path.write_text(new_content, encoding="utf-8")


def seed_demo_accounts() -> Dict[str, Any]:
    """Idempotently seed the demonstration administrator and employee accounts."""
    settings = get_settings()
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    admin_email = "admin.demo@example.com"
    emp_email = "employee.demo@example.com"

    # Read existing passwords from settings or generate new strong ones
    admin_pw = (settings.demo_admin_password or "").strip()
    emp_pw = (settings.demo_employee_password or "").strip()

    if not admin_pw:
        admin_pw = generate_secure_password(16)
        update_env_variable(env_path, "DEMO_ADMIN_PASSWORD", admin_pw)

    if not emp_pw:
        emp_pw = generate_secure_password(16)
        update_env_variable(env_path, "DEMO_EMPLOYEE_PASSWORD", emp_pw)

    with get_db_session() as db:
        # Locate or create a dedicated, isolated public demonstration workspace.
        # Never fall back to an arbitrary active customer workspace: demo credentials
        # are public by design and must not grant access to a real tenant's documents.
        demo_ws = None
        if settings.demo_workspace_id:
            demo_ws = db.query(Workspace).filter(Workspace.id == settings.demo_workspace_id).first()
        if not demo_ws:
            demo_ws = db.query(Workspace).filter(Workspace.slug == "ekn-public-demo").first()
        if not demo_ws:
            workspace_values = {
                "name": "EKN Public Demo",
                "slug": "ekn-public-demo",
                "description": "Isolated public evaluation workspace. Do not store confidential documents here.",
                "is_active": True,
            }
            if settings.demo_workspace_id:
                workspace_values["id"] = settings.demo_workspace_id
            demo_ws = Workspace(**workspace_values)
            db.add(demo_ws)
            db.flush()

        # --- 1. Demo Administrator ---
        existing_admin = db.query(User).filter(User.email == admin_email).first()
        if existing_admin:
            # Ensure it is configured as demo account
            if not verify_password(admin_pw, existing_admin.password_hash):
                existing_admin.password_hash = hash_password(admin_pw)
            existing_admin.is_active = True
            existing_admin.must_change_password = False
            existing_admin.locked_until = None
            existing_admin.failed_login_attempts = 0
            admin_user = existing_admin
        else:
            admin_user = User(
                full_name="Demo Administrator",
                email=admin_email,
                password_hash=hash_password(admin_pw),
                is_active=True,
                must_change_password=False,
            )
            db.add(admin_user)
            db.flush()

        # Public demo identities must belong only to the isolated demo workspace.
        # Remove stale memberships left by any earlier broad fallback seeding.
        for stale_membership in (
            db.query(WorkspaceMembership)
            .filter(WorkspaceMembership.user_id == admin_user.id)
            .all()
        ):
            if stale_membership.workspace_id != demo_ws.id:
                db.delete(stale_membership)

        # Ensure active admin/owner membership in demo workspace
        admin_membership = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.user_id == admin_user.id,
                WorkspaceMembership.workspace_id == demo_ws.id,
            )
            .first()
        )
        if not admin_membership:
            admin_membership = WorkspaceMembership(
                user_id=admin_user.id,
                workspace_id=demo_ws.id,
                role="admin",
                membership_status="active",
            )
            db.add(admin_membership)
        else:
            if admin_membership.role not in ("owner", "admin"):
                admin_membership.role = "admin"
            admin_membership.membership_status = "active"

        # --- 2. Demo Employee ---
        existing_emp = db.query(User).filter(User.email == emp_email).first()
        if existing_emp:
            if not verify_password(emp_pw, existing_emp.password_hash):
                existing_emp.password_hash = hash_password(emp_pw)
            existing_emp.is_active = True
            existing_emp.must_change_password = False
            existing_emp.locked_until = None
            existing_emp.failed_login_attempts = 0
            emp_user = existing_emp
        else:
            emp_user = User(
                full_name="Demo Employee",
                email=emp_email,
                password_hash=hash_password(emp_pw),
                is_active=True,
                must_change_password=False,
            )
            db.add(emp_user)
            db.flush()

        # Keep the public employee demo identity out of customer workspaces.
        for stale_membership in (
            db.query(WorkspaceMembership)
            .filter(WorkspaceMembership.user_id == emp_user.id)
            .all()
        ):
            if stale_membership.workspace_id != demo_ws.id:
                db.delete(stale_membership)

        # Ensure active employee membership in demo workspace
        emp_membership = (
            db.query(WorkspaceMembership)
            .filter(
                WorkspaceMembership.user_id == emp_user.id,
                WorkspaceMembership.workspace_id == demo_ws.id,
            )
            .first()
        )
        if not emp_membership:
            emp_membership = WorkspaceMembership(
                user_id=emp_user.id,
                workspace_id=demo_ws.id,
                role="employee",
                membership_status="active",
            )
            db.add(emp_membership)
        else:
            emp_membership.role = "employee"
            emp_membership.membership_status = "active"

        db.commit()

        # Verify authentication through auth_service
        _, admin_token = auth_service.authenticate_user(
            db=db,
            email=admin_email,
            password=admin_pw,
            portal="admin",
        )
        assert admin_token.access_token, "Admin login verification failed"

        _, emp_token = auth_service.authenticate_user(
            db=db,
            email=emp_email,
            password=emp_pw,
            portal="employee",
        )
        assert emp_token.access_token, "Employee login verification failed"

        return {
            "workspace_id": demo_ws.id,
            "workspace_name": demo_ws.name,
            "admin_email": admin_email,
            "admin_password": admin_pw,
            "employee_email": emp_email,
            "employee_password": emp_pw,
            "admin_token_type": admin_token.token_type,
            "employee_token_type": emp_token.token_type,
        }


if __name__ == "__main__":
    result = seed_demo_accounts()
    print("Demo accounts seeded successfully:")
    print(f"Workspace: {result['workspace_name']} ({result['workspace_id']})")
    print(f"Administrator: {result['admin_email']} | password configured in Railway variables")
    print(f"Employee: {result['employee_email']} | password configured in Railway variables")
