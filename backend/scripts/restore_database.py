"""Enterprise Knowledge Navigator — Database Restore and Verification Utility.

Performs:
1. Validates that the backup source file exists and passes SQLite integrity check.
2. Restores backup file into target database location safely.
3. Verifies schema integrity and row count consistency.
"""

import argparse
import logging
import os
from pathlib import Path
import sqlite3
import sys

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ekn.restore")


def restore_database(backup_path: str, target_db_path: str, force: bool = False) -> bool:
    """Restore database from backup file into target location with integrity checks."""
    backup_file = Path(backup_path).resolve()
    target_file = Path(target_db_path).resolve()

    if not backup_file.exists():
        raise FileNotFoundError(f"Backup file not found at: {backup_file}")

    # 1. Verify backup file integrity before restoring
    logger.info("Pre-flight: Checking backup file integrity: %s", backup_file)
    bck_conn = sqlite3.connect(str(backup_file))
    bck_cur = bck_conn.cursor()
    check = bck_cur.execute("PRAGMA integrity_check").fetchone()[0]
    if check != "ok":
        bck_conn.close()
        raise RuntimeError(f"Backup file failed integrity check: {check}")

    user_count = bck_cur.execute("SELECT count(1) FROM users").fetchone()[0]
    doc_count = bck_cur.execute("SELECT count(1) FROM documents").fetchone()[0]
    bck_conn.close()
    logger.info("Backup pre-check passed (Users: %d, Documents: %d)", user_count, doc_count)

    # 2. Check target path safety
    if target_file.exists() and not force:
        raise FileExistsError(
            f"Target database already exists at {target_file}. Pass --force to overwrite."
        )

    target_file.parent.mkdir(parents=True, exist_ok=True)

    # 3. Restore using SQLite backup API to ensure atomicity
    logger.info("Restoring backup to target: %s", target_file)
    src_conn = sqlite3.connect(str(backup_file))
    dst_conn = sqlite3.connect(str(target_file))
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()

    # 4. Post-restore integrity verification
    logger.info("Post-flight: Verifying restored target database...")
    verify_conn = sqlite3.connect(str(target_file))
    v_cur = verify_conn.cursor()
    v_check = v_cur.execute("PRAGMA integrity_check").fetchone()[0]
    v_users = v_cur.execute("SELECT count(1) FROM users").fetchone()[0]
    v_docs = v_cur.execute("SELECT count(1) FROM documents").fetchone()[0]
    verify_conn.close()

    if v_check != "ok" or v_users != user_count or v_docs != doc_count:
        raise RuntimeError(
            f"Restoration verification mismatch: check={v_check}, users={v_users}/{user_count}, docs={v_docs}/{doc_count}"
        )

    logger.info("SUCCESS: Database restored and verified at: %s", target_file)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore EKN database from verified backup")
    parser.add_argument("--backup", type=str, required=True, help="Path to backup .db file")
    parser.add_argument("--target", type=str, required=True, help="Path to target destination .db file")
    parser.add_argument("--force", action="store_true", help="Overwrite target if it exists")
    args = parser.parse_args()

    restore_database(args.backup, args.target, force=args.force)


if __name__ == "__main__":
    main()
