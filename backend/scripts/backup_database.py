"""Enterprise Knowledge Navigator — Automated Database and Storage Backup Utility.

Performs:
1. Safe WAL checkpoint on SQLite database if applicable.
2. Online consistent backup using SQLite backup API.
3. Verification of backup file integrity via PRAGMA integrity_check.
4. Optional backup metadata manifest with table row counts.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from typing import Optional

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ekn.backup")


def backup_database(destination_dir: Optional[str] = None) -> Path:
    """Create a verified, consistent online backup of the relational database."""
    settings = get_settings()

    if settings.is_mysql_configured:
        logger.warning("Automated backup script is designed for SQLite file storage. For MySQL, use mysqldump.")
        raise NotImplementedError("Use mysqldump for external MySQL installations.")

    db_path = settings.database_url.replace("sqlite:///", "")
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Source database not found at: {db_path}")

    # Resolve backup destination
    dest_dir = Path(destination_dir) if destination_dir else Path(settings.storage_local_root).parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = dest_dir / f"ekn_backup_{timestamp}.db"
    manifest_file = dest_dir / f"ekn_backup_{timestamp}_manifest.json"

    logger.info("Opening source database: %s", db_path)
    src_conn = sqlite3.connect(db_path)

    # 1. Force WAL checkpoint
    try:
        cur = src_conn.cursor()
        wal_result = cur.execute("PRAGMA wal_checkpoint(FULL)").fetchall()
        logger.info("WAL checkpoint completed: %s", wal_result)
    except Exception as exc:
        logger.warning("WAL checkpoint notice: %s", exc)

    # 2. Perform online backup using SQLite streaming backup API
    logger.info("Streaming online backup to: %s", backup_file)
    dst_conn = sqlite3.connect(str(backup_file))
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()

    # 3. Verify backup file integrity
    logger.info("Verifying backup integrity...")
    verify_conn = sqlite3.connect(str(backup_file))
    v_cur = verify_conn.cursor()
    integrity_result = v_cur.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity_result != "ok":
        verify_conn.close()
        raise RuntimeError(f"Backup failed integrity verification: {integrity_result}")

    # 4. Generate manifest with table counts
    tables = [r[0] for r in v_cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    table_counts = {}
    for tbl in tables:
        count = v_cur.execute(f'SELECT count(1) FROM "{tbl}"').fetchone()[0]
        table_counts[tbl] = count

    verify_conn.close()

    manifest_data = {
        "timestamp_utc": timestamp,
        "source_database": db_path,
        "backup_file": str(backup_file),
        "file_size_bytes": backup_file.stat().st_size,
        "integrity": integrity_result,
        "table_counts": table_counts,
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    logger.info("Backup successfully completed and verified: %s (%d bytes)", backup_file, backup_file.stat().st_size)
    logger.info("Manifest written: %s", manifest_file)
    return backup_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Create verified backup of EKN database")
    parser.add_argument("--dest", type=str, default=None, help="Target backup directory")
    args = parser.parse_args()
    backup_database(destination_dir=args.dest)


if __name__ == "__main__":
    main()
