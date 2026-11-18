from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import chromadb
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine.url import make_url, URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.database.base import Base
import backend.database.models  # Ensure models are registered on Base.metadata
from config.settings import get_settings

logger = logging.getLogger("ekn.database")

# Singletons for connection pools and clients
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None
_chroma_client: Optional[chromadb.ClientAPI] = None


def normalize_database_url(raw_url: str) -> Tuple[URL, str]:
    """Parse and normalize database connection URL for SQLAlchemy with PyMySQL.

    Uses SQLAlchemy's make_url parser rather than string replacement to preserve
    percent-encoded credentials, host parameters, and connection options safely.

    Returns:
        (parsed_url_object, safe_log_string_with_hidden_password)
    """
    url_obj = make_url(raw_url)
    driver = url_obj.drivername.lower()

    if driver in ("mysql", "mysql+mysqldb"):
        url_obj = url_obj.set(drivername="mysql+pymysql")
    elif driver in ("mariadb", "mariadb+mysqldb"):
        url_obj = url_obj.set(drivername="mariadb+pymysql")

    safe_log = url_obj.render_as_string(hide_password=True)
    return url_obj, safe_log


def get_engine() -> Optional[Engine]:
    """Retrieve or create the SQLAlchemy database engine.

    Supports MySQL when configured, or SQLite if configured via DATABASE_URL.
    Returns None if no database connection URL is configured.
    """
    global _engine
    if _engine is not None:
        return _engine

    settings = get_settings()
    url = settings.effective_database_url

    if not url:
        logger.info("Database URL is not configured. Relational database is offline.")
        return None

    try:
        url_obj, safe_log_url = normalize_database_url(url)

        if url_obj.drivername.startswith("sqlite"):
            _engine = create_engine(
                url_obj,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            _engine = create_engine(
                url_obj,
                pool_size=settings.mysql_pool_size,
                max_overflow=settings.mysql_max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"connect_timeout": settings.mysql_connect_timeout},
            )
        logger.info("SQLAlchemy database engine initialized successfully for: %s", safe_log_url)
        try:
            Base.metadata.create_all(bind=_engine)
        except Exception as table_err:
            logger.warning("Could not auto-create tables on engine initialization: %s", table_err)
        return _engine
    except Exception as exc:
        logger.error("Failed to initialize database engine: %s", exc)
        return None


def get_session_factory() -> Optional[sessionmaker]:
    """Retrieve or initialize the SQLAlchemy sessionmaker."""
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    engine = get_engine()
    if engine is None:
        return None

    _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding an isolated database session.

    Raises RuntimeError if the relational database is not configured.
    """
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("Relational database is not configured or engine failed to initialize.")

    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for standalone scripts, workers, and CLI commands."""
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("Relational database is not configured or engine failed to initialize.")

    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


def ensure_schema_upgrades(engine: Engine) -> None:
    """Non-destructively inspect tables and add missing columns for backward compatibility."""
    try:
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())

        with engine.begin() as conn:
            # Upgrade 'documents' table if columns from later phases are missing
            if "documents" in table_names:
                doc_columns = {col["name"] for col in inspector.get_columns("documents")}
                column_specs = [
                    ("family_id", "VARCHAR(36) NULL"),
                    ("version_number", "INTEGER DEFAULT 1"),
                    ("previous_version_id", "VARCHAR(36) NULL"),
                    ("version_status", "VARCHAR(32) DEFAULT 'current'"),
                    ("approval_status", "VARCHAR(32) DEFAULT 'unverified'"),
                    ("issued_at", "DATETIME NULL"),
                    ("effective_from", "DATETIME NULL"),
                    ("effective_until", "DATETIME NULL"),
                    # Phase 3 Ownership & Privacy Scoping
                    ("visibility_scope", "VARCHAR(50) DEFAULT 'company'"),
                    ("owner_user_id", "VARCHAR(36) NULL"),
                    ("content_hash", "VARCHAR(64) NULL"),
                ]
                for col_name, col_type in column_specs:
                    if col_name not in doc_columns:
                        logger.info("Upgrading schema: Adding column '%s' to 'documents' table", col_name)
                        try:
                            conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}"))
                        except Exception as alter_err:
                            logger.debug("Column addition note (%s): %s", col_name, alter_err)

                # Backfill historical documents to company scope and preserve raw content hash
                try:
                    conn.execute(text("UPDATE documents SET visibility_scope = 'company' WHERE visibility_scope IS NULL OR visibility_scope = ''"))
                    conn.execute(text("UPDATE documents SET content_hash = sha256_hash WHERE content_hash IS NULL"))
                except Exception as bfill_err:
                    logger.debug("Document backfill note: %s", bfill_err)

            # Upgrade 'conversations' table if knowledge_scope is missing
            if "conversations" in table_names:
                conv_columns = {col["name"] for col in inspector.get_columns("conversations")}
                if "knowledge_scope" not in conv_columns:
                    logger.info("Upgrading schema: Adding column 'knowledge_scope' to 'conversations' table")
                    try:
                        conn.execute(text("ALTER TABLE conversations ADD COLUMN knowledge_scope VARCHAR(50) DEFAULT 'company'"))
                    except Exception as alter_err:
                        logger.debug("Column addition note (knowledge_scope): %s", alter_err)
                try:
                    conn.execute(text("UPDATE conversations SET knowledge_scope = 'company' WHERE knowledge_scope IS NULL OR knowledge_scope = ''"))
                except Exception as bfill_err:
                    logger.debug("Conversation backfill note: %s", bfill_err)

            # Upgrade 'workspaces' table if owner_id is missing
            if "workspaces" in table_names:
                ws_columns = {col["name"] for col in inspector.get_columns("workspaces")}
                if "owner_id" not in ws_columns:
                    logger.info("Upgrading schema: Adding column 'owner_id' to 'workspaces' table")
                    try:
                        conn.execute(text("ALTER TABLE workspaces ADD COLUMN owner_id VARCHAR(36) NULL"))
                    except Exception as alter_err:
                        logger.debug("Column addition note (owner_id): %s", alter_err)

            # Upgrade 'entities' table if aliases_json or attributes_json are missing (Phase 7)
            if "entities" in table_names:
                ent_columns = {col["name"] for col in inspector.get_columns("entities")}
                ent_specs = [
                    ("aliases_json", "JSON NULL"),
                    ("attributes_json", "JSON NULL"),
                ]
                for col_name, col_type in ent_specs:
                    if col_name not in ent_columns:
                        logger.info("Upgrading schema: Adding column '%s' to 'entities' table", col_name)
                        try:
                            conn.execute(text(f"ALTER TABLE entities ADD COLUMN {col_name} {col_type}"))
                        except Exception as alter_err:
                            logger.debug("Column addition note (%s): %s", col_name, alter_err)

    except Exception as exc:
        logger.warning("Controlled schema upgrade check encountered note: %s", exc)


def init_db(engine: Optional[Engine] = None) -> bool:
    """Create all registered database tables in the configured database engine and run additive migrations."""
    target_engine = engine or get_engine()
    if target_engine is None:
        logger.warning("Cannot initialize tables: No database engine configured.")
        return False

    try:
        Base.metadata.create_all(bind=target_engine)
        ensure_schema_upgrades(target_engine)
        logger.info("Database tables initialized and schema verified successfully.")
        return True
    except Exception as exc:
        logger.error("Failed to create database tables: %s", exc)
        raise


def set_test_engine(engine: Optional[Engine]) -> None:
    """Testing hook to inject an isolated in-memory or temporary database engine."""
    global _engine, _session_factory
    _engine = engine
    if engine is not None:
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    else:
        _session_factory = None


def reset_db_state() -> None:
    """Reset singletons for clean test teardown."""
    global _engine, _session_factory, _chroma_client
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _chroma_client = None


def check_database_connection() -> Tuple[bool, str]:
    """Check connectivity to the configured database by executing a lightweight SELECT 1 query.

    Returns:
        (is_connected, status_message)
    """
    settings = get_settings()
    if not settings.is_mysql_configured and not settings.effective_database_url:
        return False, "Not configured (MYSQL_URL is unset)"

    engine = get_engine()
    if engine is None:
        return False, "Database engine initialization failed"

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected"
    except SQLAlchemyError as err:
        logger.warning("Database connectivity check failed: %s", err)
        return False, f"Connection failed: {str(err).splitlines()[0]}"
    except Exception as exc:
        logger.warning("Unexpected error checking database connectivity: %s", exc)
        return False, f"Connection error: {str(exc).splitlines()[0]}"


def get_chroma_client() -> chromadb.ClientAPI:
    """Retrieve or initialize the ChromaDB client.

    If chroma_host is configured, connects via HttpClient (client-server topology).
    In production (ENVIRONMENT=production), remote Chroma is strictly required.
    In development/testing, falls back to local PersistentClient.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    settings = get_settings()

    # 1. Client-server HTTP mode (preferred for multi-process / Railway topology)
    if settings.chroma_host and settings.chroma_host.strip():
        host = settings.chroma_host.strip()
        port = settings.chroma_port or 8000
        ssl = settings.chroma_ssl
        try:
            _chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
            )
            logger.info("ChromaDB HttpClient connected to %s:%s (ssl=%s)", host, port, ssl)
            return _chroma_client
        except Exception as exc:
            logger.error("Failed to connect to ChromaDB HttpClient at %s:%s: %s", host, port, exc)
            raise RuntimeError(f"ChromaDB connection to {host}:{port} failed: {exc}") from exc

    # Enforce remote ChromaDB configuration in production environment
    if settings.environment == "production":
        error_msg = (
            "ChromaDB configuration error: ENVIRONMENT=production requires a remote "
            "ChromaDB service (CHROMA_HOST and CHROMA_PORT must be configured). "
            "Silent fallback to embedded local persistence is prohibited in production."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 2. Local embedded persistent mode (for single-process development and pytest)
    persist_dir = Path(settings.chroma_persist_directory).resolve()

    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        logger.info("ChromaDB PersistentClient initialized at: %s", persist_dir)
        return _chroma_client
    except Exception as exc:
        logger.error("Failed to initialize ChromaDB PersistentClient at %s: %s", persist_dir, exc)
        raise RuntimeError(f"ChromaDB initialization failed: {exc}") from exc


def check_chroma_connection() -> Tuple[bool, str]:
    """Verify ChromaDB initialization and readiness.

    Returns:
        (is_ready, status_message)
    """
    settings = get_settings()
    if settings.environment == "production" and not (settings.chroma_host and settings.chroma_host.strip()):
        return False, "Unconfigured: CHROMA_HOST is required in production"

    try:
        client = get_chroma_client()
        heartbeat = client.heartbeat()
        return True, f"Ready (heartbeat: {heartbeat})"
    except Exception as exc:
        logger.warning("ChromaDB readiness check failed: %s", exc)
        return False, f"Initialization error: {str(exc).splitlines()[0]}"
