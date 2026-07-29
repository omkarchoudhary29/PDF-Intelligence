import time
from pathlib import Path
from typing import Any, Dict, Tuple
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from backend.database.connection import check_chroma_connection, check_database_connection
from config.settings import get_settings

router = APIRouter(tags=["Health & Readiness"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ServiceStatus(BaseModel):
    ready: bool
    message: str


class ReadinessResponse(BaseModel):
    status: str
    services: Dict[str, ServiceStatus]


def check_storage_status() -> Tuple[bool, str]:
    """Verify local storage directory access and writability."""
    settings = get_settings()
    try:
        root_dir = Path(settings.storage_local_root).resolve()
        upload_dir = Path(settings.upload_dir).resolve()
        processed_dir = Path(settings.processed_dir).resolve()

        root_dir.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        return True, f"Storage directories available at: {root_dir}"
    except Exception as exc:
        return False, f"Storage directory access error: {exc}"


def check_llm_status() -> Tuple[bool, str]:
    """Verify LLM API configuration status."""
    settings = get_settings()
    if settings.effective_gemini_api_key:
        return True, f"Configured (model: {settings.gemini_model})"
    return False, "Unconfigured (conversational answers will be unavailable until GEMINI_API_KEY is configured)"


def check_worker_status() -> Tuple[bool, str]:
    """Verify that the ingestion worker is active and reporting heartbeats."""
    settings = get_settings()
    hb_path = Path(settings.storage_local_root) / ".worker_heartbeat"
    if not hb_path.exists():
        return True, "Worker heartbeat file pending first job"
    try:
        content = hb_path.read_text(encoding="utf-8").strip()
        parts = content.split(":")
        timestamp = float(parts[0])
        age = time.time() - timestamp
        if age <= 60.0:
            return True, f"Active (last heartbeat: {age:.1f}s ago)"
        else:
            return True, f"Idle (last heartbeat: {age:.1f}s ago)"
    except Exception as exc:
        return True, f"Worker check fallback ({exc})"


@router.get("/health", response_model=HealthResponse, summary="Application Liveness Check")
def health_check() -> HealthResponse:
    """Basic health check indicating that the backend process is running."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Infrastructure Readiness Check")
def readiness_check(response: Response) -> ReadinessResponse:
    """Infrastructure readiness probe checking MySQL, ChromaDB, storage, worker, and LLM configuration."""
    settings = get_settings()

    db_ready, db_msg = check_database_connection()
    chroma_ready, chroma_msg = check_chroma_connection()
    storage_ready, storage_msg = check_storage_status()
    llm_ready, llm_msg = check_llm_status()
    worker_ready, worker_msg = check_worker_status()

    services = {
        "mysql": ServiceStatus(ready=db_ready, message=db_msg),
        "chromadb": ServiceStatus(ready=chroma_ready, message=chroma_msg),
        "storage": ServiceStatus(ready=storage_ready, message=storage_msg),
        "llm": ServiceStatus(ready=llm_ready, message=llm_msg),
        "worker": ServiceStatus(ready=worker_ready, message=worker_msg),
    }

    # If ChromaDB or storage fails, or if MySQL was explicitly configured but failed to connect:
    if not chroma_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "not_ready"
    elif not storage_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "not_ready"
    elif settings.is_mysql_configured and not db_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "not_ready"
    elif not db_ready:
        # MySQL is not configured, but ChromaDB and storage are operational
        overall_status = "degraded_without_mysql"
    else:
        overall_status = "ready"

    return ReadinessResponse(
        status=overall_status,
        services=services,
    )


@router.get("/health/llm", summary="LLM Configuration and Smoke Test")
def llm_health_check() -> Dict[str, Any]:
    """Diagnostic endpoint reporting LLM configuration status and smoke test outcome."""
    from backend.services.llm_service import llm_service
    status_info = llm_service.get_configuration_status()
    smoke_result = llm_service.test_provider_connection()
    return {
        "configuration": status_info,
        "smoke_test": smoke_result,
    }
