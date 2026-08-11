"""FastAPI Router for Automated Enterprise Knowledge Dossier Reports."""

import logging
from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    require_workspace_admin,
    require_workspace_member,
    require_workspace_owner,
)
from backend.database.connection import get_db
from backend.database.models import Report, Workspace, WorkspaceMembership
from backend.schemas.report import ReportCreateRequest, ReportListResponse, ReportResponse
from backend.services.report_generation_service import report_generation_service
from backend.services.workspace_service import WorkspaceService

logger = logging.getLogger("ekn.reports_api")

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/reports",
    tags=["Reports & Dossiers"],
)


def _get_workspace_or_404(workspace_id: str, db: Session) -> Workspace:
    """Validate workspace exists."""
    ws = WorkspaceService.get_by_id(db, workspace_id)
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )
    return ws


@router.post(
    "",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request generation of an Enterprise Knowledge Dossier PDF",
)
def create_report(
    workspace_id: str,
    request: ReportCreateRequest,
    background_tasks: BackgroundTasks,
    run_synchronous: bool = Query(default=True, description="Execute synchronously (set False for background worker)"),
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> ReportResponse:
    """Create a new report job and generate the dossier PDF."""
    report = report_generation_service.create_report_job(
        db=db,
        workspace_id=workspace_id,
        request=request,
    )
    report.generated_by_user_id = membership.user_id
    db.commit()
    db.refresh(report)

    if run_synchronous:
        finalized_report = report_generation_service.execute_report_generation(
            db=db,
            workspace_id=workspace_id,
            report_id=report.id,
        )
        return ReportResponse.model_validate(finalized_report)
    else:
        # Enqueue background task
        background_tasks.add_task(
            report_generation_service.execute_report_generation,
            db=db,
            workspace_id=workspace_id,
            report_id=report.id,
        )
        return ReportResponse.model_validate(report)


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List all knowledge dossier reports in workspace",
)
def list_reports(
    workspace_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> ReportListResponse:
    """Retrieve paginated list of reports belonging to the workspace."""
    query = (
        db.query(Report)
        .filter(Report.workspace_id == workspace_id)
        .order_by(Report.created_at.desc())
    )
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return ReportListResponse(
        total=total,
        items=[ReportResponse.model_validate(r) for r in items],
    )


@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Get report metadata and generation status",
)
def get_report(
    workspace_id: str,
    report_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> ReportResponse:
    """Retrieve report status, coverage metrics, and file size."""
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.workspace_id == workspace_id)
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' was not found in workspace '{workspace_id}'.",
        )
    return ReportResponse.model_validate(report)


@router.get(
    "/{report_id}/download",
    summary="Download the generated Knowledge Dossier PDF",
)
def download_report(
    workspace_id: str,
    report_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Download the finalized PDF file."""
    report = (
        db.query(Report)
        .filter(Report.id == report_id, Report.workspace_id == workspace_id)
        .first()
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' was not found in workspace '{workspace_id}'.",
        )

    if report.status not in ("completed", "partially_completed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report is not ready for download (status: '{report.status}').",
        )

    try:
        physical_path = report_generation_service.get_report_file_path(report)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )

    clean_filename = f"{report.report_name.replace(' ', '_')}.pdf"
    return FileResponse(
        path=str(physical_path),
        filename=clean_filename,
        media_type="application/pdf",
    )


@router.delete(
    "/{report_id}",
    summary="Delete a report and remove physical PDF storage",
)
def delete_report(
    workspace_id: str,
    report_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_owner),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete a report record and purge its underlying PDF file."""
    deleted = report_generation_service.delete_report(db, workspace_id, report_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' was not found in workspace '{workspace_id}'.",
        )
    return {"message": "Report deleted successfully.", "report_id": report_id}
