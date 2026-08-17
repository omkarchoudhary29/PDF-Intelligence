"""Ask Knowledge Base View for Enterprise Knowledge Navigator.

Provides conversational document intelligence interface connecting to LangGraph RAG workflow,
multi-turn conversation persistence, follow-up queries, personal and company document scopes,
and verified source citations with original PDF page preview.
"""

from typing import Any, Dict
from frontend.api_client import BackendAPIClient
from frontend.views.employee_workspace import render_employee_workspace


def render_ask_knowledge_base(client: BackendAPIClient, workspace: Dict[str, Any]) -> None:
    """Render the Conversational Document Intelligence interface via the modern 3-panel employee workspace."""
    render_employee_workspace(client, workspace)
