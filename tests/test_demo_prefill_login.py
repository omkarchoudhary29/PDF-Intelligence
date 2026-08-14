"""Tests for demo credential pre-filling and 1-click demo login across portals."""

import pytest
from unittest.mock import MagicMock, patch
import streamlit as st

from config.settings import get_settings
from frontend.api_client import BackendAPIClient
from frontend.views.login import _render_admin_portal, _render_employee_portal, _render_role_selection, render_login


def test_admin_portal_prefills_demo_credentials():
    """Verify that visiting the Admin portal in demo mode automatically pre-populates credentials in session state."""
    settings = get_settings()
    assert settings.ekn_demo_mode is True

    mock_client = MagicMock(spec=BackendAPIClient)

    def mock_columns(spec):
        n = len(spec) if isinstance(spec, list) else spec
        return tuple(MagicMock() for _ in range(n))

    # Initialize a clean session state mock/dict
    session_dict = {"auth_portal": "admin", "admin_mode_selector": "Sign In"}
    with patch.object(st, "session_state", session_dict), \
         patch.object(st, "columns", side_effect=mock_columns), \
         patch.object(st, "button", return_value=False), \
         patch.object(st, "markdown"), \
         patch.object(st, "radio", return_value="Sign In"), \
         patch.object(st, "container"), \
         patch.object(st, "form"), \
         patch.object(st, "text_input") as mock_text_input, \
         patch.object(st, "checkbox", return_value=False), \
         patch.object(st, "form_submit_button", return_value=False), \
         patch.object(st, "query_params", {}):

        _render_admin_portal(mock_client)

        assert session_dict.get("admin_signin_email") == "admin.demo@example.com"
        assert session_dict.get("admin_signin_password") == settings.demo_admin_password
        # Verify text_input was called with the right keys
        call_keys = [c.kwargs.get("key") for c in mock_text_input.call_args_list]
        assert "admin_signin_email" in call_keys
        assert "admin_signin_password" in call_keys


def test_employee_portal_prefills_demo_credentials():
    """Verify that visiting the Employee portal in demo mode automatically pre-populates credentials in session state."""
    settings = get_settings()
    assert settings.ekn_demo_mode is True

    mock_client = MagicMock(spec=BackendAPIClient)

    def mock_columns(spec):
        n = len(spec) if isinstance(spec, list) else spec
        return tuple(MagicMock() for _ in range(n))

    session_dict = {"auth_portal": "employee", "emp_mode_selector": "Sign In"}
    with patch.object(st, "session_state", session_dict), \
         patch.object(st, "columns", side_effect=mock_columns), \
         patch.object(st, "button", return_value=False), \
         patch.object(st, "markdown"), \
         patch.object(st, "radio", return_value="Sign In"), \
         patch.object(st, "container"), \
         patch.object(st, "form"), \
         patch.object(st, "text_input") as mock_text_input, \
         patch.object(st, "checkbox", return_value=False), \
         patch.object(st, "form_submit_button", return_value=False), \
         patch.object(st, "query_params", {}):

        _render_employee_portal(mock_client)

        assert session_dict.get("emp_signin_email") == "employee.demo@example.com"
        assert session_dict.get("emp_signin_password") == settings.demo_employee_password
        call_keys = [c.kwargs.get("key") for c in mock_text_input.call_args_list]
        assert "emp_signin_email" in call_keys
        assert "emp_signin_password" in call_keys


def test_custom_user_input_is_preserved():
    """Verify that if a user types custom credentials, the pre-fill does NOT overwrite them."""
    settings = get_settings()
    mock_client = MagicMock(spec=BackendAPIClient)

    custom_email = "custom.executive@enterprise.com"
    custom_pwd = "CustomSecretPassword123!"

    def mock_columns(spec):
        n = len(spec) if isinstance(spec, list) else spec
        return tuple(MagicMock() for _ in range(n))

    session_dict = {
        "auth_portal": "admin",
        "admin_mode_selector": "Sign In",
        "admin_signin_email": custom_email,
        "admin_signin_password": custom_pwd,
    }
    with patch.object(st, "session_state", session_dict), \
         patch.object(st, "columns", side_effect=mock_columns), \
         patch.object(st, "button", return_value=False), \
         patch.object(st, "markdown"), \
         patch.object(st, "radio", return_value="Sign In"), \
         patch.object(st, "container"), \
         patch.object(st, "form"), \
         patch.object(st, "text_input"), \
         patch.object(st, "checkbox", return_value=False), \
         patch.object(st, "form_submit_button", return_value=False), \
         patch.object(st, "query_params", {}):

        _render_admin_portal(mock_client)

        # Custom inputs must NOT be overwritten by demo credentials
        assert session_dict["admin_signin_email"] == custom_email
        assert session_dict["admin_signin_password"] == custom_pwd


def test_live_backend_authenticates_prefilled_demo_credentials():
    """Verify end-to-end that the pre-filled credentials authenticate against the running backend."""
    settings = get_settings()
    client = BackendAPIClient(settings.backend_api_url)

    # 1. Admin login
    admin_token = client.login(
        "admin.demo@example.com",
        settings.demo_admin_password,
        portal="admin",
    )
    assert admin_token is not None
    assert "access_token" in admin_token
    admin_profile = client.get_current_user_profile()
    assert admin_profile["email"] == "admin.demo@example.com"
    client.logout()

    # 2. Employee login
    emp_token = client.login(
        "employee.demo@example.com",
        settings.demo_employee_password,
        portal="employee",
    )
    assert emp_token is not None
    assert "access_token" in emp_token
    emp_profile = client.get_current_user_profile()
    assert emp_profile["email"] == "employee.demo@example.com"
    client.logout()
