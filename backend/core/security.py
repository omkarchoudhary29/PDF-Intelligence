"""Enterprise Security Utilities: Argon2id Password Hashing and JWT Management.

Provides:
- Argon2id password hashing and verification using pwdlib
- Password strength validation
- RFC-compliant JWT generation, signing, and verification using PyJWT
- Cryptographically secure refresh token generation and SHA-256 token hashing
"""

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import secrets
from typing import Any, Dict, Optional, Tuple
import uuid

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from config.settings import get_settings

logger = logging.getLogger("ekn.security")

# Initialize pwdlib with modern Argon2id hasher
_password_hasher = PasswordHash((Argon2Hasher(),))


class SecurityError(Exception):
    """Base exception for security and authentication failures."""
    pass


class TokenExpiredError(SecurityError):
    """Raised when an access token has expired."""
    pass


class InvalidTokenError(SecurityError):
    """Raised when an access token signature or claims are invalid."""
    pass


def hash_password(password: str) -> str:
    """Hash plaintext password using Argon2id algorithm."""
    return _password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against stored Argon2id hash."""
    try:
        return _password_hasher.verify(plain_password, hashed_password)
    except Exception as exc:
        logger.warning("Password verification failed with exception: %s", exc)
        return False


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """Enforce enterprise minimum password security rules."""
    settings = get_settings()
    min_len = max(settings.min_password_length, 12)

    if len(password) < min_len:
        return False, f"Password must be at least {min_len} characters long."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?/~`'\"\\")
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character."
    return True, None


def create_access_token(
    user_id: Optional[str] = None,
    email: str = "",
    expires_delta: Optional[timedelta] = None,
    subject: Optional[str] = None,
) -> str:
    """Generate a signed, short-lived JWT access token with full claims validation."""
    effective_user_id = user_id or subject
    if not effective_user_id:
        raise ValueError("Either user_id or subject must be provided.")

    settings = get_settings()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload: Dict[str, Any] = {
        "sub": effective_user_id,
        "email": email.lower().strip(),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4()),
        "type": "access",
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and strictly validate a JWT access token."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["exp", "sub", "type", "iss", "aud"]},
        )
        if payload.get("type") != "access":
            raise InvalidTokenError("Token is not an access token.")
        return payload
    except jwt.ExpiredSignatureError as err:
        raise TokenExpiredError("Access token has expired.") from err
    except jwt.PyJWTError as err:
        raise InvalidTokenError(f"Access token validation failed: {err}") from err


def generate_refresh_token() -> str:
    """Generate a high-entropy cryptographically secure refresh token string."""
    return secrets.token_urlsafe(64)


def hash_token(raw_token: str) -> str:
    """Compute SHA-256 hash of a raw token for secure database persistence."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
