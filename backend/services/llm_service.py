import logging
import time
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from config.settings import get_settings

logger = logging.getLogger("ekn.llm")


class LLMError(Exception):
    """Base exception for application LLM operations."""
    pass


class LLMNotConfiguredError(LLMError):
    """Raised when the LLM provider API key or required credentials are not configured."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit or quota has been exceeded."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the LLM provider request times out."""
    pass


class LLMService:
    """Reusable singleton LLM service for EKN.

    Manages lazy initialization of ChatGoogleGenerativeAI, handles configuration validation,
    translates provider exceptions into structured application errors, and supports mock
    injection for isolated unit tests.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm: Optional[BaseChatModel] = None
        self._is_mock: bool = False

    @property
    def is_configured(self) -> bool:
        """Check whether the application LLM has valid credentials configured."""
        if self._is_mock:
            return True
        return self.settings.gemini_key_status == "CONFIGURED"

    def get_configuration_status(self) -> Dict[str, Any]:
        """Return diagnostic configuration status without exposing sensitive credentials."""
        status_val = "MOCK" if self._is_mock else self.settings.gemini_key_status
        return {
            "status": status_val,
            "provider": self.settings.llm_provider,
            "model": self.settings.gemini_model,
            "is_configured": self.is_configured,
        }

    def test_provider_connection(self) -> Dict[str, Any]:
        """Execute a controlled smoke test against the configured LLM provider.

        Validates authentication, network connectivity, latency, and response non-emptiness.
        Masks all sensitive credentials from returned status and diagnostics.
        """
        if not self.is_configured:
            return {
                "success": False,
                "status": self.get_configuration_status()["status"],
                "error": f"LLM credentials not configured ({self.settings.gemini_key_status}).",
                "latency_ms": 0.0,
                "model": self.settings.gemini_model,
            }

        start_time = time.perf_counter()
        try:
            llm = self.get_llm()
            test_prompt = [HumanMessage(content="Respond with 'PONG' and nothing else.")]
            response = llm.invoke(test_prompt)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            raw_content = response.content if response else ""
            if isinstance(raw_content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                ).strip()
            else:
                content = str(raw_content).strip()

            if not content:
                return {
                    "success": False,
                    "status": "EMPTY_RESPONSE",
                    "error": "LLM provider returned an empty response.",
                    "latency_ms": latency_ms,
                    "model": self.settings.gemini_model,
                }

            return {
                "success": True,
                "status": "CONNECTED",
                "response_preview": content[:50],
                "latency_ms": latency_ms,
                "model": self.settings.gemini_model,
            }
        except LLMNotConfiguredError as exc:
            return {
                "success": False,
                "status": self.settings.gemini_key_status,
                "error": str(exc),
                "latency_ms": 0.0,
                "model": self.settings.gemini_model,
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_str = str(exc)
            if self.settings.gemini_api_key:
                err_str = err_str.replace(self.settings.gemini_api_key, "[REDACTED]")
            if self.settings.llm_api_key:
                err_str = err_str.replace(self.settings.llm_api_key, "[REDACTED]")
            return {
                "success": False,
                "status": "ERROR",
                "error": err_str,
                "latency_ms": latency_ms,
                "model": self.settings.gemini_model,
            }

    def get_llm(self) -> BaseChatModel:
        """Lazily initialize or return cached LangChain chat model.

        Raises:
            LLMNotConfiguredError: If API credentials are not set.
        """
        if self._llm is not None:
            return self._llm

        if not self.is_configured:
            status_desc = self.settings.gemini_key_status.lower()
            logger.warning("Attempted to initialize LLM, but GEMINI_API_KEY is %s.", status_desc)
            raise LLMNotConfiguredError(
                f"LLM provider is not configured (GEMINI_API_KEY is {status_desc}). "
                "Please set a valid GEMINI_API_KEY in your .env file to enable natural language answer generation."
            )

        api_key = self.settings.effective_gemini_api_key

        provider = (self.settings.llm_provider or "gemini").lower()
        model_name = self.settings.gemini_model

        if provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                logger.info(
                    "Initializing ChatGoogleGenerativeAI (model='%s', temp=%.2f, max_tokens=%d)...",
                    model_name,
                    self.settings.llm_temperature,
                    self.settings.llm_max_output_tokens,
                )
                self._llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=self.settings.llm_temperature,
                    max_output_tokens=self.settings.llm_max_output_tokens,
                    timeout=45.0,
                    max_retries=2,
                )
                return self._llm
            except Exception as exc:
                logger.error("Failed to initialize Google Gemini LLM: %s", exc)
                raise LLMError(f"Failed to initialize LLM provider '{provider}': {exc}") from exc
        else:
            raise LLMError(f"Unsupported LLM provider: '{provider}'")

    get_chat_model = get_llm

    def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke the configured LLM with prompt messages and return string answer.

        Handles timeouts, rate limits, and network errors safely without exposing secrets.
        """
        llm = self.get_llm()

        try:
            response = llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                # When content is returned as structured parts or list
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                return "".join(text_parts).strip()
            return str(content).strip()
        except LLMNotConfiguredError:
            raise
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                logger.error("LLM rate limit encountered during generation: %s", exc)
                raise LLMRateLimitError("LLM API rate limit or quota exceeded. Please try again shortly.") from exc
            elif "timeout" in err_str or "timed out" in err_str or "deadline" in err_str:
                logger.error("LLM request timed out during generation: %s", exc)
                raise LLMTimeoutError("LLM generation request timed out. Please retry.") from exc
            else:
                logger.error("LLM generation failed: %s", exc)
                raise LLMError(f"LLM generation failed: {exc}") from exc

    def set_mock_model(self, mock_llm: Optional[BaseChatModel]) -> None:
        """Inject a mock model for test suites to bypass live API calls."""
        self._llm = mock_llm
        self._is_mock = mock_llm is not None

    def reset(self) -> None:
        """Clear cached LLM instance."""
        self._llm = None
        self._is_mock = False


llm_service = LLMService()
