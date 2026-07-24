from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings for Enterprise Knowledge Navigator (EKN).

    Loads configuration from environment variables and an optional local `.env` file.
    Does not require external services to be configured at startup.
    """

    # Application Identity & Runtime
    app_name: str = "Enterprise Knowledge Navigator"
    environment: str = "development"
    debug: bool = True

    # Backend API Server
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    backend_api_url: str = "http://127.0.0.1:8000"

    # Relational Database (MySQL / SQLite development fallback)
    # Form for MySQL: mysql+pymysql://<user>:<password>@<host>:<port>/<database>
    # Form for SQLite: sqlite:///./data/ekn.db
    mysql_url: Optional[str] = None
    database_url: Optional[str] = None
    mysql_pool_size: int = 5
    mysql_max_overflow: int = 10
    mysql_connect_timeout: int = 5

    # Storage Paths & Upload Limits (Phase 2)
    storage_local_root: str = "./data/storage"
    upload_dir: str = "./data/storage/uploads"
    temp_upload_dir: str = "./data/storage/temp"
    max_upload_size_bytes: int = 250 * 1024 * 1024  # 250 MB default
    max_files_per_upload: int = 10                  # Max 10 PDFs per request
    upload_chunk_size_bytes: int = 1024 * 1024      # 1 MB chunk processing blocks

    # PDF Processing & Ingestion Engine (Phase 3)
    processed_dir: str = "./data/storage/processed"
    ocr_enabled: bool = True
    ocr_min_text_chars_per_page: int = 50
    tesseract_cmd: Optional[str] = "tesseract"
    worker_poll_interval_seconds: float = 2.0

    # Vector Database & Embeddings (Phase 4 & 13)
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "ekn_document_chunks"
    chroma_upsert_batch_size: int = 50
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    chroma_ssl: bool = False

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384

    # Hierarchical Chunking (Phase 4)
    chunking_parent_target_tokens: int = 800
    chunking_child_target_tokens: int = 200
    chunking_child_overlap_tokens: int = 40

    # Hybrid Retrieval & Cross-Encoder Reranking (Phase 5)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str = "cpu"
    cross_encoder_batch_size: int = 32

    retrieval_dense_top_k: int = 30
    retrieval_bm25_top_k: int = 30
    retrieval_rrf_k: int = 60
    retrieval_rrf_candidates: int = 30
    retrieval_rerank_candidates: int = 30
    retrieval_final_top_k: int = 8

    # Phase 6: Conversational RAG, LLM Integration & LangGraph Workflows
    llm_provider: str = "gemini"
    gemini_api_key: Optional[str] = None
    llm_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 2048
    rag_max_context_tokens: int = 4000
    rag_conversation_history_limit: int = 6
    rag_evidence_relevance_threshold: float = 0.0
    rag_min_evidence_count: int = 1

    # Phase 9: Automated Enterprise Knowledge Dossier PDF Generator
    app_base_url: str = "http://localhost:8501"
    reports_dir: str = "./data/storage/reports"

    # Phase 10: Enterprise Security, Authentication, RBAC & Multi-Tenant Isolation
    jwt_secret_key: str = "enterprise-knowledge-navigator-jwt-secret-key-32bytes-min"
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "ekn-platform"
    jwt_audience: str = "ekn-client"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    max_failed_login_attempts: int = 5
    account_lockout_minutes: int = 15
    min_password_length: int = 12
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # Local Demonstration Mode
    ekn_demo_mode: bool = True
    demo_admin_password: Optional[str] = None
    demo_employee_password: Optional[str] = None
    demo_workspace_id: Optional[str] = "dc5d9139-f381-4818-ab26-c2e878997b4d"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def is_mysql_configured(self) -> bool:
        """Check whether a non-empty MySQL connection URL has been provided."""
        return bool(self.mysql_url and self.mysql_url.strip())

    @property
    def effective_database_url(self) -> Optional[str]:
        """Return the effective database URL (mysql_url, database_url, or fallback)."""
        if self.mysql_url and self.mysql_url.strip():
            return self.mysql_url.strip()
        if self.database_url and self.database_url.strip():
            return self.database_url.strip()
        return None

    @property
    def effective_allowed_origins(self) -> list[str]:
        """Return list of allowed CORS origins."""
        if not self.allowed_origins:
            return ["http://localhost:8501", "http://127.0.0.1:8501"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def gemini_key_status(self) -> str:
        """Return configuration status: 'MISSING', 'PLACEHOLDER', or 'CONFIGURED'."""
        raw_key = (self.gemini_api_key or self.llm_api_key or "").strip()
        if not raw_key:
            return "MISSING"
        placeholders = {
            "your_gemini_api_key_here",
            "your_api_key_here",
            "<your-google-gemini-api-key>",
            "<your_gemini_api_key_here>",
            "placeholder",
            "changeme",
            "none",
            "null",
        }
        if raw_key.lower() in placeholders or "your_" in raw_key.lower():
            return "PLACEHOLDER"
        return "CONFIGURED"

    @property
    def effective_gemini_api_key(self) -> Optional[str]:
        """Return the effective Gemini API key only if genuinely configured (not placeholder)."""
        if self.gemini_key_status != "CONFIGURED":
            return None
        raw_key = (self.gemini_api_key or self.llm_api_key or "").strip()
        return raw_key or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached singleton instance of application settings."""
    return Settings()
