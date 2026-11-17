import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    INFERENCE_PROVIDER: str = "auto_fallback"
    OCR_PROVIDER: str = "ollama"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "qwen2.5:7b"
    OLLAMA_OCR_MODEL: str = "qwen2.5vl:7b"
    OLLAMA_EMBED_MODEL: str = "snowflake-arctic-embed2"

    # Cấu hình chuyên biệt cho AI Document Understanding (05 trường thông tin)
    LOCAL_METADATA_AI_MODEL: str = "qwen2.5:1.5b"
    LLM_METADATA_TIMEOUT: float = 25.0
    LLM_KEEP_ALIVE: int = -1
    LLM_METADATA_NUM_CTX: int = 1536
    LLM_METADATA_MAX_TOKENS: int = 384

    # Cấu hình hiệu năng OCR & phân luồng tài liệu
    OCR_HEADER_DPI: int = 175
    OCR_HEADER_HEIGHT_RATIO: float = 0.52
    OCR_TARGET_MAX_WIDTH: int = 1600
    PDF_MIN_DIGITAL_CHARS: int = 45

    # Đường dẫn công cụ OCR & lưu trữ nội bộ
    TESSERACT_CMD: str = ""
    TESSDATA_DIR: str = ""
    LOCAL_RBAC_STORE_PATH: str = "local_rbac_store.json"

    VLLM_LLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_LLM_API_KEY: str = os.environ.get("VLLM_LLM_API_KEY", "")
    VLLM_LLM_MODEL: str = "qwen2.5-llm"

    VLLM_OCR_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_OCR_API_KEY: str = os.environ.get("VLLM_OCR_API_KEY", "")
    VLLM_OCR_MODEL: str = "qwen2.5-vl-ocr"

    RAGFLOW_BASE_URL: str = "http://localhost:9380"
    RAGFLOW_API_KEY: str = os.environ.get("RAGFLOW_API_KEY", "")

    GATEWAY_API_KEY: str = os.environ.get("GATEWAY_API_KEY", "internal-gateway-secret-2026")

    RBAC_GROUP_MAPPING: str = json.dumps({
        "Lanh_Dao_Bo": ["ds_general", "ds_tccb", "ds_khtc", "ds_cds"],
        "Phong_TCCB": ["ds_general", "ds_tccb"],
        "Phong_KHTC": ["ds_general", "ds_khtc"],
        "Trung_Tam_CDS": ["ds_general", "ds_cds"],
        "DEFAULT": ["ds_general"]
    })

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def get_rbac_mapping(self) -> Dict[str, List[str]]:
        try:
            return json.loads(self.RBAC_GROUP_MAPPING)
        except Exception:
            return {"DEFAULT": []}


settings = Settings()


@dataclass(frozen=True)
class OCRPipelineConfig:
    """Ánh xạ trực tiếp từ đối tượng settings của hệ thống sang pipeline OCR 5 trường."""
    ollama_base_url: str = field(default_factory=lambda: settings.OLLAMA_BASE_URL)
    llm_model_name: str = field(default_factory=lambda: settings.LOCAL_METADATA_AI_MODEL)
    vision_slot_model_name: str = field(default_factory=lambda: settings.OLLAMA_OCR_MODEL)
    keep_alive: int = field(default_factory=lambda: settings.LLM_KEEP_ALIVE)
    request_timeout_sec: float = field(default_factory=lambda: settings.LLM_METADATA_TIMEOUT)

    tesseract_cmd: str = field(default_factory=lambda: settings.TESSERACT_CMD)
    tessdata_dir: str = field(default_factory=lambda: settings.TESSDATA_DIR)
    tesseract_lang: str = field(default_factory=lambda: os.getenv("TESSERACT_LANG", "vie"))
    tesseract_config: str = field(default_factory=lambda: os.getenv("TESSERACT_CONFIG", "--oem 1 --psm 6"))

    render_dpi_full_page: int = field(default_factory=lambda: settings.OCR_HEADER_DPI)
    render_dpi_header_strip: int = field(default_factory=lambda: max(220, settings.OCR_HEADER_DPI + 50))
    min_searchable_chars: int = field(default_factory=lambda: settings.PDF_MIN_DIGITAL_CHARS)
    header_y_ratio_limit: float = field(default_factory=lambda: settings.OCR_HEADER_HEIGHT_RATIO)
    body_snippet_max_chars: int = 450

    left_num_strip_box: Tuple[float, float, float, float] = (0.04, 0.09, 0.49, 0.25)
    right_date_strip_box: Tuple[float, float, float, float] = (0.42, 0.09, 0.96, 0.25)

    llm_temperature: float = 0.0
    llm_num_ctx: int = field(default_factory=lambda: settings.LLM_METADATA_NUM_CTX)
    llm_max_tokens: int = field(default_factory=lambda: settings.LLM_METADATA_MAX_TOKENS)