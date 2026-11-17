from typing import List, Dict, Any, Optional, Tuple
from backend_gateway.config import settings
from backend_gateway.services.ollama_service import OllamaService
from backend_gateway.services.vllm_ocr_service import VLLMService


class DualInferenceRouter:
    def __init__(self):
        self.ollama = OllamaService()
        self.vllm = VLLMService()

    def generate_chat_answer(
        self,
        messages: List[Dict[str, Any]],
        preferred_engine: Optional[str] = None,
        temperature: float = 0.1
    ) -> Tuple[str, str]:
        mode = (preferred_engine or settings.INFERENCE_PROVIDER).lower()
        if mode == "ollama":
            return self.ollama.chat_completion(messages, temperature=temperature), f"Ollama ({self.ollama.llm_model})"
        if mode == "vllm":
            return self.vllm.chat_completion(messages, temperature=temperature), f"vLLM ({self.vllm.llm_model})"

        if self.vllm.is_llm_available():
            try:
                return self.vllm.chat_completion(messages, temperature=temperature), f"vLLM ({self.vllm.llm_model})"
            except Exception:
                pass
        return self.ollama.chat_completion(messages, temperature=temperature), f"Ollama [Fallback] ({self.ollama.llm_model})"

    def perform_ocr(
        self,
        file_bytes: bytes,
        filename: str,
        preferred_engine: Optional[str] = None,
        ollama_ocr_model: Optional[str] = None
    ) -> dict:
        provider = (preferred_engine or settings.OCR_PROVIDER).lower()
        if provider == "ollama":
            return self.ollama.process_ocr_file(file_bytes, filename, custom_model=ollama_ocr_model)
        if provider == "vllm":
            return self.vllm.process_ocr_file(file_bytes, filename)
        try:
            return self.vllm.process_ocr_file(file_bytes, filename)
        except Exception:
            return self.ollama.process_ocr_file(file_bytes, filename, custom_model=ollama_ocr_model)