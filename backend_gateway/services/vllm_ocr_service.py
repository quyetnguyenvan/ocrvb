import base64
import fitz
import requests
from typing import List, Dict, Any
from backend_gateway.config import settings


class VLLMService:
    def __init__(self):
        self.llm_base_url = settings.VLLM_LLM_BASE_URL.rstrip("/")
        self.llm_api_key = settings.VLLM_LLM_API_KEY
        self.llm_model = settings.VLLM_LLM_MODEL

        self.ocr_base_url = settings.VLLM_OCR_BASE_URL.rstrip("/")
        self.ocr_api_key = settings.VLLM_OCR_API_KEY
        self.ocr_model = settings.VLLM_OCR_MODEL

    def is_llm_available(self) -> bool:
        try:
            resp = requests.get(
                f"{self.llm_base_url}/models",
                headers={"Authorization": f"Bearer {self.llm_api_key}"},
                timeout=3
            )
            return resp.status_code == 200
        except Exception:
            return False

    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        resp = requests.post(
            f"{self.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.llm_api_key}"},
            json={"model": self.llm_model, "messages": messages, "temperature": temperature},
            timeout=90
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _ocr_single_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        headers = {"Authorization": f"Bearer {self.ocr_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Bạn là hệ thống OCR chuyên nghiệp cho văn bản hành chính và nội bộ. "
                                "Hãy trích xuất toàn bộ nội dung trong ảnh sang định dạng Markdown chuẩn."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }
        resp = requests.post(f"{self.ocr_base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def process_ocr_file(self, file_bytes: bytes, filename: str) -> dict:
        ext = filename.lower().split(".")[-1]
        pages_markdown: List[str] = []
        if ext == "pdf":
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_index in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_index)
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                page_md = self._ocr_single_image_bytes(img_bytes, "image/png")
                pages_markdown.append(f"<!-- Trang {page_index + 1} -->\n{page_md}")
            pdf_doc.close()
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
            page_md = self._ocr_single_image_bytes(file_bytes, mime)
            pages_markdown.append(page_md)
        else:
            raise ValueError(f"Định dạng file .{ext} không hỗ trợ OCR trực tiếp.")

        return {
            "filename": filename,
            "total_pages": len(pages_markdown),
            "metadata": {"so_trang": len(pages_markdown)},
            "markdown_content": "\n\n---\n\n".join(pages_markdown),
            "engine": f"vLLM ({self.ocr_model})"
        }