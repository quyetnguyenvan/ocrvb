import base64
import time
import fitz
import requests
from typing import List, Dict, Any, Optional
from backend_gateway.config import settings


class OllamaService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.llm_model = settings.OLLAMA_LLM_MODEL
        self.ocr_model = settings.OLLAMA_OCR_MODEL
        self.embed_model = settings.OLLAMA_EMBED_MODEL

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/api/pull",
            json={"model": model_name, "stream": False},
            timeout=600
        )
        resp.raise_for_status()
        return resp.json()

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        target_model = model or self.llm_model
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096}
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        if resp.status_code == 404:
            raise RuntimeError(f"Model '{target_model}' chưa có trong Ollama. Hãy chạy: ollama pull {target_model}")
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def _format_raw_text_with_llm(self, raw_text: str, model_name: str) -> str:
        print(f"[OCR] Phát hiện văn bản gốc, đang chuẩn hóa Markdown bằng model '{model_name}'...", flush=True)
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Hãy trình bày lại văn bản hành chính dưới đây sang định dạng Markdown chuẩn. "
                        "Giữ nguyên quốc hiệu, tiêu ngữ, số ký hiệu, ngày tháng, nội dung và bảng biểu. "
                        "Không giải thích gì thêm:\n\n"
                        f"{raw_text[:6000]}"
                    )
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 2048}
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def _ocr_single_image_bytes(
        self,
        image_bytes: bytes,
        custom_model: Optional[str] = None,
        fallback_raw_text: str = ""
    ) -> str:
        target_model = custom_model or self.ocr_model

        # Ưu tiên 1: Nếu người dùng chọn model thuần text (VD: qwen2.5:7b) và PDF có sẵn lớp chữ
        if "vl" not in target_model.lower() and "ocr" not in target_model.lower():
            if fallback_raw_text and len(fallback_raw_text.strip()) > 30:
                return self._format_raw_text_with_llm(fallback_raw_text, target_model)

        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        print(f"[OCR] Đang gửi ảnh trang tài liệu tới Ollama Vision ('{target_model}')...", flush=True)
        t0 = time.time()

        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Trích xuất toàn bộ chữ trong ảnh văn bản này sang Markdown. "
                        "Giữ nguyên số hiệu, ngày tháng, tiêu đề và bảng biểu. Không bình luận thêm."
                    ),
                    "images": [b64_img]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 3072,
                "num_predict": 2048
            }
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        print(f"[OCR] Ollama xử lý xong trang trong {time.time() - t0:.1f} giây.", flush=True)

        if resp.status_code == 404:
            raise RuntimeError(
                f"Model OCR '{target_model}' chưa được tải trong Ollama. "
                f"Vui lòng chạy lệnh: ollama pull {target_model}"
            )

        if resp.status_code == 400:
            if fallback_raw_text and len(fallback_raw_text.strip()) > 20:
                return self._format_raw_text_with_llm(fallback_raw_text, target_model)
            err_msg = resp.json().get("error", resp.text)
            raise RuntimeError(
                f"Model '{target_model}' không hỗ trợ đọc ảnh scan ({err_msg}) và file PDF này là bản scan không có lớp chữ gốc. "
                f"Hãy tải model thị giác bằng lệnh: `ollama pull qwen2.5vl:3b` rồi điền `qwen2.5vl:3b` vào ô ollama_model."
            )

        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def process_ocr_file(self, file_bytes: bytes, filename: str, custom_model: Optional[str] = None) -> dict:
        ext = filename.lower().split(".")[-1]
        pages_markdown: List[str] = []
        used_model = custom_model or self.ocr_model

        if ext == "pdf":
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            total = len(pdf_doc)
            print(f"=== BẮT ĐẦU XỬ LÝ PDF '{filename}' (Tổng số trang: {total}) ===", flush=True)

            for page_index in range(total):
                print(f"[*] Đang xử lý trang {page_index + 1}/{total}...", flush=True)
                page = pdf_doc.load_page(page_index)
                raw_text = page.get_text("text").strip()

                # Hạ DPI xuống 120 và nén JPEG để giảm 75% dung lượng token ảnh, giúp chạy nhanh gấp 5 lần trên CPU/Laptop
                pix = page.get_pixmap(dpi=120)
                img_bytes = pix.tobytes("jpeg")

                page_md = self._ocr_single_image_bytes(
                    img_bytes,
                    custom_model=used_model,
                    fallback_raw_text=raw_text
                )
                pages_markdown.append(f"<!-- Trang {page_index + 1} -->\n{page_md}")
            pdf_doc.close()
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            page_md = self._ocr_single_image_bytes(file_bytes, custom_model=used_model)
            pages_markdown.append(page_md)
        else:
            raise ValueError(f"Định dạng file .{ext} không hỗ trợ OCR trực tiếp.")

        return {
            "filename": filename,
            "total_pages": len(pages_markdown),
            "metadata": {"so_trang": len(pages_markdown)},
            "markdown_content": "\n\n---\n\n".join(pages_markdown),
            "engine": f"Ollama ({used_model})"  # Hoặc f"vLLM ({self.ocr_model})" trong vllm_ocr_service.py
        }

    def create_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        target_model = model or self.embed_model
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": target_model, "input": texts},
            timeout=120
        )
        if resp.status_code == 404:
            raise RuntimeError(f"Model Embedding '{target_model}' chưa có trong Ollama. Hãy chạy: ollama pull {target_model}")
        resp.raise_for_status()
        return resp.json().get("embeddings", [])
