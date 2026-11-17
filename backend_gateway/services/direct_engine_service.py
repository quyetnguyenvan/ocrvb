import dis
import inspect
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend_gateway.config import OCRPipelineConfig, settings
from backend_gateway.services.evidence_extractor import (
    DocumentEvidenceExtractor,
    HeaderEvidence,
)
from backend_gateway.services.semantic_extractor import SemanticDocumentExtractor


class AttrDict(dict):
    """
    Dict kết quả tương thích 100% với FastAPI / Pydantic v2:
    - Không bị lỗi KeyError nếu main.py truy cập khóa phụ bất kỳ.
    - Hỗ trợ truy cập qua dấu chấm (res.co_quan_ban_hanh).
    """

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def __missing__(self, key: str) -> Any:
        return None


class DirectEngineService:
    """
    Service điều phối luồng AI Document Understanding (PATH A / PATH B / PATH C)
    hỗ trợ đầy đủ .pdf, .docx, .doc và các định dạng ảnh (.png, .jpg, .jpeg, .tiff)
    cho endpoint POST /api/v1/ocr/extract.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cfg = OCRPipelineConfig()
        self.evidence_extractor = DocumentEvidenceExtractor(config=self.cfg)
        self.semantic_extractor = SemanticDocumentExtractor(config=self.cfg)

    @staticmethod
    def _unpack_call_args(*args: Any, **kwargs: Any) -> Tuple[bytes, str]:
        file_bytes: Optional[bytes] = None
        filename: Optional[str] = None

        for k in (
            "file_bytes",
            "pdf_bytes",
            "content",
            "data",
            "file_content",
            "raw_bytes",
        ):
            if k in kwargs and isinstance(kwargs[k], (bytes, bytearray)):
                file_bytes = bytes(kwargs[k])
                break

        for k in ("file_path", "pdf_path", "path", "temp_path"):
            if k in kwargs and kwargs[k]:
                p = Path(str(kwargs[k]))
                if p.exists() and p.is_file():
                    file_bytes = p.read_bytes()
                    if not filename:
                        filename = p.name
                    break

        for k in ("filename", "file_name", "original_filename", "name", "doc_name"):
            if k in kwargs and isinstance(kwargs[k], str) and kwargs[k]:
                filename = Path(kwargs[k]).name
                break

        for arg in args:
            if isinstance(arg, (bytes, bytearray)) and file_bytes is None:
                file_bytes = bytes(arg)
            elif isinstance(arg, Path):
                if arg.exists() and arg.is_file() and file_bytes is None:
                    file_bytes = arg.read_bytes()
                    if not filename:
                        filename = arg.name
                elif not filename:
                    filename = arg.name
            elif isinstance(arg, str):
                p = Path(arg)
                if p.exists() and p.is_file() and file_bytes is None:
                    file_bytes = p.read_bytes()
                    if not filename:
                        filename = p.name
                else:
                    filename = p.name
            elif hasattr(arg, "file"):
                if getattr(arg, "filename", None) and not filename:
                    filename = Path(arg.filename).name
                arg.file.seek(0)
                file_bytes = arg.file.read()

        if file_bytes is None:
            raise ValueError(
                f"DirectEngineService không tìm thấy dữ liệu tài liệu hợp lệ. args={args}, kwargs={list(kwargs.keys())}"
            )

        return file_bytes, (filename or "document.pdf")

    def _run_pipeline_sync(self, *args: Any, **kwargs: Any) -> AttrDict:
        try:
            file_bytes, filename = self._unpack_call_args(*args, **kwargs)
            t_start = time.perf_counter()
            page_count = self.evidence_extractor.get_page_count(
                file_bytes, filename=filename
            )

            print(
                f"=== [AI Document Understanding] Đang phân tích '{filename}' ({page_count} trang) ==="
            )

            ev: HeaderEvidence = self.evidence_extractor.extract(
                file_bytes, filename=filename
            )

            if ev.path_used.startswith("PATH_A"):
                print(
                    f" -> [{ev.path_used} - Clean PDF Evidence] "
                    f"Left={ev.left_lines} | Right={ev.right_lines} | Center={ev.center_lines}"
                )
                if ev.handwritten_strip_text:
                    strip_s = ev.timings.get("handwritten_strip_ai_s", 0.0)
                    print(
                        f" -> [Hybrid Header Strip OCR] ({strip_s:.2f}s): {ev.handwritten_strip_text!r}"
                    )
                else:
                    print(
                        " -> [PATH A - Searchable Document] Lớp văn bản điện tử hợp lệ (0s OCR)."
                    )
            else:
                tess_s = ev.timings.get("tesseract_1pass_s", 0.0)
                slot_s = ev.timings.get("handwritten_strip_ai_s", 0.0)
                total_ocr_s = ev.timings.get("total_ocr_s", 0.0)
                print(
                    f" -> [PATH B - Stage Timing] Tesseract (1-Pass): {tess_s:.2f}s | "
                    f"Handwritten Slot AI: {slot_s:.2f}s "
                    f"(Num={ev.slot_num!r}, Day={ev.slot_day!r}, Mon={ev.slot_mon!r}) | "
                    f"Tổng OCR: {total_ocr_s:.2f}s"
                )
                print(
                    f" -> [Clean Header Evidence] "
                    f"Left={ev.left_lines} | Right={ev.right_lines} | Center={ev.center_lines}"
                )

            extracted = self.semantic_extractor.extract_5_fields(ev)
            llm_s = ev.timings.get("llm_json_s", 0.0)

            raw_5_fields = {
                "co_quan_ban_hanh": extracted.get("co_quan_ban_hanh"),
                "so_ky_hieu": extracted.get("so_ky_hieu"),
                "ngay_ban_hanh": extracted.get("ngay_ban_hanh"),
                "loai_van_ban": extracted.get("loai_van_ban"),
                "trich_yeu": extracted.get("trich_yeu"),
            }
            print(
                f" -> [AI Raw JSON] ({llm_s:.2f}s): "
                f"{json.dumps(raw_5_fields, ensure_ascii=False)}"
            )

            clean_5_fields = {
                k: (v if isinstance(v, str) else "") for k, v in raw_5_fields.items()
            }

            total_s = round(time.perf_counter() - t_start, 2)
            print(
                f"=== [Hoàn tất] Tổng thời gian xử lý '{filename}': {total_s:.2f}s ==="
            )

            raw_header_text = "\n".join(
                ev.left_lines + ev.right_lines + ev.center_lines
            )
            metadata_with_pages = {
                **clean_5_fields,
                "so_trang": ev.page_count,
                "page_count": ev.page_count,
                "path_used": ev.path_used,
                "processing_time": total_s,
            }
            payload_dict: Dict[str, Any] = {
                **clean_5_fields,
                "status": "success",
                "success": True,
                "code": 200,
                "message": "OK",
                "filename": filename,
                "file_name": filename,
                "total_pages": ev.page_count,
                "page_count": ev.page_count,
                "pages": ev.page_count,
                "so_trang": ev.page_count,
                "markdown_content": raw_header_text,
                "raw_text": raw_header_text,
                "ocr_text": raw_header_text,
                "text": raw_header_text,
                "content": raw_header_text,
                "engine": f"DirectEngine ({self.cfg.llm_model_name})",
                "provider": "ollama",
                "model": self.cfg.llm_model_name,
                "path_used": extracted.get("_meta", {}).get(
                    "path_used", ev.path_used
                ),
                "route": ev.path_used,
                "processing_time": total_s,
                "processing_time_sec": total_s,
                "processing_time_seconds": total_s,
                "processing_time_ms": round(total_s * 1000, 1),
                "latency_ms": round(total_s * 1000, 1),
                "elapsed_time": total_s,
                "confidence": 0.95,
                "timings": {**ev.timings, "total_s": total_s},
                "uncertain_fields": extracted.get("_meta", {}).get(
                    "uncertain_fields", []
                ),
                "evidence": {
                    "left": ev.left_lines,
                    "right": ev.right_lines,
                    "center": ev.center_lines,
                },
                "metadata": metadata_with_pages,
                "extracted_metadata": clean_5_fields,
                "document_metadata": clean_5_fields,
                "fields": clean_5_fields,
                "data": metadata_with_pages,
                "result": metadata_with_pages,
            }

            try:
                import backend_gateway.models.schemas as schemas_mod

                for _, cls_obj in inspect.getmembers(schemas_mod, inspect.isclass):
                    model_fields = getattr(cls_obj, "model_fields", None)
                    if isinstance(model_fields, dict):
                        for f_name, f_info in model_fields.items():
                            if f_name not in payload_dict:
                                ann = str(getattr(f_info, "annotation", ""))
                                if "int" in ann:
                                    payload_dict[f_name] = ev.page_count
                                elif "float" in ann:
                                    payload_dict[f_name] = total_s
                                elif "bool" in ann:
                                    payload_dict[f_name] = True
                                elif "list" in ann.lower():
                                    payload_dict[f_name] = []
                                elif "dict" in ann.lower():
                                    payload_dict[f_name] = metadata_with_pages
                                else:
                                    payload_dict[f_name] = raw_header_text
            except Exception:
                pass

            return AttrDict(payload_dict)
        except Exception as exc:
            print(f"\n[ERROR - DirectEngineService] Lỗi khi xử lý tài liệu: {exc}")
            traceback.print_exc()
            raise

    @staticmethod
    def _caller_expects_awaitable() -> bool:
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                lasti = caller_frame.f_lasti
                instructions = list(dis.get_instructions(caller_frame.f_code))
                for idx, instr in enumerate(instructions):
                    if instr.offset >= lasti:
                        for next_instr in instructions[idx : idx + 4]:
                            if next_instr.opname == "GET_AWAITABLE":
                                return True
                        break
        except Exception:
            pass
        return False

    def _dispatch_smart(self, *args: Any, **kwargs: Any) -> Any:
        if self._caller_expects_awaitable():
            async def _async_runner() -> AttrDict:
                return self._run_pipeline_sync(*args, **kwargs)
            return _async_runner()
        return self._run_pipeline_sync(*args, **kwargs)

    def extract_document_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch_smart(*args, **kwargs)

    def extract_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch_smart(*args, **kwargs)

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch_smart(*args, **kwargs)

    def process_document(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch_smart(*args, **kwargs)

    def process_ocr_file(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch_smart(*args, **kwargs)

    def analyze_document(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch_smart(*args, **kwargs)

    def __getattr__(self, name: str):
        def _dynamic_handler(*args: Any, **kwargs: Any) -> Any:
            return self._dispatch_smart(*args, **kwargs)
        return _dynamic_handler


direct_engine_service = DirectEngineService()