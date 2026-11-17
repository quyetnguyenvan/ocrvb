import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from backend_gateway.config import OCRPipelineConfig
from backend_gateway.services.evidence_extractor import (
    DocumentEvidenceExtractor,
    HeaderEvidence,
)


SYSTEM_PROMPT = (
    "Trích xuất 5 trường văn bản hành chính tiếng Việt từ Header sang JSON: "
    '{"co_quan_ban_hanh":str|null,"so_ky_hieu":str|null,"ngay_ban_hanh":"DD/MM/YYYY"|null,"loai_van_ban":str|null,"trich_yeu":str|null}. '
    "co_quan_ban_hanh: cơ quan trực tiếp ban hành ở [LEFT] (bỏ cơ quan chủ quản cấp trên, viết HOA). "
    "so_ky_hieu: số ở [LEFT] (null nếu khuyết số). ngay_ban_hanh: ngày ở [RIGHT] (null nếu khuyết ngày hoặc tháng). "
    "loai_van_ban: theo tiêu đề [CENTER] hoặc 'Công văn' nếu [CENTER] rỗng. "
    "trich_yeu: nội dung dưới tiêu đề [CENTER], dòng V/v ở [LEFT] (bỏ 'V/v'), hoặc mục đích mời ở [BODY]."
)


class SemanticDocumentExtractor:
    _WARMED_UP: bool = False

    def __init__(self, config: Optional[OCRPipelineConfig] = None):
        self.cfg = config or OCRPipelineConfig()
        self._http = requests.Session()
        if not SemanticDocumentExtractor._WARMED_UP:
            SemanticDocumentExtractor._WARMED_UP = True
            threading.Thread(target=self._warmup_model_async, daemon=True).start()

    def _warmup_model_async(self) -> None:
        """
        Khởi động sẵn model cùng đúng SYSTEM_PROMPT và tham số num_ctx=384, num_thread
        để Ollama lưu sẵn KV-cache của SYSTEM_PROMPT trong RAM.
        """
        try:
            base_url = self.cfg.ollama_base_url.replace(
                "localhost", "127.0.0.1"
            ).rstrip("/")
            cpu_threads = max(4, (os.cpu_count() or 8) - 1)
            self._http.post(
                f"{base_url}/api/chat",
                json={
                    "model": self.cfg.llm_model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": "[LEFT]: []\n[RIGHT]: []\n[CENTER]: []",
                        },
                    ],
                    "format": "json",
                    "stream": False,
                    "keep_alive": self.cfg.keep_alive,
                    "options": {
                        "temperature": 0.0,
                        "top_k": 1,
                        "num_ctx": 384,
                        "num_predict": 16,
                        "num_thread": cpu_threads,
                    },
                },
                timeout=15.0,
            )
        except Exception:
            pass

    @staticmethod
    def _extract_invitation_focus_from_body(body_text: str) -> str:
        """
        Trích xuất chính xác mệnh đề mục đích mời họp/hội thảo/sự kiện từ [BODY] cho Giấy mời,
        ngắt sạch trước phần 'Thời gian:' / 'Địa điểm:'.
        """
        if not body_text:
            return ""
        clean_body = re.sub(r"\s+", " ", body_text).strip()
        m_inv = re.search(
            r"((?:tổ\s+chức|tham\s+dự|tới\s+dự)\s+(?:Hội\s+thảo|Hội\s+nghị|cuộc\s+họp|Lễ\s+[^.;]{5,120})[^.;]{0,220}?)(?=\s*(?:Thời\s+gian|Địa\s+điểm)\b|[.;]|$)",
            clean_body,
            re.I,
        )
        if m_inv:
            res = m_inv.group(1).strip(" .,;")
            res = re.sub(
                r"\bgóp\s+vào\s+dự\s+thảo\b",
                "góp ý vào dự thảo",
                res,
                flags=re.I,
            )
            return res
        return clean_body[:260]

    def build_compact_user_prompt(self, ev: HeaderEvidence) -> str:
        strip_acc = DocumentEvidenceExtractor.strip_accents

        raw_agency: List[str] = []
        so_lines: List[str] = []
        vv_buf: List[str] = []
        for l_line in ev.left_lines:
            l_asc = strip_acc(l_line).strip()
            if re.match(r"^(v/v|ve\s+viec)\b", l_asc) or vv_buf:
                vv_buf.append(l_line.strip())
            elif DocumentEvidenceExtractor.is_so_ky_hieu_line(l_line):
                so_lines.append(l_line.strip())
            else:
                raw_agency.append(l_line.strip())

        direct_agency = (
            DocumentEvidenceExtractor.extract_direct_issuing_agency_lines(
                raw_agency
            )
        )
        compact_left: List[str] = []
        if direct_agency:
            compact_left.append(" ".join(direct_agency))
        compact_left.extend(so_lines)
        if vv_buf:
            compact_left.append(" ".join(vv_buf))

        compact_right = [
            r_line
            for r_line in ev.right_lines
            if not re.search(
                r"(cong\s+hoa.*viet\s+nam|doc\s+lap.*tu\s+do)",
                strip_acc(r_line),
            )
        ] or ev.right_lines[-1:]

        compact_center: List[str] = []
        if ev.center_lines:
            doc_title_asc = strip_acc(ev.center_lines[0].strip())
            compact_center.append(ev.center_lines[0].strip())
            sub_lines = [
                c.strip()
                for c in ev.center_lines[1:]
                if DocumentEvidenceExtractor._is_valid_center_subtitle_line(
                    c, doc_title_asc
                )
            ]
            if sub_lines:
                compact_center.append(" ".join(sub_lines))

        has_clear_subject_in_header = bool(vv_buf or len(compact_center) >= 2)

        parts = [
            f"[LEFT]: {json.dumps(compact_left, ensure_ascii=False)}",
            f"[RIGHT]: {json.dumps(compact_right, ensure_ascii=False)}",
            f"[CENTER]: {json.dumps(compact_center, ensure_ascii=False)}",
        ]
        if not has_clear_subject_in_header and ev.body_snippet:
            focused_body = self._extract_invitation_focus_from_body(ev.body_snippet)
            parts.append(
                f"[BODY]: {json.dumps(focused_body, ensure_ascii=False)}"
            )
        return "\n".join(parts)

    @staticmethod
    def _safe_parse_llm_json(raw_content: str) -> Dict[str, Any]:
        """Phân tích JSON an toàn, tự động phục hồi nếu chuỗi JSON bị cắt cụt do giới hạn token."""
        if not raw_content or not raw_content.strip():
            return {}
        text = raw_content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for suffix in ('"}', '" }', "}", "null}"):
            try:
                return json.loads(text + suffix)
            except json.JSONDecodeError:
                pass

        recovered: Dict[str, Any] = {}
        for key in (
            "co_quan_ban_hanh",
            "so_ky_hieu",
            "ngay_ban_hanh",
            "loai_van_ban",
            "trich_yeu",
        ):
            m_val = re.search(
                rf'"{key}"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(null))', text, re.S
            )
            if m_val:
                recovered[key] = None if m_val.group(2) == "null" else m_val.group(1)
            else:
                m_trunc = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)$', text, re.S)
                if m_trunc:
                    recovered[key] = m_trunc.group(1).strip()
        return recovered

    def extract_5_fields(self, ev: HeaderEvidence) -> Dict[str, Any]:
        t0 = time.perf_counter()
        user_prompt = self.build_compact_user_prompt(ev)

        base_url = self.cfg.ollama_base_url.replace("localhost", "127.0.0.1").rstrip(
            "/"
        )
        cpu_threads = max(4, (os.cpu_count() or 8) - 1)
        payload = {
            "model": self.cfg.llm_model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "num_ctx": 384,
                "num_predict": 100,
                "num_thread": cpu_threads,
            },
        }

        resp = self._http.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=self.cfg.request_timeout_sec,
        )
        resp.raise_for_status()
        body = resp.json()

        raw_content = body.get("message", {}).get("content", "{}")
        parsed = self._safe_parse_llm_json(raw_content)
        ev.timings["llm_json_s"] = round(time.perf_counter() - t0, 2)

        return self._validate_and_flag_uncertainty(parsed, ev)

    @staticmethod
    def _clean_agency_ocr_typos(agency_upper: str) -> str:
        """Chuẩn hóa các lỗi dấu tiếng Việt điển hình của Tesseract trên tên cơ quan IN HOA."""
        s = agency_upper
        s = re.sub(r"^BỌ\s+", "BỘ ", s)
        s = re.sub(r"^BỘ\s+TẾ$", "BỘ Y TẾ", s)
        s = re.sub(r"\bKÉ\s+HOẠCH\b", "KẾ HOẠCH", s)
        s = re.sub(r"\bBAN\s+CHÍ\s+ĐẠO\b", "BAN CHỈ ĐẠO", s)
        s = re.sub(r"\bQUÓC\s+GI[ÁA]\b", "QUỐC GIA", s)
        s = re.sub(r"\bQUỐC\s+GIÁ\b", "QUỐC GIA", s)
        s = re.sub(r"\bHÀI\s+CÓT\b", "HÀI CỐT", s)
        s = re.sub(
            r"^(?:CƠ\s+)?QUAN\s+PHÁT\s+HÀNH\s*:\s*", "", s
        )
        s = re.sub(r"\s+S[ỐÔO]\s*[:.]?$", "", s)
        s = re.sub(
            r"BỘ\s+LAO\s+ĐỘNG\s*[\-–]?\s*THƯƠNG\s+(?:BINH\s+)?VÀ\s+XÃ\s+HỘI",
            "BỘ LAO ĐỘNG - THƯƠNG BINH VÀ XÃ HỘI",
            s,
        )
        s = re.sub(
            r"TẬP\s*ĐOÀN\s*CÔNG\s*NGHIỆP\s*[\-–]?\s*VIỄN\s*THÔNG\s*QUÂN\s*ĐỘI",
            "TẬP ĐOÀN CÔNG NGHIỆP - VIỄN THÔNG QUÂN ĐỘI",
            s,
        )
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r"\s*-\s*", " - ", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _validate_and_flag_uncertainty(
        cls, data: Dict[str, Any], ev: HeaderEvidence
    ) -> Dict[str, Any]:
        strip_acc = DocumentEvidenceExtractor.strip_accents

        # 1. Xác thực Cơ quan ban hành
        agency_lines: List[str] = []
        for line in ev.left_lines:
            ascii_l = strip_acc(line).strip()
            if (
                DocumentEvidenceExtractor.is_so_ky_hieu_line(line)
                or re.search(r"^(v/v|ve\s+viec|kinh\s*gui)\b", ascii_l)
            ):
                break
            if DocumentEvidenceExtractor._SIG_NOISE_ASCII_PAT.search(ascii_l):
                continue
            clean_l = re.sub(
                r"[.\-–\s]*CỘNG\s+H[ÒOÀA]\s+XÃ\s+HỘI.*$", "", line.strip(), flags=re.I
            ).strip(" .;:_~-")
            alpha_only = re.sub(r"[^a-zA-ZÀ-ỹ]", "", clean_l)
            alpha_ascii = strip_acc(alpha_only)
            is_valid_short_prefix = bool(
                alpha_ascii == "bo"
                or re.match(r"^(s[ởỡợớờơ]|v[ụủũúùư]|y)$", alpha_only, re.I)
            )
            if len(alpha_only) >= 3 or is_valid_short_prefix:
                agency_lines.append(clean_l)

        agency_lines = (
            DocumentEvidenceExtractor.extract_direct_issuing_agency_lines(
                agency_lines
            )
        )

        header_agency = (
            cls._clean_agency_ocr_typos(" ".join(agency_lines).upper().strip(" ,"))
            if agency_lines
            else ""
        )

        cq = data.get("co_quan_ban_hanh")
        if header_agency:
            data["co_quan_ban_hanh"] = header_agency
        elif isinstance(cq, str) and cq.strip():
            data["co_quan_ban_hanh"] = cls._clean_agency_ocr_typos(
                cq.strip().upper()
            )
        else:
            data["co_quan_ban_hanh"] = None

        # 2. Xác thực Số, ký hiệu: chỉ lấy từ Left Header
        header_so: Optional[str] = None
        left_has_so_line = False
        for line in ev.left_lines:
            if not DocumentEvidenceExtractor.is_so_ky_hieu_line(line):
                continue
            left_has_so_line = True
            cleaned_line = DocumentEvidenceExtractor.clean_ocr_so_ky_hieu_line(line)
            m_so = re.search(
                r"\bSố\s*[:.]?\s*([0-9]{1,5}\s*/\s*[a-zA-ZĐđ0-9\-]+)",
                cleaned_line,
                re.I,
            )
            if m_so:
                header_so = re.sub(r"\s*/\s*", "/", m_so.group(1).strip())
                break
            m_alt = re.search(
                r"(?:^|\s)([0-9]{1,5}\s*/\s*[a-zA-ZĐđ0-9\-]{2,})", cleaned_line
            )
            if m_alt:
                header_so = re.sub(r"\s*/\s*", "/", m_alt.group(1).strip())
                break

        if header_so:
            data["so_ky_hieu"] = header_so
        elif left_has_so_line:
            data["so_ky_hieu"] = None
        else:
            data["so_ky_hieu"] = None

        # 3. Xác thực Ngày ban hành: chỉ lấy ngày đầy đủ hợp lệ từ Right Header
        header_date: Optional[str] = None
        right_has_incomplete_date = False
        for line in ev.right_lines:
            cleaned_r = DocumentEvidenceExtractor.clean_ocr_date_line(line)
            r_ascii = strip_acc(cleaned_r)
            m_d = re.search(
                r"ngày\s+([0-9]{1,2})\s+tháng\s+([0-9]{1,2})\s+năm\s+(20[0-9]{2})",
                cleaned_r,
                re.I,
            )
            if m_d:
                hd, hm, hy = int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3))
                if 1 <= hd <= 31 and 1 <= hm <= 12:
                    header_date = f"{hd:02d}/{hm:02d}/{hy:04d}"
                    break
            elif re.search(r"\b(ngay|thang|nam\s+20[0-9]{2})\b", r_ascii):
                right_has_incomplete_date = True

        ngay = data.get("ngay_ban_hanh")
        parsed_date: Optional[str] = None
        if (
            not header_date
            and not right_has_incomplete_date
            and isinstance(ngay, str)
            and ngay.strip()
        ):
            m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](20\d{2})", ngay.strip())
            if m:
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1 <= d <= 31 and 1 <= mo <= 12:
                    cand_date = f"{d:02d}/{mo:02d}/{y:04d}"
                    if cand_date in " ".join(ev.right_lines):
                        parsed_date = cand_date
        data["ngay_ban_hanh"] = header_date or parsed_date

        # 4. Xác thực Loại văn bản
        center_first_ascii = (
            strip_acc(ev.center_lines[0]).strip() if ev.center_lines else ""
        )
        so_ascii = strip_acc(data.get("so_ky_hieu") or "")

        if center_first_ascii.startswith("to trinh") or "/ttr-" in so_ascii:
            data["loai_van_ban"] = "Tờ trình"
        elif center_first_ascii.startswith("bao cao") or "/bc-" in so_ascii:
            data["loai_van_ban"] = "Báo cáo"
        elif center_first_ascii.startswith("giay moi") or "/gm-" in so_ascii:
            data["loai_van_ban"] = "Giấy mời"
        elif center_first_ascii.startswith("quyet dinh") or "/qd-" in so_ascii:
            data["loai_van_ban"] = "Quyết định"
        elif center_first_ascii.startswith("thong bao") or "/tb-" in so_ascii:
            data["loai_van_ban"] = "Thông báo"
        elif center_first_ascii.startswith("ke hoach") or "/kh-" in so_ascii:
            data["loai_van_ban"] = "Kế hoạch"
        else:
            data["loai_van_ban"] = "Công văn"

        # 5. Xác thực Trích yếu
        full_center_subj = ""
        if len(ev.center_lines) >= 2:
            subj_lines: List[str] = []
            for cline in ev.center_lines[1:]:
                if not DocumentEvidenceExtractor._is_valid_center_subtitle_line(
                    cline, center_first_ascii
                ):
                    continue
                subj_lines.append(cline.strip())
            full_center_subj = " ".join(subj_lines)
            full_center_subj = re.sub(
                r"^(v/v|v/y|về việc|ve viec)\s*[:.]?\s*",
                "",
                full_center_subj,
                flags=re.I,
            ).strip()

        left_vv_subj = ""
        vv_started = False
        vv_parts: List[str] = []
        for l_line in ev.left_lines:
            l_ascii = strip_acc(l_line).strip()
            if re.match(r"^(v/v|ve\s+viec)\b", l_ascii):
                vv_started = True
                vv_parts.append(l_line.strip())
            elif vv_started:
                if re.search(r"trân\s+trọng\b", l_line, re.I):
                    break
                vv_parts.append(l_line.strip())
        if vv_parts:
            left_vv_subj = " ".join(vv_parts)
            left_vv_subj = re.sub(
                r"^(v/v|v/y|về việc|ve viec)\s*[:.]?\s*",
                "",
                left_vv_subj,
                flags=re.I,
            ).strip()

        trich_yeu = data.get("trich_yeu")
        if full_center_subj:
            data["trich_yeu"] = full_center_subj
        elif left_vv_subj:
            data["trich_yeu"] = left_vv_subj
        else:
            body_focus = cls._extract_invitation_focus_from_body(ev.body_snippet or "")
            if body_focus and data.get("loai_van_ban") == "Giấy mời":
                data["trich_yeu"] = body_focus
            elif (
                isinstance(trich_yeu, str)
                and len(re.findall(r"[a-zA-ZÀ-ỹ]{2,}", trich_yeu)) >= 2
                and not re.search(
                    r"(kính\s*gửi|kinh\s*g[uuaoae]|thời\s*gian\s*:|\n)",
                    trich_yeu,
                    re.I,
                )
            ):
                trich_yeu = re.sub(
                    r"^(v/v|v/y|về việc|ve viec)\s*[:.]?\s*",
                    "",
                    trich_yeu.strip(),
                    flags=re.I,
                )
                data["trich_yeu"] = trich_yeu
            elif body_focus:
                data["trich_yeu"] = body_focus
            else:
                data["trich_yeu"] = None

        if isinstance(data.get("trich_yeu"), str):
            data["trich_yeu"] = re.sub(
                r"\bdi\s+chuyên\b", "di chuyển", data["trich_yeu"]
            )

        missing_fields = [
            k
            for k in (
                "co_quan_ban_hanh",
                "so_ky_hieu",
                "ngay_ban_hanh",
                "loai_van_ban",
                "trich_yeu",
            )
            if not data.get(k)
        ]
        data["_meta"] = {
            "path_used": "PATH_C_REVIEW" if missing_fields else ev.path_used,
            "uncertain_fields": missing_fields,
            "timings": ev.timings,
        }
        return data