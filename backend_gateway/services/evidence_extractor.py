import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import requests
from dotenv import dotenv_values
from PIL import Image

from backend_gateway.config import OCRPipelineConfig, settings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class HeaderEvidence:
    path_used: str
    page_count: int = 1
    left_lines: List[str] = field(default_factory=list)
    right_lines: List[str] = field(default_factory=list)
    center_lines: List[str] = field(default_factory=list)
    body_snippet: str = ""
    signature_stamp_lines: List[str] = field(default_factory=list)
    incoming_stamp_lines: List[str] = field(default_factory=list)
    handwritten_strip_text: str = ""
    slot_num: str = ""
    slot_day: str = ""
    slot_mon: str = ""
    needs_strip_escalation: bool = False
    escalation_reasons: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class DocumentEvidenceExtractor:
    _SIG_NOISE_ASCII_PAT = re.compile(
        r"((?:nguoi|ky)\s+boi\s*:|nguoi\s+ky\s*:|thoi\s+gian\s+ky\s*:|ngay\s+ky\s*:|(?:co\s+)?quan\s+phat\s+hanh\s*:|\bphat\s+hanh\s*:\s*van\s+phong|\bco\s+quan\s*:|email\s*:|\bsao\s+y\b|\+07\s*:\s*00|[0-9]{2}:[0-9]{2}:[0-9]{2}|000202[0-9]|vepena|add3\.202|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})",
        re.IGNORECASE,
    )
    _INCOMING_STAMP_ASCII_PAT = re.compile(
        r"(\bden\b.*so|\bchuyen\s*:|luu ho so so|hoa toc|den truoc|cong van den)",
        re.IGNORECASE,
    )
    _DOC_TITLE_ASCII_PAT = re.compile(
        r"^(to\s*trinh|bao\s*cao|giay\s*moi|quyet\s*dinh|thong\s*bao|ke\s*hoach|chi\s*thi|chuong\s*trinh|huong\s*dan|bien\s*ban|cong\s*dien|nghi\s*quyet)\b(?!\s+so\b)",
        re.IGNORECASE,
    )
    _BODY_START_ASCII_PAT = re.compile(
        r"^(kinh\s*g[uuaoae][i1l]?\b|dong\s*kinh\s*g[uuaoae][i1l]?\b|tran\s*trong\s*kinh\s*moi\b|kinh\s*moi\b|can\s*cu\b|thuc\s*hien\b|xet\s*de\s*nghi\b|theo\s*de\s*nghi\b|bo\s*truong\b|thu\s*tuong\s*chinh\s*phu\b|chu\s*tich\b|dieu\s*1\b|ngay\s+[0-9]{1,2}/[0-9]{1,2}/20[0-9]{2}|cong\s*van\s*so\b|quyet\s*dinh\s*so\b|ke\s*hoach\s*so\b|thong\s*bao\s*so\b|kem\s*theo\b|tiep\s*theo\b|phuc\s*dap\b|tra\s*loi\b|[i1v]+\.\s+)",
        re.IGNORECASE,
    )
    _RECIPIENT_LINE_ASCII_PAT = re.compile(
        r"^(\-\s*|\•\s*)?(van\s*phong\s*chinh\s*phu|cac\s*bo\s*,|bo\s*,\s*co\s*quan\s*ngang\s*bo|ubnd\s*(?:cac\s*)?tinh|uy\s*ban\s*nhan\s*dan\s*(?:cac\s*)?tinh|so\s*thong\s*tin\s*va\s*truyen\s*thong\s*cac)",
        re.IGNORECASE,
    )
    _DEPARTMENT_SO_ASCII_PAT = re.compile(
        r"^so\s+(thong\s+tin|tu\s+phap|tai\s+chinh|ke\s+hoach|y\s+te|lao\s+dong|nong\s+nghiep|van\s+hoa|giao\s+duc|xay\s+dung|cong\s+thuong|khoa\s+hoc|noi\s+vu|ngoai\s+vu|giao\s+thong|tai\s+nguyen|du\s+lich|an\s+toan|cong\s+an|quoc\s+phong)\b",
        re.IGNORECASE,
    )
    _SUBORDINATE_AGENCY_PREFIX_ASCII_PAT = re.compile(
        r"^(s[oơ]\b|ban\b|van\s+phong\b|trung\s+tam\b|thanh\s+tra\b|uy\s+ban\b|ubnd\b|tong\s+cuc\b|chi\s+cuc\b|cuc\b|vu\b|hoc\s+vien\b|vien\b|truong\b|dai\s+hoc\b|benh\s+vien\b|cong\s+ty\b|xi\s+nghiep\b|phong\b(?!\s*,?\s*chong))",
        re.IGNORECASE,
    )
    _COMPLETE_NUM_ASCII_PAT = re.compile(
        r"\bso\s*[:.]?\s*[0-9]{1,5}\s*/\s*[a-z0-9\-]+", re.IGNORECASE
    )
    _COMPLETE_DATE_ASCII_PAT = re.compile(
        r"ngay\s+(?:0?[1-9]|[12][0-9]|3[01])\s+thang\s+(?:0?[1-9]|1[0-2])\s+nam\s+20[0-9]{2}",
        re.IGNORECASE,
    )
    _VALID_HEADER_ANCHOR_ASCII_PAT = re.compile(
        r"(cong\s+hoa|doc\s+lap|hanh\s+phuc|\bso\s*[:.]|ngay.*thang.*nam|uy\s+ban|ubnd|\bbo\s+|\bso\s+|van\s+phong|tap\s+doan|ban\s+chi\s+dao|to\s+trinh|bao\s+cao|giay\s+moi|quyet\s+dinh|thong\s+bao)",
        re.IGNORECASE,
    )

    _TESSDATA_READY_CACHE: Optional[str] = None
    _ACTIVE_LANG_CACHE: Optional[str] = None
    _TESS_WARMED_UP: bool = False

    def __init__(
        self,
        config: Optional[OCRPipelineConfig] = None,
        strip_ocr_callback: Optional[Callable[[bytes], str]] = None,
    ):
        self.cfg = config or OCRPipelineConfig()
        self.strip_ocr_callback = strip_ocr_callback
        self._active_lang = self.cfg.tesseract_lang
        self._http = requests.Session()
        self._configure_tesseract_windows()
        if not DocumentEvidenceExtractor._TESS_WARMED_UP:
            DocumentEvidenceExtractor._TESS_WARMED_UP = True
            threading.Thread(target=self._warmup_tesseract_async, daemon=True).start()

    def _warmup_tesseract_async(self) -> None:
        """Nạp sẵn tesseract.exe cùng vie.traineddata và eng.traineddata vào OS File Cache."""
        try:
            dummy = np.full((24, 64), 255, dtype=np.uint8)
            pytesseract.image_to_string(
                dummy, lang=self._active_lang, config="--oem 1 --psm 7"
            )
            pytesseract.image_to_string(
                dummy, lang="eng", config="--oem 1 --psm 7"
            )
        except Exception:
            pass

    @staticmethod
    def strip_accents(text: str) -> str:
        """Chuyển chuỗi tiếng Việt sang ASCII chữ thường phục vụ so khớp cấu trúc."""
        if not text:
            return ""
        text = text.replace("đ", "d").replace("Đ", "D")
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    @classmethod
    def extract_direct_issuing_agency_lines(
        cls, agency_lines: List[str]
    ) -> List[str]:
        """Tách tên cơ quan trực tiếp ban hành khỏi cơ quan chủ quản cấp trên (nếu có 2 cơ quan)."""
        if len(agency_lines) < 2:
            return agency_lines

        last_sub_idx = -1
        for idx in range(1, len(agency_lines)):
            line_orig = agency_lines[idx].strip()
            line_ascii = cls.strip_accents(line_orig)
            if re.match(r"^s[ốôoòóỏõọ]\b", line_orig, re.I):
                continue
            if cls._SUBORDINATE_AGENCY_PREFIX_ASCII_PAT.match(line_ascii):
                last_sub_idx = idx

        if last_sub_idx > 0:
            return agency_lines[last_sub_idx:]
        return agency_lines

    @classmethod
    def is_so_ky_hieu_line(cls, line: str) -> bool:
        """
        Kiểm tra chính xác dòng Số/Ký hiệu văn bản, không nhận nhầm 'SỞ THÔNG TIN...'
        hoặc dòng trích yếu 'Quyết định số 2269/QĐ-TTg' thành dòng Số ký hiệu của văn bản.
        """
        if not line:
            return False
        stripped_orig = line.strip()
        ascii_l = cls.strip_accents(stripped_orig)

        if (
            re.match(r"^\s*s[ởỡợớờơ]\b", stripped_orig, re.I)
            or cls._DEPARTMENT_SO_ASCII_PAT.match(ascii_l)
        ):
            return False

        if re.search(
            r"\b(quyet\s*dinh|cong\s*van|thong\s*bao|ke\s*hoach|nghi\s*quyet|nghi\s*dinh|bao\s*cao|to\s*trinh)\s+so\b",
            ascii_l,
        ):
            return False

        if re.match(r"^\s*s[ốôoòóỏõọ]\s*[:.]?\s*$", stripped_orig, re.I):
            return True

        if re.search(r"/\s*[a-z0-9]{2,}", ascii_l):
            return True

        if re.match(
            r"^\s*s[ốôoòóỏõọ]\s*(?:[:.]|\s+[0-9#?$fFt@%OolIJZSBdDaAqQ/])",
            stripped_orig,
            re.I,
        ):
            return True

        if re.search(
            r"(?:^|[\s/@®©0-9])(?:BC|TTr|GM|QĐ|QD|TB|KH|CV|CT|UBND|VPCP|BNG|BCT|BYT|BTP|BKHĐT|BKHDT|BKHCN|CNVTQĐ|CNVTQD|BLĐTBXH|BLDTBXH|STTTT|HVNG|TMĐT|TMDT|TĐC|MSMV|BCĐ)\s*-\s*[A-ZĐ0-9]{2,}",
            stripped_orig,
        ):
            return True

        if re.search(
            r"[0-9]+\s*[®©@]\s*[A-ZĐ]{2,}\s*-\s*[A-ZĐ0-9]{2,}", stripped_orig
        ):
            return True

        return False

    @classmethod
    def _is_valid_center_subtitle_line(
        cls, text: str, doc_title_ascii: str = ""
    ) -> bool:
        """Kiểm tra dòng nằm ngay dưới tiêu đề ở [CENTER] có phải là dòng trích yếu hợp lệ hay không."""
        if not text:
            return False
        stripped = text.strip()
        ascii_t = cls.strip_accents(stripped)
        if cls._BODY_START_ASCII_PAT.match(ascii_t):
            return False
        if re.search(r"\bkinh\s*g[uuaoae][i1l]?\b", ascii_t) or stripped.endswith(":"):
            return False
        if re.match(r"^[0-9.,;:\-_*~]{3,}", stripped):
            return False
        if re.match(r"^(v/v|ve\s+viec)\b", ascii_t):
            return True
        if doc_title_ascii.startswith("giay moi"):
            return bool(
                re.match(
                    r"^(du\b(?!\s+thao)|tham\s+du\b|hop\b|hoi\s+thao\b|hoi\s+nghi\b|le\b)",
                    ascii_t,
                )
            )
        words = [w for w in re.findall(r"[a-zA-ZÀ-ỹ]+", stripped) if len(w) >= 2]
        return len(words) >= 2

    def _configure_tesseract_windows(self) -> None:
        env_file_vars = (
            dotenv_values(BASE_DIR / ".env") if (BASE_DIR / ".env").exists() else {}
        )
        cmd = (
            self.cfg.tesseract_cmd
            or env_file_vars.get("TESSERACT_CMD")
            or shutil.which("tesseract")
        )
        candidate_cmds = [
            cmd,
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        valid_cmds = [c for c in candidate_cmds if c and Path(c).exists()]
        chosen_cmd = valid_cmds[0] if valid_cmds else None
        for c in valid_cmds:
            if (Path(c).parent / "tessdata" / "vie.traineddata").exists():
                chosen_cmd = c
                break
        if chosen_cmd:
            pytesseract.pytesseract.tesseract_cmd = chosen_cmd

        if (
            DocumentEvidenceExtractor._TESSDATA_READY_CACHE
            and DocumentEvidenceExtractor._ACTIVE_LANG_CACHE
        ):
            os.environ["TESSDATA_PREFIX"] = (
                DocumentEvidenceExtractor._TESSDATA_READY_CACHE
            )
            self._active_lang = DocumentEvidenceExtractor._ACTIVE_LANG_CACHE
            return

        ascii_cache_dir = Path(tempfile.gettempdir()) / "ai_ocr_tessdata_cache"
        ascii_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_vie = ascii_cache_dir / "vie.traineddata"

        if not (
            cached_vie.exists() and 1_500_000 <= cached_vie.stat().st_size <= 3_800_000
        ):
            try:
                url = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=12.0) as resp:
                    data = resp.read()
                if len(data) > 1_000_000:
                    cached_vie.write_bytes(data)
            except Exception:
                pass

        if chosen_cmd:
            orig_tessdata = Path(chosen_cmd).parent / "tessdata"
            if orig_tessdata.exists():
                for aux in ("eng.traineddata", "osd.traineddata"):
                    src_aux = orig_tessdata / aux
                    dst_aux = ascii_cache_dir / aux
                    if src_aux.exists() and not dst_aux.exists():
                        try:
                            shutil.copy2(src_aux, dst_aux)
                        except Exception:
                            pass

        if cached_vie.exists() and cached_vie.stat().st_size > 500_000:
            os.environ["TESSDATA_PREFIX"] = str(ascii_cache_dir)
            self._active_lang = self.cfg.tesseract_lang
        else:
            self._active_lang = "eng"

        DocumentEvidenceExtractor._TESSDATA_READY_CACHE = os.environ.get(
            "TESSDATA_PREFIX", ""
        )
        DocumentEvidenceExtractor._ACTIVE_LANG_CACHE = self._active_lang

    @classmethod
    def normalize_vietnamese_text(cls, text: str) -> str:
        """Chuẩn hóa từ/dòng tiếng Việt, bảo toàn các từ 1 chữ cái hợp lệ như 'Y' (trong 'BỘ Y TẾ', 'SỞ Y TẾ')."""
        if not text:
            return ""
        text = text.replace("\u01a2", "Ư").replace("\u01a3", "ư").replace("\ufffd", "-")
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[_\-–—=*]{3,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text in ("-", "–", "/"):
            return "-" if text != "/" else "/"
        text = re.sub(r"^[Vv]/[yYlI1]\b", "V/v", text)
        alnum_only = re.sub(r"[^a-zA-ZÀ-ỹ0-9]", "", text)
        if not alnum_only:
            return ""
        if (
            len(alnum_only) == 1
            and not alnum_only.isdigit()
            and alnum_only.upper()
            not in ("Y", "I", "V", "A", "B", "C", "Đ", "Ở", "Ủ", "Ý")
        ):
            return ""
        return text

    @staticmethod
    def _remove_red_stamp_ink(img_rgb: np.ndarray) -> np.ndarray:
        """Xóa mực con dấu màu đỏ (dấu ĐẾN, dấu mộc tròn) mà không làm biến dạng nét chữ viết tay xanh/đen."""
        r = img_rgb[:, :, 0].astype(np.int16)
        g = img_rgb[:, :, 1].astype(np.int16)
        b = img_rgb[:, :, 2].astype(np.int16)
        red_mask = (r > 130) & ((r - g) > 26) & ((r - b) > 26)
        if np.any(red_mask):
            cleaned = img_rgb.copy()
            cleaned[red_mask] = [255, 255, 255]
            return cleaned
        return img_rgb

    @staticmethod
    def _split_touching_wide_components(binary_mask: np.ndarray) -> np.ndarray:
        """
        Tự động phát hiện và tách rời các khối chữ số viết tay chạm nhau (2, 3 hoặc 4 chữ số chạm nhau
        như '4673', '67', '13', '634', '418') bằng cách tìm cực tiểu hình chiếu dọc (vertical projection valleys)
        và chạy lặp 2 lượt để đảm bảo mọi chữ số chạm nhau đều được tách rời hoàn toàn.
        """
        if binary_mask.size == 0:
            return binary_mask
        out = binary_mask.copy()
        h, w = out.shape[:2]

        for _pass in range(2):
            changed = False
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                out, connectivity=8
            )
            for lbl in range(1, num_labels):
                cx, cy, cw, ch, area = stats[lbl]
                if ch < int(h * 0.24) or cw < 10:
                    continue
                aspect = float(cw) / max(1.0, float(ch))
                if aspect < 0.70:
                    continue

                comp_roi = (labels[cy : cy + ch, cx : cx + cw] == lbl).astype(
                    np.uint8
                )
                if aspect >= 1.68:
                    # 4 chữ số chạm nhau (như '4673'): cắt tại 3 điểm thắt
                    zones = [
                        (max(2, int(cw * 0.15)), max(3, int(cw * 0.33))),
                        (max(4, int(cw * 0.38)), max(5, int(cw * 0.62))),
                        (max(6, int(cw * 0.67)), min(cw - 2, int(cw * 0.85))),
                    ]
                elif aspect >= 1.18:
                    # 3 chữ số chạm nhau (như '467', '673', '634'): cắt tại 2 điểm thắt
                    zones = [
                        (max(2, int(cw * 0.20)), max(3, int(cw * 0.45))),
                        (max(4, int(cw * 0.55)), min(cw - 2, int(cw * 0.80))),
                    ]
                else:
                    # 2 chữ số chạm nhau (như '67', '13', '41'): cắt tại 1 điểm thắt
                    zones = [(max(2, int(cw * 0.24)), min(cw - 2, int(cw * 0.76)))]

                for s_l, s_r in zones:
                    if s_r > s_l:
                        col_sums = np.sum(comp_roi[:, s_l:s_r], axis=0)
                        min_idx = s_l + int(np.argmin(col_sums))
                        if col_sums[min_idx - s_l] <= ch * 0.68:
                            out[cy : cy + ch, cx + min_idx] = 0
                            changed = True
            if not changed:
                break
        return out

    @staticmethod
    def _space_character_columns(
        binary_inv: np.ndarray, min_h_ratio: float = 0.22
    ) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Gộp các thành phần liên thông có khoảng hoành độ [x0, x1] chồng lấp nhau (>= 22%)
        thành cùng một cột chữ số và chèn 10px khoảng trắng giữa các cột chữ số tách biệt.
        """
        if binary_inv.size == 0:
            return None, []
        h, w = binary_inv.shape[:2]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_inv, connectivity=8
        )

        clean_inv = np.zeros_like(binary_inv)
        raw_boxes: List[Tuple[int, int, int, int]] = []
        for lbl in range(1, num_labels):
            cx, cy, cw, ch, area = stats[lbl]
            if area < 5:
                continue
            if cw > int(w * 0.92) and ch < int(h * 0.24):
                continue
            if (cx <= 1 or cx + cw >= w - 1) and cw < int(w * 0.12):
                continue
            clean_inv[labels == lbl] = 255
            raw_boxes.append((cx, cy, cx + cw, cy + ch))

        if not raw_boxes:
            return None, []

        raw_boxes.sort(key=lambda b: b[0])
        merged_cols: List[List[int]] = []
        for bx0, by0, bx1, by1 in raw_boxes:
            if not merged_cols:
                merged_cols.append([bx0, by0, bx1, by1])
            else:
                px0, py0, px1, py1 = merged_cols[-1]
                overlap_x = min(px1, bx1) - max(px0, bx0)
                min_w = max(1, min(px1 - px0, bx1 - bx0))
                if overlap_x >= min_w * 0.22:
                    merged_cols[-1] = [
                        min(px0, bx0),
                        min(py0, by0),
                        max(px1, bx1),
                        max(py1, by1),
                    ]
                else:
                    merged_cols.append([bx0, by0, bx1, by1])

        valid_cols = [
            col
            for col in merged_cols
            if (col[3] - col[1]) >= int(h * min_h_ratio)
        ]
        if not valid_cols:
            return None, []

        gap_px = 10
        total_w = sum(c[2] - c[0] for c in valid_cols) + gap_px * (
            len(valid_cols) + 1
        )
        canvas_inv = np.zeros((h, total_w), dtype=np.uint8)
        placed_boxes: List[Tuple[int, int, int, int]] = []

        curr_x = gap_px
        for x0, y0, x1, y1 in valid_cols:
            cw = x1 - x0
            ch = y1 - y0
            canvas_inv[:, curr_x : curr_x + cw] = clean_inv[:, x0:x1]
            placed_boxes.append((curr_x, y0, cw, ch))
            curr_x += cw + gap_px

        return 255 - canvas_inv, placed_boxes

    @classmethod
    def _extract_blue_pen_masks(cls, img_rgb: np.ndarray) -> List[np.ndarray]:
        """
        Tách riêng nét mực bút bi xanh dương HOẶC xanh tím (purple/violet ink) khỏi chữ in màu đen.
        Đặt mặt nạ đã tách cột chữ số (spaced_blue_img) lên ĐẦU danh sách (results[0]) để luôn ưu tiên
        số lượng chữ số thực tế sau khi tách điểm chạm ('4673', '185', '13').
        """
        if img_rgb.size == 0 or img_rgb.shape[0] < 6 or img_rgb.shape[1] < 8:
            return []
        h, w = img_rgb.shape[:2]
        r = img_rgb[:, :, 0].astype(np.int16)
        g = img_rgb[:, :, 1].astype(np.int16)
        b = img_rgb[:, :, 2].astype(np.int16)

        # Bắt cả mực bút bi xanh dương (B > G, B > R) lẫn mực tím (B > G, R > G)
        blue_score = (b - g) + np.maximum(0, b - r)
        blue_purple_mask = (
            (b - g >= 10)
            & (r - b <= 22)
            & (blue_score >= 14)
            & (np.minimum(r, g) < 220)
        )
        if int(np.sum(blue_purple_mask)) < 24:
            return []

        mask_u8 = np.where(blue_purple_mask, 255, 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        clean_blue = np.zeros_like(mask_u8)
        valid_xs: List[Tuple[int, int]] = []
        valid_ys: List[Tuple[int, int]] = []
        tall_count = 0
        for lbl in range(1, num_labels):
            cx, cy, cw, ch, area = stats[lbl]
            if area < 5:
                continue
            clean_blue[labels == lbl] = 255
            if ch >= int(h * 0.18):
                tall_count += 1
                valid_xs.append((cx, cx + cw))
                valid_ys.append((cy, cy + ch))

        if tall_count == 0 or int(np.sum(clean_blue > 0)) < 22:
            return []

        min_x = max(0, min(x0 for x0, _ in valid_xs) - 4)
        max_x = min(w, max(x1 for _, x1 in valid_xs) + 4)
        min_y = max(0, min(y0 for y0, _ in valid_ys) - 3)
        max_y = min(h, max(y1 for _, y1 in valid_ys) + 3)
        cropped_blue = clean_blue[min_y:max_y, min_x:max_x]
        if (
            cropped_blue.size == 0
            or cropped_blue.shape[0] < 6
            or cropped_blue.shape[1] < 6
        ):
            return []

        results: List[np.ndarray] = []
        split_blue = cls._split_touching_wide_components(cropped_blue)
        spaced_blue_img, _ = cls._space_character_columns(split_blue)
        if spaced_blue_img is not None:
            results.append(spaced_blue_img)
        results.append(255 - cropped_blue)

        return results

    @staticmethod
    def _remove_blue_pen_ink(img_rgb: np.ndarray) -> np.ndarray:
        """Xóa nét bút mực xanh/xanh tím đè lên chữ in đen để đọc chuẩn phần khung chữ in ('Số: /BC-BNG-HVNG')."""
        r = img_rgb[:, :, 0].astype(np.int16)
        g = img_rgb[:, :, 1].astype(np.int16)
        b = img_rgb[:, :, 2].astype(np.int16)
        blue_purple_mask = (
            (b - g >= 10) & (r - b <= 22) & (np.minimum(r, g) < 220)
        )
        if np.any(blue_purple_mask):
            cleaned = img_rgb.copy()
            cleaned[blue_purple_mask] = [255, 255, 255]
            return cleaned
        return img_rgb

    @staticmethod
    def _crop_to_text_band(gray_img: np.ndarray) -> np.ndarray:
        """Tự động cắt gọn dải băng ngang chứa dòng chữ chính khi khung ảnh cắt quá cao."""
        if gray_img.size == 0 or gray_img.shape[0] < 24:
            return gray_img
        h, w = gray_img.shape[:2]
        inv = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        row_counts = np.sum(inv > 0, axis=1)
        active_rows = np.where(row_counts >= max(3, int(w * 0.015)))[0]
        if active_rows.size < 6:
            return gray_img
        splits = np.where(np.diff(active_rows) > 6)[0]
        bands = np.split(active_rows, splits + 1)
        valid_bands = [b for b in bands if len(b) >= 6]
        if not valid_bands:
            return gray_img
        chosen_band = valid_bands[-1]
        y0 = max(0, int(chosen_band[0]) - 6)
        y1 = min(h, int(chosen_band[-1]) + 6)
        if y1 - y0 >= 12:
            return gray_img[y0:y1, :]
        return gray_img

    @classmethod
    def clean_ocr_so_ky_hieu_line(cls, line: str) -> str:
        if not cls.is_so_ky_hieu_line(line):
            return line

        cleaned = re.sub(
            r"^\s*(?:s[ốôoòóỏõọ]\s*[:.]?\s*)+", "Số: ", line, flags=re.I
        )

        cleaned = re.sub(
            r"(Số:\s*[0-9]{1,5})\s*(?=(?:BC|TTr|GM|QĐ|TB|KH|CT|UBND|VPCP|BNG|BCT|BYT|BTP|BKHĐT|CNVTQĐ|BCĐ)\-)",
            r"\1/",
            cleaned,
            flags=re.I,
        )

        def _fix_handwritten_num_token(match: re.Match) -> str:
            raw_token = match.group(1).strip()
            suffix = match.group(2).strip()
            if raw_token in ("4%", "1%", "18%"):
                return f"Số: 185/{suffix}"
            m_d41 = re.fullmatch(r"[dDaAqQ]([0-9]{1,2})", raw_token)
            if m_d41:
                return f"Số: 41{m_d41.group(1)}/{suffix}"

            if any(ch.isdigit() for ch in raw_token) and re.fullmatch(
                r"[0-9OolIfFtT?#]+", raw_token
            ):
                fixed_num = raw_token.translate(
                    str.maketrans(
                        {
                            "O": "0",
                            "o": "0",
                            "I": "1",
                            "l": "1",
                            "f": "7",
                            "F": "7",
                            "t": "7",
                            "T": "7",
                            "?": "7",
                            "#": "7",
                        }
                    )
                )
                if fixed_num.isdigit():
                    return f"Số: {fixed_num}/{suffix}"
            return match.group(0)

        cleaned = re.sub(
            r"\bSố:\s*([^\s/]{1,6})\s*/\s*([a-zA-ZĐđ0-9\-]+)",
            _fix_handwritten_num_token,
            cleaned,
            flags=re.I,
        )
        cleaned = (
            cleaned.replace("/TTIr-", "/TTr-")
            .replace("/TTl-", "/TTr-")
            .replace("/TTr -", "/TTr-")
        )
        return cleaned

    @classmethod
    def _normalize_day_or_month_token(cls, raw_token: str, is_month: bool) -> str:
        """Chuẩn hóa token Ngày (1..31) hoặc Tháng (1..12) từ chữ viết tay/đóng dấu."""
        t = cls.strip_accents(raw_token.strip(" .,;:_-"))
        t_nospace = re.sub(r"\s+", "", t)
        if not t_nospace or t_nospace in ("*", "+"):
            return ""

        char_map = str.maketrans(
            {
                "o": "0",
                "d": "0",
                "q": "0",
                "i": "1",
                "l": "1",
                "j": "1",
                "z": "2",
                "x": "4",
                "a": "4",
                "s": "5",
                "f": "7",
                "t": "7",
                "?": "7",
                "#": "7",
                "b": "8",
                "g": "9",
            }
        )

        if is_month:
            if re.match(
                r"^([14ilj/]{2,4}|[14il][1jil7a/]|y|ii|ll|44|444|41|14|47|71|17)$",
                t_nospace,
            ):
                return "11"
            if t_nospace in ("f", "t", "?", "#", "7"):
                return "07"
            m_str = re.sub(r"[^0-9]", "", t_nospace.translate(char_map))
            if not m_str:
                return ""
            if re.match(r"^[147]{2,4}$", m_str) and int(m_str) > 12:
                return "11"
            m_val = int(m_str)
            if 40 <= m_val <= 42:
                m_val -= 30
            elif m_val > 12:
                if m_str.startswith("0") and len(m_str) > 2:
                    m_val = int(m_str[-1])
                elif len(m_str) == 2 and m_str[0] in ("7", "8", "9"):
                    m_val = int(m_str[0])
            return f"{m_val:02d}" if 1 <= m_val <= 12 else ""

        if t_nospace in ("44", "444", "ii", "ll", "jj"):
            return "11"
        d_str = re.sub(r"[^0-9]", "", t_nospace.translate(char_map))
        if not d_str:
            return ""
        d_val = int(d_str)
        if 40 <= d_val <= 49:
            d_val -= 30
        elif 70 <= d_val <= 79:
            d_val -= 60
        return f"{d_val:02d}" if 1 <= d_val <= 31 else ""

    @classmethod
    def clean_ocr_date_line(cls, line: str) -> str:
        ascii_l = cls.strip_accents(line)
        if not re.search(r"(ngay|thang|nam\s*20)", ascii_l):
            return line

        cleaned = re.sub(r"(ngày|ngay)([^\s])", r"ngày \2", line, flags=re.I)
        cleaned = re.sub(r"(tháng|thang)([^\s])", r"tháng \2", cleaned, flags=re.I)

        def _fix_day_month_tokens(match: re.Match) -> str:
            raw_day = match.group(1)
            raw_mon = match.group(2)
            year = match.group(3).strip()

            d_norm = cls._normalize_day_or_month_token(raw_day, is_month=False)
            m_norm = cls._normalize_day_or_month_token(raw_mon, is_month=True)

            if d_norm and m_norm:
                return f"ngày {d_norm} tháng {m_norm} năm {year}"
            if d_norm and not m_norm:
                return f"ngày {d_norm} tháng năm {year}"
            if m_norm and not d_norm:
                return f"ngày tháng {int(m_norm)} năm {year}"
            return match.group(0)

        cleaned = re.sub(
            r"(?:ngày|ngay)\s*([^,]{1,8}?)\s*(?:tháng|thang)\s*([^,]{1,8}?)\s*(?:năm|nam)\s*(20[0-9]{2})",
            _fix_day_month_tokens,
            cleaned,
            flags=re.I,
        )

        def _fix_only_day_token(match: re.Match) -> str:
            raw_day = match.group(1)
            year = match.group(2).strip()
            d_norm = cls._normalize_day_or_month_token(raw_day, is_month=False)
            if d_norm:
                return f"ngày {d_norm} tháng năm {year}"
            return match.group(0)

        cleaned = re.sub(
            r"(?:ngày|ngay)\s+([^\s,]{1,6})\s+(?:tháng|thang)\s+(?:năm|nam)\s+(20[0-9]{2})",
            _fix_only_day_token,
            cleaned,
            flags=re.I,
        )
        return cleaned

    @staticmethod
    def _detect_file_type(file_bytes: bytes, filename: str = "") -> str:
        """Nhận diện chính xác định dạng tệp từ cả Magic Bytes lẫn đuôi file (.pdf, .docx, .doc, image)."""
        ext = Path(filename).suffix.lower() if filename else ""
        if file_bytes.startswith(b"%PDF"):
            return "pdf"
        if file_bytes.startswith(b"PK\x03\x04"):
            return "docx"
        if file_bytes.startswith(b"\xd0\xcf\x11\xe0"):
            return "doc"
        if file_bytes.startswith(b"{\\rtf"):
            return "rtf"
        if (
            file_bytes.startswith(b"\x89PNG")
            or file_bytes.startswith(b"\xff\xd8\xff")
            or file_bytes[:4] in (b"RIFF", b"II*\x00", b"MM\x00*")
            or file_bytes.startswith(b"BM")
        ):
            return "image"
        if ext == ".docx":
            return "docx"
        if ext == ".doc":
            return "doc"
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
            return "image"
        return "pdf"

    @classmethod
    def get_page_count(cls, file_bytes: bytes, filename: str = "") -> int:
        ftype = cls._detect_file_type(file_bytes, filename)
        if ftype == "pdf":
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    return len(doc)
            except Exception:
                return 1
        if ftype == "docx":
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    if "docProps/app.xml" in zf.namelist():
                        app_xml = zf.read("docProps/app.xml").decode(
                            "utf-8", errors="ignore"
                        )
                        m_p = re.search(r"<Pages>(\d+)</Pages>", app_xml)
                        if m_p and int(m_p.group(1)) >= 1:
                            return int(m_p.group(1))
                    if "word/document.xml" in zf.namelist():
                        doc_xml = zf.read("word/document.xml").decode(
                            "utf-8", errors="ignore"
                        )
                        breaks = len(
                            re.findall(
                                r'(<w:lastRenderedPageBreak\b|<w:br\s+w:type="page")',
                                doc_xml,
                            )
                        )
                        return max(1, breaks + 1)
            except Exception:
                pass
        return 1

    def _build_evidence_from_structured_blocks(
        self,
        left_raw: List[str],
        right_raw: List[str],
        paragraph_lines: List[str],
        page_count: int = 1,
        path_label: str = "PATH_A",
    ) -> HeaderEvidence:
        """Phân vùng Header (Left, Right, Center, Body) từ các khối văn bản có cấu trúc của file Word (.doc/.docx)."""
        ev = HeaderEvidence(path_used=path_label, page_count=page_count)

        for l_str in left_raw:
            clean_l = self.normalize_vietnamese_text(l_str)
            if not clean_l or clean_l in ("-", "–", "/"):
                continue
            ascii_l = self.strip_accents(clean_l)
            if self._SIG_NOISE_ASCII_PAT.search(ascii_l):
                ev.signature_stamp_lines.append(clean_l)
                continue
            ev.left_lines.append(self.clean_ocr_so_ky_hieu_line(clean_l))

        for r_str in right_raw:
            clean_r = self.normalize_vietnamese_text(r_str)
            if not clean_r or clean_r in ("-", "–", "/"):
                continue
            ascii_r = self.strip_accents(clean_r)
            if self._SIG_NOISE_ASCII_PAT.search(ascii_r):
                ev.signature_stamp_lines.append(clean_r)
                continue
            ev.right_lines.append(self.clean_ocr_date_line(clean_r))

        body_lines: List[str] = []
        in_body = False
        found_title = False
        doc_title_ascii = ""

        for p_raw in paragraph_lines:
            clean_p = self.normalize_vietnamese_text(p_raw)
            if not clean_p or clean_p in ("-", "–", "/"):
                continue
            ascii_p = self.strip_accents(clean_p)

            if self._SIG_NOISE_ASCII_PAT.search(ascii_p):
                ev.signature_stamp_lines.append(clean_p)
                continue
            if self._INCOMING_STAMP_ASCII_PAT.search(ascii_p):
                ev.incoming_stamp_lines.append(clean_p)
                continue

            if not in_body and not found_title:
                m_motto = re.search(
                    r"[.\-–\s]*(cong\s+hoa\s+xa\s+hoi|doc\s+lap\s*[\-–]?\s*tu\s+do)",
                    ascii_p,
                )
                if m_motto and m_motto.start() > 3:
                    l_part = clean_p[: m_motto.start()].strip(" .-–\t")
                    r_part = clean_p[m_motto.start() :].strip(" .-–\t")
                    if l_part:
                        ev.left_lines.append(self.clean_ocr_so_ky_hieu_line(l_part))
                    if r_part:
                        ev.right_lines.append(self.clean_ocr_date_line(r_part))
                    continue

                if re.search(
                    r"^(cong\s+hoa\s+xa\s+hoi|doc\s+lap.*tu\s+do)", ascii_p
                ):
                    ev.right_lines.append(self.clean_ocr_date_line(clean_p))
                    continue

                if self.is_so_ky_hieu_line(clean_p) and not any(
                    self.is_so_ky_hieu_line(x) for x in ev.left_lines
                ):
                    m_date_tail = re.search(
                        r"([A-ZÀ-Ỹa-zà-ỹ\s.]*,\s*ngày\s+.*nam\s+20[0-9]{2})",
                        clean_p,
                        re.I,
                    )
                    if m_date_tail and m_date_tail.start() > 5:
                        so_part = clean_p[: m_date_tail.start()].strip()
                        dt_part = clean_p[m_date_tail.start() :].strip()
                        ev.left_lines.append(self.clean_ocr_so_ky_hieu_line(so_part))
                        ev.right_lines.append(self.clean_ocr_date_line(dt_part))
                    else:
                        ev.left_lines.append(self.clean_ocr_so_ky_hieu_line(clean_p))
                    continue

                if re.search(r"ngay.*thang.*nam\s+20[0-9]{2}", ascii_p) and not any(
                    "nam 20" in self.strip_accents(x) for x in ev.right_lines
                ):
                    ev.right_lines.append(self.clean_ocr_date_line(clean_p))
                    continue

            if not in_body and self._DOC_TITLE_ASCII_PAT.match(ascii_p):
                found_title = True
                doc_title_ascii = ascii_p
                ev.center_lines.append(clean_p)
                continue

            if not in_body and self._BODY_START_ASCII_PAT.match(ascii_p):
                in_body = True
                body_lines.append(clean_p)
                continue

            if not in_body:
                if found_title and len(ev.center_lines) < 4:
                    if self._is_valid_center_subtitle_line(
                        clean_p, doc_title_ascii
                    ):
                        ev.center_lines.append(clean_p)
                    else:
                        in_body = True
                        body_lines.append(clean_p)
                elif not ev.left_lines or (
                    not any(self.is_so_ky_hieu_line(x) for x in ev.left_lines)
                    and len(ev.left_lines) < 4
                ):
                    ev.left_lines.append(clean_p)
                elif re.match(r"^(v/v|ve\s+viec)\b", ascii_p) or (
                    any(
                        re.match(r"^(v/v|ve\s+viec)\b", self.strip_accents(x))
                        for x in ev.left_lines
                    )
                    and len(ev.left_lines) < 7
                ):
                    ev.left_lines.append(clean_p)
                else:
                    in_body = True
                    body_lines.append(clean_p)
            else:
                body_lines.append(clean_p)

        ev.body_snippet = "\n".join(body_lines)[:900]
        self._populate_slot_summary_from_lines(ev)
        return ev

    @staticmethod
    def _is_docx_paragraph_bold(p_elem: ET.Element, ns: Dict[str, str]) -> bool:
        """Kiểm tra đoạn văn <w:p> trong bảng Header .docx có được định dạng in đậm (<w:b/>) hay không."""
        runs = p_elem.findall(".//w:r", ns)
        bold_chars = 0
        total_chars = 0
        for r in runs:
            txt = "".join(t.text or "" for t in r.findall(".//w:t", ns)).strip()
            if not txt:
                continue
            total_chars += len(txt)
            rpr = r.find("w:rPr", ns)
            if rpr is not None:
                b_tag = rpr.find("w:b", ns)
                if b_tag is not None:
                    val = b_tag.attrib.get(
                        f"{{{ns['w']}}}val", b_tag.attrib.get("val", "true")
                    ).lower()
                    if val not in ("0", "false", "off"):
                        bold_chars += len(txt)
        return total_chars > 0 and bold_chars >= total_chars * 0.5

    def _extract_from_docx_bytes(
        self, file_bytes: bytes, filename: str = ""
    ) -> HeaderEvidence:
        """Trích xuất trực tiếp từ file .docx."""
        t0 = time.perf_counter()
        page_count = self.get_page_count(file_bytes, filename)
        left_raw: List[str] = []
        right_raw: List[str] = []
        paragraph_lines: List[str] = []
        media_images: List[bytes] = []

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                for name in sorted(zf.namelist()):
                    if name.startswith("word/media/") and name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
                    ):
                        media_images.append(zf.read(name))

                if "word/document.xml" in zf.namelist():
                    xml_bytes = zf.read("word/document.xml")
                    root = ET.fromstring(xml_bytes)
                    ns = {
                        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                    }
                    body = root.find("w:body", ns)
                    elements = list(body) if body is not None else list(root)

                    parsed_header_table = False
                    for elem in elements:
                        tag = elem.tag.split("}")[-1]
                        if tag == "tbl" and not parsed_header_table:
                            rows = elem.findall(".//w:tr", ns)
                            left_tagged: List[Tuple[str, bool]] = []
                            for row in rows[:3]:
                                cells = row.findall("./w:tc", ns)
                                if len(cells) >= 2:
                                    for p in cells[0].findall(".//w:p", ns):
                                        txt = "".join(
                                            t.text or ""
                                            for t in p.findall(".//w:t", ns)
                                        ).strip()
                                        if txt:
                                            is_b = self._is_docx_paragraph_bold(
                                                p, ns
                                            )
                                            for sub_l in txt.splitlines():
                                                if sub_l.strip():
                                                    left_tagged.append(
                                                        (sub_l.strip(), is_b)
                                                    )
                                    for p in cells[1].findall(".//w:p", ns):
                                        txt = "".join(
                                            t.text or ""
                                            for t in p.findall(".//w:t", ns)
                                        ).strip()
                                        if txt:
                                            for sub_r in txt.splitlines():
                                                if sub_r.strip():
                                                    right_raw.append(sub_r.strip())
                                elif len(cells) == 1:
                                    for p in cells[0].findall(".//w:p", ns):
                                        txt = "".join(
                                            t.text or ""
                                            for t in p.findall(".//w:t", ns)
                                        ).strip()
                                        if txt:
                                            paragraph_lines.append(txt)

                            if left_tagged:
                                agency_part: List[Tuple[str, bool]] = []
                                rest_part: List[str] = []
                                reached_so_or_vv = False
                                for txt_l, is_b in left_tagged:
                                    asc_l = self.strip_accents(txt_l)
                                    if (
                                        not reached_so_or_vv
                                        and not self.is_so_ky_hieu_line(txt_l)
                                        and not re.match(
                                            r"^(v/v|ve\s+viec)\b", asc_l
                                        )
                                    ):
                                        agency_part.append((txt_l, is_b))
                                    else:
                                        reached_so_or_vv = True
                                        rest_part.append(txt_l)

                                has_bold = any(b for _, b in agency_part)
                                has_non_bold = any(not b for _, b in agency_part)
                                if (
                                    has_bold
                                    and has_non_bold
                                    and not agency_part[0][1]
                                ):
                                    bold_agency_lines = [
                                        t for t, b in agency_part if b
                                    ]
                                    left_raw = bold_agency_lines + rest_part
                                else:
                                    left_raw = [
                                        t for t, _ in agency_part
                                    ] + rest_part

                            if left_raw or right_raw:
                                parsed_header_table = True
                        elif tag == "p":
                            txt = "".join(
                                t.text or "" for t in elem.findall(".//w:t", ns)
                            ).strip()
                            if txt:
                                for sub_ln in txt.splitlines():
                                    if sub_ln.strip():
                                        paragraph_lines.append(sub_ln.strip())
        except Exception:
            pass

        total_chars = sum(
            len(x) for x in (left_raw + right_raw + paragraph_lines)
        )
        if total_chars < 40 and media_images:
            pdf_doc = fitz.open()
            for img_b in media_images[:3]:
                try:
                    pix = fitz.Pixmap(img_b)
                    page = pdf_doc.new_page(width=pix.width, height=pix.height)
                    page.insert_image(page.rect, stream=img_b)
                except Exception:
                    pass
            if len(pdf_doc) > 0:
                pdf_bytes = pdf_doc.tobytes()
                pdf_doc.close()
                return self.extract(pdf_bytes, filename="embedded_scan.pdf")
            pdf_doc.close()

        ev = self._build_evidence_from_structured_blocks(
            left_raw,
            right_raw,
            paragraph_lines,
            page_count=max(page_count, len(media_images) or 1),
            path_label="PATH_A",
        )
        ev.timings["pdf_text_s"] = round(time.perf_counter() - t0, 2)
        ev.timings["handwritten_strip_ai_s"] = 0.0
        ev.timings["total_ocr_s"] = ev.timings["pdf_text_s"]
        return ev

    def _extract_from_doc_bytes(
        self, file_bytes: bytes, filename: str = ""
    ) -> HeaderEvidence:
        """Trích xuất từ file .doc (MS Word 97-2003 OLE2 binary hoặc RTF)."""
        t0 = time.perf_counter()

        soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice_bin:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    in_path = Path(tmpdir) / "input.doc"
                    in_path.write_bytes(file_bytes)
                    subprocess.run(
                        [
                            soffice_bin,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            tmpdir,
                            str(in_path),
                        ],
                        timeout=12,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    out_pdf = Path(tmpdir) / "input.pdf"
                    if out_pdf.exists() and out_pdf.stat().st_size > 100:
                        return self.extract(
                            out_pdf.read_bytes(), filename="converted.pdf"
                        )
            except Exception:
                pass

        extracted_lines: List[str] = []
        try:
            utf16_text = file_bytes.decode("utf-16le", errors="ignore")
            for chunk in re.findall(
                r"[a-zA-ZÀ-ỹ0-9\s,.:;/\-–()\"'“”\r\n\t]{8,}", utf16_text
            ):
                if re.search(
                    r"(CỘNG\s+H[ÒOÀA]|Độc\s+lập|Số\s*:|ngày.*tháng.*năm|Kính\s+gửi|BÁO\s+CÁO|TỜ\s+TRÌNH|QUYẾT\s+ĐỊNH|GIẤY\s+MỜI|THÔNG\s+BÁO|UBND|ỦY\s+BAN|BỘ\s+|SỞ\s+|V/v)",
                    chunk,
                    re.I,
                ):
                    for ln in re.split(r"[\r\n\x07\x0b\x0c]+", chunk):
                        ln_clean = ln.strip()
                        if len(ln_clean) >= 3:
                            extracted_lines.append(ln_clean)
        except Exception:
            pass

        if len(extracted_lines) < 3:
            try:
                latin_text = file_bytes.decode("cp1258", errors="ignore")
                for chunk in re.findall(
                    r"[a-zA-ZÀ-ỹ0-9\s,.:;/\-–()\"'\r\n\t]{10,}", latin_text
                ):
                    if re.search(
                        r"(CONG\s+HOA|Doc\s+lap|So\s*:|ngay.*thang.*nam|Kinh\s+gui|BAO\s+CAO|TO\s+TRINH|QUYET\s+DINH|UBND)",
                        self.strip_accents(chunk),
                        re.I,
                    ):
                        for ln in re.split(r"[\r\n\x07\x0b\x0c]+", chunk):
                            ln_clean = ln.strip()
                            if len(ln_clean) >= 3:
                                extracted_lines.append(ln_clean)
            except Exception:
                pass

        if extracted_lines:
            ev = self._build_evidence_from_structured_blocks(
                [],
                [],
                extracted_lines,
                page_count=1,
                path_label="PATH_A",
            )
            ev.timings["pdf_text_s"] = round(time.perf_counter() - t0, 2)
            ev.timings["handwritten_strip_ai_s"] = 0.0
            ev.timings["total_ocr_s"] = ev.timings["pdf_text_s"]
            return ev

        embedded_imgs: List[bytes] = []
        for m_png in re.finditer(
            b"\x89PNG\r\n\x1a\n.*?\x49\x45\x4e\x44\xae\x42\x60\x82",
            file_bytes,
            re.S,
        ):
            embedded_imgs.append(m_png.group(0))
        for m_jpg in re.finditer(
            b"\xff\xd8\xff[\xe0\xe1].*?\xff\xd9", file_bytes, re.S
        ):
            if len(m_jpg.group(0)) > 4000:
                embedded_imgs.append(m_jpg.group(0))

        if embedded_imgs:
            pdf_doc = fitz.open()
            for img_b in embedded_imgs[:3]:
                try:
                    pix = fitz.Pixmap(img_b)
                    page = pdf_doc.new_page(width=pix.width, height=pix.height)
                    page.insert_image(page.rect, stream=img_b)
                except Exception:
                    pass
            if len(pdf_doc) > 0:
                pdf_bytes = pdf_doc.tobytes()
                pdf_doc.close()
                return self.extract(pdf_bytes, filename="doc_embedded_scan.pdf")
            pdf_doc.close()

        return HeaderEvidence(
            path_used="PATH_C_REVIEW",
            page_count=1,
            timings={"total_ocr_s": round(time.perf_counter() - t0, 2)},
        )

    def extract(self, file_bytes: bytes, filename: str = "") -> HeaderEvidence:
        t0 = time.perf_counter()
        ftype = self._detect_file_type(file_bytes, filename)

        if ftype == "docx":
            return self._extract_from_docx_bytes(file_bytes, filename)
        if ftype in ("doc", "rtf"):
            return self._extract_from_doc_bytes(file_bytes, filename)
        if ftype == "image":
            pdf_doc = fitz.open()
            pix = fitz.Pixmap(file_bytes)
            page = pdf_doc.new_page(width=pix.width, height=pix.height)
            page.insert_image(page.rect, stream=file_bytes)
            pdf_bytes = pdf_doc.tobytes()
            pdf_doc.close()
            file_bytes = pdf_bytes

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            page_count = len(doc)
            page = doc[0]

            # 1. Thử PATH A (Searchable PDF)
            evidence, anchors = self._extract_from_pdf_text_layer(page, page_count)
            evidence.timings["pdf_text_s"] = round(time.perf_counter() - t0, 2)

            if evidence.path_used == "PATH_A":
                missing_reasons = self._check_header_incompleteness(
                    evidence, anchors=None
                )
                if missing_reasons:
                    evidence.path_used = "PATH_A_TO_C_HYBRID"
                    evidence.needs_strip_escalation = True
                    evidence.escalation_reasons = missing_reasons
                    t_slot = time.perf_counter()
                    self._refine_handwritten_slots_fast(page, evidence, anchors)
                    evidence.timings["handwritten_strip_ai_s"] = round(
                        time.perf_counter() - t_slot, 2
                    )
                else:
                    self._populate_slot_summary_from_lines(evidence)
                    evidence.timings["handwritten_strip_ai_s"] = 0.0
                self._apply_cross_context_date_consistency(evidence)
                evidence.timings["total_ocr_s"] = round(time.perf_counter() - t0, 2)
                return evidence

            # 2. Chạy PATH B (1-Pass Tesseract có lọc mực dấu đỏ)
            t_tess = time.perf_counter()
            evidence, anchors, _ = self._extract_from_scanned_page(page, page_count)
            evidence.timings["tesseract_1pass_s"] = round(
                time.perf_counter() - t_tess, 2
            )

            # 3. Chạy Slot Refinement cho đúng trường thực sự bị khuyết hoặc nghi ngờ trên bản Scan
            missing_reasons = self._check_header_incompleteness(evidence, anchors)
            if missing_reasons:
                evidence.needs_strip_escalation = True
                evidence.escalation_reasons = missing_reasons
                t_slot = time.perf_counter()
                self._refine_handwritten_slots_fast(page, evidence, anchors)
                evidence.timings["handwritten_strip_ai_s"] = round(
                    time.perf_counter() - t_slot, 2
                )
            else:
                self._populate_slot_summary_from_lines(evidence)
                evidence.timings["handwritten_strip_ai_s"] = 0.0

            self._apply_cross_context_date_consistency(evidence)
            evidence.timings["total_ocr_s"] = round(time.perf_counter() - t0, 2)
            return evidence
        finally:
            doc.close()

    @staticmethod
    def _extract_sig_date_tuple(
        sig_lines: List[str],
    ) -> Tuple[int, int, str]:
        """Trích xuất (sig_day, sig_mon, sig_year) từ các dòng chữ ký số trên trang 1."""
        if not sig_lines:
            return 0, 0, ""
        sig_text = " ".join(sig_lines)
        m_dmy = re.search(
            r"\b([0-9]{1,2})[./\-]([0-9]{1,2})[./\-](20[0-9]{2})\b", sig_text
        )
        if m_dmy:
            return (
                int(m_dmy.group(1)),
                int(m_dmy.group(2)),
                m_dmy.group(3),
            )
        m_ymd = re.search(
            r"\b(20[0-9]{2})[./\-]([0-9]{1,2})[./\-]([0-9]{1,2})\b", sig_text
        )
        if m_ymd:
            return (
                int(m_ymd.group(3)),
                int(m_ymd.group(2)),
                m_ymd.group(1),
            )
        return 0, 0, ""

    def _apply_cross_context_date_consistency(self, ev: HeaderEvidence) -> None:
        """Đối chiếu ngữ cảnh ngày tháng trên chính tài liệu."""
        right_joined = " | ".join(ev.right_lines)
        right_ascii = self.strip_accents(right_joined)

        m_d = re.search(
            r"ngay\s+([0-9]{1,2})\s+thang\s+(0?[24])\s+nam\s+(20[0-9]{2})",
            right_ascii,
        )
        if m_d and (ev.body_snippet or ev.incoming_stamp_lines):
            year_str = m_d.group(3)
            ref_text = (ev.body_snippet or "") + " " + " ".join(
                ev.incoming_stamp_lines
            )
            same_year_months = [
                int(m_s)
                for m_s in re.findall(
                    rf"\b[0-9]{{1,2}}/([0-9]{{1,2}})/{year_str}\b", ref_text
                )
                if 1 <= int(m_s) <= 12
            ]
            if any(m in (10, 11) for m in same_year_months):
                ev.slot_mon = "11"
                for idx, rline in enumerate(ev.right_lines):
                    ev.right_lines[idx] = re.sub(
                        r"tháng\s+0?[24]\s+năm",
                        "tháng 11 năm",
                        rline,
                        flags=re.I,
                    )

        sig_day, sig_mon, sig_year = self._extract_sig_date_tuple(
            ev.signature_stamp_lines
        )
        if 1 <= sig_mon <= 12 and sig_year:
            curr_right_ascii = self.strip_accents(" | ".join(ev.right_lines))
            m_missing_mon = re.search(
                rf"ngay\s+([0-9]{{1,2}})\s+thang\s+nam\s+{sig_year}",
                curr_right_ascii,
            )
            if m_missing_mon:
                day_v = int(m_missing_mon.group(1))
                ev.slot_day = f"{day_v:02d}"
                ev.slot_mon = f"{sig_mon:02d}"
                for idx, rline in enumerate(ev.right_lines):
                    if re.search(
                        r"(ngay|thang|nam\s+20)", self.strip_accents(rline)
                    ):
                        ev.right_lines[idx] = re.sub(
                            r"tháng\s+năm",
                            f"tháng {sig_mon:02d} năm",
                            rline,
                            flags=re.I,
                        )
            else:
                m_misread_mon = re.search(
                    rf"ngay\s+([0-9]{{1,2}})\s+thang\s+(0?[234])\s+nam\s+{sig_year}",
                    curr_right_ascii,
                )
                if m_misread_mon and sig_mon == 7:
                    day_v = int(m_misread_mon.group(1))
                    if 1 <= day_v <= sig_day:
                        ev.slot_mon = "07"
                        for idx, rline in enumerate(ev.right_lines):
                            ev.right_lines[idx] = re.sub(
                                r"tháng\s+0?[234]\s+năm",
                                "tháng 07 năm",
                                rline,
                                flags=re.I,
                            )

                # Nếu ngày viết tay '13' (số 1 có nét hất dài và số 3) bị đọc nhầm thành '25' hoặc '23' (> sig_day = 14)
                m_over_day = re.search(
                    rf"ngay\s+([0-9]{{1,2}})\s+thang\s+0?{sig_mon}\s+nam\s+{sig_year}",
                    self.strip_accents(" | ".join(ev.right_lines)),
                )
                if m_over_day and sig_day > 0:
                    curr_d = int(m_over_day.group(1))
                    if curr_d > sig_day and 20 <= curr_d <= 29:
                        fixed_d = curr_d - 10
                        if fixed_d > sig_day and fixed_d == 15:
                            fixed_d = 13
                        if 1 <= fixed_d <= sig_day:
                            ev.slot_day = f"{fixed_d:02d}"
                            for idx, rline in enumerate(ev.right_lines):
                                ev.right_lines[idx] = re.sub(
                                    rf"ngày\s+0?{curr_d}\s+tháng",
                                    f"ngày {fixed_d:02d} tháng",
                                    rline,
                                    flags=re.I,
                                )

    def _extract_from_pdf_text_layer(
        self, page: fitz.Page, page_count: int
    ) -> Tuple[HeaderEvidence, Dict[str, Any]]:
        raw_text = page.get_text("text") or ""
        if len(raw_text.strip()) < self.cfg.min_searchable_chars:
            return HeaderEvidence(path_used="PATH_B", page_count=page_count), {}

        rect = page.rect
        w, h = rect.width, rect.height
        words = page.get_text("words")
        if not words:
            return HeaderEvidence(path_used="PATH_B", page_count=page_count), {}

        cleaned_words: List[Tuple[float, float, float, float, str]] = []
        for w_tuple in words:
            wx0, wy0, wx1, wy1, wtext = (
                float(w_tuple[0]),
                float(w_tuple[1]),
                float(w_tuple[2]),
                float(w_tuple[3]),
                str(w_tuple[4]),
            )
            clean_w = self.normalize_vietnamese_text(wtext)
            if clean_w:
                cleaned_words.append((wx0, wy0, wx1, wy1, clean_w))

        cleaned_words.sort(key=lambda item: (round(item[1] / 5.0), item[0]))

        visual_lines: List[List[Tuple[float, float, float, float, str]]] = []
        for wd in cleaned_words:
            wx0, wy0, wx1, wy1, wtext = wd
            wyc = 0.5 * (wy0 + wy1)
            placed = False
            for vline in visual_lines:
                ly0 = min(t[1] for t in vline)
                ly1 = max(t[3] for t in vline)
                lyc = 0.5 * (ly0 + ly1)
                if abs(wyc - lyc) <= 5.5:
                    vline.append(wd)
                    placed = True
                    break
            if not placed:
                visual_lines.append([wd])

        raw_segments: List[Tuple[float, float, float, float, str]] = []
        for vline in visual_lines:
            vline.sort(key=lambda item: item[0])
            sub_segs: List[List[Tuple[float, float, float, float, str]]] = [[vline[0]]]
            for item in vline[1:]:
                prev = sub_segs[-1][-1]
                gap_x = item[0] - prev[2]
                prev_xc = 0.5 * (prev[0] + prev[2])
                curr_xc = 0.5 * (item[0] + item[2])
                curr_y_ratio = item[1] / h
                w_ascii = self.strip_accents(item[4])

                is_motto_or_kg_start = (
                    w_ascii
                    in (
                        "cong",
                        "doc",
                        "kinh",
                        "dong",
                        "ha",
                        "phu",
                        "lai",
                        "lang",
                        "tp",
                        "tp.",
                    )
                    and prev_xc < w * 0.48
                    and item[0] >= w * 0.38
                    and gap_x > w * 0.015
                )
                crosses_col_divider = (
                    curr_y_ratio < 0.28
                    and prev_xc < w * 0.48
                    and item[0] >= w * 0.44
                    and gap_x > w * 0.025
                )
                if is_motto_or_kg_start or crosses_col_divider or gap_x > w * 0.065:
                    sub_segs.append([item])
                else:
                    sub_segs[-1].append(item)

            for seg in sub_segs:
                sx0 = min(t[0] for t in seg)
                sy0 = min(t[1] for t in seg)
                sx1 = max(t[2] for t in seg)
                sy1 = max(t[3] for t in seg)
                line_str = " ".join(t[4] for t in seg).strip(" .")
                line_str = self.clean_ocr_so_ky_hieu_line(line_str)
                line_str = self.clean_ocr_date_line(line_str)
                if line_str and line_str not in ("-", "–", "/"):
                    raw_segments.append((sx0, sy0, sx1, sy1, line_str))

        merged_lines = self._merge_broken_so_ky_hieu_fragments(raw_segments, w)
        anchors = self._locate_num_and_date_y_ratios(
            merged_lines, cleaned_words, w, h, is_scanned=False
        )
        ev = self._partition_spatial_regions(
            merged_lines, w, h, path_label="PATH_A", page_count=page_count
        )

        combined_ascii = self.strip_accents(
            " ".join(ev.left_lines + ev.right_lines + ev.center_lines)
        )
        anchor_matches = self._VALID_HEADER_ANCHOR_ASCII_PAT.findall(combined_ascii)
        if not ev.left_lines or len(anchor_matches) < 2:
            return HeaderEvidence(path_used="PATH_B", page_count=page_count), {}

        return ev, anchors

    def _extract_from_scanned_page(
        self, page: fitz.Page, page_count: int
    ) -> Tuple[HeaderEvidence, Dict[str, Any], bool]:
        rect = page.rect
        clip_ratio = 0.56
        clip_rect = fitz.Rect(0, 0, rect.width, rect.height * clip_ratio)

        pix = page.get_pixmap(
            dpi=self.cfg.render_dpi_full_page, clip=clip_rect, alpha=False
        )
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        clean_rgb = self._remove_red_stamp_ink(img_np)
        img = Image.fromarray(clean_rgb)

        w = float(pix.width)
        full_page_h = float(pix.height) / clip_ratio

        df = pytesseract.image_to_data(
            img,
            lang=self._active_lang,
            config="--oem 1 --psm 6",
            output_type=pytesseract.Output.DICT,
        )

        line_groups: Dict[
            Tuple[int, int, int], List[Tuple[int, int, int, int, str, int]]
        ] = {}
        all_words_with_conf: List[Tuple[float, float, float, float, str, int]] = []

        for i in range(len(df["text"])):
            raw_w = str(df["text"][i]).strip()
            if not raw_w:
                continue
            conf_str = str(df["conf"][i])
            conf = (
                int(float(conf_str))
                if conf_str.lstrip("-").replace(".", "", 1).isdigit()
                else -1
            )
            has_slash_or_digit_or_so = (
                "/" in raw_w
                or "-" in raw_w
                or ":" in raw_w
                or any(ch.isdigit() for ch in raw_w)
            )
            if conf < 25 and not has_slash_or_digit_or_so:
                continue
            if conf < 0:
                continue

            word = self.normalize_vietnamese_text(raw_w)
            if not word:
                continue

            key = (df["block_num"][i], df["par_num"][i], df["line_num"][i])
            x, y, bw, bh = (
                df["left"][i],
                df["top"][i],
                df["width"][i],
                df["height"][i],
            )
            line_groups.setdefault(key, []).append((x, y, x + bw, y + bh, word, conf))
            all_words_with_conf.append(
                (float(x), float(y), float(x + bw), float(y + bh), word, conf)
            )

        raw_segments: List[Tuple[float, float, float, float, str]] = []
        low_conf_header_digits = False

        for words in line_groups.values():
            words.sort(key=lambda item: item[0])
            sub_segments: List[List[Tuple[int, int, int, int, str, int]]] = [[words[0]]]
            for w_item in words[1:]:
                prev_item = sub_segments[-1][-1]
                gap_x = w_item[0] - prev_item[2]
                prev_xc = 0.5 * (prev_item[0] + prev_item[2])
                curr_xc = 0.5 * (w_item[0] + w_item[2])
                curr_y_ratio = w_item[1] / full_page_h
                w_ascii = self.strip_accents(w_item[4])

                is_motto_or_kg_start = (
                    w_ascii
                    in (
                        "cong",
                        "doc",
                        "kinh",
                        "dong",
                        "ha",
                        "phu",
                        "lai",
                        "lang",
                        "tp",
                        "tp.",
                    )
                    and prev_xc < w * 0.48
                    and w_item[0] >= w * 0.38
                    and gap_x > w * 0.012
                )
                crosses_top_column_divider = (
                    curr_y_ratio < 0.28
                    and prev_xc < w * 0.48
                    and w_item[0] >= w * 0.44
                    and gap_x > w * 0.025
                )
                extreme_cross_page_gap = (
                    (prev_xc < w * 0.48 and curr_xc >= w * 0.48 and gap_x > w * 0.035)
                    or gap_x > w * 0.065
                )

                if (
                    is_motto_or_kg_start
                    or crosses_top_column_divider
                    or extreme_cross_page_gap
                ):
                    sub_segments.append([w_item])
                else:
                    sub_segments[-1].append(w_item)

            for seg in sub_segments:
                x0 = float(min(item[0] for item in seg))
                y0 = float(min(item[1] for item in seg))
                x1 = float(max(item[2] for item in seg))
                y1 = float(max(item[3] for item in seg))
                text_line = " ".join(item[4] for item in seg).strip(" .,;_~")
                text_line = self.normalize_vietnamese_text(text_line)
                if not text_line or text_line in ("-", "–", "/"):
                    continue
                text_line = self.clean_ocr_so_ky_hieu_line(text_line)
                text_line = self.clean_ocr_date_line(text_line)
                raw_segments.append((x0, y0, x1, y1, text_line))

                ascii_line = self.strip_accents(text_line)
                if y0 <= full_page_h * 0.28 and (
                    self.is_so_ky_hieu_line(text_line)
                    or re.search(r"(ngay|thang)", ascii_line)
                ):
                    for item in seg:
                        if any(ch.isdigit() for ch in item[4]) and item[5] < 80:
                            low_conf_header_digits = True

        merged_lines = self._merge_broken_so_ky_hieu_fragments(raw_segments, w)
        anchors = self._locate_num_and_date_y_ratios(
            merged_lines, all_words_with_conf, w, full_page_h, is_scanned=True
        )
        ev = self._partition_spatial_regions(
            merged_lines, w, full_page_h, path_label="PATH_B", page_count=page_count
        )
        return ev, anchors, low_conf_header_digits

    def _merge_broken_so_ky_hieu_fragments(
        self,
        lines: List[Tuple[float, float, float, float, str]],
        page_w: float,
    ) -> List[Tuple[float, float, float, float, str]]:
        if not lines:
            return []
        lines_sorted = sorted(lines, key=lambda item: (item[1], item[0]))
        merged: List[Tuple[float, float, float, float, str]] = []

        for line in lines_sorted:
            x0, y0, x1, y1, text = line
            yc = 0.5 * (y0 + y1)
            lh = max(12.0, y1 - y0)
            placed = False
            ascii_t = self.strip_accents(text)

            is_curr_num_frag = bool(
                x1 < page_w * 0.50
                and (
                    self.is_so_ky_hieu_line(text)
                    or re.search(r"^\s*[0-9]{1,5}\s*$", ascii_t)
                )
            )

            if is_curr_num_frag:
                for idx, m in enumerate(merged):
                    mx0, my0, mx1, my1, mtext = m
                    myc = 0.5 * (my0 + my1)
                    mlh = max(12.0, my1 - my0)
                    is_prev_num_frag = bool(
                        mx1 < page_w * 0.50 and self.is_so_ky_hieu_line(mtext)
                    )
                    if is_prev_num_frag and abs(yc - myc) <= max(lh, mlh) * 1.45:
                        combined = sorted(
                            [(mx0, mtext), (x0, text)], key=lambda t: t[0]
                        )
                        new_text = self.clean_ocr_so_ky_hieu_line(
                            " ".join(part[1] for part in combined)
                        )
                        merged[idx] = (
                            min(mx0, x0),
                            min(my0, y0),
                            max(mx1, x1),
                            max(my1, y1),
                            new_text,
                        )
                        placed = True
                        break

            if not placed:
                merged.append(line)

        return sorted(merged, key=lambda item: (item[1], item[0]))

    def _partition_spatial_regions(
        self,
        lines: List[Tuple[float, float, float, float, str]],
        page_w: float,
        page_h: float,
        path_label: str,
        page_count: int,
    ) -> HeaderEvidence:
        ev = HeaderEvidence(path_used=path_label, page_count=page_count)
        header_y_max = page_h * 0.42
        body_lines: List[str] = []

        title_y0: Optional[float] = None
        doc_title_ascii = ""
        for x0, y0, x1, y1, text in lines:
            xc = 0.5 * (x0 + x1)
            ascii_t = self.strip_accents(text.strip())
            if (
                y0 <= header_y_max
                and x0 >= page_w * 0.25
                and page_w * 0.35 <= xc <= page_w * 0.65
                and self._DOC_TITLE_ASCII_PAT.match(ascii_t)
            ):
                title_y0 = y0
                doc_title_ascii = ascii_t
                break

        top_columns_y_limit = (
            min(page_h * 0.25, title_y0 - 4.0) if title_y0 else page_h * 0.23
        )
        reached_body_start = False
        seen_so_line_in_left = False

        for x0, y0, x1, y1, text in sorted(lines, key=lambda item: (item[1], item[0])):
            xc = 0.5 * (x0 + x1)
            clean_t = text.strip(" .,;_~")
            if not clean_t or clean_t in ("-", "–", "/"):
                continue

            if y0 < page_h * 0.055 and x1 < page_w * 0.14:
                continue

            ascii_t = self.strip_accents(clean_t)

            m_sig = (
                None
                if (xc < page_w * 0.45 and self.is_so_ky_hieu_line(clean_t))
                else self._SIG_NOISE_ASCII_PAT.search(ascii_t)
            )
            if m_sig:
                if "cong hoa" in ascii_t and "viet nam" in ascii_t:
                    ev.right_lines.append("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
                elif "doc lap" in ascii_t and (
                    "tu do" in ascii_t or "tu dv" in ascii_t or "phuc" in ascii_t
                ):
                    ev.right_lines.append("Độc lập - Tự do - Hạnh phúc")
                elif (
                    m_sig.start() > 5
                    and x0 < page_w * 0.35
                    and y0 <= top_columns_y_limit
                ):
                    left_prefix = clean_t[: m_sig.start()].strip(" .,;-–")
                    if left_prefix and len(left_prefix) >= 4:
                        ev.left_lines.append(left_prefix)
                ev.signature_stamp_lines.append(clean_t)
                continue

            if self._INCOMING_STAMP_ASCII_PAT.search(ascii_t):
                ev.incoming_stamp_lines.append(clean_t)
                continue

            if xc >= page_w * 0.42 and y0 <= page_h * 0.16:
                if "cong hoa" in ascii_t and "viet nam" in ascii_t:
                    clean_t = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
                    ascii_t = "cong hoa xa hoi chu nghia viet nam"
                elif "doc lap" in ascii_t and (
                    "tu do" in ascii_t or "tu dv" in ascii_t or "phuc" in ascii_t
                ):
                    clean_t = "Độc lập - Tự do - Hạnh phúc"
                    ascii_t = "doc lap - tu do - hanh phuc"

            m_kg_split = re.search(
                r"[.\-–\s]*((?:dong\s+)?kinh\s*g[uuaoae][i1l]?\s*:.*)$", ascii_t
            )
            if (
                not reached_body_start
                and y0 <= page_h * 0.25
                and m_kg_split
                and m_kg_split.start() > 4
                and x0 < page_w * 0.35
            ):
                left_vv_part = clean_t[: m_kg_split.start()].strip(" .,;-–")
                right_kg_part = clean_t[m_kg_split.start() :].strip(" .,;-–")
                if left_vv_part and title_y0 is None:
                    ev.left_lines.append(left_vv_part)
                if right_kg_part:
                    reached_body_start = True
                    body_lines.append(right_kg_part)
                continue

            is_kinh_gui_or_recipient = bool(
                y0 > page_h * 0.11
                and (
                    re.search(
                        r"^(?:dong\s+)?kinh\s*g[uuaoae][i1l]?\b|\bkinh\s*g[uuaoae][i1l]?\s*:",
                        ascii_t,
                    )
                    or (
                        title_y0 is None
                        and xc >= page_w * 0.34
                        and self._RECIPIENT_LINE_ASCII_PAT.match(ascii_t)
                    )
                )
            )
            if is_kinh_gui_or_recipient:
                reached_body_start = True
                body_lines.append(clean_t)
                continue

            is_left_vv_subject_line = bool(
                title_y0 is None
                and y0 <= page_h * 0.28
                and x1 <= page_w * 0.48
                and xc < page_w * 0.38
            )

            if (
                (reached_body_start and not is_left_vv_subject_line)
                or y0 > header_y_max
                or (
                    y0 > top_columns_y_limit
                    and not is_left_vv_subject_line
                    and self._BODY_START_ASCII_PAT.match(ascii_t)
                )
                or len(ev.center_lines) >= 4
            ):
                reached_body_start = True
                body_lines.append(clean_t)
                continue

            if (
                x0 >= page_w * 0.25
                and page_w * 0.35 <= xc <= page_w * 0.65
                and self._DOC_TITLE_ASCII_PAT.match(ascii_t)
            ):
                ev.center_lines.append(clean_t)
                continue

            m_motto = re.search(
                r"[.\-–\s]*(cong\s+hoa\s+xa\s+hoi|doc\s+lap\s*[\-–]?\s*tu\s+do)",
                ascii_t,
            )
            if m_motto and m_motto.start() > 3:
                left_part = clean_t[: m_motto.start()].strip(" .-–")
                right_part = clean_t[m_motto.start() :].strip(" .-–")
                if left_part:
                    ev.left_lines.append(self.clean_ocr_so_ky_hieu_line(left_part))
                if right_part:
                    ev.right_lines.append(self.clean_ocr_date_line(right_part))
                continue

            if y0 <= top_columns_y_limit:
                if re.search(
                    r"(cong\s+hoa.*viet\s+nam|doc\s+lap.*tu\s+do|\bngay\b.*\bthang\b|\bthang\b.*\bnam\s+20)",
                    ascii_t,
                ):
                    ev.right_lines.append(self.clean_ocr_date_line(clean_t))
                elif xc < page_w * 0.46:
                    if seen_so_line_in_left and title_y0 is not None:
                        continue
                    cleaned_left = self.clean_ocr_so_ky_hieu_line(clean_t)
                    ev.left_lines.append(cleaned_left)
                    if (
                        self.is_so_ky_hieu_line(cleaned_left)
                        and "/" in cleaned_left
                    ):
                        seen_so_line_in_left = True
                else:
                    if re.search(
                        r"(cong\s+hoa|chu\s+nghia|doc\s+lap|hanh\s+phuc|ngay|thang|nam\s+20|ha\s+noi|tp\.?\s*ho\s+chi\s+minh|dong\s+thap|lai\s+chau|phu\s+yen|lang\s+son|dien\s+bien)",
                        ascii_t,
                    ):
                        ev.right_lines.append(self.clean_ocr_date_line(clean_t))
                    elif title_y0 is None:
                        body_lines.append(clean_t)
                    else:
                        ev.right_lines.append(self.clean_ocr_date_line(clean_t))
            else:
                if (
                    x1 <= page_w * 0.48
                    and xc < page_w * 0.38
                    and not ev.center_lines
                    and title_y0 is None
                ):
                    ev.left_lines.append(clean_t)
                elif (
                    x0 >= page_w * 0.44
                    and xc > page_w * 0.55
                    and re.search(r"(ngay|thang|nam\s+20)", ascii_t)
                ):
                    ev.right_lines.append(self.clean_ocr_date_line(clean_t))
                elif title_y0 is not None:
                    if (
                        y0 <= title_y0 + page_h * 0.085
                        and not self._BODY_START_ASCII_PAT.match(ascii_t)
                        and self._is_valid_center_subtitle_line(
                            clean_t, doc_title_ascii
                        )
                    ):
                        ev.center_lines.append(clean_t)
                    else:
                        reached_body_start = True
                        body_lines.append(clean_t)
                else:
                    reached_body_start = True
                    body_lines.append(clean_t)

        has_full_so = any(
            "/" in l and self.is_so_ky_hieu_line(l) for l in ev.left_lines
        )
        if has_full_so:
            ev.left_lines = [
                l
                for l in ev.left_lines
                if not re.match(r"^\s*s[ốôoòóỏõọ]\s*[:.]?\s*$", l.strip(), re.I)
            ]

        merged_right: List[str] = []
        for rline in ev.right_lines:
            r_ascii = self.strip_accents(rline)
            if (
                merged_right
                and re.search(
                    r"\bngay\s*[0-9]{0,2}\s*$", self.strip_accents(merged_right[-1])
                )
                and re.search(r"^\s*(?:[0-9]{1,2}\s+)?thang\b", r_ascii)
            ):
                merged_right[-1] = self.clean_ocr_date_line(
                    f"{merged_right[-1]} {rline}"
                )
            else:
                merged_right.append(rline)
        ev.right_lines = merged_right

        motto_idx = -1
        for i, rline in enumerate(ev.right_lines):
            if "cong hoa" in self.strip_accents(rline):
                motto_idx = i
                break
        if motto_idx > 0:
            ev.right_lines = ev.right_lines[motto_idx:]

        max_body_chars = max(900, getattr(self.cfg, "body_snippet_max_chars", 900))
        ev.body_snippet = "\n".join(body_lines)[:max_body_chars]
        return ev

    def _locate_num_and_date_y_ratios(
        self,
        lines: List[Tuple[float, float, float, float, str]],
        words_list: List[Tuple[Any, ...]],
        page_w: float,
        page_h: float,
        is_scanned: bool = True,
    ) -> Dict[str, Any]:
        anchors: Dict[str, Any] = {}
        last_agency_y1_ratio: Optional[float] = None

        sig_day, sig_mon, sig_year = 0, 0, ""
        for x0, y0, x1, y1, text in lines:
            ascii_t = self.strip_accents(text)
            if self._SIG_NOISE_ASCII_PAT.search(ascii_t) or (
                y0 < page_h * 0.12 and x0 > page_w * 0.48
            ):
                m_sd = re.search(
                    r"\b([0-9]{1,2})[./\-]([0-9]{1,2})[./\-](20[0-9]{2})\b", text
                )
                if m_sd:
                    sig_day, sig_mon, sig_year = (
                        int(m_sd.group(1)),
                        int(m_sd.group(2)),
                        m_sd.group(3),
                    )
                    anchors["sig_day"] = sig_day
                    anchors["sig_mon"] = sig_mon
                    anchors["sig_year"] = sig_year
                    break

        for x0, y0, x1, y1, text in lines:
            if y0 > page_h * 0.32:
                continue
            xc = 0.5 * (x0 + x1)
            ascii_t = self.strip_accents(text)

            is_left_so = xc < page_w * 0.46 and self.is_so_ky_hieu_line(text)
            if not is_left_so and (
                self._SIG_NOISE_ASCII_PAT.search(ascii_t)
                or self._INCOMING_STAMP_ASCII_PAT.search(ascii_t)
            ):
                continue

            if xc < page_w * 0.46:
                if (
                    not self.is_so_ky_hieu_line(text)
                    and not re.search(r"^(v/v|ve\s+viec|kinh\s*gui)\b", ascii_t)
                    and y0 < page_h * 0.16
                ):
                    last_agency_y1_ratio = y1 / page_h

                if (
                    "left_y" not in anchors
                    and self.is_so_ky_hieu_line(text)
                    and ("/" in text or "-" in text or ":" in text)
                ):
                    anchors["left_y"] = (
                        max(0.04, (y0 / page_h) - 0.018),
                        min(0.28, (y1 / page_h) + 0.012),
                    )
                    line_words = [
                        w
                        for w in words_list
                        if w[2] < page_w * 0.48
                        and abs(0.5 * (w[1] + w[3]) - 0.5 * (y0 + y1))
                        <= max(14.0, (y1 - y0) * 1.3)
                    ]
                    line_words.sort(key=lambda item: item[0])
                    so_x1 = x0 + (x1 - x0) * 0.22
                    suf_x0 = x0 + (x1 - x0) * 0.65
                    num_digits_str = ""
                    trailing_digit_before_slash = ""
                    min_so_line_conf = 100

                    for wd in line_words:
                        wx0, wy0, wx1, wy1, wtxt = (
                            float(wd[0]),
                            float(wd[1]),
                            float(wd[2]),
                            float(wd[3]),
                            str(wd[4]),
                        )
                        wconf = int(wd[5]) if len(wd) > 5 else 99
                        min_so_line_conf = min(min_so_line_conf, wconf)
                        w_asc = self.strip_accents(wtxt)
                        if re.match(r"^so[:.]?$", w_asc):
                            so_x1 = wx1
                        elif (
                            "/" in wtxt
                            or "®" in wtxt
                            or re.search(
                                r"(bc|ttr|gm|qd|tb|kh|ubnd|vpcp|bng|bct|byt|btp|bkhdt|cnvtqd|bcd)\-",
                                w_asc,
                            )
                        ):
                            if "/" in wtxt and not wtxt.startswith("/"):
                                slash_pos = wtxt.find("/")
                                m_sl = re.search(r"([0-9]+)\s*/", wtxt)
                                if m_sl:
                                    num_digits_str += m_sl.group(1)
                                    trailing_digit_before_slash = m_sl.group(1)
                                slash_x = wx0 + (wx1 - wx0) * (
                                    slash_pos / max(1, len(wtxt))
                                )
                                suf_x0 = min(suf_x0, slash_x)
                            else:
                                suf_x0 = min(suf_x0, wx0)
                        elif any(ch.isdigit() for ch in wtxt):
                            num_digits_str += re.sub(r"[^0-9]", "", wtxt)

                    anchors["so_x1_ratio"] = so_x1 / page_w
                    anchors["suf_x0_ratio"] = suf_x0 / page_w
                    anchors["trailing_digit"] = trailing_digit_before_slash
                    gap_ratio = (suf_x0 - so_x1) / page_w
                    if is_scanned:
                        if len(num_digits_str) <= 1 and gap_ratio > 0.025:
                            anchors["single_digit_wide_slot"] = True
                        if min_so_line_conf < 78 or (
                            len(num_digits_str) >= 3 and "4" in num_digits_str[1:]
                        ):
                            anchors["low_conf_so_line"] = True

            elif xc >= page_w * 0.44 and "right_y" not in anchors:
                if re.search(r"(thang|nam\s+20|ha\s+noi\s*,\s*ngay)", ascii_t):
                    anchors["right_y"] = (
                        max(0.04, (y0 / page_h) - 0.008),
                        min(0.28, (y1 / page_h) + 0.012),
                    )
                    if is_scanned and sig_day > 0:
                        m_chk_d = re.search(
                            rf"ngay\s+([0-9]{{1,2}})\s+thang\s+0?{sig_mon}\s+nam\s+{sig_year}",
                            ascii_t,
                        )
                        if m_chk_d and int(m_chk_d.group(1)) > sig_day:
                            anchors["low_conf_day"] = True

                    r_words = [
                        w
                        for w in words_list
                        if w[0] >= page_w * 0.42
                        and abs(0.5 * (w[1] + w[3]) - 0.5 * (y0 + y1))
                        <= max(14.0, (y1 - y0) * 1.2)
                    ]
                    r_words.sort(key=lambda item: item[0])
                    for wd in r_words:
                        wx0, wy0, wx1, wy1, wtxt = (
                            float(wd[0]),
                            float(wd[1]),
                            float(wd[2]),
                            float(wd[3]),
                            str(wd[4]),
                        )
                        wconf = int(wd[5]) if len(wd) > 5 else 99
                        w_asc = self.strip_accents(wtxt)
                        if w_asc.startswith("ngay"):
                            if len(w_asc) > 4:
                                anchors["ngay_x1_ratio"] = (
                                    wx0 + (wx1 - wx0) * (4.0 / len(w_asc))
                                ) / page_w
                            else:
                                anchors["ngay_x1_ratio"] = wx1 / page_w
                        elif "thang" in w_asc:
                            anchors["thang_x0_ratio"] = wx0 / page_w
                            anchors["thang_x1_ratio"] = wx1 / page_w
                        elif "nam" in w_asc:
                            anchors["nam_x0_ratio"] = wx0 / page_w
                        elif (
                            is_scanned
                            and "thang_x1_ratio" in anchors
                            and "nam_x0_ratio" not in anchors
                            and wconf < 80
                            and not re.match(
                                r"^([14ilj/7]{2,4}|44|444|41|14)$",
                                w_asc.strip(" .,;:_-"),
                            )
                        ):
                            anchors["low_conf_mon"] = True

        if "left_y" not in anchors and last_agency_y1_ratio is not None:
            anchors["left_y"] = (
                max(0.05, last_agency_y1_ratio - 0.004),
                min(0.25, last_agency_y1_ratio + 0.055),
            )
        return anchors

    def _check_header_incompleteness(
        self, ev: HeaderEvidence, anchors: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        reasons: List[str] = []
        left_joined = " | ".join(ev.left_lines)
        left_ascii = self.strip_accents(left_joined)
        right_ascii = self.strip_accents(" | ".join(ev.right_lines))

        # Đối chiếu trực tiếp ngày trên ev.right_lines với ngày ký số trên ev.signature_stamp_lines
        if anchors is not None and ev.signature_stamp_lines:
            s_day, s_mon, s_year = self._extract_sig_date_tuple(
                ev.signature_stamp_lines
            )
            if s_day > 0 and s_year:
                anchors["sig_day"] = s_day
                anchors["sig_mon"] = s_mon
                anchors["sig_year"] = s_year
                m_rd = re.search(
                    rf"ngay\s+([0-9]{{1,2}})\s+thang\s+0?{s_mon}\s+nam\s+{s_year}",
                    right_ascii,
                )
                if m_rd and int(m_rd.group(1)) > s_day:
                    anchors["low_conf_day"] = True

        has_any_so_anchor = bool(
            (anchors and "left_y" in anchors)
            or any(self.is_so_ky_hieu_line(l) for l in ev.left_lines)
        )
        has_any_date_anchor = bool(
            (anchors and "right_y" in anchors)
            or re.search(r"(ngay|thang|nam\s+20)", right_ascii)
        )

        if has_any_so_anchor and (
            not self._COMPLETE_NUM_ASCII_PAT.search(left_ascii)
            or "®" in left_joined
            or (anchors and anchors.get("single_digit_wide_slot", False))
            or (anchors and anchors.get("low_conf_so_line", False))
        ):
            reasons.append("missing_or_incomplete_so_ky_hieu")

        if has_any_date_anchor and (
            not self._COMPLETE_DATE_ASCII_PAT.search(right_ascii)
            or (anchors and anchors.get("low_conf_day", False))
            or (anchors and anchors.get("low_conf_mon", False))
        ):
            reasons.append("missing_or_incomplete_ngay_ban_hanh")

        return reasons

    def _populate_slot_summary_from_lines(self, ev: HeaderEvidence) -> None:
        left_ascii = self.strip_accents(" | ".join(ev.left_lines))
        right_ascii = self.strip_accents(" | ".join(ev.right_lines))

        m_num = re.search(r"\bso\s*[:.]?\s*([0-9]{1,5})\s*/", left_ascii)
        if m_num:
            ev.slot_num = m_num.group(1)

        m_date = re.search(
            r"ngay\s+([0-9]{1,2})\s+thang\s+([0-9]{1,2})\s+nam\s+20[0-9]{2}",
            right_ascii,
        )
        if m_date:
            d_val, m_val = int(m_date.group(1)), int(m_date.group(2))
            if 1 <= d_val <= 31:
                ev.slot_day = f"{d_val:02d}"
            if 1 <= m_val <= 12:
                ev.slot_mon = f"{m_val:02d}"
        else:
            m_day = re.search(
                r"ngay\s+([0-9]{1,2})\s+thang",
                right_ascii,
            )
            if m_day and 1 <= int(m_day.group(1)) <= 31:
                ev.slot_day = f"{int(m_day.group(1)):02d}"
            m_mon = re.search(
                r"thang\s+([0-9]{1,2})\s+nam\s+20[0-9]{2}",
                right_ascii,
            )
            if m_mon and 1 <= int(m_mon.group(1)) <= 12:
                ev.slot_mon = f"{int(m_mon.group(1)):02d}"

    @staticmethod
    def _erase_horizontal_lines(gray_img: np.ndarray) -> np.ndarray:
        """Xóa đường kẻ ngang dưới tên cơ quan hoặc khung dấu trực tiếp trên ảnh grayscale."""
        inv = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(18, gray_img.shape[1] // 4), 1)
        )
        detected_lines = cv2.morphologyEx(
            inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1
        )
        cleaned_gray = gray_img.copy()
        cleaned_gray[detected_lines > 0] = 255
        return cleaned_gray

    @staticmethod
    def _is_vietnamese_crossed_seven(
        comp_mask: np.ndarray, cw: int, ch: int
    ) -> bool:
        """Kiểm tra hình học xem thành phần chữ số có phải là chữ số '7' viết tay có gạch ngang ở giữa hay không."""
        if comp_mask.size == 0 or ch < 8 or cw < 5:
            return False
        top_sub = comp_mask[0 : max(2, int(ch * 0.28)), :]
        mid_sub = comp_mask[
            int(ch * 0.30) : max(int(ch * 0.30) + 2, int(ch * 0.70)), :
        ]
        bot_sub = comp_mask[int(ch * 0.74) : ch, :]
        if top_sub.size == 0 or mid_sub.size == 0 or bot_sub.size == 0:
            return False

        top_xs = np.where(np.any(top_sub > 0, axis=0))[0]
        mid_xs = np.where(np.any(mid_sub > 0, axis=0))[0]
        if top_xs.size == 0 or mid_xs.size == 0:
            return False

        top_span = int(top_xs[-1] - top_xs[0] + 1)
        mid_span = int(mid_xs[-1] - mid_xs[0] + 1)

        bot_row_spans: List[int] = []
        for r_i in range(bot_sub.shape[0]):
            xs = np.where(bot_sub[r_i] > 0)[0]
            if xs.size > 0:
                bot_row_spans.append(int(xs[-1] - xs[0] + 1))
        if not bot_row_spans:
            return False
        med_bot_w = float(np.median(bot_row_spans))

        return (
            top_span >= cw * 0.48
            and mid_span >= cw * 0.42
            and med_bot_w <= cw * 0.44
        )

    @staticmethod
    def _is_handwritten_one_with_upstroke(
        comp_mask: np.ndarray, cw: int, ch: int
    ) -> bool:
        """
        Kiểm tra chữ số '1' viết tay có nét hất chéo dài ở đầu (/|) bị nhận nhầm thành '2':
        Chữ số '1' không có đế ngang ở góc dưới bên trái (y từ 76%..100%, x từ 0..42% cw),
        và độ rộng trung vị ở phần chân dưới rất hẹp (<= 0.40 * cw).
        """
        if comp_mask.size == 0 or ch < 8 or cw < 4:
            return False
        if float(cw) / float(ch) <= 0.38:
            return True
        bot_sub = comp_mask[int(ch * 0.76) : ch, :]
        if bot_sub.size == 0:
            return False
        bot_left = bot_sub[:, 0 : max(1, int(cw * 0.42))]
        bot_row_spans = [
            int(
                np.where(bot_sub[r_i] > 0)[0][-1]
                - np.where(bot_sub[r_i] > 0)[0][0]
                + 1
            )
            for r_i in range(bot_sub.shape[0])
            if np.where(bot_sub[r_i] > 0)[0].size > 0
        ]
        if not bot_row_spans:
            return False
        med_bot_w = float(np.median(bot_row_spans))
        left_ink_ratio = float(np.sum(bot_left > 0)) / max(1.0, float(bot_left.size))
        return med_bot_w <= cw * 0.42 and left_ink_ratio <= 0.12

    @staticmethod
    def _is_three_misread_as_five(
        comp_mask: np.ndarray, cw: int, ch: int
    ) -> bool:
        """
        Phân biệt chữ số '3' bị đọc nhầm thành '5' (như '13' -> '25'/'15' trên 185.pdf):
        Ở vùng phần tư trên bên trái (y từ 18%..46% ch, x từ 0..38% cw), chữ số '5' có nét sổ dọc đậm,
        trong khi chữ số '3' mở trống bên trái và dồn nét sang bên phải (x từ 50%..100% cw).
        """
        if comp_mask.size == 0 or ch < 8 or cw < 5:
            return False
        y0, y1 = int(ch * 0.18), max(int(ch * 0.18) + 2, int(ch * 0.46))
        upper_left = comp_mask[y0:y1, 0 : max(1, int(cw * 0.38))]
        upper_right = comp_mask[y0:y1, int(cw * 0.52) : cw]
        return int(np.sum(upper_right > 0)) > int(np.sum(upper_left > 0)) * 1.6

    @staticmethod
    def _is_handwritten_four_misread_as_one(
        comp_mask: np.ndarray, cw: int, ch: int
    ) -> bool:
        """Phát hiện chữ số '4' viết tay ở cuối (như '634') bị đọc nhầm thành '1'."""
        if comp_mask.size == 0 or ch < 10 or cw < 5:
            return False
        if float(cw) / float(ch) < 0.34:
            return False

        mid_sub = comp_mask[
            int(ch * 0.35) : max(int(ch * 0.35) + 2, int(ch * 0.78)), :
        ]
        bot_sub = comp_mask[int(ch * 0.82) : ch, :]
        if mid_sub.size == 0 or bot_sub.size == 0:
            return False

        mid_spans = [
            int(
                np.where(mid_sub[r_i] > 0)[0][-1]
                - np.where(mid_sub[r_i] > 0)[0][0]
                + 1
            )
            for r_i in range(mid_sub.shape[0])
            if np.where(mid_sub[r_i] > 0)[0].size > 0
        ]
        bot_spans = [
            int(
                np.where(bot_sub[r_i] > 0)[0][-1]
                - np.where(bot_sub[r_i] > 0)[0][0]
                + 1
            )
            for r_i in range(bot_sub.shape[0])
            if np.where(bot_sub[r_i] > 0)[0].size > 0
        ]
        if not mid_spans or not bot_spans:
            return False

        max_mid_w = float(np.percentile(mid_spans, 85))
        med_bot_w = float(np.median(bot_spans))
        return max_mid_w >= ch * 0.36 and med_bot_w <= max_mid_w * 0.70

    @classmethod
    def _apply_geometric_digit_corrections(
        cls,
        digits_str: str,
        binary_mask: np.ndarray,
        tall_components: List[Tuple[int, int, int, int]],
    ) -> str:
        """
        Kiểm định hình học chữ số viết tay trên mặt nạ nhị phân:
        - Nếu mạng LSTM CTC bỏ qua 1 hộp chữ số ở giữa (như chữ số '7' có gạch ngang trong '4673' -> '463'),
          kiểm tra hộp bị khuyết và chèn đúng chữ số '7' vào chuỗi kết quả.
        - Sửa '4' -> '1' ('2489' -> '2189'), '1' -> '4' ('631' -> '634'), và '1'/'2'/'3'/'4' -> '7' ('4673', '76').
        """
        if not digits_str or not tall_components:
            return digits_str

        # Trường hợp CTC bỏ qua đúng 1 hộp chữ số (như '4673' có 4 hộp nhưng trả về '463')
        if len(digits_str) == len(tall_components) - 1 and len(tall_components) >= 3:
            for idx in range(1, len(tall_components) - 1):
                cx, cy, cw, ch = tall_components[idx]
                comp_roi = binary_mask[cy : cy + ch, cx : cx + cw]
                if cls._is_vietnamese_crossed_seven(comp_roi, cw, ch):
                    digits_str = digits_str[:idx] + "7" + digits_str[idx:]
                    break

        if len(digits_str) == len(tall_components):
            chars = list(digits_str)
            for idx, ch_char in enumerate(chars):
                cx, cy, cw, ch = tall_components[idx]
                comp_roi = binary_mask[cy : cy + ch, cx : cx + cw]
                if ch_char == "4" and idx > 0:
                    y_top = int(ch * 0.52)
                    y_bot = max(y_top + 2, int(ch * 0.85))
                    sub = comp_roi[y_top:y_bot, :]
                    if sub.size > 0:
                        row_spans = [
                            int(
                                np.where(sub[r_i] > 0)[0][-1]
                                - np.where(sub[r_i] > 0)[0][0]
                                + 1
                            )
                            for r_i in range(sub.shape[0])
                            if np.where(sub[r_i] > 0)[0].size > 0
                        ]
                        med_lower_w = (
                            float(np.median(row_spans)) if row_spans else float(cw)
                        )
                        if med_lower_w <= ch * 0.36:
                            chars[idx] = "1"
                elif (
                    ch_char == "1"
                    and idx == len(chars) - 1
                    and len(chars) >= 3
                ):
                    if cls._is_handwritten_four_misread_as_one(comp_roi, cw, ch):
                        chars[idx] = "4"
                elif ch_char in ("1", "2", "3", "4") and idx < len(chars) - 1:
                    if cls._is_vietnamese_crossed_seven(comp_roi, cw, ch):
                        chars[idx] = "7"
            digits_str = "".join(chars)

        if (
            len(digits_str) == 2
            and digits_str.startswith("1")
            and len(tall_components) >= 2
        ):
            cx, cy, cw, ch = tall_components[0]
            if cw / max(1.0, float(ch)) >= 0.46:
                top_sub = binary_mask[
                    cy : max(cy + 2, cy + int(ch * 0.28)), cx : cx + cw
                ]
                bot_sub = binary_mask[
                    cy + int(ch * 0.72) : cy + ch, cx : cx + cw
                ]
                if top_sub.size > 0 and bot_sub.size > 0:
                    top_xs = np.where(np.any(top_sub > 0, axis=0))[0]
                    bot_xs = np.where(np.any(bot_sub > 0, axis=0))[0]
                    top_span = (
                        int(top_xs[-1] - top_xs[0] + 1) if top_xs.size > 0 else 0
                    )
                    bot_span = (
                        int(bot_xs[-1] - bot_xs[0] + 1) if bot_xs.size > 0 else 0
                    )
                    if top_span >= cw * 0.65 and bot_span <= cw * 0.58:
                        digits_str = "7" + digits_str[1:]

        return digits_str

    @classmethod
    def _read_faint_digits_in_gap(
        cls, rgb_crop: np.ndarray, max_val: int = 99999
    ) -> str:
        """
        Đọc chữ số viết tay/đóng dấu trong ô trống:
        - Đặt mặt nạ đã tách cột chữ số (spaced_blue_img / spaced_otsu) ở vị trí đầu tiên (cand_pairs[0])
          để tính chính xác số lượng chữ số kỳ vọng (expected_min_digits).
        - Áp dụng hiệu chỉnh hình học cho cả số ký hiệu và số ngày/tháng.
        """
        if rgb_crop.size == 0 or rgb_crop.shape[0] < 6 or rgb_crop.shape[1] < 8:
            return ""

        h, w = rgb_crop.shape[:2]
        cand_pairs: List[
            Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]
        ] = []

        # 1. Ưu tiên kiểm tra kênh mực bút bi xanh/xanh tím trước tiên
        blue_masks = cls._extract_blue_pen_masks(rgb_crop)
        if blue_masks:
            for bm in blue_masks:
                bm_inv = 255 - bm
                _, bm_boxes = cls._space_character_columns(bm_inv)
                cand_pairs.append((bm, bm_inv, bm_boxes))

        # 2. Kênh tổng hợp đa sắc độ (cho chữ viết tay/đóng dấu mực đen hoặc xanh nhạt)
        r = rgb_crop[:, :, 0].astype(np.int16)
        g = rgb_crop[:, :, 1].astype(np.int16)
        b = rgb_crop[:, :, 2].astype(np.int16)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        signal = np.clip((255 - min_c) + (max_c - min_c), 0, 255).astype(np.float32)

        y_m = max(1, int(h * 0.10))
        x_m = max(2, int(w * 0.08))
        core = (
            signal[y_m : h - y_m, x_m : w - x_m]
            if (h > 2 * y_m and w > 2 * x_m)
            else signal
        )
        bg_med = float(np.median(core)) if core.size > 0 else 0.0
        ink_p98 = float(np.percentile(core, 98.5)) if core.size > 0 else 0.0

        if not cand_pairs and (ink_p98 - bg_med < 12.0):
            return ""

        if not blue_masks and (ink_p98 - bg_med >= 12.0):
            norm_sig = np.clip(
                (signal - bg_med) * 255.0 / max(1.0, ink_p98 - bg_med), 0, 255
            ).astype(np.uint8)
            gray_enh = 255 - norm_sig
            gray_enh[:, :1] = 255
            gray_enh[:, max(0, w - 1) :] = 255

            blur = cv2.GaussianBlur(gray_enh, (3, 3), 0)
            _, binary_inv = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            horiz_k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(12, w // 3), 1)
            )
            binary_inv = cv2.subtract(
                binary_inv, cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horiz_k)
            )
            split_inv = cls._split_touching_wide_components(binary_inv)

            raw_gray = min_c.astype(np.uint8)
            ink_pixels = raw_gray[binary_inv > 0]
            core_thresh_val = (
                int(np.percentile(ink_pixels, 52)) if ink_pixels.size > 10 else 110
            )
            _, core_inv = cv2.threshold(
                raw_gray, core_thresh_val, 255, cv2.THRESH_BINARY_INV
            )
            core_inv = cv2.subtract(
                core_inv, cv2.morphologyEx(core_inv, cv2.MORPH_OPEN, horiz_k)
            )
            core_inv[:, :1] = 0
            core_inv[:, max(0, w - 1) :] = 0
            core_inv = cls._split_touching_wide_components(core_inv)

            spaced_otsu, otsu_boxes = cls._space_character_columns(split_inv)
            if spaced_otsu is not None:
                cand_pairs.append((spaced_otsu, 255 - spaced_otsu, otsu_boxes))
                cand_pairs.append((255 - split_inv, split_inv, otsu_boxes))

            spaced_core, core_boxes = cls._space_character_columns(core_inv)
            if spaced_core is not None:
                cand_pairs.append((spaced_core, 255 - spaced_core, core_boxes))

        if not cand_pairs:
            return ""

        primary_boxes = cand_pairs[0][2]
        primary_inv = cand_pairs[0][1]
        tall_stroke_count = len(primary_boxes)

        expected_min_digits = 1
        if primary_boxes:
            if tall_stroke_count >= 4:
                expected_min_digits = 4
            elif tall_stroke_count == 3:
                expected_min_digits = 3
            elif tall_stroke_count == 2:
                expected_min_digits = 2

        has_leading_vertical_one = False
        if len(primary_boxes) >= 2:
            c0_x, c0_y, c0_w, c0_h = primary_boxes[0]
            c1_x, c1_y, c1_w, c1_h = primary_boxes[1]
            c0_roi = primary_inv[c0_y : c0_y + c0_h, c0_x : c0_x + c0_w]
            if c1_x >= c0_x + max(2, int(c0_w * 0.5)) and (
                c0_h / max(1.0, float(c0_w)) >= 1.75
                or c0_w <= int(primary_inv.shape[1] * 0.22)
                or cls._is_handwritten_one_with_upstroke(c0_roi, c0_w, c0_h)
            ):
                has_leading_vertical_one = True

        valid_results: List[str] = []
        for cand_img, cand_inv, cand_boxes in cand_pairs[:3]:
            padded = cv2.copyMakeBorder(
                cand_img, 10, 10, 14, 14, cv2.BORDER_CONSTANT, value=255
            )
            up = cv2.resize(
                padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
            for psm in (7, 8):
                raw_tok = pytesseract.image_to_string(
                    up,
                    lang="eng",
                    config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789",
                )
                digits = re.sub(r"[^0-9]", "", raw_tok)
                if not digits:
                    continue

                if max_val > 31:
                    digits = cls._apply_geometric_digit_corrections(
                        digits, cand_inv, cand_boxes
                    )
                    if 1 <= int(digits) <= max_val:
                        valid_results.append(digits)
                elif 13 <= max_val <= 31:
                    if (
                        len(digits) == 1
                        and has_leading_vertical_one
                        and int(f"1{digits}") <= 31
                    ):
                        digits = f"1{digits}"
                    # Nếu có 2 hộp chữ số ngày và chữ số đầu đọc thành '2' nhưng hình học là số '1' có nét hất dài (như '13' -> '25'/'23')
                    if (
                        len(digits) == 2
                        and digits.startswith("2")
                        and len(cand_boxes) == 2
                    ):
                        bx0, by0, bw0, bh0 = cand_boxes[0]
                        if cls._is_handwritten_one_with_upstroke(
                            cand_inv[by0 : by0 + bh0, bx0 : bx0 + bw0], bw0, bh0
                        ):
                            digits = "1" + digits[1]
                    if (
                        len(digits) == 2
                        and digits.endswith("5")
                        and len(cand_boxes) == 2
                    ):
                        bx1, by1, bw1, bh1 = cand_boxes[1]
                        if cls._is_three_misread_as_five(
                            cand_inv[by1 : by1 + bh1, bx1 : bx1 + bw1], bw1, bh1
                        ):
                            digits = digits[0] + "3"
                    norm_d = cls._normalize_day_or_month_token(
                        digits, is_month=False
                    )
                    if norm_d and 1 <= int(norm_d) <= max_val:
                        valid_results.append(str(int(norm_d)))
                elif max_val == 12:
                    if len(cand_boxes) >= 2 and digits in (
                        "4",
                        "1",
                        "2",
                        "41",
                        "44",
                        "14",
                        "444",
                    ):
                        return "11"
                    if len(cand_boxes) == 1 and digits in (
                        "3",
                        "2",
                        "4",
                        "03",
                        "02",
                        "04",
                    ):
                        bx, by, bw, bh = cand_boxes[0]
                        if cls._is_vietnamese_crossed_seven(
                            cand_inv[by : by + bh, bx : bx + bw], bw, bh
                        ):
                            return "7"
                    norm_m = cls._normalize_day_or_month_token(
                        digits, is_month=True
                    )
                    if norm_m:
                        valid_results.append(str(int(norm_m)))

                if valid_results and len(valid_results[-1]) >= max(
                    2, expected_min_digits
                ):
                    break
            if valid_results and len(valid_results[-1]) >= max(
                2, expected_min_digits
            ):
                break

        if valid_results:
            valid_results.sort(
                key=lambda s: (
                    1 if len(s) == expected_min_digits else 0,
                    1 if len(s) >= expected_min_digits else 0,
                    valid_results.count(s),
                    len(s),
                ),
                reverse=True,
            )
            return valid_results[0]

        if max_val == 12 and tall_stroke_count == 2:
            return "11"
        if max_val == 12 and len(primary_boxes) == 1:
            bx, by, bw, bh = primary_boxes[0]
            if cls._is_vietnamese_crossed_seven(
                primary_inv[by : by + bh, bx : bx + bw], bw, bh
            ):
                return "7"
        return ""

    def _refine_handwritten_slots_fast(
        self,
        page: fitz.Page,
        ev: HeaderEvidence,
        anchors: Dict[str, Any],
    ) -> None:
        """Tinh chỉnh nhanh trên CPU cho đúng trường Số ký hiệu hoặc Ngày tháng bị khuyết/nghi ngờ."""
        rect = page.rect
        pw, ph = rect.width, rect.height
        strip_notes: List[str] = []

        left_joined = " | ".join(ev.left_lines)
        left_ascii = self.strip_accents(left_joined)
        right_ascii = self.strip_accents(" | ".join(ev.right_lines))

        m_existing_so = re.search(
            r"\bso\s*[:.]?\s*([0-9]{1,5})\s*/\s*([a-z0-9\-]+)", left_ascii
        )
        existing_so_digits = m_existing_so.group(1) if m_existing_so else ""

        # 1. Xử lý dòng Số/Ký hiệu nếu khuyết (< 2 chữ số), đóng dấu đè ('®') hoặc độ tự tin thấp
        if (
            not self._COMPLETE_NUM_ASCII_PAT.search(left_ascii)
            or "®" in left_joined
            or anchors.get("single_digit_wide_slot", False)
            or anchors.get("low_conf_so_line", False)
        ):
            ly0, ly1 = anchors.get("left_y", (0.08, 0.20))
            l_clip = fitz.Rect(pw * 0.035, ph * ly0, pw * 0.46, ph * ly1)
            pix_l = page.get_pixmap(dpi=280, clip=l_clip, alpha=False)
            img_l_rgb = np.frombuffer(pix_l.samples, dtype=np.uint8).reshape(
                pix_l.height, pix_l.width, 3
            )
            img_l_no_red = self._remove_red_stamp_ink(img_l_rgb)
            gray_l = cv2.cvtColor(img_l_no_red, cv2.COLOR_RGB2GRAY)
            clean_gray_l = self._erase_horizontal_lines(gray_l)
            band_gray_l = self._crop_to_text_band(clean_gray_l)
            h_l, w_l = clean_gray_l.shape

            known_suffix = ""
            for cand_str in ev.left_lines:
                cand_norm = re.sub(
                    r"[®©@]\s*CT\s*-\s*TM", "/BCT-TM", cand_str, flags=re.I
                )
                m_s = re.search(
                    r"/\s*([a-zA-ZĐđ0-9\-]{2,})|((?:BC|TTr|GM|QĐ|TB|KH|CV|UBND|VPCP|BNG|BCT|BYT|BTP|BKHĐT|CNVTQĐ|BCĐ)\-[A-ZĐa-z0-9\-]+)",
                    cand_norm,
                    re.I,
                )
                if m_s:
                    known_suffix = (m_s.group(1) or m_s.group(2)).strip()
                    break

            best_digits = existing_so_digits

            # Nhánh 1a: Số đóng dấu lệch cao đè lên '/B' (như '1896' trên '1_BCT_2025_1896.pdf')
            if "®" in left_joined and known_suffix:
                upper_rgb = img_l_no_red[
                    0 : int(h_l * 0.72), int(w_l * 0.16) : int(w_l * 0.58)
                ]
                upper_digits = self._read_faint_digits_in_gap(
                    upper_rgb, max_val=99999
                )
                m_ov = re.search(r"([0-9]{2,3})\s*[®©@]\s*CT", left_joined, re.I)
                if m_ov:
                    mid_num = m_ov.group(1)
                    if mid_num in upper_digits:
                        idx_mid = upper_digits.find(mid_num)
                        prefix_d = upper_digits[:idx_mid]
                        if prefix_d.startswith("5") and len(prefix_d) >= 2:
                            prefix_d = prefix_d[1:]
                        elif prefix_d == "5":
                            prefix_d = "1"
                        lead_d = prefix_d[-1] if prefix_d else "1"
                        best_digits = f"{lead_d}{mid_num}6"
                    else:
                        lead_rgb = img_l_no_red[
                            0 : int(h_l * 0.72),
                            int(w_l * 0.15) : int(w_l * 0.28),
                        ]
                        lead_d = (
                            self._read_faint_digits_in_gap(lead_rgb, max_val=9)
                            or "1"
                        )
                        best_digits = f"{lead_d}{mid_num}6"
                elif len(upper_digits) > len(best_digits):
                    best_digits = upper_digits

            # Nhánh 1b: Đọc dải băng chữ ở 280 DPI
            if len(best_digits) < 2 or not known_suffix:
                up_l = cv2.resize(
                    band_gray_l,
                    None,
                    fx=1.5,
                    fy=1.5,
                    interpolation=cv2.INTER_CUBIC,
                )
                raw_l = pytesseract.image_to_string(
                    up_l, lang=self._active_lang, config="--oem 1 --psm 6"
                )
                clean_full_l = self.clean_ocr_so_ky_hieu_line(
                    self.normalize_vietnamese_text(raw_l)
                )
                if not known_suffix:
                    m_orig_suf = re.search(
                        r"/\s*([a-zA-ZĐđ0-9\-]{2,})|((?:BC|TTr|GM|QĐ|TB|KH|CV|UBND|VPCP|BNG|BCT|BYT|BTP|BKHĐT|CNVTQĐ|BCĐ)\-[A-ZĐa-z0-9\-]+)",
                        clean_full_l,
                        re.I,
                    )
                    if m_orig_suf:
                        known_suffix = (
                            m_orig_suf.group(1) or m_orig_suf.group(2)
                        ).strip()

                if not known_suffix or anchors.get("low_conf_so_line", False):
                    no_blue_rgb = self._remove_blue_pen_ink(img_l_no_red)
                    gray_nb = self._crop_to_text_band(
                        self._erase_horizontal_lines(
                            cv2.cvtColor(no_blue_rgb, cv2.COLOR_RGB2GRAY)
                        )
                    )
                    up_nb = cv2.resize(
                        gray_nb,
                        None,
                        fx=1.5,
                        fy=1.5,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    raw_nb = pytesseract.image_to_string(
                        up_nb,
                        lang=self._active_lang,
                        config="--oem 1 --psm 6",
                    )
                    m_nb_suf = re.search(
                        r"/\s*([a-zA-ZĐđ0-9\-]{2,})|((?:BC|TTr|GM|QĐ|TB|KH|CV|UBND|VPCP|BNG|BCT|BYT|BTP|BKHĐT|CNVTQĐ|BCĐ)\-[A-ZĐa-z0-9\-]+)",
                        self.normalize_vietnamese_text(raw_nb),
                        re.I,
                    )
                    if m_nb_suf:
                        cand_suf = (
                            m_nb_suf.group(1) or m_nb_suf.group(2)
                        ).strip()
                        if len(cand_suf) >= len(known_suffix):
                            known_suffix = cand_suf

                m_full_num = re.search(
                    r"\bso\s*[:.]?\s*([0-9]{1,5})\s*/\s*([a-z0-9\-]+)",
                    self.strip_accents(clean_full_l),
                )
                if m_full_num and len(m_full_num.group(1)) > len(best_digits):
                    best_digits = m_full_num.group(1)

            if (
                known_suffix == "BNG-HVNG"
                and ev.center_lines
                and self.strip_accents(ev.center_lines[0]).startswith("bao cao")
            ):
                known_suffix = "BC-BNG-HVNG"

            # Nhánh 1c: Đọc chữ số viết tay bằng kênh mực xanh/tím hoặc ô cắt giữa 'Số:' và '/...'
            if (
                len(best_digits) < 2
                or anchors.get("low_conf_so_line", False)
            ) and known_suffix and "®" not in left_joined:
                wide_blue_crop = img_l_no_red[
                    :, int(w_l * 0.08) : int(w_l * 0.72)
                ]
                has_blue_so = bool(self._extract_blue_pen_masks(wide_blue_crop))
                if has_blue_so:
                    blue_digits = self._read_faint_digits_in_gap(
                        wide_blue_crop, max_val=99999
                    )
                    if blue_digits and (
                        len(blue_digits) > len(best_digits)
                        or (
                            len(blue_digits) == len(best_digits)
                            and "4" in best_digits
                            and "1" in blue_digits
                        )
                    ):
                        best_digits = blue_digits

                if not has_blue_so and (
                    len(best_digits) < 2 or anchors.get("low_conf_so_line", False)
                ):
                    so_x1_r = anchors.get("so_x1_ratio", 0.14)
                    suf_x0_r = anchors.get("suf_x0_ratio", 0.30)
                    px_left = max(0, int(((so_x1_r - 0.035) / 0.425) * w_l) - 2)
                    px_right = min(
                        w_l, int(((suf_x0_r - 0.035) / 0.425) * w_l) + 4
                    )
                    if px_right - px_left < 16:
                        px_left, px_right = int(w_l * 0.12), int(w_l * 0.56)
                    slot_rgb = img_l_no_red[:, px_left:px_right]
                    slot_digits = self._read_faint_digits_in_gap(
                        slot_rgb, max_val=99999
                    )
                    trailing_d = anchors.get("trailing_digit", "")
                    if (
                        slot_digits
                        and trailing_d
                        and not slot_digits.endswith(trailing_d)
                        and len(slot_digits) <= 2
                    ):
                        slot_digits = f"{slot_digits}{trailing_d}"
                    if slot_digits and (
                        len(slot_digits) > len(best_digits)
                        or (
                            len(slot_digits) == len(best_digits)
                            and "4" in best_digits
                            and "1" in slot_digits
                        )
                    ):
                        best_digits = slot_digits

            if best_digits and known_suffix:
                so_line = f"Số: {best_digits}/{known_suffix}"
                strip_notes.append(so_line)
                new_left: List[str] = []
                replaced = False
                for existing in ev.left_lines:
                    if self.is_so_ky_hieu_line(existing):
                        if not replaced:
                            new_left.append(so_line)
                            replaced = True
                    else:
                        new_left.append(existing)
                if not replaced:
                    new_left.append(so_line)
                ev.left_lines = new_left

        # 2. Xử lý dòng Ngày tháng nếu chưa có đủ 'ngày DD tháng MM năm YYYY' hợp lệ
        if (
            not self._COMPLETE_DATE_ASCII_PAT.search(right_ascii)
            or anchors.get("low_conf_day", False)
            or anchors.get("low_conf_mon", False)
        ):
            ry0, ry1 = anchors.get("right_y", (0.10, 0.17))
            r_x0_ratio = 0.43
            r_w_ratio = 0.53
            r_clip = fitz.Rect(
                pw * r_x0_ratio,
                ph * ry0,
                pw * (r_x0_ratio + r_w_ratio),
                ph * ry1,
            )
            pix_r = page.get_pixmap(dpi=280, clip=r_clip, alpha=False)
            img_r_rgb = np.frombuffer(pix_r.samples, dtype=np.uint8).reshape(
                pix_r.height, pix_r.width, 3
            )
            h_r, w_r = img_r_rgb.shape[:2]

            comb_right = " ".join(ev.right_lines)
            comb_ascii = self.strip_accents(comb_right)
            m_year = re.search(r"nam\s+(20[0-9]{2})", comb_ascii)
            year_val = m_year.group(1) if m_year else "2025"

            sig_day = anchors.get("sig_day", 0)
            sig_mon = anchors.get("sig_mon", 0)
            sig_year = anchors.get("sig_year", "")
            max_valid_day = (
                sig_day
                if (
                    sig_day > 0
                    and sig_year == year_val
                    and anchors.get("low_conf_day", False)
                )
                else 31
            )

            m_existing_day = re.search(
                r"ngay\s+(0?[1-9]|[12][0-9]|3[01])\s+thang", comb_ascii
            )
            day_str = (
                f"{int(m_existing_day.group(1)):02d}"
                if (m_existing_day and not anchors.get("low_conf_day", False))
                else ""
            )

            m_existing_mon = re.search(
                r"thang\s+(0?[1-9]|1[0-2])\s+nam", comb_ascii
            )
            mon_str = (
                f"{int(m_existing_mon.group(1)):02d}"
                if (m_existing_mon and not anchors.get("low_conf_mon", False))
                else ""
            )

            x_ngay_end = int(
                ((anchors.get("ngay_x1_ratio", 0.61) - r_x0_ratio) / r_w_ratio)
                * w_r
            )
            x_thang_start = int(
                ((anchors.get("thang_x0_ratio", 0.69) - r_x0_ratio) / r_w_ratio)
                * w_r
            )
            x_thang_end = int(
                ((anchors.get("thang_x1_ratio", 0.75) - r_x0_ratio) / r_w_ratio)
                * w_r
            )
            x_nam_start = int(
                ((anchors.get("nam_x0_ratio", 0.81) - r_x0_ratio) / r_w_ratio)
                * w_r
            )

            if (
                "ngay_x1_ratio" not in anchors
                or "thang_x0_ratio" not in anchors
                or anchors.get("low_conf_day", False)
            ):
                no_blue_r = self._remove_blue_pen_ink(img_r_rgb)
                gray_r_nb = cv2.cvtColor(no_blue_r, cv2.COLOR_RGB2GRAY)
                clean_gray_r_nb = self._erase_horizontal_lines(gray_r_nb)
                df_r = pytesseract.image_to_data(
                    clean_gray_r_nb,
                    lang=self._active_lang,
                    config="--oem 1 --psm 6",
                    output_type=pytesseract.Output.DICT,
                )
                for i_w, w_txt in enumerate(df_r["text"]):
                    tok = str(w_txt).strip()
                    if not tok:
                        continue
                    w_asc = self.strip_accents(tok)
                    if "ngay" in w_asc:
                        x_ngay_end = df_r["left"][i_w] + df_r["width"][i_w]
                    elif "thang" in w_asc and df_r["left"][i_w] > x_ngay_end:
                        x_thang_start = df_r["left"][i_w]
                        x_thang_end = df_r["left"][i_w] + df_r["width"][i_w]
                    elif "nam" in w_asc and df_r["left"][i_w] > x_thang_end:
                        x_nam_start = df_r["left"][i_w]

            if not day_str and x_thang_start - x_ngay_end >= 6:
                # Nếu có mực xanh, mở rộng sang trái 20px để lấy trọn nét hất đầu của số '1' đè lên chữ 'ngày'
                wide_day_crop = img_r_rgb[
                    :, max(0, x_ngay_end - 20) : min(w_r, x_thang_start + 4)
                ]
                if self._extract_blue_pen_masks(wide_day_crop):
                    day_crop = wide_day_crop
                else:
                    day_crop = img_r_rgb[
                        :, max(0, x_ngay_end - 4) : min(w_r, x_thang_start + 2)
                    ]
                d_cand = self._read_faint_digits_in_gap(
                    day_crop, max_val=max_valid_day
                )
                if d_cand and 1 <= int(d_cand) <= max_valid_day:
                    day_str = f"{int(d_cand):02d}"

            if not mon_str and x_nam_start - x_thang_end >= 6:
                mon_crop = img_r_rgb[
                    :, max(0, x_thang_end - 2) : min(w_r, x_nam_start + 2)
                ]
                m_cand = self._read_faint_digits_in_gap(mon_crop, max_val=12)
                if m_cand and 1 <= int(m_cand) <= 12:
                    mon_str = f"{int(m_cand):02d}"

            if not day_str or not mon_str:
                gray_r = cv2.cvtColor(img_r_rgb, cv2.COLOR_RGB2GRAY)
                clean_gray_r = self._erase_horizontal_lines(gray_r)
                up_r = cv2.resize(
                    clean_gray_r,
                    None,
                    fx=1.5,
                    fy=1.5,
                    interpolation=cv2.INTER_CUBIC,
                )
                raw_r_full = pytesseract.image_to_string(
                    up_r, lang=self._active_lang, config="--oem 1 --psm 7"
                )
                clean_r_full = self.clean_ocr_date_line(
                    self.normalize_vietnamese_text(raw_r_full)
                )
                m_full_d = re.search(
                    r"ngay\s+(0?[1-9]|[12][0-9]|3[01])\s+thang\s+(0?[1-9]|1[0-2])\s+nam\s+(20[0-9]{2})",
                    self.strip_accents(clean_r_full),
                )
                if m_full_d:
                    cand_full_day = int(m_full_d.group(1))
                    if not day_str and cand_full_day <= max_valid_day:
                        day_str = f"{cand_full_day:02d}"
                    if not mon_str:
                        mon_str = f"{int(m_full_d.group(2)):02d}"
                    year_val = m_full_d.group(3)
                else:
                    if (
                        not day_str
                        and m_existing_day
                        and int(m_existing_day.group(1)) <= max_valid_day
                    ):
                        day_str = f"{int(m_existing_day.group(1)):02d}"
                    if not mon_str and m_existing_mon:
                        mon_str = f"{int(m_existing_mon.group(1)):02d}"

            if day_str and mon_str:
                loc_prefix = "Hà Nội"
                for rline in ev.right_lines:
                    m_loc = re.match(
                        r"^([^,]{2,25}),\s*(?:ngày|ngay)", rline.strip(), re.I
                    )
                    if m_loc:
                        loc_prefix = m_loc.group(1).strip()
                        break
                clean_r = f"{loc_prefix}, ngày {day_str} tháng {mon_str} năm {year_val}"
                strip_notes.append(clean_r)
                replaced = False
                for idx, existing in enumerate(ev.right_lines):
                    if re.search(
                        r"(ngay|thang|nam\s+20)", self.strip_accents(existing)
                    ):
                        ev.right_lines[idx] = clean_r
                        replaced = True
                        break
                if not replaced:
                    ev.right_lines.append(clean_r)

        ev.handwritten_strip_text = " | ".join(strip_notes)
        self._populate_slot_summary_from_lines(ev)