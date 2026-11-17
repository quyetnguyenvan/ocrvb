from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    co_quan_ban_hanh: str = Field(default="", description="Cơ quan/Đơn vị ban hành văn bản")
    so_ky_hieu: str = Field(default="", description="Số và ký hiệu văn bản")
    ngay_ban_hanh: str = Field(default="", description="Ngày, tháng, năm ban hành")
    loai_van_ban: str = Field(default="", description="Tên loại văn bản (Công văn, Tờ trình, Quyết định, Giấy mời...)")
    trich_yeu: str = Field(default="", description="Trích yếu nội dung văn bản")
    so_trang: int = Field(default=1, description="Tổng số trang của tài liệu")


class OCRResponse(BaseModel):
    filename: str
    total_pages: int
    metadata: DocumentMetadata
    markdown_content: str
    engine: str


class UserContext(BaseModel):
    user_id: str = Field(..., description="ID của người dùng")
    email: Optional[str] = None
    name: Optional[str] = "User"
    role: str = Field(default="user")
    groups: List[str] = Field(default_factory=list)


class RBACQueryRequest(BaseModel):
    question: str
    user: UserContext
    engine_preference: Optional[str] = Field(default=None)
    top_k: int = Field(default=6, ge=1, le=20)
    similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    vector_similarity_weight: float = Field(default=0.3)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class CitationItem(BaseModel):
    document_id: str
    document_name: str
    dataset_id: str
    content_snippet: str
    similarity: float


class RBACQueryResponse(BaseModel):
    answer: str
    engine_used: str
    allowed_datasets: List[str]
    citations: List[CitationItem]


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: Optional[str] = None