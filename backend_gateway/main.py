from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from backend_gateway.config import settings
from backend_gateway.models.schemas import (
    RBACQueryRequest,
    RBACQueryResponse,
    CitationItem,
    OCRResponse,
    UserContext,
    EmbeddingRequest
)
from backend_gateway.services.rbac_service import RBACService
from backend_gateway.services.inference_router import DualInferenceRouter
from backend_gateway.services.ragflow_service import RAGFlowService
from backend_gateway.services.direct_engine_service import DirectEngineService

app = FastAPI(
    title="Enterprise AI OCR & RBAC RAG Gateway (Direct Code + Ollama/vLLM)",
    description="Hệ thống hỗ trợ chạy trực tiếp bằng mã nguồn (engine='direct') không cần Model LLM, hoặc kết nối Ollama/vLLM.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router_service = DualInferenceRouter()
ragflow_service = RAGFlowService()
direct_service = DirectEngineService()


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if settings.GATEWAY_API_KEY and x_api_key != settings.GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key header")
    return True


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "direct_code_engine": "ready (No external AI model required)",
        "ollama_online": router_service.ollama.is_available(),
        "vllm_online": router_service.vllm.is_llm_available()
    }


@app.post("/api/v1/ocr/extract", response_model=OCRResponse, dependencies=[Depends(verify_api_key)])
async def ocr_document(
    file: UploadFile = File(...),
    engine: Optional[str] = Form(
        default="direct",
        description="Mặc định 'direct' (chạy trực tiếp bằng mã nguồn không cần model). Hoặc chọn 'ollama', 'vllm'."
    ),
    ollama_model: Optional[str] = Form(default=None)
):
    try:
        content = await file.read()
        mode = (engine or "direct").strip().lower()
        if mode in ["direct", "direct_llm", "string", ""]:
            result = direct_service.extract_document_direct(content, file.filename, use_llm_refiner=(mode != "direct_fast"))
            return OCRResponse(**result)

        result = router_service.perform_ocr(
            file_bytes=content,
            filename=file.filename,
            preferred_engine=mode,
            ollama_ocr_model=ollama_model
        )
        return OCRResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý OCR: {str(e)}")


@app.post("/api/v1/documents/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_document_with_rbac(
    dataset_id: str = Form(default="ds_cds", description="Mã kho phòng ban: ds_general, ds_tccb, ds_khtc, ds_cds"),
    user_id: str = Form(default="nvquy"),
    user_role: str = Form(default="user"),
    user_groups: str = Form(default="Trung_Tam_CDS", description="Nhóm quyền: Lanh_Dao_Bo, Phong_TCCB, Phong_KHTC, Trung_Tam_CDS"),
    engine: str = Form(default="direct", description="'direct' (lưu & bóc tách trực tiếp) hoặc 'ragflow'"),
    file: UploadFile = File(...)
):
    groups_list = [g.strip() for g in user_groups.split(",") if g.strip()]
    user_ctx = UserContext(user_id=user_id, role=user_role, groups=groups_list)

    if not RBACService.can_upload_to_dataset(user_ctx, dataset_id):
        raise HTTPException(
            status_code=403,
            detail=f"Người dùng thuộc nhóm {groups_list} không có quyền tải tài liệu lên kho '{dataset_id}'."
        )

    file_bytes = await file.read()
    if engine.strip().lower() in ["direct", "string", ""]:
        extracted = direct_service.extract_document_direct(file_bytes, file.filename)
        return direct_service.ingest_to_local_store(dataset_id, file.filename, extracted["markdown_content"])

    return ragflow_service.upload_and_parse_document(dataset_id, file_bytes, file.filename)


@app.post("/api/v1/retrieval/rbac-query", response_model=RBACQueryResponse, dependencies=[Depends(verify_api_key)])
def query_internal_documents(req: RBACQueryRequest):
    allowed_datasets = RBACService.get_allowed_datasets(req.user)
    if not allowed_datasets:
        raise HTTPException(status_code=403, detail="Tài khoản chưa được cấp quyền truy cập kho văn bản nào.")

    pref = (req.engine_preference or "direct").strip().lower()

    # Nhánh 1: Chạy trực tiếp bằng mã nguồn (BM25 + Trích xuất nguyên văn theo phân quyền RBAC)
    if pref in ["direct", "string", ""]:
        chunks = direct_service.search_local_rbac(req.question, allowed_datasets, top_k=req.top_k)
        if not chunks:
            return RBACQueryResponse(
                answer="Không tìm thấy văn bản nội bộ nào khớp từ khóa trong phạm vi phân quyền của bạn.",
                engine_used="Direct Code (BM25 RBAC Search - No LLM)",
                allowed_datasets=allowed_datasets,
                citations=[]
            )

        citations = []
        summary_lines = [f"**Kết quả tra cứu trực tiếp từ kho văn bản ({', '.join(allowed_datasets)}):**\n"]
        for idx, c in enumerate(chunks, 1):
            summary_lines.append(f"**[{idx}] Nguồn: {c['document_keyword']} (Kho: `{c['kb_id']}`):**\n{c['content']}\n")
            citations.append(
                CitationItem(
                    document_id=c["document_id"],
                    document_name=c["document_keyword"],
                    dataset_id=c["kb_id"],
                    content_snippet=c["content"][:300],
                    similarity=c["similarity"]
                )
            )
        return RBACQueryResponse(
            answer="\n---\n".join(summary_lines),
            engine_used="Direct Code (BM25 RBAC Search - No LLM)",
            allowed_datasets=allowed_datasets,
            citations=citations
        )

    # Nhánh 2: Chạy qua RAGFlow + Ollama/vLLM nếu người dùng chỉ định
    chunks = ragflow_service.retrieve_chunks(
        question=req.question,
        dataset_ids=allowed_datasets,
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        vector_similarity_weight=req.vector_similarity_weight
    )
    if not chunks:
        return RBACQueryResponse(
            answer="Không tìm thấy văn bản nội bộ nào phù hợp trong phạm vi phân quyền của bạn.",
            engine_used="none",
            allowed_datasets=allowed_datasets,
            citations=[]
        )

    context_blocks = [f"[Tài liệu {i} - Nguồn: {c.get('document_keyword')}]\n{c.get('content', '').strip()}" for i, c in enumerate(chunks, 1)]
    citations = [
        CitationItem(
            document_id=c.get("document_id", ""),
            document_name=c.get("document_keyword", ""),
            dataset_id=c.get("kb_id", ""),
            content_snippet=c.get("content", "")[:300],
            similarity=float(c.get("similarity", 0.0))
        ) for c in chunks
    ]
    messages = [
        {"role": "system", "content": "Chỉ dựa vào ngữ cảnh sau để trả lời:\n\n" + "\n\n---\n\n".join(context_blocks)},
        {"role": "user", "content": req.question}
    ]
    answer, engine_used = router_service.generate_chat_answer(messages=messages, preferred_engine=pref)
    return RBACQueryResponse(answer=answer, engine_used=engine_used, allowed_datasets=allowed_datasets, citations=citations)
