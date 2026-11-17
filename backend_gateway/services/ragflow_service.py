import requests
from typing import List, Dict, Any
from backend_gateway.config import settings


class RAGFlowService:
    def __init__(self):
        self.base_url = settings.RAGFLOW_BASE_URL.rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.RAGFLOW_API_KEY}"}

    def create_dataset(self, name: str, description: str = "", chunk_method: str = "naive") -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/datasets"
        payload = {
            "name": name,
            "description": description,
            "permission": "team",
            "chunk_method": chunk_method,
            "parser_config": {
                "layout_recognize": "DeepDoc",
                "chunk_token_num": 512,
                "delimiter": "\n!?;。；！？",
                "auto_keywords": 5,
                "auto_questions": 2,
                "html4excel": False
            }
        }
        resp = requests.post(url, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_datasets(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/api/v1/datasets?page=1&page_size=100"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", []) if data.get("code") == 0 else []

    def upload_and_parse_document(self, dataset_id: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        upload_url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents"
        files = {"file": (filename, file_bytes)}
        resp = requests.post(upload_url, headers=self.headers, files=files, timeout=120)
        resp.raise_for_status()
        res_json = resp.json()
        if res_json.get("code") != 0:
            raise RuntimeError(f"RAGFlow Upload Error: {res_json.get('message')}")
        doc_ids = [d["id"] for d in res_json.get("data", [])]

        parse_url = f"{self.base_url}/api/v1/datasets/{dataset_id}/chunks"
        parse_resp = requests.post(
            parse_url,
            headers={**self.headers, "Content-Type": "application/json"},
            json={"document_ids": doc_ids},
            timeout=30
        )
        parse_resp.raise_for_status()
        return {"dataset_id": dataset_id, "document_ids": doc_ids, "status": "deepdoc_ocr_and_chunking_started"}

    def retrieve_chunks(
        self,
        question: str,
        dataset_ids: List[str],
        top_k: int = 6,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        if not dataset_ids:
            return []
        url = f"{self.base_url}/api/v1/retrieval"
        payload = {
            "question": question,
            "dataset_ids": dataset_ids,
            "page": 1,
            "page_size": top_k,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": 1024
        }
        resp = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, json=payload, timeout=30)
        resp.raise_for_status()
        res_json = resp.json()
        if res_json.get("code") == 0 and "data" in res_json:
            return res_json["data"].get("chunks", [])
        return []