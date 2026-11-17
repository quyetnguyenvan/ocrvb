import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend_gateway.config import settings
from backend_gateway.services.ollama_service import OllamaService

def main():
    ollama = OllamaService()
    print(f"=== KIỂM TRA KẾT NỐI OLLAMA SERVER ({settings.OLLAMA_BASE_URL}) ===")
    if not ollama.is_available():
        print("[!] Chưa kết nối được Ollama. Hãy đảm bảo ứng dụng Ollama đang chạy trên máy.")
        return

    for model_name in [settings.OLLAMA_LLM_MODEL, settings.OLLAMA_OCR_MODEL, settings.OLLAMA_EMBED_MODEL]:
        print(f"[*] Đang tải model '{model_name}'...")
        try:
            res = ollama.pull_model(model_name)
            print(f"[OK] Hoàn tất '{model_name}': {res.get('status', 'success')}")
        except Exception as e:
            print(f"[!] Lỗi khi tải '{model_name}': {e}")

if __name__ == "__main__":
    main()