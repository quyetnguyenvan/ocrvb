import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend_gateway.services.ragflow_service import RAGFlowService

DEPARTMENTS = [
    {"group_key": "DEFAULT", "dataset_name": "01_Van_Ban_Quy_Che_Chung", "description": "Kho văn bản chung", "chunk_method": "laws"},
    {"group_key": "Phong_TCCB", "dataset_name": "02_Phong_To_Chuc_Can_Bo", "description": "Văn bản Phòng TCCB", "chunk_method": "laws"},
    {"group_key": "Phong_KHTC", "dataset_name": "03_Phong_Ke_Hoach_Tai_Chinh", "description": "Văn bản Phòng KHTC", "chunk_method": "naive"},
    {"group_key": "Trung_Tam_CDS", "dataset_name": "04_Trung_Tam_Chuyen_Doi_So", "description": "Tài liệu Trung tâm CĐS", "chunk_method": "manual"}
]

def main():
    service = RAGFlowService()
    existing = {d["name"]: d["id"] for d in service.list_datasets()}
    group_mapping = {"Lanh_Dao_Bo": []}
    default_ds_id = None

    for dept in DEPARTMENTS:
        ds_name = dept["dataset_name"]
        if ds_name in existing:
            ds_id = existing[ds_name]
        else:
            res = service.create_dataset(name=ds_name, description=dept["description"], chunk_method=dept["chunk_method"])
            ds_id = res.get("data", {}).get("id", f"mock_id_{ds_name}")
        group_mapping["Lanh_Dao_Bo"].append(ds_id)
        if dept["group_key"] == "DEFAULT":
            default_ds_id = ds_id
            group_mapping["DEFAULT"] = [ds_id]
        else:
            group_mapping[dept["group_key"]] = [default_ds_id, ds_id] if default_ds_id else [ds_id]

    print(f"RBAC_GROUP_MAPPING='{json.dumps(group_mapping, ensure_ascii=False)}'")

if __name__ == "__main__":
    main()