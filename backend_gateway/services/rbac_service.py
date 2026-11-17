from typing import List, Set
from backend_gateway.config import settings
from backend_gateway.models.schemas import UserContext


class RBACService:
    @staticmethod
    def get_allowed_datasets(user: UserContext) -> List[str]:
        mapping = settings.get_rbac_mapping()
        if user.role == "admin":
            all_datasets: Set[str] = set()
            for ds_list in mapping.values():
                all_datasets.update(ds_list)
            return sorted(list(all_datasets))

        allowed: Set[str] = set(mapping.get("DEFAULT", []))
        for group_name in user.groups:
            if group_name in mapping:
                allowed.update(mapping[group_name])
        return sorted(list(allowed))

    @staticmethod
    def can_upload_to_dataset(user: UserContext, target_dataset_id: str) -> bool:
        if user.role == "admin":
            return True
        return target_dataset_id in RBACService.get_allowed_datasets(user)