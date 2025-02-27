import torch

from config import *
from model.ddqn import QNet
from typing import TYPE_CHECKING, Any, List, Dict
from network.processing_node import ProcessingNode

if TYPE_CHECKING:
    from network.network_sim import Network


class RLAlgorithm:
    def __init__(
        self,
        alloc_type: str,
        splits_amount: int,
        filter_num: int,
        model_path: str,
        network: "Network",
    ) -> None:
        self.alloc_type = alloc_type
        self.splits_amount = splits_amount
        self.model_path = model_path
        self.network = network

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = QNet(
            n_observations=len(self.get_state(0)),
            n_actions=self.splits_amount,
            filter_num=filter_num,
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

    def run(self, active_ru_num: int) -> int:
        state = self.get_state(active_ru_num)
        state = torch.tensor(state).unsqueeze(0).to(torch.float32).to(self.device)

        with torch.no_grad():
            action = self.model(state).argmax(dim=1).squeeze().cpu().numpy().item()

        return action

    def get_state(self, active_ru_num: int) -> List[float]:
        if self.alloc_type.lower() == "wavelength":
            cloud_state = []
            fog_state = []

            for cu in self.network.processing_elements:
                if cu.aType == ProcessingNode.A_TYPE_CLOUD:
                    cloud_state.extend([w / cu.wavelengths_capacity for w in cu.wavelengths])
                elif cu.aType == ProcessingNode.A_TYPE_FOG:
                    fog_state.extend([w / cu.wavelengths_capacity for w in cu.wavelengths])
        else:
            cloud_state = []
            fog_state = []

            for cu in self.network.processing_elements:
                if cu.aType == ProcessingNode.A_TYPE_CLOUD:
                    cloud_state.append(cu.load / cu.capacity)
                elif cu.aType == ProcessingNode.A_TYPE_FOG:
                    fog_state.append(cu.load / cu.capacity)

        return [active_ru_num / self.network.number_of_ru, *cloud_state, *fog_state]
