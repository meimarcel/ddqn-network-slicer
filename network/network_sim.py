from typing import List, TYPE_CHECKING
from config import *

if TYPE_CHECKING:
    from network.control_plane import ControlPlane
    from network.ru import RU
    from network.processing_node import ProcessingNode


class Network:
    def __init__(self, number_of_ru: int) -> None:
        self.number_of_ru = number_of_ru
        self.enabled_RUs: List["RU"] = []
        self.disabled_RUs: List["RU"] = []
        self.processing_elements: List["ProcessingNode"] = []
        self.consumed_memory: List[float] = []
        self.consumed_cpu: List[float] = []
        self.trafics: List[int] = []
        self.lost_datas: List[float] = []

        self.control_plane: "ControlPlane" = None

    def reset_RUs(self) -> None:
        enabled_RUs = len(self.enabled_RUs)

        for _ in range(enabled_RUs):
            ru = self.enabled_RUs.pop(0)
            ru.reset_placement()
            ru.disable()
            self.disabled_RUs.append(ru)

    def reset_CUs(self) -> None:
        for cu in self.processing_elements:
            cu.reset()
