import simpy
import algorithm as algs

from network.network_component import NetworkComponent
from typing import Any, Dict, Generator, List, NoReturn, TYPE_CHECKING
from config import *
from network.processing_node import ProcessingNode

if TYPE_CHECKING:
    from network.network_sim import Network


class ControlPlane(NetworkComponent):
    A_TYPE = "ControlPlane"

    def __init__(
        self,
        env: simpy.Environment,
        aId: str,
        model_path: str,
        alloc_type: str,
        algorithm_parameters: Dict[str, Any],
        network: "Network",
    ) -> None:
        super().__init__(aId, ControlPlane.A_TYPE)

        self.env = env
        self.model_path = model_path
        self.alloc_type = alloc_type
        self.network = network

        self.lost_data = 0
        self.lost_data_count = []
        self.allocated_splits = []
        self.cu_states = []
        self.requests = simpy.Store(self.env)

        self.init_algorithm(algorithm_parameters)

        self.placement_process = self.env.process(self.run_placement())

    def run_placement(self) -> Generator[simpy.Timeout, Any, NoReturn]:
        while True:
            active_ru_num = yield self.requests.get()

            self.network.reset_CUs()
            self.lost_data = 0
            splits = []

            for index in range(active_ru_num):
                split = self.algorithm.run(active_ru_num)
                self.allocate_resource(index, split)

                splits.append(split)

            self.allocated_splits.append(splits)
            self.cu_states.append(self.get_cu_states())
            self.lost_data_count.append(self.lost_data)

    def allocate_resource(self, index: int, split: int) -> None:
        ru = self.network.enabled_RUs[index]

        if self.alloc_type == "wavelength":
            ru.process_wavelength(split)
        else:
            ru.process_load(split)

    def get_cu_states(self) -> List[float]:
        clouds = []
        fogs = []

        if self.alloc_type == "wavelength":
            for cu in self.network.processing_elements:
                if cu.aType == ProcessingNode.A_TYPE_CLOUD:
                    clouds.extend(cu.wavelengths)
                elif cu.aType == ProcessingNode.A_TYPE_FOG:
                    fogs.extend(cu.wavelengths)
        else:
            for cu in self.network.processing_elements:
                if cu.aType == ProcessingNode.A_TYPE_CLOUD:
                    clouds.append(cu.load)
                elif cu.aType == ProcessingNode.A_TYPE_FOG:
                    fogs.append(cu.load)

        return [*clouds, *fogs]

    def init_algorithm(self, algorithm_parameters: Dict[str, Any]) -> None:
        if algorithm_parameters[NAME].upper() == "RL":
            self.algorithm = algs.RLAlgorithm(
                self.alloc_type, 4, algorithm_parameters[FILTER_NUM], self.model_path, self.network
            )
        else:
            raise Exception("Algorithm not found!")
