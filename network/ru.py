from network.network_component import NetworkComponent
from network.processing_node import ProcessingNode
from typing import TYPE_CHECKING, List
from config import *

if TYPE_CHECKING:
    from network.network_sim import Network


class RU(NetworkComponent):
    A_TYPE = "RU"

    def __init__(
        self,
        aId: str,
        cpri_frame_generation_time: float,
        transmission_time: float,
        local_transmission_time: float,
        network_sim: "Network",
    ) -> None:
        super().__init__(aId, RU.A_TYPE)

        self.network = network_sim
        self.cpri_frame_generation_time = cpri_frame_generation_time
        self.transmission_time = transmission_time
        self.local_transmission_time = local_transmission_time

        self.enabled = False
        self.life_time = 0
        self.processing_node: ProcessingNode = None
        self.second_processing_node: ProcessingNode = None
        self.allocated_cloud_wavelength: int = None
        self.allocated_fog_wavelength: int = None
        self.split: int = None

    def get_cloud_fog_split(self, split: int) -> List[float]:
        cloud_part = 0
        fog_part = 0

        if split == 0:
            cloud_part = 1966
        elif split == 1:
            cloud_part = 74
            fog_part = 1892
        elif split == 2:
            cloud_part = 119
            fog_part = 1847
        else:
            cloud_part = 674.4
            fog_part = 1291.6

        return cloud_part, fog_part

    def process_wavelength(self, split: int) -> None:
        self.split = split
        cloud_part, fog_part = self.get_cloud_fog_split(self.split)

        for cu in filter(lambda x: x.aType == ProcessingNode.A_TYPE_CLOUD, self.network.processing_elements):
            for index in range(len(cu.wavelengths)):
                if cu.wavelengths[index] + cloud_part <= cu.wavelengths_capacity:
                    self.allocated_cloud_wavelength = index
                    self.processing_node = cu
                    cu.allocate_wavelength(index, cloud_part)
                    cloud_part = 0
                    break

        if split != 0:
            for cu in filter(lambda x: x.aType == ProcessingNode.A_TYPE_FOG, self.network.processing_elements):
                for index in range(len(cu.wavelengths)):
                    if cu.wavelengths[index] + fog_part <= cu.wavelengths_capacity:
                        self.allocated_fog_wavelength = index
                        self.second_processing_node = cu
                        cu.allocate_wavelength(index, fog_part)
                        fog_part = 0
                        break

        if cloud_part > 0 or fog_part > 0:
            self.network.control_plane.lost_data += cloud_part + fog_part

    def process_load(self, split: int) -> None:
        self.split = split
        cloud_part, fog_part = self.get_cloud_fog_split(self.split)

        for cu in filter(lambda x: x.aType == ProcessingNode.A_TYPE_CLOUD, self.network.processing_elements):
            if cu.load + cloud_part <= cu.capacity:
                self.processing_node = cu
                cu.allocate_resource(cloud_part)
                cloud_part = 0
                break

        if split != 0:
            for cu in filter(lambda x: x.aType == ProcessingNode.A_TYPE_FOG, self.network.processing_elements):
                if cu.load + fog_part <= cu.capacity:
                    self.second_processing_node = cu
                    cu.allocate_resource(fog_part)
                    fog_part = 0
                    break

        if cloud_part > 0 or fog_part > 0:
            self.network.control_plane.lost_data += cloud_part + fog_part

    def reset_placement(self) -> None:
        self.processing_node: ProcessingNode = None
        self.second_processing_node: ProcessingNode = None
        self.allocated_cloud_wavelength: int = None
        self.allocated_fog_wavelength: int = None

    def disable(self) -> None:
        self.enabled = False
        self.life_time = 0

    def enable(self) -> None:
        self.enabled = True
