from network.network_component import NetworkComponent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from network.network_sim import Network


class ProcessingNode(NetworkComponent):
    A_TYPE_CLOUD = "Cloud"
    A_TYPE_FOG = "Fog"

    def __init__(
        self,
        aId: str,
        aType: str,
        wavelengths_capacity: float,
        wavelengths_amount: int,
        proc_time: float,
        transmission_time: float,
        network: "Network",
    ) -> None:
        super().__init__(aId, aType)

        self.wavelengths_capacity = wavelengths_capacity
        self.proc_time = proc_time
        self.transmission_time = transmission_time
        self.wavelengths_amount = wavelengths_amount
        self.network = network
        self.capacity = self.wavelengths_amount * self.wavelengths_capacity
        self.load = 0

        self.wavelengths = [0] * self.wavelengths_amount

    def allocate_wavelength(self, wavelength_index: int, amount: float) -> None:
        self.wavelengths[wavelength_index] += amount

    def allocate_resource(self, amount: float) -> None:
        self.load += amount

    def reset(self) -> None:
        self.wavelengths = [0] * self.wavelengths_amount
        self.load = 0
