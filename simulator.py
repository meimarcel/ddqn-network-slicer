import simpy

from network.control_plane import ControlPlane
from network.network_sim import Network
from network.ru import RU
from network.processing_node import ProcessingNode
from xml.etree.ElementTree import Element

from network.traffic_generator import TrafficGenerator
from typing import Any, Dict
from config import *


class Simulator:
    def __init__(
        self,
        parameters: Dict[str, Any],
        output_path: str,
        generation_type: str,
        model_path: str,
    ) -> None:
        self.env = simpy.Environment()
        self.output_path = output_path
        self.generation_type = generation_type
        self.model_path = model_path

        self.input_parameters = parameters[INPUT_PARAMETERS]
        self.algorithm = parameters[ALGORITHM]
        self.processing_node_parameters = parameters[PROCESSING_NODES]

        self.network = Network(self.input_parameters[NUMBER_OF_RU])
        self.traffic_generator = TrafficGenerator(
            self.env,
            self.generation_type,
            self.output_path,
            self.input_parameters[NUMBER_OF_RU],
            12,
            4,
            self.network,
        )

        for index in range(self.input_parameters[NUMBER_OF_RU]):
            ru = RU(
                f"{RU.A_TYPE}:{index}",
                self.input_parameters[CPRI_FRAME_GENERATION_TIME],
                self.input_parameters[TRANSMISSION_TIME],
                self.input_parameters[LOCAL_TRANSMISSION_TIME],
                self.network,
            )

            self.network.disabled_RUs.append(ru)

        for node in self.processing_node_parameters:
            node_obj = ProcessingNode(
                node[A_ID],
                node[A_TYPE],
                node[WAVELENGTHS_CAPACITY],
                node[WAVELENGTHS],
                self.input_parameters[FRAME_PROC_TIME],
                self.input_parameters[TRANSMISSION_TIME],
                self.network,
            )

            self.network.processing_elements.append(node_obj)

        self.network.control_plane = ControlPlane(
            self.env,
            f"{ControlPlane.A_TYPE}:0",
            self.model_path,
            self.input_parameters[ALLOC_TYPE],
            self.algorithm,
            self.network,
        )

    def run(self, limit: int) -> None:
        self.env.run(until=limit)
