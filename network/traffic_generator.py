import simpy
import numpy as np

from typing import TYPE_CHECKING, Any, Generator, NoReturn

if TYPE_CHECKING:
    from network.network_sim import Network


class TrafficGenerator:
    def __init__(
        self,
        env: simpy.Environment,
        generation_type: str,
        output_path: str,
        peak_rate: int,
        mean_hour: int,
        std_dev: int,
        network: "Network",
    ) -> None:
        self.env = env
        self.generation_type = generation_type
        self.output_path = output_path
        self.peak_rate = peak_rate
        self.mean_hour = mean_hour
        self.std_dev = std_dev
        self.network = network

        self.day = 1
        self.hour = 0
        self.interval = 60
        self.total_time_sim = 60
        self.change_time = 60
        self.requests_count = []
        self.activation_sequence = 1
        self.activation_rate = self.get_activation_rate(self.hour)

        if self.generation_type.lower() == "static":
            self.traffic_generation_process = self.env.process(self.generate_static_traffic())
        else:
            self.traffic_generation_process = self.env.process(self.generate_dynamic_traffic())

        self.change_load_rate_process = self.env.process(self.change_load_rate())

    def generate_static_traffic(self) -> Generator[simpy.Timeout, Any, NoReturn]:
        while True:
            if self.activation_sequence > self.network.number_of_ru:
                self.activation_sequence = 1

            self.network.reset_RUs()

            for _ in range(self.activation_sequence):
                if self.network.disabled_RUs:
                    ru = self.network.disabled_RUs.pop(0)
                    ru.enable()
                    self.network.enabled_RUs.append(ru)
                else:
                    break

            enabled_rus = len(self.network.enabled_RUs)

            self.requests_count.append(enabled_rus)
            self.network.control_plane.requests.put(enabled_rus)

            self.activation_sequence += 1

            yield self.env.timeout(self.change_time / self.network.number_of_ru)

    def generate_dynamic_traffic(self) -> Generator[simpy.Timeout, Any, NoReturn]:
        while True:
            new_arrivals = np.random.poisson(self.activation_rate * self.interval / self.total_time_sim)

            for _ in range(new_arrivals):
                if self.network.disabled_RUs:
                    ru = self.network.disabled_RUs.pop(0)
                    ru.enable()
                    ru.life_time = np.random.exponential(scale=5 * self.total_time_sim / self.activation_rate)

                    self.network.enabled_RUs.append(ru)
                else:
                    break

            enabled_rus = len(self.network.enabled_RUs)

            for _ in range(enabled_rus):
                ru = self.network.enabled_RUs.pop(0)
                ru.reset_placement()

                if ru.life_time <= 0:
                    ru.disable()
                    self.network.disabled_RUs.append(ru)
                else:
                    ru.life_time -= self.interval
                    self.network.enabled_RUs.append(ru)

            enabled_rus = len(self.network.enabled_RUs)

            self.requests_count.append(enabled_rus)
            self.network.control_plane.requests.put(enabled_rus)

            yield self.env.timeout(1)

    def change_load_rate(self) -> Generator[simpy.Timeout, Any, NoReturn]:
        print(f"========= Running Day {self.day} =========")

        while True:
            yield self.env.timeout(self.change_time)

            print(f"Generaged traffic at hour {self.hour}: {self.requests_count}")
            for i in range(len(self.requests_count)):
                print("================================")
                print(f"Traffic {self.requests_count[i]}")
                print(f"Lost data: {self.network.control_plane.lost_data_count[i]}")
                print(f"Allocated splits: {self.network.control_plane.allocated_splits[i]}")
                print(f"CU states: {self.network.control_plane.cu_states[i]}")
                print("================================")

            self.network.trafics.append(self.requests_count)
            self.network.lost_datas.append(self.network.control_plane.lost_data_count)
            self.network.reset_RUs()

            self.requests_count = []
            self.network.control_plane.lost_data_count = []
            self.network.control_plane.allocated_splits = []
            self.network.control_plane.cu_states = []

            self.hour += 1

            if self.hour == 24:
                self.day += 1
                self.hour = 0

                print(f"========= Running Day {self.day} =========")

            if self.generation_type.lower() == "dynamic":
                self.activation_rate = self.get_activation_rate(self.hour)

    def get_activation_rate(self, hour: int) -> float:
        act_rate = self.peak_rate * np.exp(-((hour - self.mean_hour) ** 2) / (2 * self.std_dev**2))
        act_rate *= np.random.uniform(0.8, 1.2)

        return act_rate
