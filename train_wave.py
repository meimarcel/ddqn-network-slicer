import argparse
import json
import torch
import os
import matplotlib.pyplot as plt
import model.functions as F

from typing import Any, Dict, List
from datetime import datetime
from dateutil import tz
from log import init_logs
from config import *
from model.ddqn import DDQN


class Env:
    def __init__(self, input_parameters: Dict[str, Any], processing_node_parameters: Dict[str, Any]) -> None:
        self.ru_number = input_parameters[NUMBER_OF_RU]
        self.lost_data = 0

        self.clouds_capacity = []
        self.fogs_capacity = []

        self.clouds = []
        self.fogs = []

        for p in processing_node_parameters:
            if p[A_TYPE].lower() == "cloud":
                self.clouds_capacity.append(p[WAVELENGTHS_CAPACITY])
                self.clouds.append([0] * p[WAVELENGTHS])
            elif p[A_TYPE].lower() == "fog":
                self.fogs_capacity.append(p[WAVELENGTHS_CAPACITY])
                self.fogs.append([0] * p[WAVELENGTHS])

    def reset(self) -> None:
        for index in range(len(self.clouds)):
            self.clouds[index] = [0] * len(self.clouds[index])

        for index in range(len(self.fogs)):
            self.fogs[index] = [0] * len(self.fogs[index])

        self.lost_data = 0

    def get_state(self, active_ru_num: int) -> List[float]:
        clouds = []
        fogs = []

        for c_index in range(len(self.clouds)):
            clouds.extend(
                [
                    self.clouds[c_index][index] / self.clouds_capacity[c_index]
                    for index in range(len(self.clouds[c_index]))
                ]
            )

        for f_index in range(len(self.fogs)):
            fogs.extend(
                [self.fogs[f_index][index] / self.fogs_capacity[f_index] for index in range(len(self.fogs[f_index]))]
            )

        rus = active_ru_num / self.ru_number

        return [rus, *clouds, *fogs]

    def get_reward(self, ru_index: int, active_ru_num: int, action: int) -> float:
        if self.lost_data > 0:
            return -100

        fit_cloud = False
        total_cloud_capacity = 0

        for index in range(len(self.clouds)):
            total_cloud_capacity += len(self.clouds[index]) * self.clouds_capacity[index]

        if active_ru_num * 1966 < total_cloud_capacity:
            fit_cloud = True

        if fit_cloud and action != 0:
            return -100

        if ru_index == active_ru_num:
            percent = []

            if sum([sum(f) for f in self.fogs]) > 0:
                for index in range(len(self.clouds)):
                    percent.append(sum(self.clouds[index]) / (len(self.clouds[index]) * self.clouds_capacity[index]))

                return sum([p * 100 for p in percent])
            else:
                return 100

        total_energy = 0
        consumed_energy = 0

        for cu in self.clouds:
            total_energy += len(cu) * 150
            for w in cu:
                if w > 0:
                    consumed_energy += 150

        for cu in self.fogs:
            total_energy += len(cu) * 150
            for w in cu:
                if w > 0:
                    consumed_energy += 150

        return 0.01 + (total_energy - consumed_energy) / total_energy

    def alloc_resource(self, split: int, ru_index: int, active_ru_num: int) -> List[Any]:
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

        for index in range(len(self.clouds)):
            for w in range(len(self.clouds[index])):
                if self.clouds[index][w] + cloud_part <= self.clouds_capacity[index]:
                    self.clouds[index][w] += cloud_part
                    cloud_part = 0
                    break

        if split != 0:
            for index in range(len(self.fogs)):
                for w in range(len(self.fogs[index])):
                    if self.fogs[index][w] + fog_part <= self.fogs_capacity[index]:
                        self.fogs[index][w] += fog_part
                        fog_part = 0
                        break

        if cloud_part > 0 or fog_part > 0:
            self.lost_data += cloud_part + fog_part

        reward = self.get_reward(ru_index, active_ru_num, split)

        if reward < 0 or ru_index == active_ru_num:
            return None, reward
        else:
            return self.get_state(active_ru_num), reward


def run_eval(env: Env, ddqn: DDQN, number_of_ru: int, device: torch.device) -> None:
    ddqn.Q_model.eval()

    print("========= Model Evaluation =========")
    for active_ru_num in range(1, number_of_ru + 1):
        env.reset()
        splits = []
        allocated = 0

        state = env.get_state(active_ru_num)

        for ru_index in range(1, active_ru_num + 1):
            action = F.get_action(ddqn, state, 0, device)
            next_state, reward = env.alloc_resource(action, ru_index, active_ru_num)

            splits.append(action)

            if reward < 0:
                break

            allocated += 1
            state = next_state

        print(f"Total RU:{active_ru_num}. {'V' if reward >= 0 else 'X'} Allocated: {allocated} {splits}")

    print("====================================")

    ddqn.Q_model.train()


def train(path: str, arguments: argparse.Namespace, parameters: Dict[str, Any]) -> None:
    input_parameters = parameters[INPUT_PARAMETERS]
    algorithm = parameters[ALGORITHM]
    processing_node_parameters = parameters[PROCESSING_NODES]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = Env(input_parameters, processing_node_parameters)
    ddqn = DDQN(
        num_input=1 + sum([len(c) for c in env.clouds]) + sum([len(f) for f in env.fogs]),
        num_actions=4,
        target_update=algorithm[TARGET_UPDATE],
        memory_capacity=algorithm[MEMORY_CAPACITY],
        batch_size=algorithm[BATCH_SIZE],
        filter_num=algorithm[FILTER_NUM],
        start_learning=algorithm[START_LEARNING],
        device=device,
    )

    history = {"training": []}
    states = []
    rewards = []

    if arguments.checkpoint:
        ddqn.load(arguments.checkpoint, arguments.start_episode)
        with open(os.path.join(arguments.checkpoint, "train_history.json"), "r") as f:
            history = json.load(f)

    plt.ion()

    for episode in range(arguments.start_episode, arguments.episodes + 1):
        epsilon = F.get_epsilon(episode, algorithm[EPSILON_DECAY])
        run = []

        for active_ru_num in range(1, input_parameters[NUMBER_OF_RU] + 1):
            states = []
            rewards = []
            allocated = 0

            env.reset()
            state = env.get_state(active_ru_num)

            for ru_index in range(1, active_ru_num + 1):
                action = F.get_action(ddqn, state, epsilon, device)
                next_state, reward = env.alloc_resource(action, ru_index, active_ru_num)

                ddqn.append_to_replay(state, action, reward, next_state)
                ddqn.update(episode)

                states.append([state, action])
                rewards.append(reward)

                state = next_state

                if reward < 0:
                    break

                allocated += 1

            history["training"].append(
                {
                    "ru_num": active_ru_num,
                    "ru_allocated": allocated,
                    "states": states,
                    "rewards": rewards,
                    "total_reward": sum(rewards),
                }
            )

            run.append((active_ru_num, allocated))

        print(f"Episode {episode}: {run}")

        if episode % 24 == 0 and episode >= algorithm[START_LEARNING]:
            run_eval(env, ddqn, input_parameters[NUMBER_OF_RU], device)
            ddqn.save(path, episode)
            with open(os.path.join(path, "train_history.json"), "w") as f:
                json.dump(history, f, indent=2)

    ddqn.save(output_path, "FINAL")
    with open(os.path.join(output_path, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Configuration file path")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint model to resume training")
    parser.add_argument("--out_path", type=str, help="Name of the output path folder")
    parser.add_argument("--start_episode", type=int, help="Episode to start iteration", default=0)
    parser.add_argument("--episodes", type=int, help="Number of days to simulate", default=10000)
    arguments = parser.parse_args()

    with open(arguments.config_path, "r") as f:
        parameters = json.load(f)

    if arguments.checkpoint:
        output_path = arguments.checkpoint
    else:
        timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")

        if arguments.out_path:
            timestamp += "_" + arguments.out_path

        output_path = os.path.join("trained-models", timestamp)

        os.makedirs(output_path)

        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(parameters, f, indent=4)

    init_logs(os.path.join(output_path, "log.txt"), "at")

    print(arguments)

    train(output_path, arguments, parameters)
