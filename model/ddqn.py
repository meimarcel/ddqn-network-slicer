import torch
import torch.nn as nn
import os
import json
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.memory import PrioritizedReplayMemory


class QNet(nn.Module):

    def __init__(self, n_observations, n_actions, filter_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(n_observations, filter_num)

        self.fc_value = nn.Linear(filter_num, filter_num)
        self.fc_advantage = nn.Linear(filter_num, filter_num)

        self.out_value = nn.Linear(filter_num, 1)
        self.out_advantage = nn.Linear(filter_num, n_actions)

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.fc1(x))

        value = self.act(self.fc_value(out))
        value = self.out_value(value)

        advantage = self.act(self.fc_advantage(out))
        advantage = self.out_advantage(advantage)

        return value + advantage - torch.mean(advantage, dim=1, keepdim=True)


class DDQN:
    def __init__(
        self, num_input, num_actions, target_update, memory_capacity, batch_size, filter_num, start_learning, device
    ) -> None:
        self.target_net_update_freq = target_update
        self.experience_replay_size = memory_capacity
        self.batch_size = batch_size
        self.device = device

        self.lr = 1e-4
        self.gamma = 0.99
        self.update_count = 0

        self.start_learning = start_learning
        self.num_feats = num_input
        self.num_actions = num_actions

        self.train_hist = {"loss": []}

        self.memory = PrioritizedReplayMemory(self.experience_replay_size)

        self.Q_model = QNet(self.num_feats, self.num_actions, filter_num).to(self.device)
        self.Q_T_model = QNet(self.num_feats, self.num_actions, filter_num).to(self.device)
        self.Q_T_model.load_state_dict(self.Q_model.state_dict())

        self.optimizer = optim.Adam(self.Q_model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def append_to_replay(self, state, action, reward, next_state):
        self.memory.push((state, action, reward, next_state))

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.Q_model.state_dict(), f"{path}/Q_model_{epoch}.dump")
        pickle.dump(self.memory, open(f"{path}/exp_replay_agent_{epoch}.dump", "wb"))

        with open(f"{path}/loss_{epoch}.json", "w") as f:
            json.dump(self.train_hist, f, indent=4)

    def load(self, path, epoch):
        fname_Q_model = f"{path}/Q_model_{epoch}.dump"
        fname_replay = f"{path}/exp_replay_agent_{epoch}.dump"
        fname_hist = f"{path}/loss_{epoch}.json"

        self.Q_model.load_state_dict(torch.load(fname_Q_model, weights_only=True, map_location=self.device))
        self.Q_T_model.load_state_dict(torch.load(fname_Q_model, weights_only=True, map_location=self.device))
        self.memory = pickle.load(open(fname_replay, "rb"))

        with open(fname_hist, "r") as f:
            self.train_hist = json.load(f)

    def plot_loss(self):
        plt.figure(2)
        plt.clf()
        plt.title("Training loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.plot(self.train_hist["loss"], "r")
        plt.legend(["loss"])
        plt.pause(0.001)

    def get_minibatch(self):
        transitions, indexes, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float32)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.int64).view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float32).view(-1, 1)
        batch_next_state_indexes = torch.tensor(
            tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.bool
        )
        batch_next_state = torch.tensor(
            np.array(tuple(filter(lambda x: x is not None, batch_next_state))), device=self.device, dtype=torch.float32
        )

        return batch_state, batch_action, batch_reward, batch_next_state_indexes, batch_next_state, indexes, weights

    def update_target_model(self):
        self.update_count += 1
        self.update_count %= self.target_net_update_freq

        if self.update_count == 0:
            self.Q_T_model.load_state_dict(self.Q_model.state_dict())

    def update(self, frame):
        if len(self.memory._storage) < self.batch_size or frame < self.start_learning:
            return

        self.optimizer.zero_grad(set_to_none=True)

        batch_state, batch_action, batch_reward, batch_next_state_indexes, batch_next_state, indexes, weights = (
            self.get_minibatch()
        )

        current_q_values_samples = self.Q_model(batch_state)
        current_q_values_samples = current_q_values_samples.gather(1, batch_action)

        max_next_action_q_value = torch.zeros(self.batch_size, device=self.device).unsqueeze(1)

        if batch_next_state.shape[0] > 0:
            with torch.no_grad():
                max_next_actions = self.Q_model(batch_next_state).argmax(dim=1, keepdim=True)
                max_next_action_q_value[batch_next_state_indexes] = self.Q_T_model(batch_next_state).gather(
                    1, max_next_actions
                )

        target_q_values = batch_reward + (self.gamma * max_next_action_q_value)

        loss = torch.mean((current_q_values_samples - target_q_values) ** 2 * weights)
        loss.backward()

        self.train_hist["loss"].append(loss.item())

        td_error = torch.abs(current_q_values_samples - target_q_values).detach().squeeze(1)

        self.memory.update_priorities(indexes, td_error.cpu().numpy())

        for param in self.Q_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.update_target_model()
