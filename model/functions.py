import torch
import torch.nn as nn
import numpy as np
import scipy.special as sp
import math

from typing import List, TYPE_CHECKING
from config import *

if TYPE_CHECKING:
    from model.ddqn import DDQN


def initialize_weights(net: nn.Module) -> None:
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()


def get_epsilon(episode: int, decay: float) -> float:
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1.0 * episode / decay)


def get_action(model: "DDQN", state: List[float], epsilon: float, device: torch.device, use_softmax=False) -> int:
    with torch.no_grad():
        X = torch.tensor(state).unsqueeze(0).to(torch.float32).to(device)
        action_q_values = model.Q_model(X).detach().cpu().numpy().squeeze()

    if use_softmax:
        p = sp.softmax(action_q_values / epsilon).squeeze()
        p /= np.sum(p)

        return np.random.choice(model.num_actions, p=p)
    else:
        if np.random.random() >= epsilon:

            return int(np.argmax(action_q_values, axis=0))
        else:
            return np.random.randint(0, model.num_actions)
