import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append("../")

from itertools import count
from model.ddqn import DDQN


env = gym.make('LunarLander-v3', render_mode="human")

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()

if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

model = DDQN(
    num_input=n_observations,
    num_actions=n_actions,
    target_update=400,
    memory_capacity=10000,
    batch_size=256,
    start_learning=100,
    device=device,
)

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            X = torch.tensor(state).unsqueeze(0).to(torch.float32).to(device)
            return model.Q_model(X).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


frame = 0

for episode in range(600):
    state, info = env.reset()
    state = state

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        print(state, reward)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation

        # Store the transition in memory
        model.append_to_replay(state, action, reward, next_state)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        model.update(frame)
        frame += 1

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
