import gymnasium as gym
import math
import random
from itertools import count
from tqdm import tqdm, trange
import torch
import torch.optim as optim
import torch.nn as nn
from network import *
import warnings
import time
from copy import copy

warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")

EPOCHS = 1000
BATCH_SIZE = 128
GAMMA = 0.99 # Discount factor
EPS_START = 0.9 # Initial probability of taking random action
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-4
N = 100 # Video recording interval

device = None

def main():  
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    global device
    device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

    # Get number of actions 
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    episode_durations = []

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=".videos",
        episode_trigger=lambda episode_id: episode_id % N == 0 and episode_id != 0,
        name_prefix="CartPole"
    )
    global steps_done
    total_rewards = 0
    steps_done = 0
    start_time = time.time()
    for e in trange(1, EPOCHS+1, desc="Epochs"):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(env, policy_net, state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_rewards += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform optimization on the policy network
            optimize_model(policy_net, target_net, optimizer, memory)

            # Soft update of the target network's weights:  θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

        if e % N == 0:
            tqdm.write(f"> {e} epochs completed after {time.time() - start_time:.2f}s")
            tqdm.write(f"  - Saved video to file.")
            tqdm.write(f"  - Total rewards: {total_rewards:.2f}")
            tqdm.write(f"  - Avg reward: {total_rewards/e:.2f}")

def select_action(env, policy_net, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    

def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    main()