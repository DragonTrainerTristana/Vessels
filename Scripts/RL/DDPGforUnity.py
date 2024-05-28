# Unity Env 호출
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Unity 예외처리
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)

# 신경망 처리
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 일반 라이브러리
import numpy as np
import copy
import datetime
import platform
import random
import matplotlib.pyplot as plt
from collections import deque

# weight(theta) 초기화
def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)

# Actor 클래스
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic 클래스
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        initialize_weights(self)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# Ornstein-Uhlenbeck Process for Noise
class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Agent 정의
class Agent:
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3, weight_decay=0)

        self.noise = OUNoise(action_size, random_seed)

        self.memory = ReplayBuffer(buffer_size=int(1e5), batch_size=64, seed=random_seed)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, gamma=0.99)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if state.size(1) == 0:
            return np.zeros(self.action_size)  # 상태가 비어 있으면 무작위 행동 반환
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, tau=1e-3)
        self.soft_update(self.actor_local, self.actor_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# 학습 루프
def ddpg(n_episodes=1000, max_t=1000):
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes + 1):
        env.reset()
        state = env.states.cpu().numpy()
        if state.size == 0:
            continue  # 상태가 비어있는 경우 건너뜀
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            if action.size == 0:
                break  # 행동이 비어있는 경우 반복문 종료
            action = np.reshape(action, (1, -1))  # 2차원으로 reshape
            next_state, reward, done = env.step(action)
            if next_state.size(0) == 0:
                break  # 다음 상태가 비어있는 경우 반복문 종료
            agent.step(state, action, reward, next_state, done)
            state = next_state.cpu().numpy()
            score += reward
            if done[0]:
                break
        scores_window.append(score)
        scores.append(score)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        # 에피소드마다 로그 추가
        print(f'Episode {i_episode}: Score: {score}, Average Score: {np.mean(scores_window):.2f}')
    return scores

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Environment:
    def __init__(self, env_filepath):
        print("Env 호출")
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=env_filepath,
            side_channels=[self.engine_configuration_channel],
            worker_id=5)
        self.env.reset()
        print("1번")
        self.behavior_name = list(self.env.behavior_specs)[0]
        print("behavior_name 호출")
        print(self.behavior_name)
        self.spec = self.env.behavior_specs[self.behavior_name]
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps
        self.agent_count = len(decision_steps) + len(terminal_steps)
        print("에이전트 호출")
        print(self.agent_count)
        self.action_size = self.spec.action_spec.continuous_size
        self.state_size = self.spec.observation_specs[0].shape[0]

    def reset(self, train=True):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps
        self.agent_count = len(decision_steps) + len(terminal_steps)

    def close(self):
        self.env.close()


    # 여기 부분 에러가 있음.
    def step(self, actions):
        if self.agent_count == 0:
            return torch.empty(0), np.array([0]), np.array([True])  # 에이전트가 없는 경우 빈 상태와 보상 반환
        continuous_actions = ActionTuple()
        continuous_actions.add_continuous(np.array(actions))
        self.env.set_actions(self.behavior_name, continuous_actions)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps
        next_states = self.states
        rewards = decision_steps.reward if len(decision_steps) > 0 else np.zeros(1)
        dones = np.array([agent_id in terminal_steps for agent_id in decision_steps.agent_id]) if len(decision_steps) > 0 else np.zeros(1)

        # 행동, 보상, 완료 여부 로그 추가
        print(f"Action: {actions}, Reward: {rewards}, Done: {dones}")

        return next_states, rewards, dones

    @property
    def states(self):
        if len(self.env_info) == 0:
            return torch.empty(0)
        return torch.from_numpy(self.env_info.obs[0]).float()

env_filepath = "C:/Users/sengh/OneDrive/Desktop/Github/Unity/RLjjjj/BuildSettings/Version4/RLjjjj.exe"
env = Environment(env_filepath)

agent = Agent(state_size=env.state_size, action_size=env.action_size, random_seed=0)
scores = ddpg()
