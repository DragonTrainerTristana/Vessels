# 필요한 라이브러리 임포트
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import datetime
import platform
import random
import matplotlib.pyplot as plt
from collections import deque
import time
import pandas as pd

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

# 환경 파일 경로 설정
env_filepath = "C:/Users/sengh/OneDrive/Desktop/Github/Unity/RLjjjj/BuildSettings/Version6/RLjjjj.exe"
env = Environment(env_filepath)

# 에이전트 생성
agent = Agent(state_size=env.state_size, action_size=env.action_size, random_seed=0)

# 이동 평균 계산 함수
def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# DDPG 학습 함수
def ddpg(n_episodes=2000, max_t=1000):
    scores = []
    scores_window = deque(maxlen=100)
    times = []  # 학습 시간 저장
    successes = []  # 성공 여부 저장
    rewards_per_episode = []  # 에피소드별 보상 저장
    
    for i_episode in range(1, n_episodes + 1):
        start_time = time.time()  # 에피소드 시작 시간 기록
        env.reset()
        state = env.states.cpu().numpy()
        if state.size == 0:
            continue  # 상태가 비어있는 경우 건너뜀
        agent.reset()
        score = 0
        success = False  # 초기 성공 여부
        
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
            if done[0] or t >= 100:  # Step이 100을 넘어가면 종료
                success = done[0]  # 완료 여부에 따라 success 설정
                break
                
        scores_window.append(score)
        scores.append(score)
        end_time = time.time()  # 에피소드 종료 시간 기록
        times.append(end_time - start_time)
        successes.append(success)
        rewards_per_episode.append(score)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {float(np.mean(scores_window)):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {float(np.mean(scores_window)):.2f}')
    
    return scores, times, successes, rewards_per_episode

# 학습 수행
scores, times, successes, rewards_per_episode = ddpg()

# 모델 저장 함수
def save_model(agent, actor_path, critic_path):
    torch.save(agent.actor_local.state_dict(), actor_path)
    torch.save(agent.critic_local.state_dict(), critic_path)
    print(f"Model saved to {actor_path} and {critic_path}")

# 모델 로드 함수
def load_model(agent, actor_path, critic_path):
    agent.actor_local.load_state_dict(torch.load(actor_path))
    agent.critic_local.load_state_dict(torch.load(critic_path))
    print(f"Model loaded from {actor_path} and {critic_path}")

# 모델 저장 경로
actor_path = 'actor_model.pth'
critic_path = 'critic_model.pth'

# 모델 저장
save_model(agent, actor_path, critic_path)

# 모델을 ONNX로 저장
def save_model_to_onnx(agent, actor_onnx_path, critic_onnx_path):
    dummy_input = torch.randn(1, agent.state_size).to(device)

    # Actor 모델 저장
    torch.onnx.export(agent.actor_local, dummy_input, actor_onnx_path, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # Critic 모델 저장
    dummy_input = (torch.randn(1, agent.state_size).to(device), torch.randn(1, agent.action_size).to(device))
    torch.onnx.export(agent.critic_local, dummy_input, critic_onnx_path, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['state', 'action'], output_names=['output'], dynamic_axes={'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print(f"Model saved to {actor_onnx_path} and {critic_onnx_path}")

# 모델 저장 경로
actor_onnx_path = 'actor_model.onnx'
critic_onnx_path = 'critic_model.onnx'

# 모델을 ONNX로 저장
save_model_to_onnx(agent, actor_onnx_path, critic_onnx_path)

# 이동 평균 계산
window_size = 100
rewards_moving_average = moving_average(rewards_per_episode, window_size)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame({
    'Episode': range(1, len(rewards_per_episode) + 1),
    'Reward': rewards_per_episode,
    'Moving_Average_Reward': np.concatenate((np.zeros(window_size - 1), rewards_moving_average)),
    'Time': times,
    'Success': successes
})

# CSV 파일로 저장
results_df.to_csv('ddpg_training_results.csv', index=False)

# 그래프 시각화 함수
def plot_metrics(scores, times, successes, rewards_per_episode, rewards_moving_average):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # 에피소드별 학습 시간
    axs[0].plot(times)
    axs[0].set_title('Episode Learning Time')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Time (seconds)')

    # 에피소드별 성공 여부
    axs[1].plot(successes)
    axs[1].set_title('Episode Success')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Success (True/False)')

    # 에피소드별 보상
    axs[2].plot(rewards_per_episode)
    axs[2].set_title('Episode Reward')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Reward')

    # 이동 평균 보상
    axs[3].plot(np.arange(window_size, len(rewards_per_episode) + 1), rewards_moving_average)
    axs[3].set_title('Moving Average of Episode Reward')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel(f'Reward (Moving Average, Window Size = {window_size})')

    plt.tight_layout()
    plt.show()

# 그래프 호출
plot_metrics(scores, times, successes, rewards_per_episode, rewards_moving_average)
