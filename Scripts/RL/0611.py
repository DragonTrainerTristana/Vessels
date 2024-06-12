import copy
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Actor 클래스 정의
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

# Critic 클래스 정의
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ReplayBuffer 클래스 정의
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

# OUNoise 클래스 정의
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

# Agent 클래스 정의
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
        state = torch.from_numpy(state).float().to(device)
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
print(f"Using device: {device}")  # 여기에 출력 추가
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Environment 클래스 정의
class Environment:
    def __init__(self, env_filepath):
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=env_filepath,
            side_channels=[self.engine_configuration_channel],
            worker_id=0)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps
        self.agent_count = len(decision_steps) + len(terminal_steps)
        self.action_size = self.spec.action_spec.continuous_size
        self.state_size = self.spec.observation_specs[0].shape[0]

    def reset(self):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps
        self.agent_count = len(decision_steps) + len(terminal_steps)
        print(f"Environment reset with {self.agent_count} agents")

    def close(self):
        self.env.close()

    def step(self, actions):
        continuous_actions = ActionTuple()
        continuous_actions.add_continuous(np.array(actions))
        self.env.set_actions(self.behavior_name, continuous_actions)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps

        next_states = self.states
        rewards = np.array([decision_steps[agent_id].reward for agent_id in decision_steps.agent_id])
        dones = np.array([agent_id in terminal_steps for agent_id in decision_steps.agent_id])

        print(f"Step completed: rewards = {rewards}, dones = {dones}")

        return next_states, rewards, dones

    @property
    def states(self):
        if len(self.env_info) == 0:
            return np.empty((0, self.state_size))
        return np.array([self.env_info[agent_id].obs[0] for agent_id in self.env_info.agent_id])

def train_agents(agents, env, n_episodes=2000, max_t=1000):
    scores_window = deque(maxlen=100)
    all_scores = [[] for _ in range(len(agents))]
    episode_counts = np.zeros(len(agents), dtype=int)

    for i_episode in range(1, n_episodes + 1):
        env.reset()
        states = env.states
        if len(states) < len(agents):
            print(f"Not enough states for all agents: {len(states)} available, {len(agents)} required.")
            continue

        for agent in agents:
            agent.reset()

        scores = np.zeros(len(agents))
        dones = np.zeros(len(agents), dtype=bool)

        for t in range(max_t):
            actions = [agents[i].act(states[i]) for i in range(len(agents))]
            next_states, rewards, new_dones = env.step(actions)

            if len(next_states) < len(agents):
                print(f"Not enough next states for all agents at step {t}. Breaking loop.")
                break

            for i in range(len(agents)):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], new_dones[i])
                scores[i] += rewards[i]
                dones[i] = dones[i] or new_dones[i]

                if new_dones[i]:
                    episode_counts[i] += 1
                    states[i] = env.states[i]
                    agents[i].reset()

            states = next_states

            if all(dones):
                break

        for i in range(len(agents)):
            avg_score = scores[i]
            all_scores[i].append(avg_score)
            if len(all_scores[i]) > 100:
                all_scores[i] = all_scores[i][-100:]
            print(f'\rAgent {i} Episode {episode_counts[i]} Average Score: {np.mean(all_scores[i]):.2f}', end="")
            if episode_counts[i] % 100 == 0:
                print(f'\rAgent {i} Episode {episode_counts[i]} Average Score: {np.mean(all_scores[i]):.2f}')

    return all_scores

# 멀티 에이전트 학습 함수
def train_multi_agent(env_filepath, n_episodes=2000, max_t=1000):
    env = Environment(env_filepath)
    agents = [Agent(env.state_size, env.action_size, random_seed=i) for i in range(env.agent_count)]
    results = train_agents(agents, env, n_episodes, max_t)
    env.close()
    return results

if __name__ == '__main__':
    env_filepath = "C:/Users/sengh/OneDrive/Desktop/Github/Unity/RLjjjj/BuildSettings/Version8/RLjjjj.exe"
    results = train_multi_agent(env_filepath)
