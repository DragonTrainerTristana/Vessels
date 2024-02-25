import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel


# ------------ 함수 모음 ------------

# 신경망 weight 초기화
def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)
        
def train_maddpg(multi_agent, args, env):
    multi_agent.initialize_memory(args.pretrain_length, env)
    scores_deque = deque(maxlen=args.print_every)
    scores = []
    highest_avg_score = 0

    for episode in range(1, args.num_episodes + 1):
        env.reset()
        obs = env.states
        score = np.zeros(multi_agent.agent_count)
        while True:
            actions = multi_agent.act(obs)
            next_obs, rewards, dones = env.step(actions)
            score += rewards
            multi_agent.store((obs, next_obs, actions, rewards, dones))
            multi_agent.learn()
            obs = next_obs
            if np.any(dones):
                break
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)), end="")
        if np.mean(scores_deque) >= highest_avg_score:
            for i, a in enumerate(multi_agent.agents):
                torch.save(a.actor_local.state_dict(), f'checkpoint_actor_a{i}.pth')
                torch.save(a.critic_local.state_dict(), f'checkpoint_critic_a{i}.pth')
            highest_avg_score = np.mean(scores_deque)
        if episode % args.print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
        multi_agent.new_episode(scores)
    env.close()
    return scores

# ------------ 클래스 모음 ------------

# Actor 클래스
class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units = 400, fc2_units = 300):
        super(Actor, self).__init__()  # 부모 클래스의 생성자를 호출
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x).tanh()

# Critic 클래스
class Critic(nn.Module):

    def __init__(self, state_size, action_size, num_atoms, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, num_atoms)
        initialize_weights(self)

    def forward(self, state, action, log=False):
        xs = F.relu(self.fcs1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        if log:
            return F.log_softmax(self.fc3(x), dim=-1)
        else:
            return F.softmax(self.fc3(x), dim=-1)
        
# Unity 환경 연동 
# class Environment:
    
#     def __init__ (self, env_filepath):
#         self.env = UnityEnvironment(file_name=env_filepath)
#         #self.brain_name = self.env.brain_names[0]
#         self.brain_name = self.env.behavior_specs[0]
#         self.brain = self.env.brains[self.brain_name]
#         self.reset()
#         self.action_size = self.brain.vector_action_space_size
#         self.state_size = self.states.shape[1]
#         self.agent_count = len(self.env_info.agents)
        
#         def reset(self, train=True):
#             self.env_info = self.env.reset(train_mode=train)[self.brain_name]

#         def close(self):
#             self.env.close()

#         def step(self, actions):
#             self.env_info = self.env.step(actions)[self.brain_name]
#             next_obs = self.states
#             rewards = np.array(self.env_info.rewards)
#             dones = self.env_info.local_done
#             return next_obs, rewards, dones

#         @property
#         def states(self):
#             return torch.from_numpy(self.env_info.vector_observations).float()
        
        
class Environment:
    
    def __init__(self, env_filepath):
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=env_filepath,
                                    side_channels=[self.engine_configuration_channel],
                                    worker_id= 5)
        self.env.reset()
        #self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.behavior_name = list(self.env.behavior_specs)[0] 
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        # Get the number of agents in the environment
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps 
        self.agent_count = len(decision_steps) + len(terminal_steps)
        print(self.agent_count)
        
        # Get the action size and state size from the behavior spec
        self.action_size = self.spec.action_spec.continuous_size
        self.state_size = self.spec.observation_specs[0].shape[0]
        
    def reset(self, train=True):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps

    def close(self):
        self.env.close()

    def step(self, actions):
        
        continuous_actions = ActionTuple()
        continuous_actions.add_continuous(np.array(actions))
        
        # 변환된 continuous_actions를 사용하여 행동을 설정
        self.env.set_actions(self.behavior_name, continuous_actions)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        self.env_info = decision_steps  # or terminal_steps based on your needs
        
        next_states = self.states
        rewards = decision_steps.reward
        dones = terminal_steps.interrupted  # Note: 'interrupted' might be used for 'dones'
        return next_states, rewards, dones

    @property
    def states(self):
        return torch.from_numpy(self.env_info.obs[0]).float()
        

# MADDPG 

class MADDPG_Net:
    
    
    # 기존 arg Class의 데이터 그대로 할당하기
    def __init__(self, env, args):
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        self.t_step = 0
        self.avg_score = 0
        self.update_every = args.update_every
        self._e = args.e
        self.e_min = args.e_min
        self.e_decay = args.e_decay
        self.anneal_max = args.anneal_max
        self.update_type = args.update_type
        self.tau = args.tau
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.agent_count = env.agent_count
        
        # 각 에이전트를 DDPG_Agent로 불러와서 할당하기
        self.agents = [DDPG_Agent(self.state_size, self.action_size, args, self.agent_count) for _ in range(self.agent_count)]
        self.batch_size = args.batch_size
        
        # ReplayBuffer 분석할 필요가 있음
        self.memory = ReplayBuffer(args.device, args.buffer_size, args.gamma, args.rollout, self.agent_count)
        self.memory.init_n_step()
        
       
        
        for agent in self.agents:
            self.update_networks(agent, force_hard=True)
    def act(self, obs, training=True):
        with torch.no_grad():
            actions = np.array([agent.act(o) for agent, o in zip(self.agents, obs)])
        if training:
            actions += self._gauss_noise(actions.shape)
        return np.clip(actions, -1, 1)

    def store(self, experience):
        self.memory.store(experience)

    # def learn(self):
        
   
    #     if len(self.memory) == 0:
    #         print("Memory is empty")
    #         return
 
    #     self.t_step += 1
    #     try:
    #         batch = self.memory.sample(self.batch_size)
    #         obs, next_obs, actions, rewards, dones = batch
    #         target_actions = [agent.actor_target(next_obs[i]) for i, agent in enumerate(self.agents)]
    #         predicted_actions = [agent.actor_local(obs[i]) for i, agent in enumerate(self.agents)]
    #         target_actions = torch.cat(target_actions, dim=-1)
    #         predicted_actions = torch.cat(predicted_actions, dim=-1)
    #         obs = obs.transpose(1, 0).contiguous().view(self.batch_size, -1)
    #         next_obs = next_obs.transpose(1, 0).contiguous().view(self.batch_size, -1)
    #         for i, agent in enumerate(self.agents):
    #             agent.learn(obs, next_obs, actions, target_actions, predicted_actions, rewards[i], dones[i])
    #             self.update_networks(agent)
    #     except IndexError as e:
    #         print(f"An error occurred: {e}")


    def learn(self):
        if len(self.memory) < self.batch_size:
            print("Not enough memory to sample")
            return

        self.t_step += 1

        # 배치 샘플링
        batch = self.memory.sample(self.batch_size)
        for agent_idx, agent in enumerate(self.agents):
            try:
                # 에이전트별 데이터 분리
                obs, next_obs, actions, rewards, dones = self._extract_agent_data(batch, agent_idx)

                # 타깃 액션과 예측 액션 계산
                target_actions = agent.actor_target(next_obs)
                predicted_actions = agent.actor_local(obs)

                # 에이전트별 학습
                agent.learn(obs, next_obs, actions, target_actions, predicted_actions, rewards, dones)
                self.update_networks(agent)
            except Exception as e:
                print(f"An error occurred while training agent {agent_idx}: {e}")
                continue




# 초기화 및 할당 (이슈 없음 확인)
    def initialize_memory(self, pretrain_length, env):
        print("Initializing memory.")
        obs = env.states
        while len(self.memory) < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_obs, rewards, dones = env.step(actions)
            self.store((obs, next_obs, actions, rewards, dones))
            obs = next_obs
            if np.any(dones):
                # 유니티 exe 파일 초기화 및 실행
                env.reset()
                obs = env.states
                self.memory.init_n_step()
        # 학습 시작 준비 완료
        print("memory initialized.")

    @property
    def e(self):
        ylow = self.e_min
        yhigh = self._e
        xlow = 0
        xhigh = self.anneal_max
        steep_mult = 8
        steepness = steep_mult / (xhigh - xlow)
        offset = (xhigh + xlow) / 2
        midpoint = yhigh - ylow
        x = np.clip(self.avg_score, 0, xhigh)
        x = steepness * (x - offset)
        e = ylow + midpoint / (1 + np.exp(x))
        return e

    def new_episode(self, scores):
        avg_across = np.clip(len(scores), 1, 50)
        self.avg_score = np.array(scores[-avg_across:]).mean()
        self.memory.init_n_step()
    
    def update_networks(self, agent, force_hard=False):
        if self.update_type == "soft" and not force_hard:
            self._soft_update(agent.actor_local, agent.actor_target)
            self._soft_update(agent.critic_local, agent.critic_target)
        elif self.t_step % self.update_every == 0 or force_hard:
            self._hard_update(agent.actor_local, agent.actor_target)
            self._hard_update(agent.critic_local, agent.critic_target)

    def _soft_update(self, active, target):
        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def _hard_update(self, active, target):
        target.load_state_dict(active.state_dict())

    def _gauss_noise(self, shape):
        n = np.random.normal(0, 1, shape)
        return self.e * n
        
class DDPG_Agent:
    
    # 각 에이전트에 대한 할당
    def __init__(self, state_size, action_size, args, agent_count=1, l2_decay=0.0001):
        self.device = args.device
        self.eval = args.eval
        self.actor_learn_rate = args.actor_learn_rate
        self.critic_learn_rate = args.critic_learn_rate
        self.gamma = args.gamma
        self.rollout = args.rollout
        self.num_atoms = args.num_atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atoms = torch.linspace(self.vmin, self.vmax, self.num_atoms).to(self.device)
        self.atoms = self.atoms.unsqueeze(0)
        self.actor_local = Actor(state_size, action_size, args.random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, args.random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learn_rate, weight_decay=l2_decay)
        all_states_size = state_size * agent_count
        all_actions_size = action_size * agent_count
        self.critic_local = Critic(all_states_size, all_actions_size, self.num_atoms, args.random_seed).to(self.device)
        self.critic_target = Critic(all_states_size, all_actions_size, self.num_atoms, args.random_seed).to(self.device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=self.critic_learn_rate, weight_decay=l2_decay)
    
    # observation 기준으로 action 실행
    def act(self, obs, eval=False): 
        obs = obs.to(self.device)
        with torch.no_grad():
            actions = self.actor_local(obs).cpu().numpy()
        return actions

    def learn(self, obs, next_obs, actions, target_actions, predicted_actions, rewards, dones):
        log_probs = self.critic_local(obs, actions, log=True)
        target_probs = self.critic_target(next_obs, target_actions).detach()
        target_dist = self._categorical(rewards, target_probs, dones)
        critic_loss = -(target_dist * log_probs).sum(-1).mean()
        probs = self.critic_local(obs, predicted_actions)
        expected_reward = (probs * self.atoms).sum(-1)
        actor_loss = -expected_reward.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def _categorical(self, rewards, probs, dones):
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        gamma = self.gamma
        rollout = self.rollout
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1).type(torch.float)
        delta_z = (vmax - vmin) / (num_atoms - 1)
        projected_atoms = rewards + gamma ** rollout * atoms * (1 - dones)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin) / delta_z
        precision = 1
        b = torch.round(b * 10 ** precision) / 10 ** precision
        lower_bound = b.floor()
        upper_bound = b.ceil()
        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs
        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()


class ReplayBuffer:
    """
    A replay buffer class is implemented for storing experiences for agents to learn.
    """
    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5, agent_count=1):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout
        self.agent_count = agent_count

    def store(self, experience):
        if self.rollout > 1:
            self.n_step.append(experience)
            if len(self.n_step) < self.rollout:
                return
            experience = self._n_stack()
        obs, next_obs, actions, rewards, dones = experience
        actions = np.concatenate(actions)
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.tensor(dones).float()

        self.buffer.append((obs, next_obs, actions, rewards, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)
        obs, next_obs, actions, rewards, dones = zip(*batch)
        obs = torch.stack(obs).transpose(1, 0).to(self.device)
        next_obs = torch.stack(next_obs).transpose(1, 0).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).transpose(1, 0).to(self.device)
        dones = torch.stack(dones).transpose(1, 0).to(self.device)
        return (obs, next_obs, actions, rewards, dones)

    def init_n_step(self):
        self.n_step = deque(maxlen=self.rollout)

    def _n_stack(self):
        obs, next_obs, actions, rewards, dones = zip(*self.n_step)
        summed_rewards = rewards[0]
        for i in range(1, self.rollout):
            summed_rewards += self.gamma ** i * rewards[i]
            if np.any(dones[i]):
                break
        obs = obs[0]
        nstep_obs = next_obs[i]
        actions = actions[0]
        dones = dones[i]
        return (obs, nstep_obs, actions, summed_rewards, dones)

    def __len__(self):
        return len(self.buffer)

# 하이퍼파라미터 정의
class args:
    eval = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 학습 튜닝
    num_episodes = 1200
    pretrain_length = 25000
    
    #신경망 변수들
    batch_size = 128
    actor_learn_rate = 5e-4
    critic_learn_rate = 3e-4
    gamma = 0.99
    tau = 1e-4
    e = 0.3
    e_decay = 1
    e_min = 0.00
    anneal_max = 0.7
    rollout = 5
    num_atoms = 100
    vmin = 0.0
    vmax = 2.0
    update_every = 2500
    print_every = 100
    update_type = 'soft'
    random_seed = 0
    train = not eval
    buffer_size = int(3e5)
    
# main 부분

# 나의 파일 위치
env_filepath = "C:/Users/sengh/OneDrive/Desktop/Github/Unity/RLBuildManagement/Version11/Drone.exe"

# 유니티 환경 초기화 (문제 없음 확인)
env = Environment(env_filepath)

# 
multi_agent = MADDPG_Net(env, args)

# 학습쪽에서 계속해서 에러 뜨고 있음  (train_maddpg 부분)
# Traceback (most recent call last):
#   File "c:\Users\sengh\OneDrive\Desktop\Github\Unity\RLBuildManagement\RLCode2\0201.py", line 436, in <module>
#     scores = train_maddpg(multi_agent, args, env)
#   File "c:\Users\sengh\OneDrive\Desktop\Github\Unity\RLBuildManagement\RLCode2\0201.py", line 40, in train_maddpg
#     multi_agent.learn()
#   File "c:\Users\sengh\OneDrive\Desktop\Github\Unity\RLBuildManagement\RLCode2\0201.py", line 221, in learn
#     agent.learn(obs, next_obs, actions, target_actions, predicted_actions, rewards[i], dones[i])
# IndexError: index 0 is out of bounds for dimension 0 with size 0
scores = train_maddpg(multi_agent, args, env)

# ChatGPT4의 문제 해결 방법
# 문제는 train_maddpg 함수 내의 multi_agent.learn() 호출 시 발생하는 IndexError: index 0 is out of bounds for dimension 0 with size 0 오류입니다. 이 오류는 일반적으로 텐서 또는 배열에 액세스하려 할 때 해당 텐서 또는 배열이 예상보다 작아 인덱스가 범위를 벗어날 때 발생합니다.
# 이 경우, multi_agent.learn() 메서드 내에서 rewards[i]와 dones[i]에 액세스하려고 할 때 발생합니다. 여기서 i는 각 에이전트의 인덱스를 나타내며, rewards와 dones는 환경으로부터 반환된 보상과 완료 상태입니다. 오류 메시지는 rewards 또는 dones 배열이 비어 있어(size 0) i 인덱스로 액세스할 수 없음을 나타냅니다.
# 문제 해결을 위해 다음 점을 확인해야 합니다:
# 환경 초기화 및 스텝 응답: Environment 클래스에서 step 메서드가 올바르게 구현되어 있으며, 각 스텝에서 유효한 rewards와 dones 배열을 반환하는지 확인합니다. 특히, self.env.get_steps(self.behavior_name) 호출이 예상대로 작동하여 올바른 데이터를 제공하는지 검증합니다.
# 에피소드 및 에이전트의 수: Environment 클래스가 올바른 에이전트 수(self.agent_count)를 반환하며, 이 값이 rewards와 dones 배열의 길이와 일치하는지 확인합니다. 만약 self.agent_count가 실제 환경의 에이전트 수보다 크거나 작다면, 배열 인덱싱 중에 문제가 발생할 수 있습니다.
# 배치 샘플링: MADDPG_Net.learn 메서드에서 배치 샘플링 과정이 적절히 수행되고 있는지 확인합니다. self.memory.sample(self.batch_size) 호출이 유효한 배치를 반환하며, 이 배치가 모든 에이전트에 대한 데이터를 포함하고 있는지 검사합니다.
# 보상 및 종료 상태 처리: MADDPG_Net.learn 내에서 rewards와 dones 처리가 모든 에이전트에 대해 올바르게 수행되는지 확인합니다. rewards와 dones가 각 에이전트별로 개별적으로 처리되어야 합니다. 이 경우, 배치에서 추출된 rewards와 dones의 차원과 구조를 검토하고, 모든 에이전트에 대해 올바른 인덱싱이 이루어지는지 확인합니다.
# 이 문제를 해결하기 위한 첫 단계로, Environment.step 메서드와 MADDPG_Net.learn 메서드에서 rewards와 dones의 길이와 내용을 로깅하여, 예상대로 동작하는지 확인해보세요. 이를 통해 문제의 원인을 좁혀나갈 수 있습니다.

# 결과 보고
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
