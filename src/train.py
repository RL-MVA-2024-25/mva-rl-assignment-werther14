import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import os
import time


from evaluate import evaluate_HIV, evaluate_HIV_population


import random
from collections import deque
import numpy as np






import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os
import time

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.position = 0
        self.size = 0

    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)

    def get_leaf(self, value):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):
                break

            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.position + self.capacity - 1
        self.data[self.position] = data
        self.update(idx, priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=1e-4):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size):
        batch_indices = []
        batch = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(value)
            
            batch_indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, batch_indices, is_weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = float(priority)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self):
        return self.tree.size

class DuelingQNetwork(nn.Module):
   def __init__(self, state_dim=6, action_dim=4):
       super(DuelingQNetwork, self).__init__()

       self.feature = nn.Sequential(
           nn.Linear(state_dim, 256),
           nn.SiLU(),
           nn.Linear(256, 512),
           nn.SiLU(),
           nn.Linear(512, 1024),
           nn.SiLU(),
           nn.Linear(1024, 1024),
           nn.SiLU(),
           nn.Linear(1024, 1024),
           nn.SiLU(),
       )
       self.value_stream = nn.Sequential(
           nn.Linear(1024, 512),
           nn.SiLU(),
           nn.Linear(512, 256),
           nn.SiLU(),
           nn.Linear(256, 1)
       )
       self.advantage_stream = nn.Sequential(
           nn.Linear(1024, 512),
           nn.SiLU(),
           nn.Linear(512, 256),
           nn.SiLU(),
           nn.Linear(256, action_dim)
       )

       self.apply(self._init_weights)

   def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

   def forward(self, x):
       features = self.feature(x)
       value = self.value_stream(features)              
       advantage = self.advantage_stream(features)      
       q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
       return q_values



class ProjectAgent:
    def __init__(self, state_dim=6, n_actions=4):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = 0.85
        self.save_path = "project_agent_test.pt"
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9965 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DuelingQNetwork(state_dim, n_actions).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5)
        
        self.replay_buffer = PrioritizedReplayBuffer(capacity=60000)
        
        self.batch_size = 1024
        self.target_update_freq = 1000
        self.update_count = 0

    def act(self, observation, use_random=False):
       if use_random and random.random() < self.epsilon:
           return np.random.randint(0, self.n_actions)
       state_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
       q_values = self.q_network(state_t)
       return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        
        self.replay_buffer.update_priorities(indices, td_errors)
        
        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def step_scheduler(self):
        self.scheduler.step()
        
    def update_epsilon(self):
       if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.save_path
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

def train_episode(env, agent):
   state, _ = env.reset()
   episode_reward = 0.0
   actions = {0: 0, 1: 0, 2: 0, 3: 0}
  
   for step in range(200):
       action = agent.act(state, use_random=True)
       actions[action] += 1
       next_state, reward, done, truncated, _info = env.step(action)
      
       agent.replay_buffer.push(state, action, reward, next_state, done)
       agent.train_step()
      
       state = next_state
       episode_reward += reward
      
       if done or truncated:
           break
          
   return episode_reward, actions



def main():

   num_episodes = 1500
   EVAL_THRESHOLD = 3.5e10
   SAVE_INTERVAL = 100  
  
   env = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
   agent = ProjectAgent()
   reward_history = []
   BEST_VALIDATION_SCORE = 0.0
   start_time = time.time()
  
   try:
       for episode in range(num_episodes):
           episode_reward, actions = train_episode(env, agent)
           reward_history.append(episode_reward)
          
           agent.step_scheduler()
           agent.update_epsilon()
          
           print(
               f"Episode {episode:4d}, "
               f"Reward: {int(episode_reward):11d}, "
           )
          
           if episode_reward > EVAL_THRESHOLD:
               validation_score = evaluate_HIV(agent=agent, nb_episode=3)
               validation_score_dr = evaluate_HIV_population(agent=agent, nb_episode=5)
               avg_score = (validation_score + validation_score_dr) / 2
              
               if avg_score > BEST_VALIDATION_SCORE:
                   BEST_VALIDATION_SCORE = avg_score
                   print(f"New model with validation score: {BEST_VALIDATION_SCORE:.2f}")
                   agent.save("best_" + agent.save_path)
              
               print(f"Validation score: {validation_score:.2f}, Validation score DR: {validation_score_dr:.2f}")
              
               if validation_score > 4e10 and validation_score_dr > 2.5e10:
                   agent.save("best_top_score_" + agent.save_path)
                   break
  
   except KeyboardInterrupt:
       print("\nTraining interrupted by user")
  
   finally:
       agent.save(agent.save_path)
      
       elapsed_time = time.time() - start_time
       print(f"\nTraining completed in {elapsed_time:.1f} seconds")
       print(f"Best validation score: {BEST_VALIDATION_SCORE:.2e}")


if __name__ == "__main__":
   main()

