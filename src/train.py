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


class PrioritizedReplayBuffer:
   def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-4, alpha_decrement_per_sampling=1e-6):
       self.capacity = capacity
       self.buffer = []
       self.pos = 0
       self.priorities = np.zeros((capacity,), dtype=np.float32)
       self.alpha = alpha
       self.beta = beta
       self.beta_increment_per_sampling = beta_increment_per_sampling
       self.alpha_decrement_per_sampling = alpha_decrement_per_sampling
       self.eps = 1e-5 
       self.max_priority = 1.0 
       self.max_priority_limit = 10.0 


   def push(self, state, action, reward, next_state, done):
       if len(self.buffer) < self.capacity:
           self.buffer.append((state, action, reward, next_state, done))
       else:
           self.buffer[self.pos] = (state, action, reward, next_state, done)
      
       self.priorities[self.pos] = min(self.max_priority_limit, 0.9 * self.max_priority)
       self.pos = (self.pos + 1) % self.capacity


   def sample(self, batch_size = 512):
       priority_size = int(0.5 * batch_size) 
       random_size = int(0.3 * batch_size)  
       recent_size = batch_size - priority_size - random_size 

       priorities = self.priorities[:len(self.buffer)] + self.eps
       priorities = priorities / (priorities.max() + self.eps) 
       probs = priorities ** self.alpha 
       probs /= probs.sum() 
       
       priority_indices = np.random.choice(len(self.buffer), priority_size, p=probs)
       random_indices = np.random.choice(len(self.buffer), random_size, replace=False)
       recent_indices = np.arange(max(0, len(self.buffer) - recent_size), len(self.buffer))
      
       indices = np.concatenate([priority_indices, random_indices, recent_indices])
       np.random.shuffle(indices)

       batch = [self.buffer[idx] for idx in indices]

       total = len(self.buffer)
       weights = (total * probs[indices] + self.eps) ** (-self.beta)
       weights /= weights.max() 
       
       return (
           np.array([s for s, _, _, _, _ in batch]), 
           np.array([a for _, a, _, _, _ in batch]), 
           np.array([r for _, _, r, _, _ in batch], dtype=np.float32),
           np.array([n_s for _, _, _, n_s, _ in batch]),
           np.array([d for _, _, _, _, d in batch], dtype=np.float32), 
           indices, 
           weights 
       )


   def update_priorities(self, batch_indices, batch_priorities):
       for idx, prio in zip(batch_indices, batch_priorities):
           self.priorities[idx] = min(self.max_priority_limit, prio + self.eps)
           self.max_priority = max(self.max_priority, self.priorities[idx])

   def __len__(self):
       return len(self.buffer)


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
       self.save_path = "project_agent.pt"
       self.replay_buffer = PrioritizedReplayBuffer(capacity=60000)


       self.lr = 1e-3 
       self.batch_size = 1024 
       self.epsilon = 1.0
       self.epsilon_min = 0.01
       self.epsilon_decay = 0.9965 
       self.target_update_freq = 1000
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.update_count = 0
       self.q_network = DuelingQNetwork(self.state_dim, self.n_actions).to(self.device)
       self.target_network = DuelingQNetwork(self.state_dim, self.n_actions).to(self.device)
              
       self.target_network.load_state_dict(self.q_network.state_dict())
       self.target_network.eval()
       self.optimizer = optim.Adam(self.q_network.parameters(),lr=self.lr,betas=(0.5, 0.999))
       self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5)


   def act(self, observation, use_random=False):
       if use_random and random.random() < self.epsilon:
           return np.random.randint(0, self.n_actions)
       state_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
       q_values = self.q_network(state_t)
       return q_values.argmax(dim=1).item()


   def train_step(self):
       if len(self.replay_buffer) < self.batch_size:
           return
       sample = self.replay_buffer.sample(self.batch_size)
       if sample is None:
               return
       states, actions, rewards, next_states, dones, indices, weights = sample
       weights_t = torch.FloatTensor(weights).to(self.device).unsqueeze(1)


       states_t = torch.FloatTensor(states).to(self.device)
       actions_t = torch.LongTensor(actions).to(self.device)
       rewards_t = torch.FloatTensor(rewards).to(self.device)
       next_states_t = torch.FloatTensor(next_states).to(self.device)
       dones_t = torch.FloatTensor(dones).to(self.device)

       q_values = self.q_network(states_t)
       q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

       with torch.no_grad():
           next_actions = self.q_network(next_states_t).argmax(1)
           max_next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
           target = rewards_t + self.gamma * max_next_q * (1 - dones_t)

       loss = (q_values - target) ** 2
       loss = loss * weights_t.squeeze()  
       loss = loss.mean()


       self.optimizer.zero_grad()
       loss.backward()
       nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
       self.optimizer.step()

       if indices is not None:
           td_errors = (q_values - target).detach().cpu().numpy()
           new_priorities = np.abs(td_errors)
           self.replay_buffer.update_priorities(indices, new_priorities)


       self.update_count += 1
       if self.update_count % self.target_update_freq == 0:
           self.target_network.load_state_dict(self.q_network.state_dict())

   def update_epsilon(self):
       if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay
  
   def step_scheduler(self):
       self.scheduler.step()


   def save(self, path):
       torch.save(self.q_network.state_dict(), path)
       print(f"DQN model saved to {path}")


   def load(self, path = None):
       if path is None:
           path = self.save_path
       self.q_network.load_state_dict(torch.load(path, map_location="cpu"))
       print(f"DQN model loaded from {path}")


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
          
           agent.update_epsilon()
           agent.step_scheduler()
          
           print(
               f"Episode {episode:4d}, "
               f"Reward: {int(episode_reward):11d}, "
           )
          
           if episode % SAVE_INTERVAL == 0 and episode > 0:
               agent.save(f"checkpoint_{episode}.pt")
          
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
       print(f"Final epsilon: {agent.epsilon:.3f}")


if __name__ == "__main__":
   main()
