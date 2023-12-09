#!/usr/bin/python3


#%% Python imports

import torch
from torch import nn
from torch.nn import functional as F

import gymnasium as gym

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm


#%% Define MLP and policy networks

class MLP(nn.Module):
   def __init__(self, input_dim, hidden_dim=64, out_dim=1):
      super().__init__()
      self.mlp = nn.Sequential(
         nn.Linear(input_dim, hidden_dim),
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(),
        #  nn.Linear(hidden_dim, hidden_dim),
        #  nn.ReLU(),
         nn.Linear(hidden_dim, out_dim)
      )
   def forward(self, x):
      return self.mlp(x)

class PolicyDiscrete(nn.Module):
   def __init__(self, input_dim, hidden_dim=64, num_cat=2, cat_labels=None):
      super().__init__()
      self.mlp = MLP(input_dim, hidden_dim=hidden_dim, out_dim=num_cat)
      self.num_cat = num_cat
      self.cat_labels = None
      self.label_to_idx = None
      if cat_labels is not None:
         self.cat_labels = cat_labels
         self.label_to_idx = dict()
         for idx in range(len(cat_labels)):
            self.label_to_idx[cat_labels[idx]] = idx
   def forward(self, x):
      x = self.mlp(x)
      x = F.softmax(x, dim=-1)
      return x
   def sample(self, x):
      with torch.no_grad():
         x = self(x).numpy()
      action = np.random.choice(self.num_cat, 1, p=x)[0]
      if self.cat_labels is not None:
         action = self.cat_labels[action]
      return action
   def log_prob(self, x, action):
      x = self(x)
      if self.label_to_idx is not None:
         action = self.label_to_idx[action]
      action = torch.tensor(action)
      prob = torch.gather(x, -1, action)
      return torch.log(prob)

class PolicyContinuous(nn.Module):
   def __init__(self, input_dim, hidden_dim=64, num_out=1, constrained=False):
      super().__init__()
      self.mlp = MLP(input_dim, hidden_dim=hidden_dim, out_dim=num_out*2)
      self.num_out = num_out
      self.constrained = constrained
   def forward(self, x):
      x = self.mlp(x)
      mu = x[:self.num_out]
      sigma = torch.exp(x[self.num_out:])
      return mu, sigma
   def sample(self, x):
      with torch.no_grad():
         mu, sigma = self(x)
         action = torch.randn_like((mu))*sigma+mu
         if self.constrained:
            action = torch.tanh(action)
      return action.numpy()
   def log_prob(self, x, action):
      mu, sigma = self(x)
      action = torch.tensor(action)
      if self.constrained:
         action = torch.atanh(action)
      prob = 1/torch.sqrt(2*torch.pi*sigma**2)*torch.exp(-(action-mu)**2/(2*sigma**2))
      return torch.log(prob)


#%% Initialize environment and get sample observation and action
env = gym.make("CartPole-v1")#, render_mode="human")
observation, info = env.reset(seed=1)
action = env.action_space.sample()


#%% Initialize policy and baseline value networks
policy = PolicyDiscrete(observation.shape[0], hidden_dim=32, cat_labels=[0, 1])
value = MLP(observation.shape[0], hidden_dim=32)

optimizer_policy = torch.optim.Adam(policy.parameters(), lr=3e-5)
optimizer_value = torch.optim.Adam(value.parameters(), lr=1e-4)

discount = 1#0.99


#%% Run training loop

num_episodes = 3000

observation, info = env.reset(seed=0)
returns_data = []
for _ in (pbar := tqdm(range(num_episodes))):
   # loop episode, store all (S,A,R) tuples
   S_list = []
   A_list = []
   R_list = []

   while True:
      action = policy.sample(torch.tensor(observation))
      next_observation, reward, terminated, truncated, info = env.step(action)

      S_list.append(observation)
      A_list.append(action)
      R_list.append(reward)
      observation = next_observation

      if terminated or truncated:
         break
   
   G_list = []
   G_list.append(R_list[-1])
   for t in range(len(R_list)-1):
      G_list.append(G_list[-1]*discount + R_list[-(t+2)])
   G_list.reverse()
   
   episode_len = len(S_list)
   
   with torch.no_grad():
      printx = policy(torch.tensor(S_list[0])).numpy()
      a = printx[0]
      b = printx[1]
      pbar.set_postfix({'return': "{:03.2f}".format(G_list[0]),
   'estimate': "{:03.2f}".format(value(torch.tensor(S_list[0])).numpy()[0]), 'predict': "{:01.4f} {:01.4f}".format(a,b)})
   returns_data.append(G_list[0])
   if _ % 100 == 0:
      plt.plot(list(range(len(returns_data))), returns_data)
      plt.savefig("returns_cartpole_adam_baseline.png")

   for S, A, G in zip(S_list, A_list, G_list):
      S = torch.tensor(S)
      G = torch.tensor(G)
      with torch.no_grad():
         delta = G - value(S)


      optimizer_value.zero_grad()
      optimizer_policy.zero_grad()
      
      value_loss = 0
      policy_loss = 0


      value_loss += -delta * value(S)
      policy_loss += -delta * policy.log_prob(S, A)
   

      # value_loss /= episode_len
      # policy_loss /= episode_len

      value_loss.backward()
      policy_loss.backward()

      optimizer_value.step()
      optimizer_policy.step()

   observation, info = env.reset()

#%%

plt.xlabel('Episodes')
plt.ylabel('Accumulated Rewards')
plt.plot(list(range(len(returns_data))), returns_data)
plt.savefig("returns_cartpole_adam_baseline.png")
torch.save(policy.state_dict(), 'cartpoleadambaselinepolicy.pth')
torch.save(value.state_dict(), 'cartpoleadambaselinevalue.pth')
import pickle
with open('cartpoleadambaselinereturns.pickle', 'wb') as f:
    pickle.dump(returns_data, f)


#%%
