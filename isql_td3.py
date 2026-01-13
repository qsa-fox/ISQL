import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from isql_main import device


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action,hidden_dim=256):
		super(Actor, self).__init__()

		self.pi = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, action_dim),
		)

		self.max_action = max_action

	def forward(self, state):
		a = self.pi(state)
		return self.max_action * torch.tanh(a)


class ValueNet(nn.Module):
	def __init__(self, state_dim, hidden_dim=256):
		super(ValueNet, self).__init__()
		self.vnet = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

			
	def forward(self, state):
		v = self.vnet(state)
		return v
	

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Critic, self).__init__()
		# Q1 architecture
		self.q1 = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

		self.q2 = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)
			
	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.q1(sa)
		q2 = self.q2(sa)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.q1(sa)
		return q1


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class TD3(object):
	def __init__(
			self,
			state_dim,
			action_dim,
			max_action,
			discount=0.99,
			tau=0.005,
			policy_freq=2,
			beta_list=[1.0],
			critic_adv_beta=1.0,
	):
		self.beta_list = beta_list
		self.actor_list, self.actor_optimizer_list, self.actor_lr_schedule_list = {}, {}, {}
		for actor_type in ['ddpg', 'awr']:
			self.actor_list[actor_type], self.actor_optimizer_list[actor_type], self.actor_lr_schedule_list[actor_type] = [], [], []
			for i in range(len(beta_list)):
				actor = Actor(state_dim, action_dim, max_action).to(device)
				actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
				self.actor_list[actor_type].append(actor)
				self.actor_optimizer_list[actor_type].append(actor_optimizer)
				self.actor_lr_schedule_list[actor_type].append(CosineAnnealingLR(actor_optimizer, int(1e6)//policy_freq))

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.vnet = ValueNet(state_dim, hidden_dim=256).to(device)
		self.vnet_target = copy.deepcopy(self.vnet)
		self.vnet_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq
		self.total_it = 0
		self.action_dim = action_dim
		self.beta_list = beta_list
		self.critic_adv_beta = critic_adv_beta

	def select_action(self, state, actor_indice, actor_type):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor_list[actor_type][actor_indice](state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1
		# Sample replay buffer
		state, action, next_state, reward, not_done, _ = replay_buffer.sample(batch_size)

		with torch.no_grad():
			target_v = self.vnet_target(next_state)
			target_v = reward + not_done * self.discount * target_v

		# Get current Q estimates
		q1, q2 = self.critic(state, action)
		critic_loss = F.mse_loss(q1, target_v) + F.mse_loss(q2, target_v)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step() 

		v1 = self.vnet(state)
		adv1 = torch.min(q1, q2).detach() - v1

		## ========= ISQL-Clip
		# v_loss = asymmetric_l2_loss(adv1, tau=0.9)

		## ========= ISQL-Sigmoid
		# weight = 2 * torch.sigmoid(self.critic_adv_beta * adv1).detach()
		# weight = weight.clip(min=0.2, max=10)
		# v_loss = (adv1**2 * weight).mean()
		
		## ========= ISQL-KL
		# exp_adv1 = torch.exp(self.critic_adv_beta * adv1).clamp(0.2, 10).detach()
		# v_loss = (adv1**2 * exp_adv1).mean()

		## ========= ISQL-Kai_square
		weight = 1 + adv1 * self.critic_adv_beta
		weight = weight.detach().clip(min=0.2, max=10)
		v_loss = (adv1**2 * weight).mean()

		# Optimize the value function
		self.vnet_optimizer.zero_grad()
		v_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.vnet.parameters(), max_norm=10.0)
		self.vnet_optimizer.step()

		for param, target_param in zip(self.vnet.parameters(), self.vnet_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		if (self.total_it-1) % self.policy_freq == 0:
			for actor_type in ['ddpg', 'awr']:
				for i in range(len(self.actor_list[actor_type])):
					pi = self.actor_list[actor_type][i](state)

					if actor_type == 'ddpg':
						## ddpg actor
						pi_q1 = self.critic.Q1(state, pi)
						scale = torch.abs(pi_q1).detach().mean() + 1e-6
						actor_loss = -pi_q1.mean() / scale + self.beta_list[i] * F.mse_loss(pi, action)
					else:
						## awr actor
						exp_adv = torch.exp(adv1.squeeze().detach() / self.beta_list[i]).clamp(0, 100)
						actor_loss = (exp_adv * ((pi - action)**2).sum(-1)).mean()

					self.actor_optimizer_list[actor_type][i].zero_grad()
					actor_loss.backward()
					self.actor_optimizer_list[actor_type][i].step()
					self.actor_lr_schedule_list[actor_type][i].step()

		return {'Q': q1[0].mean().item(),
				'V': v1.mean().item(),
				'adv/adv': adv1.mean().item(),
				'adv/max_adv': adv1.max().item(),
				'adv/min_adv': adv1.min().item(),	
				'weight/weight': weight.mean().item(),
				'weight/max_weight': weight.max().item(),
				'weight/min_weight': weight.min().item(),
				'critic_loss': critic_loss.item(),
				'v_loss': v_loss.item(),}

	def save(self, filename):
		torch.save(self.critic_ensemble.state_dict(), filename + "_critic_ensemble")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic_ensemble.load_state_dict(torch.load(filename + "_critic_ensemble"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target_ensemble = copy.deepcopy(self.critic_ensemble)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

	def save_all(self, filename):
		torch.save(self, filename + '_all.pth')

	def load_all(self, filename):
		return torch.load(self, filename + '_all.pth')
