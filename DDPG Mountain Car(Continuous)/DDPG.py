import gym
import numpy as np
from collections import deque
import torch
import torch.autograd
import torch.nn.functional as F 
import torch.optim as optim
import torch.nn as nn
import random
import matplotlib.pyplot as plt 
class Memory():



	def __init__(self,max_length):
		self.length = max_length
		self.buffer = deque(maxlen = self.length)

	def push(self,state,action,rewards,next_state,done):
		experience = state,action,next_state,rewards,done
		self.buffer.append(experience)

	def sample(self,batch_size):
		state_batch = []
		action_batch = []
		next_state_batch = []
		rewards_batch = []
		done_batch = []
		batch = random.sample(self.buffer, batch_size)
		for experiences in batch:
			state,action,next_state,rewards,done = experiences
			state_batch.append(state)
			action_batch.append(action)
			next_state_batch.append(next_state)
			rewards_batch.append(rewards)
			done_batch.append(done)
		return state_batch,action_batch,next_state_batch,rewards_batch,done_batch



class DDPG():
	def __init__(self,num_states,num_actions,tau,actor_lr,critic_lr,hidden_size,max_length,gamma):
		self.num_states = num_states
		self.num_actions = num_actions
		self.critic_lr = critic_lr
		self.actor_lr = actor_lr
		self.hidden_size = hidden_size
		self.tau = tau
		self.max_length = max_length
		self.gamma = gamma

		## Networks
		self.actor = Actor(self.num_states,self.hidden_size,self.num_actions)
		self.actor_target = Actor(self.num_states,self.hidden_size,self.num_actions)
		self.critic = Critic(self.num_states + self.num_actions,self.hidden_size,self.num_actions)
		self.critic_target = Critic(self.num_states + self.num_actions,self.hidden_size,self.num_actions)

		## Copy Parameter
		for target_param,param in zip(self.actor_target.parameters(),self.actor.parameters()):
			target_param.data.copy_(param.data)
		for target_param,param in zip(self.critic_target.parameters(),self.critic.parameters()):
			target_param.data.copy_(param.data)

		## Training
		self.memory = Memory(self.max_length)
		self.critic_criterion = nn.MSELoss()
		self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.actor_lr)
		self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.critic_lr)


	def get_action(self,state):
		state = torch.Tensor(state)
		action = self.actor.forward(state)
		action = action.detach().numpy()[0]
		return action


	def update_values(self,batch_size):
		state_batch,action_batch,next_state_batch,rewards_batch,done_batch = self.memory.sample(batch_size)
		state = torch.Tensor(state_batch)
		action = torch.Tensor(action_batch)
		next_state = torch.Tensor(next_state_batch)
		rewards = torch.Tensor(rewards_batch)
		done = torch.Tensor(done_batch)
		Qval = self.critic.forward(state,action)
		next_action = self.actor_target.forward(next_state)
		next_Q = self.critic_target.forward(next_state, next_action.detach())
		rewards.resize_((next_Q.size()))
		Q_prime = rewards + self.gamma*next_Q
		# print(Q_prime.shape,rewards.shape)
		critic_loss = self.critic_criterion(Qval,Q_prime)

		policy_loss = -self.critic.forward(state,self.actor.forward(state)).mean()


		## Reset Model 
		self.actor_optimizer.zero_grad()
		policy_loss.backward()
		self.actor_optimizer.step()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		## Update Parameters
		for target_param,param in zip(self.actor_target.parameters(),self.actor.parameters()):
			target_param.data.copy_((self.tau*param.data) + (1-self.tau)*target_param.data)
		for target_param,param in zip(self.critic_target.parameters(),self.critic.parameters()):
			target_param.data.copy_((self.tau*param.data) + (1-self.tau)*target_param.data)



class Critic(nn.Module):
	def __init__(self,inputs,hidden,outputs):
		super().__init__()
		self.linear1 = nn.Linear(inputs,hidden)
		self.linear2 = nn.Linear(hidden,hidden)
		self.linear3 = nn.Linear(hidden , outputs)


	def forward(self,state, action):
		x = torch.cat([state,action],1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		return self.linear3(x)


class Actor(nn.Module):
	def __init__(self,inputs,hidden,outputs):
		super().__init__()
		self.linear1 = nn.Linear(inputs,hidden)
		self.linear2 = nn.Linear(hidden,hidden)
		self.linear3 = nn.Linear(hidden , outputs)

		
	def forward(self,state):
		#x = torch.cat([state,action],1)
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		return torch.tanh(self.linear3(x))

class Noise():
	def __init__(self,action_space,mu = 0.0,theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 1000000):
		#self.num_states = num_states
		#self.state = state
		self.num_actions = action_space.shape[0]
		self.min_action = action_space.low
		self.max_action = action_space.high
		self.mu = mu
		self.sigma = max_sigma
		self.min_sigma = min_sigma
		self.max_sigma = max_sigma
		self.theta = theta
		self.decay_period = decay_period
		self.reset()

	def reset(self):
		self.state = np.ones(self.num_actions)*self.mu

	def evolve_state(self):
		x  = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)
		self.state = x + dx
		return self.state

	def get_action(self, action, t=0):
		ou_state = self.evolve_state()
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
		return np.clip(action + ou_state, self.min_action, self.max_action)





def main():

	env = gym.make('MountainCarContinuous-v0')
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]
	tau = 0.01
	actor_lr = 0.00001
	critic_lr = 0.0001
	hidden_size = 250
	max_length = 500000
	gamma = 0.990 
	batch_size = 128
	agent = DDPG(num_states,num_actions,tau,actor_lr,critic_lr,hidden_size,max_length,gamma)
	noise = Noise(env.action_space)
	num_episodes = 500
	ep_rewards = []
	avg = []
	avg_reward = 0
	for i in range(num_episodes):
		observation = env.reset()
		noise.reset()
		episode_reward = 0

		#print(env.action_space.shape[0])

		for _ in range(500):
			# if (i %10 ==0):
			# 	env.render()
			#action = env.action_space.sample() # your agent here (this takes random actions)
			action = agent.get_action(observation)
			# print('1',action)
			action = noise.get_action(action)
			# print('2',action)
			next_state, reward, done, info = env.step(action)
			agent.memory.push(observation,action,reward,next_state,done)
			if len(agent.memory.buffer) > batch_size:
				agent.update_values(batch_size)
			observation = next_state
			episode_reward += reward
			if done:
				observation = env.reset()
				break
		ep_rewards.append(episode_reward)
		avg_reward = ((avg_reward*i)+episode_reward)/(i+1)
		avg.append(avg_reward)
		print("episode",i,"reward", episode_reward,'avg_reward',avg_reward)
	x_axis = np.arange(np.size(ep_rewards))
	plt.plot(x_axis,ep_rewards,'r')
	plt.plot(x_axis,avg,'g')
	plt.show()
    		
    #env.close()


if __name__ == '__main__':
	main()
    	

	
	