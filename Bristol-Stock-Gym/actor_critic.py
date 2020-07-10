import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
import os

class OUActionNoise(object):
    def __init__(self, mu,sigma = 0.15, theta = 0.2, dt = 1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0= x0
        self.dt = dt
        self.reset()
        
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev ) * self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size = self.mu.shape)
        self.x_prev = x
        return x
        
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)




class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size , n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.float32)

    
    def store_transition(self,state,action,reward,state_,done):
        state = state.flatten()
        state_ = state.flatten()
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward 
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done 
        self.mem_cntr +=1
    
    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal_memory = self.terminal_memory[batch]
        
        
        return states,actions,rewards,new_states,terminal_memory



class CriticNetwork(nn.Module):
    def __init__(self,lr, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = os.path.join("Models", name)
        
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #Constraining the weights to -f1 to f1
        f1_weights = 1/np.sqrt(self.fc1.weight.data.size()[0])   
        torch.nn.init.uniform_(self.fc1.weight.data, -f1_weights, f1_weights)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1_weights, f1_weights)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2_weights = 1/np.sqrt(self.fc2.weight.data.size()[0])   
        torch.nn.init.uniform_(self.fc2.weight.data, -f2_weights, f2_weights)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2_weights, f2_weights)        
        
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3,f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3,f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = F.relu(self.action_value(action))  
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
    
        return state_action_value

        
    def save_checkpoint(self):
        print("Saving Model..")
        torch.save(self.state_dict(),self.name)
        
    def load_checkpoint(self):
        print("Loading model..")
        self.load_state_dict(torch.load(self.name))
        

class ActorNetwork(nn.Module):
    def __init__(self,lr, input_dims, fc1_dims, fc2_dims, n_actions, name, MAX_ACTION):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = os.path.join("Models", name)
        self.MAX_ACTION = MAX_ACTION
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #Constraining the weights to -f1 to f1
        f1_weights = 1/np.sqrt(self.fc1.weight.data.size()[0])   
        torch.nn.init.uniform_(self.fc1.weight.data, -f1_weights, f1_weights)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1_weights, f1_weights)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2_weights = 1/np.sqrt(self.fc2.weight.data.size()[0])   
        torch.nn.init.uniform_(self.fc2.weight.data, -f2_weights, f2_weights)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2_weights, f2_weights)        
        
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3,f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3,f3)
        

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    
    def forward(self, state):
        x = self.fc1(state)
        
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))
        
        return x 
    
    def save_checkpoint(self):
        print("Saving Model..")
        torch.save(self.state_dict(),self.name)
        
    def load_checkpoint(self):
        print("Loading model..")
        self.load_state_dict(torch.load(self.name))
        
        



class Agent(object):
    def __init__(self,alpha, beta, input_dims, tau, gamma = 0.99, n_actions = 2,
                  max_size = 1000000, layer1_size = 400, layer2_size = 300, batch_size = 64, 
                  MAX_ACTION = 1000, actor_name ='Actor-DDPG', target_actor_name ='TargetActor-DDPG',
                  critic_name ='Critic-DDPG', target_critic_name = 'TargetCritic-DDPG' ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.MAX_ACTION = MAX_ACTION
        
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = actor_name, MAX_ACTION = MAX_ACTION)
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = target_actor_name, MAX_ACTION = MAX_ACTION)
        
        self.critic = CriticNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = critic_name)
        self.target_critic = CriticNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name = target_critic_name)
        
        self.noise = OUActionNoise(mu = np.zeros(n_actions))
        
        self.update_network_parameters(tau = 1)
        
    
    def choose_action(self,state):
        self.actor.eval()
        state = torch.tensor(state, dtype = torch.float).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype = torch.float).to(self.actor.device)
        self.actor.train()
        
        mu_prime = mu_prime.cpu().detach().numpy()
        action = mu_prime[0]
        if round(action) < -1:
            action = -1
        print(action) 
        return round(mu_prime[0]), mu_prime[1]*self.MAX_ACTION
    
    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state, action,reward,new_state, done)
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state,action,reward,new_state,done = self.memory.sample_buffer(self.batch_size)
        reward = torch.tensor(reward, dtype = torch.float).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype= torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype= torch.float).to(self.critic.device)
        state = torch.tensor(state, dtype= torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
    
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        
        critic_value = self.critic.forward(state,action)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] * self.gamma*critic_value_[j]*done[j])
        target = torch.Tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)
        
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target,critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = - self.critic.forward(state,mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
    
    def update_network_parameters(self, tau= None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params) 
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)
                                    
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()                              
        self.target_actor.load_state_dict(actor_state_dict)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        
        
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()