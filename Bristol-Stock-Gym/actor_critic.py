import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 


class GenericNetwork(nn.Module):
    
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        T.set_default_tensor_type('torch.DoubleTensor')

        super(GenericNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=  self.lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        
        
    def forward(self, observation):
        state = T.tensor(observation).to(self.device)
        state = state.flatten()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    
class Agent(object):
    
    def __init__(self, actor_lr, critic_lr, input_dims, gamma = 0.9,
                  l1_size = 256, l2_size = 256, n_actions = 3):
        self.gamma = gamma
        self.log_probs = None 
        self.actor = GenericNetwork(lr = actor_lr, input_dims=input_dims, 
                                    fc1_dims=l1_size, fc2_dims=l1_size, n_actions= n_actions)
        
        self.critic = GenericNetwork(lr = critic_lr, input_dims=input_dims, 
                                    fc1_dims=l1_size, fc2_dims=l1_size, n_actions= 1)
        
    def choose_action(self,observation):       
        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        
        return action.item()
        
    
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)
        
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - critic_value)
        
        actor_loss = - self.log_probs * delta
        critic_loss = delta**2
        
        (actor_loss + critic_loss).backward()
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        
    def save_models(self, actor_outfile = "actor", critic_outfile = "critic"):
        T.save(self.actor.state_dict(), f"Models/{actor_outfile}.pth")
        T.save(self.critic.state_dict(), f"Models/{critic_outfile}.pth")
    
    def load_models(self, actor_outfile = "actor", critic_outfile = "critic"):
        try:
            self.actor.load_state_dict(T.load(f'Models/{actor_outfile}', map_location = 'cpu'))
            self.actor.load_state_dict(T.load(f'Models/{critic_outfile}', map_location = 'cpu'))
        except Exception as e:
            print(e)
            print("No models found")