import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from torch.distributions import Categorical




    
    
class Actor(nn.Module):
    
    def __init__(self, input_shape, action_size):
        super(Actor,self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        
        self.Layer1 = nn.Linear(8*2, 128)
        self.Layer2 = nn.Linear(128,256)
        self.Layer3 = nn.Linear(256,self.action_size)
        
    def forward(self,state):
        state = state.view(1,-1)
        x = F.relu(self.Layer1(state))
        x = F.relu(self.Layer2(x))
        x = self.Layer3(x)
        distribution = Categorical(F.softmax(x, dim = -1))
        return distribution

class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.linear1 = nn.Linear(8*2, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        state = state.view(1,-1)
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value     
    
def save_models(actor, critic, actor_outfile = "actor", critic_outfile = "critic"):
    torch.save(actor, f"Models/{actor_outfile}.pkl")
    torch.save(critic, f"Models/{critic_outfile}.pkl")
        