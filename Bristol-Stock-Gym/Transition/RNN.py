import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

bsz = 200
epochs = 500


z_size = 17
n_hidden = 256
n_gaussians = 5
seqlen = 4


class MDNRNN(nn.Module):
    def __init__(self, z_size = 17, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(z_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc2 = nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians*z_size)
        
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
     
        
        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)
        
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h):
        # Forward propagate LSTM
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))
    
    def save_model(self):
        torch.save(self.state_dict(), 'Models/RNN' )
        




    def mdn_loss_fn(self,y, pi, mu, sigma):
        
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss)
        
        return loss.mean()



    def criterion(self,y, pi, mu, sigma):
        y = y.unsqueeze(2)
        return self.mdn_loss_fn(y, pi, mu, sigma)


# optimizer = torch.optim.Adam(model.parameters())

# for epoch in range(epochs):
#     # Set initial hidden and cell states
#     hidden = model.init_hidden(bsz)
    
#     for i in range(0, z.size(1) - seqlen, seqlen):
#         # Get mini-batch inputs and targets
#         inputs = z[:, i:i+seqlen, :]
#         targets = z[:, (i+1):(i+1)+seqlen, :]
        
#         # Forward pass
#         hidden = detach(hidden)
#         (pi, mu, sigma), hidden = model(inputs, hidden)
#         loss = criterion(targets, pi, mu, sigma)
        
#         # Backward and optimize
#         model.zero_grad()
#         loss.backward()
#         # clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
        
#     if epoch % 100 == 0:
#         print ('Epoch [{}/{}], Loss: {:.4f}'
#                .format(epoch, epochs, loss.item()))


