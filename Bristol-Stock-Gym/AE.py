import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#Designed for a LOB input of 5 * 8 input


class Autoencoder(nn.Module):
    def __init__(self, input_dims, l1_size, l2_size,l3_size):
        super(Autoencoder,self).__init__()
        
        
        #Layer information 
        self.l1_in = input_dims
        self.l1_out = l1_size
        self.l2_in = l1_size        
        self.l2_out = l2_size
        self.l3_in = l2_size
        self.l3_out = l3_size
        
        self.encoder = nn.Sequential(
                            nn.Linear(self.l1_in, self.l1_out),
                            nn.Tanh(),
                            nn.Linear(self.l2_in,self.l2_out),
                            nn.Tanh(),
                            nn.Linear(self.l3_in, self.l3_out)
                        )
            
        self.decoder = nn.Sequential(             
                            nn.Linear(self.l3_out, self.l3_in),
                            nn.Tanh(),
                            nn.Linear(self.l2_out,self.l2_in),
                            nn.Tanh(),
                            nn.Linear(self.l1_out, self.l1_in),
                            nn.Sigmoid()
                        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    



class LOB_trainer(object):
    
    def __init__(self, lr = 1e-3, input_dims = [9,5], l1_size = 32, l2_size = 16 , l3_size = 8, window_size = 5):
        self.lob = np.zeros((input_dims[0],input_dims[1],1))

        self.autoencoder = Autoencoder(input_dims=input_dims[0]*input_dims[1], 
                                    l1_size=l1_size, l2_size=l1_size, l3_size= l3_size)
        
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay = 1e-5)
        
        self.loss = nn.MSELoss()
        
    def learn(self):
        self.optimizer.zero_grad()
        LOB = self.lob.flatten()
        
        
        x = Variable(torch.from_numpy(LOB))    
        y = x
    
        encoded, decoded = self.autoencoder.forward(x)
        loss = self.loss(decoded, y)
        self.optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        self.optimizer.step()       

    def get_lob_snapshot(self,column, time):
        #get latest 5 time step snapshot
        #if latest time step is unchanged, no amendment, otherwise, next step 
        #delete latest oldest time step
        #add new time step lob
        #append to the stack of time step lobs
    
        rows, cols, depth = self.lob.shape
        
        snapshot = self.lob[:,:,depth-1]
        row, cols = snapshot.shape
        latest_time_step = snapshot[:, cols-1]
        #latest_time_step = np.reshape(latest_time_step, (9,1))
        
        if (latest_time_step[0:8]!=column).any():
            column = np.append(column,time) 
            column = np.expand_dims(column, axis = 1)
            snapshot = np.delete(snapshot, 0, axis = 1)
            snapshot = np.append(snapshot, column, axis = 1)
            snapshot = np.expand_dims(snapshot, axis = 2)
            
            self.lob = np.append(self.lob, np.atleast_3d(snapshot), axis=2)

    
    
    def save_lob_data(self):
        np.save("Data/lob_data.npy", self.lob)
        #print(self.lob)
    
        