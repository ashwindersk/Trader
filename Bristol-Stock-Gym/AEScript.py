import os
import time

import torch

from torch import nn 
torch.set_default_tensor_type('torch.DoubleTensor')
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as Functional
from torch.autograd import Variable

from sklearn import preprocessing



from AE import Autoencoder 
import numpy as np

import argparse

parser = argparse.ArgumentParser(description = "LOB autoencoder")
parser.add_argument("--epochs", type = int, default = 50)
parser.add_argument("--batch-size", type = int, default = 256)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--data-file", type = str, default = "Data/lob_data.npy")
args = parser.parse_args()

input_dims = 8*5
model = Autoencoder(input_dims = 8*5, l1_size = 32, l2_size = 16, l3_size = 8).cuda()

epochs = args.epochs

dataset = np.load(args.data_file)
row,cols,num_images = dataset.shape
lobs = []
for i in range(num_images):
    lob = dataset[:,:,i]
    lob = lob.flatten()
    lobs.append(lob)


lobs = preprocessing.normalize(x)

# mean = 0.
# std = 0.
# nb_samples = 0.
# for data in lobs:
#     mean += np.mean(data)
#     std += np.std(data)
#     nb_samples += 1

# mean /= nb_samples
# std /= nb_samples

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-5)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
dataloader = DataLoader(lobs, batch_size = args.batch_size, shuffle = True)







for epoch in range(epochs):
    for data in dataloader:    
        lob = Variable(data).cuda()
        
        #===============forward==================
        
        output = model(lob)    
        loss = criterion(output, lob)
        
        
        #===============backward=================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #===============log======================
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, epochs, loss.item()))
        
torch.save(model.state_dict(), 'Models/autoencoder.pth')


