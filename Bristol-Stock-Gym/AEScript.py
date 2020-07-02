import os
import time
import argparse

parser = argparse.ArgumentParser(description = "LOB autoencoder")
parser.add_argument("--epochs", type = int, default = 50)
parser.add_argument("--batch-size", type = int, default = 256)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--data-file", type = str, default = "Data/lob_data.npy")
parser.add_argument("--outfile", type = str, default = "autoencoder.pth", required = True)
args = parser.parse_args()


import torch

from torch import nn 
torch.set_default_tensor_type('torch.DoubleTensor')
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as Functional
from torch.autograd import Variable

from sklearn import preprocessing



from AE import Autoencoder 
import numpy as np



input_dims = 9*5
model = Autoencoder(input_dims = 9*5, l1_size = 32, l2_size = 16, l3_size = 8)#.cuda()

epochs = args.epochs

dataset = np.load(args.data_file)
row,cols,num_images = dataset.shape
lobs = []
for i in range(num_images):
    lob = dataset[:,:,i]
    lob = lob.flatten()
    lobs.append(lob)


#lobs = preprocessing.normalize(lobs)
len_data = len(lobs)
train_len = int(0.7*len_data)

# mean = 0.
# std = 0.
# nb_samples = 0.
# for data in lobs:
#     mean += np.mean(data)
#     std += np.std(data)
#     nb_samples += 1

# mean /= nb_samples
# std /= nb_samples


minmax_scale = preprocessing.MinMaxScaler().fit(lobs)
df_minmax = minmax_scale.transform(lobs)
for index, data in enumerate(lobs):
    print(data)
    print(df_minmax[index].shape)


time.sleep(10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-5)

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainloader = DataLoader(lobs, batch_size = args.batch_size, shuffle = True)
#testloader  = DataLoader(lobs[train_len: len_data-1], batch_size = args.batch_size, shuffle = True)






for epoch in range(epochs):
    running_loss = 0.0
    for data in trainloader:    
        lob = Variable(data)#.cuda()
        
        #===============forward==================
        
        output = model(lob)    
        loss = criterion(output, lob)
        
        
        #===============backward=================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #===============log======================
        running_loss += loss.item()
    loss = running_loss / len(trainloader)
    print('epoch [{}/{}], Train loss:{:.4f}'
        .format(epoch + 1, epochs, loss))
 
 
    
           
torch.save(model.state_dict(), f'Models/{args.outfile}')


