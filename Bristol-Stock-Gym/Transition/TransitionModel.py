#!/usr/bin/env python
# coding: utf-8

# In[16]:


from RNN import MDNRNN
import torch
import numpy as np 
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################
#  DATALOADING #
################


# In[3]:



def sliding_windows(data,seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[(i+1):(i+1)+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


# In[22]:


bsz = 64

data = np.load('latent.npy')

#Reshaping data and seperating training and test set
sc = MinMaxScaler()
data = sc.fit_transform(data.squeeze(axis = 0))

seq_length = 4
x, y = sliding_windows(data, seq_length)



train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])).float())
trainY = Variable(torch.Tensor(np.array(y[0:train_size])).float())


testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])).float())
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])).float())


trainingset =[]
for i in range(len(trainY)):
    trainingset.append((trainX[i,:,:], trainY[i]))
    


trainloader = torch.utils.data.DataLoader(trainingset, batch_size=bsz, shuffle=False)




# In[23]:


num_epochs = 2000
learning_rate = 1e-7

input_size = 17
hidden_size = 256
num_layers = 1
seq_length = 4
num_classes = 1
gaussians = 5

mdnrnn = MDNRNN(z_size = input_size, n_hidden = hidden_size, n_gaussians = gaussians, n_layers = num_layers).to(device)


def detach(states):
    return [state.detach() for state in states]


# In[ ]:


optimizer = torch.optim.Adam(mdnrnn.parameters(), lr = learning_rate)

epochs = 500
random = torch.zeros(200,4,17)

for epoch in range(epochs):
    # Set initial hidden and cell states
    hidden = mdnrnn.init_hidden(bsz)

    # Get mini-batch inputs and targets
    total_loss = 0
    j = 0
    for i, (inputs, targets) in enumerate(trainloader):
        # Forward pass
        if inputs.size(0) != bsz:
            continue
        hidden = detach(hidden)
        (pi, mu, sigma), hidden = mdnrnn(inputs.to(device), hidden)

        loss = mdnrnn.criterion(targets.to(device), pi, mu, sigma)
        # Backward and optimize
        mdnrnn.zero_grad()
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(mdnrnn.parameters(), 0.5)
        optimizer.step()
        j += 1
    total_loss /=j
    if epoch % 2 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs, total_loss))
        torch.save(mdnrnn.state_dict(), '../Models/transition-regression')

# In[ ]:


import matplotlib.pyplot as plt

mdnrnn.eval()
size = train_size+test_size
train_predict = mdnrnn(dataX[size-1000:size])
data_predict = train_predict.data.numpy()[0:1000]
dataY_truth = dataY.data.numpy()[size-1000:size]
data_predict = sc_y.inverse_transform(data_predict)
dataY_truth = sc_y.inverse_transform(dataY_truth)
print(data_predict[0:5])
print("---------------------------------------")
print(data_truth[0:5])


# In[ ]:


torch.save(mdnrnn.state_dict(), '../Models/transition-regression')


# In[ ]:


import pickle
with open('sc_transition', 'wb') as f:
    pickle.dump(sc_y, f)
    


# In[ ]:




