#!/usr/bin/env python
# coding: utf-8

# In[1]:


from RNN2 import LSTM
import torch
import numpy as np 
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


################
#  DATALOADING #
################


# In[3]:



def sliding_windows(data,seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[(i+seq_length):(i+1)+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


# In[4]:


bsz = 200

data = np.load('latent-action.npy')
print(data[:,23233,:])
#Reshaping data and seperating training and test set
sc = MinMaxScaler()
data = sc.fit_transform(data.squeeze(axis = 0))
print(data[23233,:])
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




# In[5]:



learning_rate = 1e-7
input_size = 18
hidden_size = 256
num_layers = 1
num_classes = 1
fc1_out = 128
output_size = 18

lstm = LSTM(output_size, input_size, hidden_size, num_layers, seq_length, fc1_out)


def detach(states):
    return [state.detach() for state in states]


# In[ ]:


lstm.train()
test_hist = []
criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(500):

        outputs = lstm(trainX).to(device)

        optimizer.zero_grad()
    
    # obtain the loss function
        loss = criterion(outputs, trainY)
    
        loss.backward()
    
        optimizer.step()
        if testX is not None:
            with torch.no_grad():
                y_test_pred = lstm(testX).to(device)
                test_loss = criterion(y_test_pred.float(), testY).to(device)
                test_hist.append( test_loss.item())

        print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_hist[epoch]))

        torch.save(lstm.state_dict(), '../Models/test-transition')



# In[ ]:


mdnrnn.load_state_dict(torch.load('../Models/transition-regression-test', map_location='cpu'))

mdnrnn.eval()
size = train_size+test_size
zero = np.random.randint(testX.size(0))

one = np.random.randint(testX.size(1))


x = testX[zero:zero+1, one:one+1, :]
y = testX[zero:zero+1, one+1:one+2, :]
print(x.shape)
hidden = mdnrnn.init_hidden(1)
(pi, mu, sigma), _ = mdnrnn(x, hidden)

y_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(5)]

new = [tens.detach().numpy().reshape(1,18) for tens in y_preds]


print(y.shape)
data_predict = new[2]
dataY_truth = y.flatten()
print(data_predict)
print("---------------------------------------")
print(dataY_truth)


# In[ ]:


x = torch.Tensor([0.9825, 0.4345, 0.9836, 0.2939, 0.3526, 0.7257, 0.4492, 0.1332,
          0.5326, 0.5326, 0.5326, 0.5349, 0.5337, 0.5326, 0.0012, 0.0000,
          0.4744, 0.5000])
x = x.reshape((1,1,18))
(pi,mu,sigma), _ = mdnrnn(x,hidden)
pi = pi.reshape((1,5,18))
y_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(5)]
y_preds = [tens.reshape(1,18) for tens in y_preds]
y_preds = torch.cat(y_preds, dim = 0)

y_pred = pi * y_preds
            

y_pred = torch.sum(y_pred, dim = 1)

            #print(y_pred)
y_pred = y_pred.detach().numpy()
y_pred = y_pred.reshape((1,18))
y_pred = sc.inverse_transform(y_pred)

print(y_pred)


# In[ ]:


torch.save(mdnrnn.state_dict(), '../Models/transition-regression-new2')


# In[ ]:


import pickle
with open('sc', 'wb') as f:
    pickle.dump(sc, f)
    


# In[ ]:




