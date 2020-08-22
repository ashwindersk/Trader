import matplotlib.pyplot as plt
import numpy as np
import csv


balance = []
reward = []
num_trades = []
profitable_trades = []
bad_trades = []
with open('rewards-delta.csv', 'r') as datafile:
    reader = csv.reader(datafile, delimiter=",")
    i = 0 
    for row in reader:
        print(row[0], row[1], row[2], row[3])
        if i== 100:
            break
        reward.append(row[0])
        balance.append(row[1]) 
        num_trades.append(row[2])
        print(i,row[3])
        
        profitable_trades.append(row[3])
        bad_trades.append(int(row[2])-int(row[3]))
        i+=1
        


x = range(100)

plt.xlabel('Episode')
plt.ylabel('Number of Trades')
#plt.plot(x, reward, color='olive', linewidth=2, label ='Reward' )
#plt.plot(x, balance, color='blue', linewidth=2, label ='Balance')
plt.plot(x, profitable_trades, color='green', linewidth=2, label='Profitable Trades')
plt.plot(x, bad_trades, color='blue', linewidth=2, label='Negative Trades')
plt.legend()
plt.show()
