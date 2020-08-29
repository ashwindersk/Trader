import matplotlib.pyplot as plt
import numpy as np
import csv


balance = []
reward = []
num_trades = []
profitable_trades = []
bad_trades = []
average_profit = []
average_loss = []
average_balance = 0 
with open('rewards-DDPG-synthetic.csv', 'r') as datafile:
    reader = csv.reader(datafile, delimiter=",") 
    i = 0
    j = 0 
    for row in reader:
        
        reward.append(row[0])
        balance.append(row[1])
        num_trades.append(row[2])
        profitable_trades.append(row[3])
        bad_trades.append(row[4])
        average_profit.append(row[5])
        average_loss.append(row[6])
        i +=1
        print(i)
        
print(average_balance, j)

x = range(104)

plt.xlabel('Episode')
plt.ylabel('Number of Trades')
#plt.plot(x, reward, color='olive', linewidth=2, label ='Reward' )
#plt.plot(x, balance, color='blue', linewidth=2)
plt.plot(x, profitable_trades, color='green', linewidth=2, label='Profitable Trades')
plt.plot(x, bad_trades, color='blue', linewidth=2, label='Negative Trades')
#plt.plot(x, average_profit, color = 'blue', linewidth = 2, label = 'Average Profit')
#plt.plot(x, average_loss, color = 'red', linewidth = 2, label = 'Average Loss')
plt.legend()
plt.show()
