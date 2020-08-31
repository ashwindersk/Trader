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
with open('rewards-DDPG-Sparsity-Synthetic.csv', 'r') as datafile:
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
        if int(row[4]) != 0 and float(row[1]) < 0:
            print(row[0], row[1], row[2], row[3], row[4], row[5])
            average = -(-int(float(row[1])) + int(float(row[3]))*int(float(row[5])) + 500)/int(float(row[4])) 
        elif int(row[4]) != 0 and float(row[1]) > 0:
            print(row[4])
            average = (int(float(row[1]) + int(float(row[3])) * int(float(row[5]))-500)/int(float(row[4])))
        else:
            average = 0 
        average_loss.append(average)
        i +=1
        print(i)
        
print(average_balance, j)

x = range(116)

plt.xlabel('Episode')
#plt.ylabel('Number of Trades')
plt.ylabel('Average GBP')
#plt.ylabel('Reward')

#plt.plot(x, reward, color='olive', linewidth=2 )
#plt.plot(x, balance, color='blue', linewidth=2)
#plt.plot(x, profitable_trades, color='green', linewidth=2, label='Profitable Trades')
#plt.plot(x, bad_trades, color='blue', linewidth=2, label='Negative Trades')
plt.plot(x, average_profit, color = 'blue', linewidth = 2, label = 'Average Profit')
plt.plot(x, average_loss, color = 'red', linewidth = 2, label = 'Average Loss')
plt.legend()
plt.show()
