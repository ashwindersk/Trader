import random
import time as ti
import signal
import sys
import pickle
import math
import os

from Exchange import Exchange
from Order import OType, Order


import numpy as np
from sklearn import preprocessing
import torch

# Import Trader strategies:
from Trader import TType, Trader
from Giveaway import Giveaway
from ZIU import ZIU
from ZIC import ZIC
from ZIP import ZIP
from DeeplyReinforced import DeeplyReinforced, Position
from AE import LOB_trainer, Autoencoder

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description = "LOB autoencoder")
parser.add_argument("--suffix", type = str, default = "")
args = parser.parse_args()

from actor_critic import Agent


class Environment:

    def __init__(self, traders_spec, order_schedule, time_step = 1,max_time = 1000, min_price = 1, max_price = 1000, replenish_orders = True):
        self.maxtime = max_time
        self.minprice = min_price
        self.maxprice = max_price
        self.replenish_orders = replenish_orders
        self.init = False
        self.traders_spec = traders_spec
        self.order_schedule = order_schedule
        self.time_step = time_step

    def _get_observation(self):
        observation = {
            'lob': self.exchange.get_public_lob(self.time),
            'trader': self.traders['PLAYER']
        }
        return observation

    def reset(self):
        self.exchange = Exchange(self.minprice, self.maxprice)
        self.time = 1
        self.done = False
        self.trader_stats, self.traders = self._populate_traders(self.traders_spec)
        self.init = True
        self.pending_cust_orders = []
        return self._get_observation()

    #
    def step(self, player_action):
        if not self.init:
            raise RuntimeError('Error: step() function in environment called before reset()')
        


        trade = None

        self.pending_cust_orders, kills = self._customer_orders(self.time, self.traders, self.trader_stats,
                                                 self.order_schedule, self.pending_cust_orders, player_action)

        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0 :
                for kill in kills :
                        if self.traders[kill].lastquote != None :
                            
                            self.exchange.del_order(self.traders[kill].lastquote,self.time )
                            
        

        ## Shuffle traders in a random order
        trader_keys =  list(self.traders.keys())
        random.shuffle(trader_keys)

        ## Update the traders with the latest public lob
        for trader_key in trader_keys:
            if trader_key == 'PLAYER':
                self.traders[trader_key].update(self.exchange.bids, self.exchange.asks)    
            else:
                self.traders[trader_key].update(self.exchange.get_public_lob(self.time))
            
        ## Process trader actions

        # In their random order, traders take an action
        for trader_key in trader_keys:
            trader = self.traders[trader_key]
            order = trader.action(player_action, self.time) 
            output = None
            if order != None: # If an order is placed, process it
                output = self.exchange.process_order(order, self.time)
                
            if output != None: # If a trade occurred due to the order being placed, notify the parties involved
                trader1 = self.traders[output['party1']]
                trader2 = self.traders[output['party2']]
                trader1.notify_transaction(output)
                trader2.notify_transaction(output)
            
            
                


                
        if self.time >= self.maxtime:
            self.done = True
        self.time += self.time_step

        observation = self._get_observation()
        
        #Getting all relevant data from trader
        
            
        
        
        done = self.done
        info = ""
        if self.done: # Return the balance of each trader
            info = "BALANCES: \n"
            trader_keys =  list(self.traders.keys())
            for trader_key in trader_keys:
                trader = self.traders[trader_key]
                balance = trader.balance
                string = trader_key + ":" + str(trader.balance) + "\n"
                info = info + string

        position = Position.NONE.value
        if self.traders['PLAYER'].prev_order_price is not None:
            position = self.traders['PLAYER'].position.value * self.traders['PLAYER'].prev_order_price/1000
        num_trades =self.traders['PLAYER'].num_trades    
        reward = self.traders['PLAYER'].get_reward()
        balance = self.traders['PLAYER'].get_balance()
        return observation, reward, done, info, balance, position, num_trades

    def _populate_traders(self, traders_spec):

        # Create and return a trader of the specified type
        def create_trader(trader_type, trader_id, min_price, max_price):
            if trader_type == TType.GVWY:
                trader = Giveaway(trader_type, trader_id, min_price, max_price)
            elif trader_type == TType.ZIU:
                trader = ZIU(trader_type, trader_id, min_price, max_price)
            elif trader_type == TType.ZIC:
                trader = ZIC(trader_type, trader_id, min_price, max_price)
            elif trader_type == TType.ZIP:
                trader = ZIP(trader_type, trader_id, min_price, max_price)
            elif trader_type ==  TType.PLAYER:
                trader = DeeplyReinforced(trader_type, trader_id, min_price, max_price, balance = 500)
            else:
                trader = Trader(trader_type, trader_id, min_price, max_price)
            return trader

        # Generates a trader and assigns an order to it
        def generate_trader(self, trader_id, trader_type, min_price, max_price):
            #trader = Trader(trader_type, trader_id, min_price, max_price) # TODO: Here for reference. Remove.
            trader = create_trader(trader_type, trader_id, min_price, max_price)
            #new_order = self._generate_order(trader.tid, order_type, 0)
            #trader.assign_order(new_order)
            return trader

        traders = {}

        tid = 'PLAYER'
        
        traders[tid] = generate_trader(self, tid, TType.PLAYER, self.minprice, self.maxprice)

        def shuffle_traders(ttype_char, n, traders):
                for swap in range(n):
                        t1 = (n - 1) - swap
                        t2 = random.randint(0, t1)
                        t1name = '%c%02d' % (ttype_char, t1)
                        t2name = '%c%02d' % (ttype_char, t2)
                        traders[t1name].tid = t2name
                        traders[t2name].tid = t1name
                        temp = traders[t1name]
                        traders[t1name] = traders[t2name]
                        traders[t2name] = temp


        n_buyers = 0
        for bs in traders_spec['buyers']:
                ttype = bs[0]
                for b in range(bs[1]):
                        tname = 'B%02d' % n_buyers  # buyer i.d. string
                        if ttype ==TType.PLAYER:
                            pass
                        else:
                            traders[tname] = generate_trader(self,tname, ttype, min_price=self.minprice, max_price=self.maxprice)
                        n_buyers = n_buyers + 1

        if n_buyers < 1:
                raise RuntimeError('FATAL: no buyers specified\n')

        shuffle_traders('B', n_buyers-1, traders)


        n_sellers = 0
        for ss in traders_spec['sellers']:
                ttype = ss[0]
                for s in range(ss[1]):
                        tname = 'S%02d' % n_sellers  # buyer i.d. string
                        if ttype ==TType.PLAYER:
                            tname = 'PLAYER'
                        traders[tname] = generate_trader(self,tname, ttype, min_price=self.minprice, max_price=self.maxprice)
                        n_sellers = n_sellers + 1

        if n_sellers < 1:
                raise RuntimeError('FATAL: no sellers specified\n')

        shuffle_traders('S', n_sellers, traders)

        return {'n_buyers':n_buyers, 'n_sellers':n_sellers}, traders


    def _customer_orders(self, time, traders, trader_stats,os, pending, player_action):
        def sysmin_check(price):
            if price < self.minprice:
                    raise RuntimeWarning('WARNING: price < bse_sys_min -- clipped')
                    price = self.minprice
            return price


        def sysmax_check(price):
            if price > self.maxprice:
                    raise RuntimeWarning('WARNING: price > bse_sys_max -- clipped')
                    price = self.maxprice
            return price

        

        def getorderprice(i, sched, n, mode, issuetime):
                # does the first schedule range include optional dynamic offset function(s)?
                if len(sched[0]) > 2:
                        offsetfn = sched[0][2]
                        if callable(offsetfn):
                                # same offset for min and max
                                offset_min = offsetfn(issuetime)
                                offset_max = offset_min
                        else:
                                raise RuntimeError('FAIL: 3rd argument of sched in getorderprice() not callable')
                        if len(sched[0]) > 3:
                                # if second offset function is specfied, that applies only to the max value
                                offsetfn = sched[0][3]
                                if callable(offsetfn):
                                        # this function applies to max
                                        offset_max = offsetfn(issuetime)
                                else:
                                        raise RuntimeError('FAIL: 4th argument of sched in getorderprice() not callable')
                else:
                        offset_min = 0.0
                        offset_max = 0.0

                pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
                pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
                prange = pmax - pmin
                stepsize = prange / (n - 1)
                halfstep = round(stepsize / 2.0)

                if mode == 'fixed':
                        orderprice = pmin + int(i * stepsize) 
                elif mode == 'jittered':
                        orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
                elif mode == 'random':
                        if len(sched) > 1:
                                # more than one schedule: choose one equiprobably
                                s = random.randint(0, len(sched) - 1)
                                pmin = sysmin_check(min(sched[s][0], sched[s][1]))
                                pmax = sysmax_check(max(sched[s][0], sched[s][1]))
                        orderprice = random.randint(pmin, pmax)
                else:
                        raise RuntimeError('FAIL: Unknown mode in schedule')
                orderprice = sysmin_check(sysmax_check(orderprice))
                return orderprice



        def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
                interval = float(interval)
                if n_traders < 1:
                        raise RuntimeError('FAIL: n_traders < 1 in getissuetime()')
                elif n_traders == 1:
                        tstep = interval
                else:
                        tstep = interval / (n_traders - 1)
                arrtime = 0
                issuetimes = []
                for t in range(n_traders):
                        if mode == 'periodic':
                                arrtime = interval
                        elif mode == 'drip-fixed':
                                arrtime = t * tstep
                        elif mode == 'drip-jitter':
                                arrtime = t * tstep + tstep * random.random()
                        elif mode == 'drip-poisson':
                                # poisson requires a bit of extra work
                                interarrivaltime = random.expovariate(n_traders / interval)
                                arrtime += interarrivaltime
                        else:
                                raise RuntimeError('FAIL: unknown time-mode in getissuetimes()')
                        issuetimes.append(arrtime) 
                        
                # at this point, arrtime is the last arrival time
                if fittointerval and ((arrtime > interval) or (arrtime < interval)):
                        # generated sum of interarrival times longer than the interval
                        # squish them back so that last arrival falls at t=interval
                        for t in range(n_traders):
                                issuetimes[t] = interval * (issuetimes[t] / arrtime)
                # optionally randomly shuffle the times
                if shuffle:
                        for t in range(n_traders):
                                i = (n_traders - 1) - t
                                j = random.randint(0, i)
                                tmp = issuetimes[i]
                                issuetimes[i] = issuetimes[j]
                                issuetimes[j] = tmp
                return issuetimes
        

        def getschedmode(time, os):
                got_one = False
                for sched in os:
                        if (sched['from'] <= time) and (time < sched['to']) :
                                # within the timezone for this schedule
                                schedrange = sched['ranges']
                                mode = sched['stepmode']
                                got_one = True
                                exit  # jump out the loop -- so the first matching timezone has priority over any others
                if not got_one:
                        raise RuntimeError('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
                return (schedrange, mode)
        

        n_buyers = trader_stats['n_buyers']
        n_sellers = trader_stats['n_sellers']

        shuffle_times = True

        cancellations = []

        trader_keys =  list(self.traders.keys())
               
        for trader_key in trader_keys:
            trader = traders[trader_key]
            if trader.ttype == TType.PLAYER:
                response = trader.add_order(player_action)
                if response == 'LOB_Cancel':
                    cancellations.append(trader.tid)
                
        
        if len(pending) < 1:
                # list of pending (to-be-issued) customer orders is empty, so generate a new one
                new_pending = []

                # demand side (buyers)
                issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)
                
                ordertype = OType.BID
                (sched, mode) = getschedmode(time, os['dem'])             
                for t in range(n_buyers-1):
                        issuetime = time + issuetimes[t]
                        tname = 'B%02d' % t
                        orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
                        order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
                        new_pending.append(order)
                        
                # supply side (sellers)
                issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
                ordertype = OType.ASK
                (sched, mode) = getschedmode(time, os['sup'])
                for t in range(n_sellers):
                        issuetime = time + issuetimes[t]
                        tname = 'S%02d' % t
                        orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
                        order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
                        new_pending.append(order)
        else:
                # there are pending future orders: issue any whose timestamp is in the past
                new_pending = []
                for order in pending:
                        if order.time < time:
                                # this order should have been issued by now
                                # issue it to the trader
                                tname = order.tid                    
                                response = traders[tname].add_order(order)
                                
                                if response == 'LOB_Cancel' :
                                    cancellations.append(tname)
                
                                # and then don't add it to new_pending (i.e., delete it)
                        else:
                                # this order stays on the pending list
                                new_pending.append(order)
        return new_pending, cancellations




#------- All functionality to do with varying supply and demand schedule   ------------
def get_traders_schedule():
    def schedule_offsetfn(t):
        pi2 = math.pi * 2
        c = math.pi * 3000
        wavelength = t / c
        gradient = 2 * t / (c / pi2)
        amplitude = 2 * t / (c / pi2)
        offset = gradient + amplitude * math.sin(wavelength * t)
        return int(round(offset, 0))
    low = random.choice(range(100,600))
    high = random.choice(range(300,800))
    intervals = 10
    supply_schedule = []
    sigma = 5
    for i in range(0,intervals):
        low = random.gauss(low,sigma)
        high = random.gauss(high,sigma)
        found = False
        while not found:
            if (high > low) and (low > 0):
                    found = True
            
            else:
                    low = random.gauss(low,sigma)
                    high = random.gauss(high,sigma)
        
        
        low = int(low)
        high = int(high)
        
        range_i = (low, high, schedule_offsetfn, schedule_offsetfn)
        supply_schedule.append({'from': i*end_time/intervals , 'to': (i+1)*end_time/intervals, 'ranges':[range_i], 'stepmode':'fixed'})                
        
        

    demand_schedule = supply_schedule
        
    order_sched = {'sup':supply_schedule, 'dem':demand_schedule,
                       'interval':30, 'timemode':'drip-poisson'}

    

    buyers_spec = [(TType.GVWY,10),(TType.ZIC,9),(TType.ZIP,10), (TType.PLAYER, 1)]
    

    sellers_spec = [(TType.GVWY,10),(TType.ZIC,10),(TType.ZIP,10)]
    
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
    
    return traders_spec, order_sched
 
def get_state(observation, position):
        
         #Get the latest 5 changes to the lob - autoencoded 
        def get_lob():
            bids = observation['lob']['bids']
            asks = observation['lob']['asks']
            column = np.zeros(8)

            len_bids = len(bids)
            len_asks = len(asks)
            for i in range(min(2,len_asks)):
                column[4*i] = asks[i][0]
                column[4*i +1 ] = asks[i][1]

            for i in range(min(2,len_bids)):
                column[4*i + 2] = bids[i][0]
                column[4*i + 3] = bids[i][1] 

            time = observation['lob']['time']

            column = column.reshape((8,))

            snapshot = lob_trainer.get_lob_snapshot(column, time)
            snapshot = snapshot.flatten()
            snapshot = snapshot.reshape(1,-1)

            snapshot = scalar.transform(snapshot)
            snapshot = torch.FloatTensor(snapshot)
            snapshot = Autoencoder.encoder(snapshot)
            snapshot = snapshot.cpu().detach().numpy()
            snapshot = snapshot.reshape(8)
            
            return snapshot
        
        def get_trades():
            tape = observation['lob']['tape']
            trades = np.zeros([1,8])
            i = 0
            for event in reversed(tape):
                if event['type'] == 'event':
                    try:
                        trades[i] = event['price']
                    except IndexError:
                        pass
                    i +=1
                    
            min_max = preprocessing.MinMaxScaler()
            trades = min_max.fit_transform(trades)
            trades = trades.reshape(8)
            return trades
        
        lob    = np.array(get_lob())
        
        
        trades = np.array(get_trades())
        input = np.concatenate((lob, trades, [position]))
        
        return input 

def trader_strategy(state):
        
    
        #Model chooses an action based on oberservation
        action, price = agent.choose_action(state)
        current_position = observation['trader'].position
        price = round(price)
        if action == 0:
            return None, action
        
        if current_position == Position.NONE:
            
            if action == 1:
                order_type = OType.BID
            elif action == -1:
                order_type = OType.ASK
            else:
                return None, action
        elif current_position == Position.BOUGHT:
            if action == 1 or action == 0:
                return None, action
            elif action == -1:
                order_type = OType.ASK
        elif current_position == Position.SOLD: 
            if action == 1:
                order_type = OType.BID
            elif action == -1 or action == 0:
                return None, action
            
        tid = observation['trader'].tid
        time =  observation['lob']['time']
        if price < 0:
            price = 1
        if price > 1000:
            price = 1000
        order = Order(tid, order_type, price, 1, time)
        
        
        
        return order, action

if __name__ == "__main__":
    
    end_time = 1000.0    
    
    traders_spec, order_sched = get_traders_schedule()

    #------ Getting mixnmax scalar to transform lob input for AE -----
    filehandler = open('Objects/scalar', 'rb')
    scalar = pickle.load(filehandler)
    filehandler.close()
    #==================================================================
    
    
    #------ Autoencoder ----------------------------------------
    
    lob_trainer = LOB_trainer() 
    
    Autoencoder = Autoencoder(input_dims = 9*5, l1_size = 32, l2_size = 16, l3_size = 8)
    Autoencoder.load_state_dict(torch.load('Models/autoencoder.pth', map_location=torch.device('cpu')))
    
    
    agent = Agent(alpha = 2.5e-5, beta = 2.5e-4, input_dims =[17], tau = 0.001,
                  batch_size = 64, layer1_size = 400, layer2_size = 300, n_actions = 2)
    
    np.random.seed(0)
    
    
    #==================================================================    
    
    states = []
    for i in range(5):
        time_step = 1.0/60.0
        environment = Environment(traders_spec, order_sched,time_step = time_step, max_time = end_time, min_price = 1, max_price = end_time, replenish_orders = True)
        totalreward = 0
        done = False
        balance = 500 
        observation = environment.reset()
        position = Position.NONE.value
        num_trades = 0
        j  = 0
        while not done:
            state = get_state(observation, position)
                
            order, action = trader_strategy(state.flatten())
            latent = np.concatenate((state.flatten(), np.array([action]), np.array([order.price if order is not None else 0])))
            states.append(latent)
            print(len(states))
            #a sequence of the last N (Observation_, a) pairs are used to predict the next state 
            #state_prediction = RNN_model((state,a))
            
            #Instead of the reward being calculated here, a LSTM regression model (trained), calculates the reward beased on the midprice x position
            #reward = reward_model(state, state_prediction)
            
            if order is not None:
                print(action,order, balance, position, num_trades)
            observation_, reward, done, info, balance, position, num_trades = environment.step(order)
            #
            
            
            
            new_state  = get_state(observation_, position)
            #agent.remember(state,action,reward,new_state, int(done))
            #agent.learn(j)
            totalreward += reward
            observation = observation_
            j+=1
            
        
        #print(f"End of trading session{i} with Total Reward: {totalreward}, Total Balance: {balance}, number of trades: {num_trades} ")
            
        #with open(f'rewards-{args.suffix}.csv', 'a') as rewardfile:
        #    rewardfile.write(f"{i}: {totalreward}, {balance}, {num_trades}\n")
        
        
        #if i % 5 == 0:
        #   agent.save_models()
        
    states = np.stack([states])
    with open('latent.npy', 'wb') as f:
        np.save(f, states)  
                    
        
        
       
        
    
    
    

   
