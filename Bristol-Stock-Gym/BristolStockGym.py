import random
import time as ti
import signal
import sys

from Exchange import Exchange
from Order import OType, Order
import math


import numpy as np

# Import Trader strategies:
from Trader import TType, Trader
from Giveaway import Giveaway
from ZIU import ZIU
from ZIC import ZIC
from ZIP import ZIP
from DeeplyReinforced import DeeplyReinforced, Position

import torch
from torch.autograd import Variable

from actor_critic import Agent
from AE import LOB_trainer, Autoencoder

import matplotlib.pyplot as plt


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
                            
        benefit = 0

        ## Shuffle traders in a random order
        trader_keys =  list(self.traders.keys())
        random.shuffle(trader_keys)

        ## Update the traders with the latest public lob
        for trader_key in trader_keys:
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
                # if output['party1'] == 'PLAYER' or output['party2'] == 'PLAYER': # If the player trader was involved in the trade, this step's reward becomes the balance of the trade
                #     player = self.traders['PLAYER']
                #     benefit = player.benefit

        # Assign new orders to the traders who completed the previous ones if the exchange(experiment) rules say so
        # Here we could try preserving which trader bids and which
        # trader asks (chosen), or we could give them random orders (thus upsetting the balance)
        # if self.replenish_orders == True:
        #     for trader_key in trader_keys:
        #         trader = self.traders[trader_key]
        #         if trader.order == None:
        #             new_order = self._generate_order(trader.tid, trader.otype, self.time)
        #             trader.assign_order(new_order)

        # Increment timestep
        benefit = self.traders['PLAYER'].get_benefit()
                
        if self.time >= self.maxtime:
            self.done = True
        self.time += self.time_step

        observation = self._get_observation()
        reward = benefit
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

        return observation, reward, done, info

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


def save_lob(signum, frame):
    lob_trainer.save_lob_data("unnormalized_data")
    sys.exit()
    
signal.signal(signal.SIGINT,save_lob)

if __name__ == "__main__":
    
    end_time = 1000.0
  
  
  
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
    
        low = 100
        high = 150
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
    
    traders_spec, order_sched = get_traders_schedule()

    #----------------------------------------------------------------
    
    
    trader = Agent(actor_lr=1e-3, critic_lr=1e-3, input_dims = [2], gamma = 0.99,
                   n_actions = 3, l1_size = 32, l2_size= 32)
    
    lob_trainer = LOB_trainer() 
    
    #Autoencoder = Autoencoder(input_dims = 9*5, l1_size = 32, l2_size = 16, l3_size = 8)
    #Autoencoder.load_state_dict(torch.load('Models/autoencoder.pth', map_location=torch.device('cpu')))
    
    
    def get_lob(observation):
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

        lob_trainer.get_lob_snapshot(column, time)
        lob = lob_trainer.lob

        


                
        
        

    def get_observation(observation):
            index_error = False
            try:
                best_ask= observation['lob']['asks'][0][0]
            except IndexError:
                best_ask = 0
                index_error = True
            try:
                best_bid= observation['lob']['bids'][0][0]
            except IndexError:
                best_bid = 0
                index_error = True

            
            
            midprice = int((best_ask + best_bid)/2)
            tape = observation['lob']['tape']
            transaction = None
            for event in tape:
                if event['type'] == 'Trade':
                    transaction = event['price']
                    
            latest_transaction = transaction
            if latest_transaction is None:
                latest_transaction = 0
            input = np.array([midprice,latest_transaction], dtype = np.double)
            
            return input, index_error
        
      #------------------------------------------------------------------  
    
    #Autoencoder to reduce lob observation down to a reasonable dimensionality 
    
    # Modelling the state space as S_t = {ae(ob_t−T :t), u_t, po_t}
    # Where ae(ob_t-T : t) is a latent representation of the LOB 
    #       u_t is a vector of prior trades by the agent
    #       po_t is the traders position at time t
    # Unsure how we will use these as input considering they arent of fixed length
    
    #Reward function of R(t) = delta (midprice)_s_t, s_t+1 x po_t
    
    
    
    
    def trader_strategy(observation):
        
        
        #Based off've the LOB, get the current observation
        
        input, no_midprice = get_observation(observation)
        
        
        #Model chooses an action based on oberservation
        action = trader.choose_action(input)

        
        current_position = observation['trader'].position
        
        
        if no_midprice == True:
            return None, input

        if action == 0:
            return None,input
        
        if current_position == Position.NONE:
            
            if action == 1:
                order_type = OType.BID
            elif action == 2:
                order_type = OType.ASK
            else:
                return None, input
        elif current_position == Position.BOUGHT:
            if action == 1 or action == 0:
                return None,input
            elif action == 2:
                order_type = OType.ASK
        elif current_position == Position.SOLD: 
            if action == 1:
                order_type = OType.BID
            elif action == 2 or action == 0:
                return None,input
        tid = observation['trader'].tid
        time =  observation['lob']['time']
        
        price = input[0]
        order = Order(tid, order_type, price, 1, time)
        
        
        
        return order, input
    
    
    
    for i in range(1000):
        time_step = 1.0/60.0
        environment = Environment(traders_spec, order_sched,time_step = time_step, max_time = end_time, min_price = 1, max_price = end_time, replenish_orders = True)
        
        totalreward = 0
        done = False
        observation = environment.reset()
        
    
        j = 0
        while not done:
            lob =  get_lob(observation)
            #lob_trainer.learn()
            action, state = trader_strategy(observation)
            action = None
            observation_, reward, done, info = environment.step(action)
            new_state, empty_flag = get_observation(observation_)
            totalreward += reward
            if(j % 1000 == 0):
                print(f"Reward after {j}'th step in {i}'th Episode': {reward}, Total Reward: {totalreward}")
            #trader.learn(state, reward, new_state, done)
            observation = observation_
            
            j+=1
            
        if done:
            print(f"End of trading session{i} with Total Reward: {totalreward} ")
    lob_trainer.save_lob_data()

   