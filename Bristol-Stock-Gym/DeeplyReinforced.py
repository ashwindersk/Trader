import random
from Order import OType, Order
from Trader import Trader
from enum import Enum
import time as ti

class Position(Enum):
    BOUGHT  = 1
    SOLD = -1
    NONE = 0

class DeeplyReinforced(Trader):
    def __init__(self, trader_type, trader_id, min_price=1, max_price=1000, balance=500):
        self.position = Position.NONE
        self.prev_trade_price = None
        self.orders = []
        self.reward = 0
        self.num_trades = 0
        super().__init__(trader_type, trader_id, min_price=min_price, max_price=max_price)
        self.balance = balance
        self.lob_range = None
        self.prev_order = None
    
    def assign_order(self, order):
        pass
    
    
    def add_order(self,order):
        if order is not None:
            if self.order is not None :
                response = 'LOB_Cancel'
                self.lastquote = self.order
            else:
                response = 'Proceed'
          
            
                
                        
            self.order = order
            return response 
            
                
            
        return None 
    
    def action(self,player_action, time):

            if not self.order:
                return None
            self.lastquote = self.order
            return self.order
    
        
    def notify_transaction(self, transaction_record):
        def calculate_reward(order, trade_price):  
            
            reward = 0
            profit = 0
            if self.position == Position.NONE:
                if order.otype == OType.ASK:
                    print("SOLD", trade_price)
                    self.num_trades += 1
                    self.balance += trade_price
                    self.position = Position.SOLD
                    self.prev_trade_price = trade_price
                    self.prev_order = order
                    reward += trade_price
                    
            
                if order.otype == OType.BID:
                    print("BOUGHT", trade_price)
                    self.num_trades += 1
                    self.balance -= trade_price
                    self.position = Position.BOUGHT
                    self.prev_trade_price = trade_price
                    self.prev_order = order
                    reward -= trade_price
                    
                    
            elif self.position == Position.BOUGHT:
                self.balance +=trade_price
                self.num_trades +=1
                profit = (trade_price - self.prev_trade_price)
                reward += trade_price
                print(f"Profit: {profit} -> {trade_price} - {self.prev_trade_price} ")
                self.position = Position.NONE
                self.prev_trade_price = None
                self.prev_order = order
            elif self.position == Position.SOLD:
                self.balance -=trade_price 
                reward -= trade_price
                profit = (self.prev_trade_price - trade_price)
                self.num_trades += 1
                print(f"Profit: {profit} -> {self.prev_trade_price} - {trade_price}")
                self.position = Position.NONE
                self.prev_order_price = None
                
                
                
            self.lastquote = None
            
            
                
            
            return reward

        if transaction_record['type'] == 'Trade':
            reward = calculate_reward(self.order, transaction_record['price'])
            self.reward += reward
            self.order = None
            self.lastquote = self.order
            if self.position == Position.NONE:
                self.prev_order_price = None
        
    #Total reward for one step in the environment  
    #reset the reward   
    def get_reward(self):
        
        reward = self.reward
        self.reward = 0
        return reward
            
    def get_balance(self):
        return self.balance
    
    def update(self,bids, asks, time):
        self.time = time
        worst_ask,_ = bids.get_worst()
            
        worst_bid,_ = asks.get_worst()

        max_price = int(self.exchange_rules['maxprice'])
        min_price = int(self.exchange_rules['minprice'])
        
        if worst_ask is None and worst_bid is not None:
            self.lob_range = range(int(worst_bid), max_price)
        elif worst_ask is not None and worst_bid is None:
            self.lob_range = range(min_price, int(worst_ask))
        elif worst_ask is None and worst_bid is None:
            self.lob_range = None
        else:
            self.lob_range = range(int(worst_bid), int(worst_ask))
        
