import random
from Order import OType, Order
from Trader import Trader
from enum import Enum
import time as ti

class Position(Enum):
    BOUGHT  = 1
    SOLD = 2
    NONE = 0

class DeeplyReinforced(Trader):
    def __init__(self, trader_type, trader_id, min_price=1, max_price=1000, balance=500):
        self.balance = balance
        self.position = Position.NONE
        self.prev_order_price = None
        self.orders = []
        self.benefit = 0
        super().__init__(trader_type, trader_id, min_price=min_price, max_price=max_price)
    
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
            if self.order is None:
                self.benefit = - 1e-4
            if self.order:
                self.benefit = 5
            if not self.order:
                return None

            self.lastquote = self.order
            return self.order
    
        
    def notify_transaction(self, transaction_record):
        def calculate_benefit(order, trade_price):  
            print("Calculating benefit", order, self.tid)   
            ti.sleep(10)   
            self.benefit = 0
            benefit = 0
            if self.position == Position.NONE:
                if order.otype == OType.ASK:
                    print("SOLD", order)
                    self.balance += trade_price
                    self.position = Position.SOLD
                    
            
                if order.otype == OType.BID:
                    print("BOUGHT", order)
                    self.balance -= trade_price
                    self.position = Position.BOUGHT
                    
            elif self.position == Position.BOUGHT:
                print("SOLD", order)
                self.balance +=trade_price
                benefit += trade_price - self.prev_order_price
                self.position = Position.NONE
            elif self.position == Position.SOLD:
                print("BOUGHT", order)
                self.balance -=trade_price
                benefit += self.prev_order_price - trade_price 
                self.position = Position.NONE
            
            self.prev_order_price = trade_price
            self.lastquote = None
            return benefit

        if transaction_record['type'] == 'Trade':
            benefit = calculate_benefit(self.order, transaction_record['price'])
            self.benefit = benefit
            self.order = None
            self.lastquote = self.order
            if self.position == Position.NONE:
                self.prev_order_price = None
        
        
    def get_benefit(self):
        return self.benefit
            