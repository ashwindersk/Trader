import random
from Order import OType, Order
from Trader import Trader

class DeeplyReinforced(Trader):
    def __init__(self, trader_type, trader_id, min_price=1, max_price=1000, balance=500):
        self.balance = balance
        self.position = 0
        self.prev_order_price = None
        self.orders = []
        super().__init__(trader_type, trader_id, min_price=min_price, max_price=max_price)
    
    def assign_order(self, order):
        pass
    
    
    def add_order(self,order):
        if order is not None:
            
            if self.order is not None :
                response = 'LOB_Cancel'
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
        def calculate_benefit(order, trade_price):        
            self.benefit = 0
            benefit = 0
            if self.position == 0:
                if order.otype == OType.ASK:
                    print("SOLD", order)
                    self.balance += trade_price
                    self.position = 2
                    
            
                if order.otype == OType.BID:
                    print("BOUGHT", order)
                    self.balance -= trade_price
                    self.position = 1
                    
            elif self.position == 1:
                print("SOLD", order)
                self.balance +=trade_price
                benefit += trade_price - self.prev_order_price
                self.position = 0
            elif self.position == 2:
                print("BOUGHT", order)
                self.balance -=trade_price
                benefit += self.prev_order_price - trade_price 
                self.position = 0
            
            self.prev_order_price = trade_price
            self.lastquote = None
            return benefit

        if transaction_record['type'] == 'Trade':
            benefit = calculate_benefit(self.lastquote, transaction_record['price'])
            self.benefit = benefit
            self.lastquote = None
            
            
            