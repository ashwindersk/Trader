import random
from Order import OType, Order
from Trader import Trader

class DeeplyReinforced(Trader):
    def __init__(self, trader_type, trader_id, min_price=1, max_price=1000, balance=500):
        self.balance = balance
        self.position = 0
        self.prev_order_price = None
        super().__init__(trader_type, trader_id, min_price=min_price, max_price=max_price)
    
    def assign_order(self, order):
        pass
    
    
    
    def notify_transaction(self, transaction_record):
        def calculate_benefit(order, trade_price):        
            if order.otype == OType.BID:
                if self.position == 2:
                    self.position = 0
                    benefit = self.prev_order_price - trade_price
                    self.prev_order_price = None
                else:
                    self.position = 1
                    self.prev_order_price = trade_price
            elif order.otype == OType.ASK:
                if self.position == 1:
                    self.position = 0
                    benefit = trade_price - self.prev_order_price
                    self.prev_order_price = None
                else:
                    self.position = 2
                    self.prev_order_price = trade_price
            else:
                raise RuntimeError('Error: Wrong order type stored in trader')
            return benefit

        if transaction_record['type'] == 'Trade':
            benefit = calculate_benefit(self.order, transaction_record['price'])
            self.balance += benefit
            self.order = None