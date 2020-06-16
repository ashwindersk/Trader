import random

from enum import Enum
from Order import OType, Order

# A trader can be multiple types, including a player agent
class TType(Enum):
    PLAYER = 'PLAYER'
    GVWY = 'GIVEAWAY'
    ZIU = 'ZIU' # After Gode & Sunder 1993
    ZIC = 'ZIC' # After Gode & Sunder 1993
    ZIP = 'ZIP' # After Cliff 1997

class Trader:

    def __init__(self, trader_type, trader_id, min_price = 1, max_price = 1000):
        # Trader attributes:
        self.ttype = trader_type
        self.tid = trader_id
        self.order = None
        self.otype = None # Type of the order last assigned to the trader. Used if we want to keep a trader exclusively bidding or asking
        self.balance = 0
        self.n_quotes = 0
        self.lastquote = None
        # Exchange rules: # TODO: maybe change it to storing a local copy of the exchange if necessary?
        self.exchange_rules = {
            'minprice' : min_price,
            'maxprice' : max_price
        }

    # Assigns a new order to the trader, replacing a previous one if there was one
    def add_order(self, order):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0 :
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
            self.order = order
            return response

    # Regress to a price and place an order in the exchange
    # player_action input is for the player action
    def action(self, player_action, time):
        # If the trader has no pending trade orders, do nothing 
        return self.order
        



    # Called whenever a transaction involving the trader has occurred
    def notify_transaction(self, transaction_record):
        def calculate_benefit(order, trade_price):
            if order.otype == OType.BID:
                benefit = order.price - trade_price
            elif order.otype == OType.ASK:
                benefit = trade_price - order.price
            else:
                raise RuntimeError('Error: Wrong order type stored in trader')
            return benefit

        if transaction_record['type'] == 'Trade':
            benefit = calculate_benefit(self.order, transaction_record['price'])
            self.balance += benefit
            self.order = None

    # Called to update the trader with the latest LOB information after each timestep
    def update(self, public_lob):
        None
