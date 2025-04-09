from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import statistics
import numpy as np

kelp_prices = []
squid_prices = []

class Trader:
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "SQUID_INK":
                cap=0
                vol=0
                vwap = 1972.46561081
                limit=50
                window=100 #window for calculatins sma and sd
                if(product in state.market_trades.keys() and len(state.market_trades[product])!=0):
                    kelp_trades = state.market_trades[product]
                    for t in kelp_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    kelp_prices.append(vwap) #we keep track of the historic trade vwap to calculate sma and sd
                #make orders
                if len(kelp_prices) > window:
                    kelp_prices.pop(0)
                    stdev = statistics.stdev(kelp_prices)
                    sma = statistics.mean(kelp_prices)
                    d = stdev 
                    best_bid = max(list(order_depth.buy_orders.keys()))
                    best_ask = min(list(order_depth.sell_orders.keys()))
                    spread = best_ask - best_bid
                    p=0
                    if product in state.position.keys():
                        p = state.position[product]
                    if vwap > sma+d:
                        #-p-limit for sell
                        #limit-p for buy
                        orders.append(Order(product,best_bid,-p-limit))
                        print("sold squid_ink for: "+str(best_bid))
                    if vwap < sma-d:
                        orders.append(Order(product,best_ask,limit-p))
                        print("bought squid_ink for: "+str(best_ask))
                
            
            if product == "KELP":
                print()

            if product == "RAINFOREST_RESIN":
                print()
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
