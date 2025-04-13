from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import statistics
import numpy as np

kelp_prices = []
squid_prices = []
jams_prices=[]
djembes_prices = []
c_prices = []

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
                    best_bid_amt = order_depth.buy_orders[best_bid]
                    best_ask_amt = order_depth.sell_orders[best_ask]
                    spread = best_ask - best_bid
                    p=0
                    if product in state.position.keys():
                        p = state.position[product]
                    if vwap > sma+d:
                        #-p-limit for sell
                        #limit-p for buy
                        orders.append(Order(product,int(vwap+1),-7)) #sell
                    if vwap < sma-d:
                        orders.append(Order(product,int(vwap-1),7)) #buy
                        
            if product == "KELP":
                print()

            if product == "RAINFOREST_RESIN":
                if product in state.position.keys():
                    p=state.position[product]
                else:
                    p=0
                d = .5
                arb = False
                limit=50
                if len(order_depth.sell_orders) !=0:
                    for ask, q in list(order_depth.sell_orders.items()):
                        if ask < 10000:
                            orders.append(Order(product,int(ask),-q))
                            arb = True
                        elif not arb and ask > 10000 + d:
                            orders.append(Order(product,int(ask-d),-7))

                if len(order_depth.buy_orders) !=0:
                    for bid, q in list(order_depth.buy_orders.items()):
                        if bid > 10000:
                            orders.append(Order(product,int(bid),-q))
                        elif not arb and bid < 10000-d:
                            orders.append(Order(product,int(bid+d),7))

            if product == 'PICNIC_BASKET2':
                #we want to calculate the inav
                #we do this by adding up the previous prices for djembes, croissant, and jams
                pb2_price = 0
                if 'JAMS' in state.market_trades.keys() and len(state.market_trades['JAMS'])!=0: #get jams vwap
                    jams_trades = state.market_trades['JAMS']
                    cap=0
                    vol=0
                    for t in jams_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    pb2_price += 2*vwap
                else:
                    continue
                
                if 'CROISSANTS' in state.market_trades.keys() and len(state.market_trades['CROISSANTS'])!=0: #get croissant vwap
                    c_trades = state.market_trades['CROISSANTS']
                    cap=0
                    vol=0
                    for t in c_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    pb2_price += 4*vwap
                else:
                    continue

                if product in state.market_trades.keys() and len(state.market_trades[product])!=0:
                    pb1_trades = state.market_trades[product]
                    cap = 0
                    vol=0
                    for t in pb1_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                
                    #find mispricing
                    nav = pb2_price
                    premium = vwap - nav
                    percent_diff = premium / nav
                    d = .05
                    limit = 100
                    if product in state.position.keys():
                        p = state.position[product]
                    else:
                        p=0

                    if abs(percent_diff) > .0:
                        if len(order_depth.buy_orders)!=0:
                            best_bid = max(order_depth.buy_orders.keys())
                            best_bid_amt = order_depth.buy_orders[best_bid]
                        if len(order_depth.sell_orders)!=0:
                            best_ask = min(order_depth.sell_orders.keys())
                            best_ask_amt = order_depth.sell_orders[best_ask]

                        if vwap > nav:
                            #we want to sell, sell basket buy underlying
                            orders.append(Order('PICNIC_BASKET2',best_bid + 1,-p-limit))
                        else:
                            # we want to buy, buy basket sell underlying
                            orders.append(Order('PICNIC_BASKET2',best_ask - 1,limit-p))

            if product == "JAMS":
                cap=0
                vol=0
                limit=350
                window=20 #window for calculatins sma and sd
                if(product in state.market_trades.keys() and len(state.market_trades[product])!=0):
                    trades = state.market_trades[product]
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    jams_prices.append(vwap) #we keep track of the historic trade vwap to calculate sma and sd
                #make orders
                if len(jams_prices) > window:
                    jams_prices.pop(0)
                    stdev = statistics.stdev(jams_prices)
                    sma = statistics.mean(jams_prices)
                    d = 1.5*stdev 
                    best_bid = max(list(order_depth.buy_orders.keys()))
                    best_ask = min(list(order_depth.sell_orders.keys()))
                    best_bid_amt = order_depth.buy_orders[best_bid]
                    best_ask_amt = order_depth.sell_orders[best_ask]
                    spread = best_ask - best_bid
                    p=0
                    if product in state.position.keys():
                        p = state.position[product]
                    
                    #PAIRS TRADE JAMS AND CROISSANTS IDEA, FINISH LATER
                    if('CROISSANTS' in state.market_trades.keys() and len(state.market_trades['CROISSANTS'])!=0):
                        trades = state.market_trades['CROISSANTS']
                        for t in trades:
                            cap+=t.price*t.quantity
                            vol+=t.quantity
                        c_vwap = cap/vol

                    if vwap > sma+d:
                        #-p-limit for sell
                        #limit-p for buy
                        orders.append(Order(product,int(vwap+1),-30)) #sell
                        #result['CROISSANTS'] = [Order('CROISSANTS',int(c_vwap-1),30)]
                    if vwap < sma-d:
                        orders.append(Order(product,int(vwap-1),30)) #buy
                        #result['CROISSANTS'] = [Order('CROISSANTS',int(c_vwap+1),-30)]
            
            if product == "DJEMBES":
                cap=0
                vol=0
                limit=60
                window=500 #window for calculatins sma and sd
                if(product in state.market_trades.keys() and len(state.market_trades[product])!=0):
                    trades = state.market_trades[product]
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    djembes_prices.append(vwap) #we keep track of the historic trade vwap to calculate sma and sd
                #make orders
                if len(djembes_prices) > window:
                    djembes_prices.pop(0)
                    stdev = statistics.stdev(djembes_prices)
                    sma = statistics.mean(djembes_prices)
                    d = 2*stdev 
                    best_bid = max(list(order_depth.buy_orders.keys()))
                    best_ask = min(list(order_depth.sell_orders.keys()))
                    best_bid_amt = order_depth.buy_orders[best_bid]
                    best_ask_amt = order_depth.sell_orders[best_ask]
                    spread = best_ask - best_bid
                    p=0
                    if product in state.position.keys():
                        p = state.position[product]
                    if vwap > sma+d:
                        #-p-limit for sell
                        #limit-p for buy
                        orders.append(Order(product,int(vwap+1),-6)) #sell
                    if vwap < sma-d:
                        orders.append(Order(product,int(vwap-1),6)) #buy

            if product == 'PICNIC_BASKET1':
                #we want to calculate the inav
                #we do this by adding up the previous prices for djembes, croissant, and jams
                pb1_price=0
                if 'DJEMBES' in state.market_trades.keys() and len(state.market_trades['DJEMBES'])!=0: #get djembes vwap
                    djembe_trades = state.market_trades['DJEMBES']
                    cap=0
                    vol=0
                    for t in djembe_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    pb1_price += vwap
                else:
                    continue

                if 'JAMS' in state.market_trades.keys() and len(state.market_trades['JAMS'])!=0: #get jams vwap
                    jams_trades = state.market_trades['JAMS']
                    cap=0
                    vol=0
                    for t in jams_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    pb1_price += 3*vwap
                else:
                    continue
                
                if 'CROISSANTS' in state.market_trades.keys() and len(state.market_trades['CROISSANTS'])!=0: #get croissant vwap
                    c_trades = state.market_trades['CROISSANTS']
                    cap=0
                    vol=0
                    for t in c_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    pb1_price += 6*vwap
                else:
                    continue

                if product in state.market_trades.keys() and len(state.market_trades[product])!=0:
                    pb1_trades = state.market_trades[product]
                    cap = 0
                    vol=0
                    for t in pb1_trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                
                    #find mispricing
                    nav = pb1_price
                    premium = vwap - nav
                    percent_diff = premium / nav
                    d = .05
                    limit = 60
                    if product in state.position.keys():
                        p = state.position[product]
                    else:
                        p=0

                    if abs(percent_diff) > 0:
                        if len(order_depth.buy_orders)!=0:
                            best_bid = max(order_depth.buy_orders.keys())
                            best_bid_amt = order_depth.buy_orders[best_bid]
                        if len(order_depth.sell_orders)!=0:
                            best_ask = min(order_depth.sell_orders.keys())
                            best_ask_amt = order_depth.sell_orders[best_ask]

                        if('CROISSANTS' in state.market_trades.keys() and len(state.market_trades['CROISSANTS'])!=0):
                            trades = state.market_trades['CROISSANTS']
                            for t in trades:
                                cap+=t.price*t.quantity
                                vol+=t.quantity
                            c_vwap = cap/vol

                        if vwap > nav:
                            #sell
                            orders.append(Order('PICNIC_BASKET1',best_bid + 1,-p-limit))
                                         
                        else:
                            #buy
                            orders.append(Order('PICNIC_BASKET1',best_ask - 1,limit-p))

            result[product] = orders
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
