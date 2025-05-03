from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
import string
import statistics
import json
from math import log,sqrt,exp,pi
import numpy as np

kelp_prices = []
squid_prices = []
jams_prices=[]
djembes_prices = []
c_prices = []
vr_prices = []
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()
norm = statistics.NormalDist()

def bs_call(S, K, T, r, vol):
    d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - exp(-r * T) * K * norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return S * norm.pdf(d1) * sqrt(T)

def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 1000
    PRECISION = 1e-6
    sigma = 0.2
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        if vega < 1e-6:
            break
        if sigma <=0 or sigma > 3:
            return np.nan
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

class Trader:
    def run(self, state: TradingState):
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
                window=20 #short window seems to work well for jams, my guess is that jams is too volatile
                if(product in state.market_trades.keys() and len(state.market_trades[product])!=0):
                    trades = state.market_trades[product]
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    vwap = cap/vol
                    jams_prices.append(vwap)
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

            if product == 'VOLCANIC_ROCK':
                limit = 400
                if product in state.market_trades.keys() and len(state.market_trades[product])!=0:
                    trades = state.market_trades[product]
                    cap = 0
                    vol = 0
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    if vol ==0:
                        continue
                    vwap = cap/vol
                    pred = vwap #spot price
                    vr_prices.append(pred)
                    if len(vr_prices) == 500:
                        pred = statistics.mean(vr_prices)
                        vr_prices.pop(0)
                        if product in state.position.keys():
                                p = state.position[product]
                        else:
                            p=0
                        """
                        for bid,quantity in order_depth.buy_orders.items():
                            if bid > pred:
                                orders.append(Order(product,bid,-quantity))
                        for ask,quantity in order_depth.sell_orders.items():
                            if ask < pred:
                                orders.append(Order(product,ask,quantity))
                        """
                        best_bid = max(list(order_depth.buy_orders.keys()))
                        best_ask = min(list(order_depth.sell_orders.keys()))
                        if best_bid > pred:
                            orders.append(Order(product,int(pred)+1,-p-limit))
                        if best_ask < pred:
                            orders.append(Order(product,int(pred)-1,limit-p))
                        
                    

            if product == 'VOLCANIC_ROCK_VOUCHER_10500':
                u = 'VOLCANIC_ROCK'
                limit = 200
                if u in state.market_trades.keys():
                    trades = state.market_trades[u]
                    cap=0
                    vol=0
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    if vol == 0:
                        continue
                    vwap = cap/vol
                    underlying = vwap
                    if product in state.market_trades.keys():
                        trades = state.market_trades[product]
                        cap=0
                        vol=0
                        for t in trades:
                            cap+=t.price*t.quantity
                            vol+=t.quantity
                        if vol==0:
                            continue
                        option_mp = cap/vol 
                        ts = ((5) - (state.timestamp/1000000))/252
                        ir = 0
                        od = state.order_depths[product]
                        ivol = find_vol(option_mp,underlying,10500,ts,ir)
                        if np.isnan(ivol) or ivol == 0:
                            continue
                        pred = bs_call(underlying,10500,ts,ir,ivol)
                        if np.isnan(pred) or pred == 0:
                            continue
                        if product in state.position.keys():
                            p = state.position[product]
                        else:
                            p=0
                        for bid,quantity in order_depth.buy_orders.items():
                            if bid > pred*1.1:
                                orders.append(Order(product,bid,-quantity))
                        for ask,quantity in order_depth.sell_orders.items():
                            if ask < pred*.9:
                                orders.append(Order(product,ask,quantity))
            
            if product == 'VOLCANIC_ROCK_VOUCHER_10250':
                u = 'VOLCANIC_ROCK'
                limit = 200
                if u in state.market_trades.keys():
                    trades = state.market_trades[u]
                    cap=0
                    vol=0
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    if vol ==0:
                        continue
                    vwap = cap/vol
                    underlying = vwap
                    if product in state.market_trades.keys():
                        trades = state.market_trades[product]
                        cap=0
                        vol=0
                        for t in trades:
                            cap+=t.price*t.quantity
                            vol+=t.quantity
                        if vol==0:
                            continue
                        option_mp = cap/vol 
                        ts = ((5) - (state.timestamp/1000000))/252
                        ir = 0
                        od = state.order_depths[product]
                        ivol = find_vol(option_mp,underlying,10250,ts,ir)
                        if np.isnan(ivol) or ivol == 0:
                            continue
                        pred = bs_call(underlying,10250,ts,ir,ivol)
                        if np.isnan(pred) or pred == 0:
                            continue
                        if product in state.position.keys():
                            p = state.position[product]
                        else:
                            p=0
                        for bid,quantity in order_depth.buy_orders.items():
                            if bid > pred:
                                orders.append(Order(product,bid,-quantity))
                        for ask,quantity in order_depth.sell_orders.items():
                            if ask < pred:
                                orders.append(Order(product,ask,quantity))
            
            if product == 'VOLCANIC_ROCK_VOUCHER_9750':
                u = 'VOLCANIC_ROCK'
                limit = 200
                if u in state.market_trades.keys():
                    trades = state.market_trades[u]
                    cap=0
                    vol=0
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    if vol == 0:
                        continue
                    vwap = cap/vol
                    underlying = vwap
                    if product in state.market_trades.keys():
                        trades = state.market_trades[product]
                        cap=0
                        vol=0
                        for t in trades:
                            cap+=t.price*t.quantity
                            vol+=t.quantity
                        if vol==0:
                            continue
                        option_mp = cap/vol
                        ts = ((5) - (state.timestamp/1000000))/252
                        ir = 0
                        od = state.order_depths[product]
                        ivol = find_vol(option_mp,underlying,9750,ts,ir)
                        if np.isnan(ivol) or ivol == 0:
                            continue
                        pred = bs_call(underlying,9750,ts,ir,ivol)
                        if np.isnan(pred) or pred == 0:
                            continue
                        if product in state.position.keys():
                            p = state.position[product]
                        else:
                            p=0
                        for bid,quantity in order_depth.buy_orders.items():
                            if bid > pred:
                                orders.append(Order(product,bid+1,-quantity))
                        for ask,quantity in order_depth.sell_orders.items():
                            if ask < pred:
                                orders.append(Order(product,ask-1,quantity))

            if product == 'VOLCANIC_ROCK_VOUCHER_10000':
                u = 'VOLCANIC_ROCK'
                limit = 200
                if u in state.market_trades.keys():
                    trades = state.market_trades[u]
                    cap=0
                    vol=0
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    if vol ==0:
                        continue
                    vwap = cap/vol
                    underlying = vwap
                    if product in state.market_trades.keys():
                        trades = state.market_trades[product]
                        cap=0
                        vol=0
                        for t in trades:
                            cap+=t.price*t.quantity
                            vol+=t.quantity
                        if vol==0:
                            continue
                        option_mp = cap/vol 
                        ts = ((5) - (state.timestamp/1000000))/30
                        ir = 0
                        od = state.order_depths[product]
                        ivol = find_vol(option_mp,underlying,10000,ts,ir)
                        if np.isnan(ivol) or ivol == 0:
                            continue
                        pred = bs_call(underlying,10000,ts,ir,ivol)
                        if np.isnan(pred) or pred == 0:
                            continue
                        if product in state.position.keys():
                            p = state.position[product]
                        else:
                            p=0
                        option_mp = int(option_mp)
                        for bid,quantity in order_depth.buy_orders.items():
                            if bid > pred*1.1:
                                orders.append(Order(product,bid,-p-limit))
                        for ask,quantity in order_depth.sell_orders.items():
                            if ask < pred-.1*pred:
                                orders.append(Order(product,ask,limit-p))

            if product == 'VOLCANIC_ROCK_VOUCHER_9500':
                u = 'VOLCANIC_ROCK'
                limit = 200
                if u in state.market_trades.keys():
                    trades = state.market_trades[u]
                    cap=0
                    vol=0
                    for t in trades:
                        cap+=t.price*t.quantity
                        vol+=t.quantity
                    if vol == 0:
                        continue
                    vwap = cap/vol
                    underlying = vwap
                    if product in state.market_trades.keys():
                        trades = state.market_trades[product]
                        cap=0
                        vol=0
                        for t in trades:
                            cap+=t.price*t.quantity
                            vol+=t.quantity
                        if vol==0:
                            continue
                        option_mp = cap/vol 
                        ts = ((5) - (state.timestamp/1000000))/252
                        ir = 0
                        od = state.order_depths[product]
                        ivol = find_vol(option_mp,underlying,9500,ts,ir)
                        if np.isnan(ivol) or ivol == 0:
                            continue
                        pred = bs_call(underlying,9500,ts,ir,ivol)
                        if np.isnan(pred) or pred == 0:
                            continue
                        if product in state.position.keys():
                            p = state.position[product]
                        else:
                            p=0
                        option_mp = int(option_mp)
                        for bid,quantity in order_depth.buy_orders.items():
                            if bid > pred*1.1:
                                orders.append(Order(product,bid,-p-limit))
                        for ask,quantity in order_depth.sell_orders.items():
                            if ask < pred-.1*pred:
                                orders.append(Order(product,ask,limit-p))

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
        logger.flush(state,result,conversions,traderData)
        return result, conversions, traderData
