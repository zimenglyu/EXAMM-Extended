import pandas as pd
import os
from Logger import Logger

class Stock:

    def __init__(self, name):
        self.name = name
        self.share = 0
        self.bought_price = 0
        self.return_prediction = []
        self.return_price = []
        self.sold_price = 0
        self.use_TC = False
        self.trade_with_bid_ask = False
        self.long_waiting_period = 20
        self.short_waiting_period = 20
        
    def reset(self):
        self.share = 0
        self.bought_price = 0
        self.sold_price = 0
        self.long_waiting_period = 20
        self.short_waiting_period = 20

    def read_return_prediction(self, file_path):
        self.return_prediction = pd.read_csv(file_path, usecols=['predicted_RET']).to_numpy().flatten()
        # self.return_prediction = pd.read_csv(file_path, usecols=['predicted_ret']).to_numpy().flatten()
        # self.return_prediction = pd.read_csv(file_path, usecols=['expected_RET']).to_numpy().flatten()
        self.testing_period = len(self.return_prediction)

    def read_stock_price(self, file_path):
        self.stock_price = pd.read_csv(file_path, usecols=['PRC']).to_numpy().flatten()
        self.transaction_cost = pd.read_csv(file_path, usecols=['TRAN_COST']).to_numpy().flatten()
        # ASK/BID are read ONLY when the file provides them. They are consumed exclusively
        # by the trade_with_bid_ask execution mode; the default (PRC) and use_TC modes
        # never touch them. Some datasets ship PRC/TRAN_COST without quote levels -- and
        # for those, ASK/BID are NOT derivable: PRC is the closing TRADE price, which
        # (measured on a dataset that has all four) equals ASK only 27% of the time and
        # falls outside [BID, ASK] entirely 30% of the time. Only the SPREAD is recoverable
        # (ASK - BID == 2 * TRAN_COST exactly), not the level -- so inventing them would be
        # fabrication. Left as None instead; buy/sell/short/return_stock raise a clear
        # error below if bid-ask mode is requested without the data.
        cols = set(pd.read_csv(file_path, nrows=0).columns)
        has_quotes = {'ASK', 'BID'} <= cols
        self.ask_price = pd.read_csv(file_path, usecols=['ASK']).to_numpy().flatten() if has_quotes else None
        self.bid_price = pd.read_csv(file_path, usecols=['BID']).to_numpy().flatten() if has_quotes else None

        # if len(self.return_prediction) != 0:
        #     if len(self.return_prediction) + 1 != len(self.stock_price):
        #         self.logger.log('Length of predicted_return is not equal to price')
        #         self.logger.log('stock name: ', self.name)
        #         self.logger.log('Length of predicted_return: ', len(self.return_prediction))
        #         self.logger.log('Length of price: ', len(self.stock_price))
        #         exit()

    def print_current_holdings(self, time):
        if abs(self.share) > 1e-10:
            self.logger.log(f"[End of time {time}] holdings: {self.name}, share: {self.share}, stock price: {self.stock_price[time]}", level='DEBUG')
                
    def buy_stock(self, cash_amount, time):
        # buy as many shares as possible, when buying, their might be existing shares
        new_share = 0
        if self.use_TC:
            new_share = cash_amount / (self.stock_price[time] + self.transaction_cost[time])
            self.bought_price = self.stock_price[time] + self.transaction_cost[time]
            self.logger.log(f"[time {time}]: Buying company: {self.name} with ${cash_amount}, stock price: {self.stock_price[time]}, transaction cost:{self.transaction_cost[time]}, shares: {new_share}", level='DEBUG')
        elif self.trade_with_bid_ask:
            self._require_quotes()
            # buy at ask price (high)
            new_share = cash_amount / (self.ask_price[time])
            self.bought_price = self.ask_price[time]
            self.logger.log(f"[time {time}]: Buying company: {self.name} with ${cash_amount}, stock asking price: {self.ask_price[time]}, shares: {new_share}", level='DEBUG')
        else:
            new_share = cash_amount / (self.stock_price[time])
            self.bought_price = self.stock_price[time] 
            self.logger.log(f"[time {time}]: Buying company: {self.name} with ${cash_amount}, stock price: {self.stock_price[time]}, shares: {new_share}", level='DEBUG')
        self.share += new_share
        
    
    def sell_stock(self, time): 
        # sell all shares
        cash_amount = 0
        if self.share > 0:
            if self.use_TC:
                cash_amount = (self.stock_price[time] - self.transaction_cost[time]) * self.share
                self.logger.log(f"[time {time}]: Selling company: {self.name}, stock price: {self.stock_price[time]}, transaction cost:{self.transaction_cost[time]}, shares: {self.share}, cash amount: {cash_amount}", level='DEBUG')
            elif self.trade_with_bid_ask:
                self._require_quotes()
                cash_amount = (self.bid_price[time]) * self.share
                self.logger.log(f"[time {time}]: Selling company: {self.name}, stock bid price: {self.bid_price[time]}, shares: {self.share}, cash amount {cash_amount}", level='DEBUG')
            else:
                cash_amount = (self.stock_price[time]) * self.share
                self.logger.log(f"[time {time}]: Selling company: {self.name}, stock price: {self.stock_price[time]}, shares: {self.share}, cash amount {cash_amount}", level='DEBUG')
            self.share = 0
        return cash_amount
    
    def short_stock(self, cash_amount, time):
        # short as many shares as possible, when shorting, their might be existing shares.
        # Returns the actual cash credit for this trade -- equal to cash_amount exactly
        # when TC is off or trading at bid/ask (unchanged behavior). With TC on, TRAN_COST
        # already reduces the share count (divided into the price), so crediting the flat
        # cash_amount target regardless was a bug: the open leg ignored the fee entirely
        # while the close leg paid it twice (fewer shares, then TC again on cover),
        # making transaction costs spuriously INCREASE short-side returns. Crediting
        # shares_shorted * stock_price[time] (the raw-price value of what was ACTUALLY
        # shorted) instead makes TC a pure drag, symmetric with the long side.
        if self.use_TC:
            shares_shorted = cash_amount / (self.stock_price[time] + self.transaction_cost[time])
            self.share -= shares_shorted
            self.sold_price = self.stock_price[time] + self.transaction_cost[time]
            credited = shares_shorted * self.stock_price[time]
            self.logger.log(f"[time {time}]: Shorting company: {self.name} with ${cash_amount}, stock price: {self.stock_price[time]}, transaction cost: {self.transaction_cost[time]}, shares: {self.share}, credited: {credited}", level='DEBUG')
            return credited
        elif self.trade_with_bid_ask:
            self._require_quotes()
            self.share -= cash_amount / (self.bid_price[time])
            self.sold_price = self.bid_price[time]
            self.logger.log(f"[time {time}]: Shorting company: {self.name}, cash amount: {cash_amount}, stock ask price: {self.bid_price[time]}, shares: {self.share}", level='DEBUG')
            return cash_amount
        else:
            self.share -= cash_amount / (self.stock_price[time])
            self.sold_price = self.stock_price[time]
            self.logger.log(f"[time {time}]: Shorting company: {self.name}, cash amount: {cash_amount}, stock price: {self.stock_price[time]}, shares: {self.share}", level='DEBUG')
            return cash_amount
    
    def return_stock(self, time):
        # return all shares
        # returned cash amount can be negative
        if self.share < 0:
            if self.use_TC:
                cash_amount = (self.stock_price[time] - self.transaction_cost[time]) * self.share
                self.logger.log(f"[time {time}]: Returning borrowed: {self.name}, shares {self.share}, price  {self.stock_price[time]}, transaction cost {self.transaction_cost[time]}, cash amount: {cash_amount}", level='DEBUG')
            elif self.trade_with_bid_ask:
                self._require_quotes()
                cash_amount = (self.ask_price[time]) * self.share
                self.logger.log(f"[time {time}]: Returning borrowed: {self.name}, shares {self.share}, stock ask price: {self.ask_price[time]}, cash amount: {cash_amount}", level='DEBUG')
            else:
                cash_amount = (self.stock_price[time]) * self.share
                self.logger.log(f"[time {time}]: Returning borrowed: {self.name}, shares {self.share}, stock price: {self.stock_price[time]}, cash amount: {cash_amount}", level='DEBUG')
            self.share = 0
        else:
            self.logger.log(f"[time {time}]: Company {self.name}: Error: trying to return shares when there is no shorted shares", level='DEBUG')
            exit()

        return cash_amount 
    
    def simple_return(self, money_pool):
        """
        Calculates the return on investment for a stock based on predicted returns.
        """
        self.money_pool = money_pool
        for i in range(len(self.return_prediction)-1):
            if self.return_prediction[i] > 0:
                if self.share == 0:
                    self.share = money_pool / self.stock_price[i]
                    money_pool = 0
                    self.bought_price = self.stock_price[i]
                    self.logger.log(f"Stock {self.name}, bought price: {self.bought_price}, number of shares: {self.share}", level='DEBUG')
            else:
                if self.share != 0:
                    if self.bought_price < self.stock_price[i]:
                        money_pool = self.stock_price[i] * self.share
                        self.logger.log(f"Stock {self.name}, bought price: {self.bought_price}, sold price: {self.stock_price[i]}, number of shares: {self.share} money_pool: {money_pool}", level='DEBUG') 
                        self.share = 0
        if self.share != 0:
            money_pool = self.stock_price[-2] * self.share
            self.share = 0
        return money_pool
    
    def get_sold_price(self):
        return self.sold_price
    
    def get_current_liquid(self, time):
        # it can be positive or negative
        current_liquid = self.share * self.stock_price[time]
        if abs(current_liquid) > 1e-5:  
            self.logger.log(f"[time {time}]: Company {self.name}, current liquid: {current_liquid}", level='DEBUG')
        return current_liquid
        

    def get_predicted_return(self, time):
        # print(f"company {self.name}, time {time}")
        return self.return_prediction[time]

    def get_past_return(self, time, lookback=1):
        # realized return from `lookback` days ago up to `time`, used as a
        # reversal signal; 0 if there isn't enough price history yet
        if time - lookback < 0:
            return 0.0
        past_price = self.stock_price[time - lookback]
        if past_price == 0:
            return 0.0
        return (self.stock_price[time] - past_price) / past_price
    
    def get_stock_price(self, time):
        return self.stock_price[time]
    
    def get_transaction_cost(self, time):
        return self.transaction_cost[time]
    
    def get_stock_name(self):
        return self.name

    def get_share(self):
        return self.share
    
    def set_share(self, share):
        self.share = share
    
    def set_bought_price(self, price):
        self.bought_price = price
    
    def get_bought_price(self):
        return self.bought_price
    
    def clear_holdings(self, time):
        cash_amount = 0
        if self.share < 0:
            return self.return_stock(time)
        elif self.share > 0:
            return self.sell_stock(time)
        else:
            return cash_amount
    

    def get_notional(self, time):
        """Signed dollar value of the current position, negative if short."""
        return self.share * abs(self.stock_price[time])

    def rebalance_to(self, target_notional, time):
        """Move the position to `target_notional` dollars; return the cash flow.

        buy_stock/short_stock open a position and sell_stock/return_stock
        close it entirely, so a name whose desired size merely *changes*
        can only be expressed as a full close followed by a full re-open,
        which pays the spread on the whole position instead of on the part
        that actually traded. The overlapping rule holds several sleeves at
        once and rolls one of them per day, so it needs the difference.

        Only |delta| shares cross the spread, which is what "transaction
        costs are charged only on the change in aggregate notional per
        name" means. Cash flow is negative when buying and positive when
        selling, matching the sign convention of the methods above.

        PRC is taken in absolute value: CRSP stores a negative price when
        the day's figure is a bid/ask midpoint rather than a traded close,
        and that sign carries no economic meaning.
        """
        price = abs(self.stock_price[time])
        if not price > 0:
            self.logger.log(
                f"[time {time}]: Company {self.name}: no usable price, "
                "leaving the position unchanged", level='DEBUG')
            return 0.0

        target_share = target_notional / price
        delta = target_share - self.share
        if delta == 0:
            return 0.0

        tc = self.transaction_cost[time] if self.use_TC else 0.0
        # buying lifts the offer, selling hits the bid
        exec_price = price + tc if delta > 0 else price - tc
        cash_flow = -delta * exec_price
        self.share = target_share
        if delta > 0:
            self.bought_price = exec_price
        else:
            self.sold_price = exec_price
        self.logger.log(
            f"[time {time}]: Rebalancing {self.name} to ${target_notional:.2f} "
            f"({delta:+.4f} shares at {exec_price:.4f}, cash {cash_flow:+.2f})",
            level='DEBUG')
        return cash_flow

    def set_use_TC(self, use_TC):
        # self.logger.log(f"Company {self.name}, use transaction cost: {use_TC}", level='DEBUG')
        self.use_TC = use_TC

    def _require_quotes(self):
        if self.ask_price is None or self.bid_price is None:
            raise ValueError(
                f"{self.name}: bid-ask execution requested but this dataset has no "
                f"ASK/BID columns. They are not derivable from PRC/TRAN_COST (only the "
                f"spread is: ASK-BID == 2*TRAN_COST), so re-export the data with real "
                f"quote levels or use PRC/TC execution instead.")

    def set_trade_with_bid_ask(self, trade_with_bid_ask):
        # self.logger.log(f"Company {self.name}, buy with ASK and sell with BID: {trade_with_bid_ask}", level='DEBUG')
        self.trade_with_bid_ask = trade_with_bid_ask

    def set_logger(self, logger):
        self.logger = logger
    
    def set_waiting_period(self, waiting_period):
        self.long_waiting_period = waiting_period
        self.short_waiting_period = waiting_period
    
    def check_waiting_period(self):
        if self.share > 0:
            if self.long_waiting_period > 0:
                self.long_waiting_period -= 1
            else:
                self.logger.log(f"Company {self.name}: long waiting period is over, selling", level='DEBUG')
                return True
        elif self.share < 0:
            if self.short_waiting_period > 0:
                self.short_waiting_period -= 1
            else:
                self.logger.log(f"Company {self.name}: short waiting period is over, returning", level='DEBUG')
                return True
        else:
            return False
        return False