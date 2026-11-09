"""
This module contains the `Portfolio` class, which provides methods 
for tracking spending and earnings, managing stocks in the portfolio, 
and calculating returns on investment using various investment strategies.

"""
from stock import Stock
from Logger import Logger
import numpy as np
class Portfolio:
    """
    A class to manage financial portfolios including spending, earning, and stock investments.

    Attributes:
        money_spend (float): Total amount spent.
        money_earn (float): Total amount earned.
        portfolio_names (list): List of stock names in the portfolio.
    """

    def __init__(self, stock_names):
        """
        Initializes a Portfolio object with default attributes.
        """
        self.initial_spend_per_stock = 0
        self.initial_capital = 0
        self.stock_names = stock_names
        self.money_spend = 0
        self.money_earn = 0
        self.portfolio_list = []
        self.portfolio_return = 0
        self.borrow_ratio = 0.5
        self.use_TC = False
        self.calculate_bound = False
        print("Portfolio created")

    def reset(self):
        """
        Resets the portfolio by clearing spending, earning, and the list of stock names.
        """
        self.money_earn = 0

        for company in self.portfolio_list:
            company.reset()
        
        if (self.strategy == 'simple_return'):
            self.money_spend = self.initial_spend_per_stock * len(self.stock_names)
        else:
            self.money_spend = self.initial_capital
        self.daily_values = []
    
    def set_logger(self, logger):
        self.logger = logger
        for company in self.portfolio_list:
            company.set_logger(logger)

    def add_company_to_protfolio(self, company):
        self.portfolio_list.append(company)
        self.logger.log(f"added company {company.get_stock_name()} to portfolio list", 'DEBUG')

    def set_initial_capital(self, money_pool):
        self.initial_capital = money_pool
    
    def set_initial_spend_per_stock(self, spend_per_stock):
        self.initial_spend_per_stock = spend_per_stock
    
    def set_risk_tolenrance(self, risk_tolenrance):
        self.risk_tolenrance = risk_tolenrance

    def trade(self, strategy, long_company_number=10, short_company_number=5, lookback=1,
              hold_days=10, entry_rank=10, exit_rank=35):
        portfolio_return = 0
        self.strategy = strategy
        self.logger.log(f"Trading with {self.strategy} strategy", 'INFO')
        self.reset()
        if self.strategy == 'simple_return':
            portfolio_return = self.simple_return()
        elif self.strategy == 'portfolio_simple_return':
            portfolio_return = self.portfolio_simple_return()
        elif self.strategy == 'portfolio_shorting_return':
            portfolio_return = self.portfolio_shorting_return()
        elif self.strategy == 'long_short_return':
            portfolio_return = self.long_short_return(long_company_number, short_company_number)
        elif self.strategy == 'daily_long_return':
            portfolio_return = self.daily_long_return(long_company_number, short_company_number)
        elif self.strategy == 'daily_long_short_return':
            portfolio_return = self.daily_long_short_return(long_company_number, short_company_number)
        elif self.strategy == 'daily_zimeng_long_short_return':
            portfolio_return = self.daily_zimeng_long_short_return(long_company_number, short_company_number)
        elif self.strategy == 'buy_and_hold_return':
            portfolio_return = self.buy_and_hold_return()
        elif self.strategy == 'daily_equal_weight_return':
            portfolio_return = self.daily_equal_weight_return()
        elif self.strategy == 'short_term_reversal_return':
            portfolio_return = self.short_term_reversal_return(long_company_number, short_company_number, lookback)
        elif self.strategy == 'overlapping_return':
            portfolio_return = self.overlapping_return(long_company_number, hold_days)
        elif self.strategy == 'banded_return':
            portfolio_return = self.banded_return(entry_rank, exit_rank)
        elif self.strategy == 'conditional_long_short_return':
            portfolio_return = self.conditional_long_short_return(long_company_number, short_company_number)
        else:
            self.logger.log("Invalid trading strategy", 'ERROR')
            exit()
        return portfolio_return
    
    def simple_return(self):
        self.logger.log("Calculating simple return", 'INFO')
        for company in self.portfolio_list:
            self.money_earn += company.simple_return(self.initial_spend_per_stock)
        return self.calculate_return()
    
    def set_waiting_period(self, waiting_period):
        for company in self.portfolio_list:
            company.set_waiting_period(waiting_period)

    def calculate_return(self):
        """
        Calculates and prints the portfolio's return on investment.

        Returns:
            float: Portfolio return on investment.
        """
        self.logger.log(f"Total money spent: {self.money_spend}", 'DEBUG')
        self.logger.log(f"Total money earned: {self.money_earn}", 'DEBUG')
        self.portfolio_return = ((self.money_earn - self.money_spend) / self.money_spend) * 100
        self.logger.log(f"Portfolio return for {self.strategy} is {self.portfolio_return:.2f}%", 'INFO')
        values = np.array(getattr(self, 'daily_values', []), dtype=float)
        if len(values) > 2:
            daily = values[1:] / values[:-1] - 1
            self.sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float('nan')
            self.mdd = (1 - values / np.maximum.accumulate(values)).max() * 100
            self.logger.log(f"Sharpe for {self.strategy} is {self.sharpe:.2f}", 'INFO')
            self.logger.log(f"MDD for {self.strategy} is {self.mdd:.2f}%", 'INFO')
        self.daily_values = []
        return self.portfolio_return

    def add_stock(self, stock_name):
        """
        Adds a stock to the portfolio if it's not already present.

        Args:
            stock_name (str): Name of the stock.
        """
        if not self.is_in_portfolio(stock_name):
            self.portfolio_names.append(stock_name)
        else:
            self.logger.log(f"{stock_name} is already in the portfolio", 'ERROR')

    def remove_stock(self, stock_name):
        """
        Removes a stock from the portfolio if it's present.

        Args:
            stock_name (str): Name of the stock.
        """
        if self.is_in_portfolio(stock_name):
            self.portfolio_names.remove(stock_name)
        else:
            self.logger.log(f"{stock_name} is not in the portfolio", "ERROR")
    
    def set_use_TC(self, use_TC):
        self.use_TC = use_TC
        for company in self.portfolio_list:
            company.set_use_TC(use_TC)

    def set_trade_with_bid_ask(self, trade_with_bid_ask):
        self.trade_with_bid_ask = trade_with_bid_ask
        for company in self.portfolio_list:
            company.set_trade_with_bid_ask(trade_with_bid_ask)
    
    def get_sorted_return_list(self, time):
        predicted_return_list = []
        for company in self.portfolio_list:
            predicted_return_list.append(company.get_predicted_return(time))
        sorted_index = np.argsort(predicted_return_list)
        # sorted_index = sorted(range(len(predicted_return_list)), key=lambda k: predicted_return_list[k])
        return sorted_index, predicted_return_list

    def get_sorted_reversal_signal_list(self, time, lookback=1):
        # reversal signal = negative of each stock's past `lookback`-day
        # return, so biggest recent losers get the highest signal
        signal_list = []
        for company in self.portfolio_list:
            signal_list.append(-company.get_past_return(time, lookback))
        sorted_index = np.argsort(signal_list)
        return sorted_index, signal_list

    def daily_long_return(self, number_top = 5, number_bottom = 5):
        """
        Calculates the return on investment for a stock using a long strategy.
        """
        self.reset()
        self.current_cash_amount = self.initial_capital
        testing_period = self.portfolio_list[0].testing_period
        for time in range(testing_period - 1):
            self.logger.log(f"[time {time}]: current cash amount: {self.current_cash_amount}", 'DEBUG')
            sorted_index, predicted_return_list = self.get_sorted_return_list(time)
            daily_long_quota = self.current_cash_amount / number_top

            sold_company = 0
            for i in range(30):
                if sold_company < number_bottom:
                    company = self.portfolio_list[sorted_index[i]]
                    if company.get_share() > 0:
                        self.current_cash_amount += company.sell_stock(time)
                        sold_company += 1
            self.logger.log(f"[time {time}]: current cash amount after selling stocks: {self.current_cash_amount}", 'DEBUG')
            
            for i in range(number_top):
                index = -(i+1)
                company = self.portfolio_list[sorted_index[index]]
                if self.current_cash_amount >= daily_long_quota:
                    company.buy_stock(daily_long_quota, time)
                    self.current_cash_amount -= daily_long_quota
                elif abs(self.current_cash_amount - daily_long_quota) < 1:
                    company.buy_stock(self.current_cash_amount, time)
                    self.current_cash_amount = 0
                else:
                    self.logger.log(f"Can't buy stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')
        self.logger.log("/*-----Clearing all stocks-------*/", 'DEBUG')
        for company in self.portfolio_list:
            self.money_earn += company.clear_holdings(-2)
        return self.calculate_return()


    def portfolio_simple_return(self):
        testing_period = self.portfolio_list[0].testing_period
        self.current_cash_amount = self.initial_capital

        for time in range(testing_period - 1):
            self.logger.log("------------------------------------------", 'DEBUG')
            self.logger.log(f"[time {time}]: current cash amount: {self.current_cash_amount}", 'DEBUG')
            # sell stocks with negative predicted returns
            for company in self.portfolio_list:
                if company.get_predicted_return(time) < 0:
                    if company.get_bought_price() < company.get_stock_price(time):
                        self.current_cash_amount += company.sell_stock(time)
            companies_to_buy = []
            self.logger.log(f"[time {time}]: current cash amount after selling stocks: {self.current_cash_amount}", 'DEBUG')
            # find all stocks to buy at time t
            for company in self.portfolio_list:
                if company.get_predicted_return(time) > 0:
                    companies_to_buy.append(company)
            if len(companies_to_buy) == 0:
                self.logger.log(f"No stock to buy at time {time}", 'DEBUG')
                continue
            else:
                # just give each stock equal amount of money
                if self.current_cash_amount > 0.0001:
                    quota_per_stock = self.current_cash_amount / len(companies_to_buy)
                    self.logger.log(f"[time {time}]: number of company to buy: {len(companies_to_buy)}, quota per stock: {quota_per_stock}", 'DEBUG')
                    for company in companies_to_buy:
                        if self.current_cash_amount >= quota_per_stock:
                            company.buy_stock(quota_per_stock, time)
                            self.current_cash_amount -= quota_per_stock
                        elif abs(self.current_cash_amount - quota_per_stock) < 1:
                            company.buy_stock(self.current_cash_amount, time)
                            self.current_cash_amount = 0
                        else:
                            self.logger.log(f"Can't buy stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')
                            self.logger.log("current money pool: ", self.current_cash_amount, 'DEBUG')
                            self.logger.log("quota per stock: ", quota_per_stock, 'DEBUG')
                    if self.current_cash_amount > 1:
                        self.logger.log(f"current money pool {self.current_cash_amount} at time {time}", 'DEBUG')
                self.logger.log(f"[End of time {time}], current cash amount: {self.current_cash_amount}", 'DEBUG')
            for company in self.portfolio_list:
                company.print_current_holdings(time)
            
        # sell all stocks at the end
        self.logger.log("/*-----Clearing all stocks-------*/", 'DEBUG')
        for company in self.portfolio_list:
            self.money_earn += company.clear_holdings(testing_period - 2)
        return self.calculate_return()

    def portfolio_shorting_return(self):
        testing_period = self.portfolio_list[0].testing_period
        # current money pool is the current cash amount
        self.current_cash_amount = self.initial_capital
        # current liquid is currently how much we have minus how much we borrowed
        current_liquid = self.initial_capital
        self.logger.log(f"[time 0]: current liquid: {current_liquid}", 'DEBUG')

        for time in range(testing_period - 1):
            for company in self.portfolio_list:
                if company.get_predicted_return(time) > 0:
                    if company.get_share() < 0:
                        # if company.get_sold_price() > company.get_stock_price(time):
                        self.current_cash_amount += company.return_stock(time)
            self.logger.log(f"[time {time}]: current cash amount after covering stocks {self.current_cash_amount}", 'DEBUG')
            current_liquid = 0
            for company in self.portfolio_list:
                current_liquid += company.get_current_liquid(time)
            current_liquid += self.current_cash_amount
            self.logger.log(f"[time {time}]: current liquid: {current_liquid}", 'DEBUG')
            # --------- start to short stocks ------------
            company_to_short = []
            for company in self.portfolio_list:
                if company.get_predicted_return(time) < 0 and company.get_share() >= 0:
                    company_to_short.append(company)
            if len(company_to_short) == 0:
                self.logger.log(f"[time {time}]: No stock to short today", 'DEBUG')
                continue
            else:
                self.logger.log(f"[time {time}]: number of company to short: {len(company_to_short)}", 'DEBUG' )
                borrow_cash_amount = current_liquid
                # borrow_cash_amount = self.borrow_ratio * current_liquid
                quota_per_stock = borrow_cash_amount / len(company_to_short)
                # 
                self.logger.log(f"[time {time}]: borrow cash amount:{borrow_cash_amount}, quota per stock {quota_per_stock}", 'DEBUG')
                for company in company_to_short:
                    # self.logger.log("shorting stock: ", company.get_stock_name())
                    if borrow_cash_amount >= quota_per_stock:
                        credited = company.short_stock(quota_per_stock, time)
                        borrow_cash_amount -= quota_per_stock
                        self.current_cash_amount += credited
                        self.logger.log(f"[time {time}]: current cash: {borrow_cash_amount}, current cash amount: {self.current_cash_amount}", 'DEBUG')
                    elif abs(borrow_cash_amount - quota_per_stock) < 1:
                        credited = company.short_stock(borrow_cash_amount, time)
                        self.current_cash_amount += credited
                        borrow_cash_amount = 0
                        self.logger.log(f"[time {time}]: current cash: {borrow_cash_amount}, current cash amount: {self.current_cash_amount}", 'DEBUG')
                    else:
                        self.logger.log(f"[time {time}]: Can't short stock {company.get_stock_name()} because of insufficient money", 'DEBUG')
                        self.logger.log(f"[time {time}]: current money pool: {borrow_cash_amount}", 'DEBUG')
                self.logger.log(f"[time {time}]: current cash amount at end {self.current_cash_amount}".format(time, self.current_cash_amount), 'DEBUG')

        for company in self.portfolio_list:
            self.current_cash_amount += company.clear_holdings(-2)
        self.money_earn = self.current_cash_amount
        return self.calculate_return()
    
    def daily_long_short_return(self, long_company_number=10, short_company_number=5):
        """
        Calculates the return on investment for a stock using a long-short strategy.
        simple strategies only "long" the stocks, but long-short strategy can "short" the stocks
        """

        self.logger.log(f"Using {long_company_number} companies to long and {short_company_number} companies to short", 'INFO')
        self.reset()
        self.current_cash_amount = self.initial_capital
        # self.long_short_return_helper(list_length)
        testing_period = self.portfolio_list[0].testing_period
        # current money pool is the current cash amount
        self.current_cash_amount = self.initial_capital
        # current liquid is currently how much we have minus how much we borrowed
        longed_stock_yesterday = []
        shorted_stock_yesterday = []
        for time in range(testing_period - 1):
            do_trade = True
            longed_stock_today = []
            shorted_stock_today = []
# ----------------------- find company -------------------------------------
            sorted_index, predicted_return_list = self.get_sorted_return_list(time)
            for i in range(long_company_number):
                index = -(i+1) # get the last 10 stocks
                company = self.portfolio_list[sorted_index[index]]
                # if company.get_predicted_return(time) > 0:
                longed_stock_today.append(company)
                # else:
                #     self.logger.log(f"[time {time}]: No Trade Today!", 'DEBUG')
                #     do_trade = False
                #     break

            for i in range(short_company_number):
                company = self.portfolio_list[sorted_index[i]]
                # if company.get_predicted_return(time) < 0:
                shorted_stock_today.append(company)
                # else:
                #     self.logger.log(f"[time {time}]: No Trade Today!", 'DEBUG')
                #     do_trade = False
                #     break
# ---------------------------- trade -------------------------------------
            if do_trade:
                # clear all holdings
                self.logger.log(f"[time {time}]: clearing all holdings", 'DEBUG')
                for company in longed_stock_yesterday:
                    self.current_cash_amount += company.clear_holdings(time)
                    self.logger.log(f"[time {time}]: current cash amount: {self.current_cash_amount}", 'DEBUG')
                for company in shorted_stock_yesterday:
                    self.current_cash_amount += company.clear_holdings(time)
                    self.logger.log(f"[time {time}]: current cash amount: {self.current_cash_amount}", 'DEBUG')
                self.logger.log(f"[time {time}]: current cash amount after clearing all holdings: {self.current_cash_amount}", 'DEBUG')
                self.daily_values.append(self.current_cash_amount)

                # sorted_index, predicted_return_list = self.get_sorted_return_list(time)
                borrow_cash_amount = self.current_cash_amount
                quota_per_stock = self.current_cash_amount / long_company_number

                self.logger.log(f"[time {time}]: longing {long_company_number} stocks, current cash amount: {self.current_cash_amount}, quota per stock: {quota_per_stock}", 'DEBUG')

                for company in longed_stock_today:
                    self.logger.log(f"[time {time}]: longing stock {company.get_stock_name()}, quota per stock: {quota_per_stock}, predicted return: {company.get_predicted_return(time)}", 'DEBUG')
                    if self.current_cash_amount > quota_per_stock:
                        company.buy_stock(quota_per_stock, time)
                        self.current_cash_amount -= quota_per_stock
                    elif abs(self.current_cash_amount - quota_per_stock) < 1:
                        company.buy_stock(self.current_cash_amount, time)
                        self.current_cash_amount = 0
                    else:
                        self.logger.log(f"[time {time}]: Can't buy stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')

                # short the last 10 stocks
                quota_per_stock = borrow_cash_amount / short_company_number
                self.logger.log(f"[time {time}]: shorting {short_company_number} stocks, quota per stock: {quota_per_stock}", 'DEBUG')

                for company in shorted_stock_today:
                    self.logger.log(f"[time {time}]: shorting stock {company.get_stock_name()}, quota per stock: {quota_per_stock}, predicted return: {company.get_predicted_return(time)}", 'DEBUG')
                    if borrow_cash_amount >= quota_per_stock:
                        credited = company.short_stock(quota_per_stock, time)
                        borrow_cash_amount -= quota_per_stock
                        self.current_cash_amount += credited
                        # self.current_cash_amount -= quota_per_stock
                    elif abs(borrow_cash_amount - quota_per_stock) < 1:
                        credited = company.short_stock(borrow_cash_amount, time)
                        self.current_cash_amount += credited
                        borrow_cash_amount = 0
                    else:
                        self.logger.log(f"[time {time}]: Can't short stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')
                longed_stock_yesterday = longed_stock_today
                shorted_stock_yesterday = shorted_stock_today

        for company in self.portfolio_list:
            self.current_cash_amount += company.clear_holdings(time)
        self.money_earn = self.current_cash_amount
        self.daily_values.append(self.money_earn)
        return self.calculate_return()
    
    def daily_hybrid_long_short_return(self, long_company_number=10, short_company_number=5):
        """
        Calculates the return on investment for a stock using a long-short strategy.
        simple strategies only "long" the stocks, but long-short strategy can "short" the stocks
        """

        self.logger.log(f"Using {long_company_number} companies to long and {short_company_number} companies to short", 'INFO')
        self.reset()
        self.current_cash_amount = self.initial_capital
        # self.long_short_return_helper(list_length)
        testing_period = self.portfolio_list[0].testing_period
        # current money pool is the current cash amount
        self.current_cash_amount = self.initial_capital
        # current liquid is currently how much we have minus how much we borrowed
        longed_stock_yesterday = []
        shorted_stock_yesterday = []
        for time in range(testing_period - 1):
            do_trade = True
            longed_stock_today = []
            shorted_stock_today = []
# ----------------------- find company -------------------------------------
            sorted_index, predicted_return_list = self.get_sorted_return_list(time)
            for i in range(long_company_number):
                index = -(i+1) # get the last 10 stocks
                company = self.portfolio_list[sorted_index[index]]
                if company.get_predicted_return(time) > 0:
                    longed_stock_today.append(company)
                else:
                    self.logger.log(f"[time {time}]: No Trade Today!", 'DEBUG')
                    do_trade = False
                    break

            for i in range(short_company_number):
                company = self.portfolio_list[sorted_index[i]]
                if company.get_predicted_return(time) < 0:
                    shorted_stock_today.append(company)
                else:
                    self.logger.log(f"[time {time}]: No Trade Today!", 'DEBUG')
                    do_trade = False
                    break
# ---------------------------- trade -------------------------------------
            if do_trade:
                # clear all holdings
                self.logger.log(f"[time {time}]: clearing all holdings", 'DEBUG')
                for company in longed_stock_yesterday:
                    self.current_cash_amount += company.clear_holdings(time)
                    self.logger.log(f"[time {time}]: current cash amount: {self.current_cash_amount}", 'DEBUG')
                for company in shorted_stock_yesterday:
                    self.current_cash_amount += company.clear_holdings(time)
                    self.logger.log(f"[time {time}]: current cash amount: {self.current_cash_amount}", 'DEBUG')
                self.logger.log(f"[time {time}]: current cash amount after clearing all holdings: {self.current_cash_amount}", 'DEBUG')

                # sorted_index, predicted_return_list = self.get_sorted_return_list(time)
                borrow_cash_amount = self.current_cash_amount
                quota_per_stock = self.current_cash_amount / long_company_number

                self.logger.log(f"[time {time}]: longing {long_company_number} stocks, current cash amount: {self.current_cash_amount}, quota per stock: {quota_per_stock}", 'DEBUG')

                for company in longed_stock_today:
                    self.logger.log(f"[time {time}]: longing stock {company.get_stock_name()}, quota per stock: {quota_per_stock}, predicted return: {company.get_predicted_return(time)}", 'DEBUG')
                    if self.current_cash_amount > quota_per_stock:
                        company.buy_stock(quota_per_stock, time)
                        self.current_cash_amount -= quota_per_stock
                    elif abs(self.current_cash_amount - quota_per_stock) < 1:
                        company.buy_stock(self.current_cash_amount, time)
                        self.current_cash_amount = 0
                    else:
                        self.logger.log(f"[time {time}]: Can't buy stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')

                # short the last 10 stocks
                quota_per_stock = borrow_cash_amount / short_company_number
                self.logger.log(f"[time {time}]: shorting {short_company_number} stocks, quota per stock: {quota_per_stock}", 'DEBUG')

                for company in shorted_stock_today:
                    self.logger.log(f"[time {time}]: shorting stock {company.get_stock_name()}, quota per stock: {quota_per_stock}, predicted return: {company.get_predicted_return(time)}", 'DEBUG')
                    if borrow_cash_amount >= quota_per_stock:
                        credited = company.short_stock(quota_per_stock, time)
                        borrow_cash_amount -= quota_per_stock
                        self.current_cash_amount += credited
                        # self.current_cash_amount -= quota_per_stock
                    elif abs(borrow_cash_amount - quota_per_stock) < 1:
                        credited = company.short_stock(borrow_cash_amount, time)
                        self.current_cash_amount += credited
                        borrow_cash_amount = 0
                    else:
                        self.logger.log(f"[time {time}]: Can't short stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')
                longed_stock_yesterday = longed_stock_today
                shorted_stock_yesterday = shorted_stock_today

        for company in self.portfolio_list:
            self.current_cash_amount += company.clear_holdings(time)
        self.money_earn = self.current_cash_amount
        return self.calculate_return()
    

    def long_short_return(self, long_company_number=10, short_company_number=5):
        """
        Calculates the return on investment for a stock using a long-short strategy.
        simple strategies only "long" the stocks, but long-short strategy can "short" the stocks
        """

        self.reset()
        testing_period = self.portfolio_list[0].testing_period
        # current money pool is the current cash amount
        self.current_cash_amount = self.initial_capital
        current_liquid = self.initial_capital   
        # current liquid is currently how much we have minus how much we borrowed

        for time in range(testing_period - 1):
            # self.logger.log(f"[time {time}]: checking waiting time", 'DEBUG')
            # for company in self.portfolio_list:
            #     if company.check_waiting_period():
            #         self.current_cash_amount += company.clear_holdings(time)

            self.logger.log(f"[time {time}]: clearing holdings based on predicted returns", 'DEBUG')
            for company in self.portfolio_list:
                self.current_cash_amount += company.clear_holdings(time)
            self.logger.log(f"[time {time}]: current cash amount after clearing all holdings: {self.current_cash_amount}", 'DEBUG')
            # sorted_index, predicted_return_list = self.get_sorted_return_list(time)
            
            current_liquid = 0
            for company in self.portfolio_list:
                current_liquid += company.get_current_liquid(time)
            current_liquid += self.current_cash_amount
            self.logger.log(f"[time {time}]: current liquid: {current_liquid}", 'DEBUG')
            borrow_cash_amount = current_liquid


                # --------- start to long and short stocks ------------
            company_to_long = []
            for company in self.portfolio_list:
                if company.get_predicted_return(time) > 0:
                    company_to_long.append(company)
            if len(company_to_long) != 0:
                long_quota_per_stock = borrow_cash_amount / len(company_to_long)
                for company in company_to_long:
                    if self.current_cash_amount > long_quota_per_stock:
                        company.buy_stock(long_quota_per_stock, time)
                        self.current_cash_amount -= long_quota_per_stock
                    elif abs(self.current_cash_amount - long_quota_per_stock) < 1:
                        company.buy_stock(self.current_cash_amount, time)
                        self.current_cash_amount = 0
                    else:
                        self.logger.log(f"Can't buy stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')
                       
                # short the last 10 stocks
            company_to_short = []
            for company in self.portfolio_list:
                if company.get_predicted_return(time) < 0:
                    company_to_short.append(company)
            
            if len(company_to_short) != 0:
                short_quota_per_stock = borrow_cash_amount / len(company_to_short)
                for company in company_to_short:
                    if borrow_cash_amount >= short_quota_per_stock:
                        credited = company.short_stock(short_quota_per_stock, time)
                        borrow_cash_amount -= short_quota_per_stock
                        self.current_cash_amount += credited
                        # self.current_cash_amount -= quota_per_stock
                    elif abs(borrow_cash_amount - short_quota_per_stock) < 1:
                        credited = company.short_stock(borrow_cash_amount, time)
                        self.current_cash_amount += credited
                        borrow_cash_amount = 0
                    else:
                        self.logger.log(f"Can't short stock {company.get_stock_name()} at time {time} because of insufficient money", 'DEBUG')
            else:
                self.logger.log(f"[time {time}]: No Trade Today!", 'DEBUG')
                continue


        for company in self.portfolio_list:
            self.current_cash_amount += company.clear_holdings(time)
        self.money_earn = self.current_cash_amount
        return self.calculate_return()

    def buy_and_hold_return(self):
        """
        Buys equal dollar amounts of every stock on day one of the testing
        period and never trades again. Since each stock gets the same
        starting dollar amount, the portfolio's total return works out to
        the average of the 50 stocks' cumulative returns.
        """
        self.reset()
        self.current_cash_amount = self.initial_capital
        quota_per_stock = self.initial_capital / len(self.portfolio_list)
        self.logger.log(f"[time 0]: buying {len(self.portfolio_list)} stocks equally, quota per stock: {quota_per_stock}", 'DEBUG')
        for company in self.portfolio_list:
            company.buy_stock(quota_per_stock, 0)
            self.current_cash_amount -= quota_per_stock

        self.logger.log("/*-----Clearing all stocks-------*/", 'DEBUG')
        for company in self.portfolio_list:
            self.money_earn += company.clear_holdings(-2)
        return self.calculate_return()

    def daily_equal_weight_return(self):
        """
        Rebalances the portfolio back to equal weight (1/N per stock) at
        every time step. Each day all holdings are sold at that day's price
        and immediately re-split evenly across all stocks, so the daily
        portfolio return is the mean of the stocks' daily returns, and the
        total return is that daily mean return compounded over the period.
        """
        self.reset()
        self.current_cash_amount = self.initial_capital
        testing_period = self.portfolio_list[0].testing_period
        num_stocks = len(self.portfolio_list)

        for time in range(testing_period - 1):
            for company in self.portfolio_list:
                self.current_cash_amount += company.clear_holdings(time)
            self.logger.log(f"[time {time}]: current cash amount after clearing all holdings: {self.current_cash_amount}", 'DEBUG')
            self.daily_values.append(self.current_cash_amount)

            quota_per_stock = self.current_cash_amount / num_stocks
            for company in self.portfolio_list:
                company.buy_stock(quota_per_stock, time)
                self.current_cash_amount -= quota_per_stock
            self.logger.log(f"[time {time}]: rebalanced to equal weight, quota per stock: {quota_per_stock}", 'DEBUG')

        self.logger.log("/*-----Clearing all stocks-------*/", 'DEBUG')
        for company in self.portfolio_list:
            self.money_earn += company.clear_holdings(-2)
        self.daily_values.append(self.money_earn)
        return self.calculate_return()


    def overlapping_return(self, top_k=10, hold_days=10):
        """Jegadeesh-Titman overlapping portfolios; the ONE-NAS headline rule.

        Each day a new portfolio goes long the `top_k` highest-ranked names
        and short the `top_k` lowest-ranked, equal notional per name, and is
        held for `hold_days`; the one formed `hold_days` ago is closed. So
        `hold_days` portfolios are open at any time, each funded with
        1/hold_days of the capital.

        The sleeves are netted before trading rather than run side by side:
        a name held by several consecutive sleeves is held once, at the sum
        of their notionals, so it is not sold and re-bought each day. Only
        the change in that aggregate is traded, and only it pays the spread.

        Fully invested the book is long the capital and short the capital,
        i.e. twice the capital in gross notional.
        """
        n = len(self.portfolio_list)
        if 2 * top_k > n:
            raise ValueError(
                f"overlapping_return needs at least {2 * top_k} names to take "
                f"a top-{top_k} and a bottom-{top_k} side without overlap, "
                f"but the portfolio holds {n}")
        if hold_days < 1:
            raise ValueError("hold_days must be at least 1")
        self.logger.log(
            f"Overlapping: top {top_k} long / bottom {top_k} short, "
            f"held {hold_days} days, {hold_days} sleeves netted", 'INFO')
        self.reset()
        cash = self.initial_capital
        testing_period = self.portfolio_list[0].testing_period

        # one sleeve's worth of notional on each side, per name
        per_name = (self.initial_capital / hold_days) / top_k
        sleeves = []
        time = 0

        for time in range(testing_period - 1):
            sorted_index, _ = self.get_sorted_return_list(time)
            longs = [self.portfolio_list[sorted_index[-(i + 1)]]
                     for i in range(top_k)]
            shorts = [self.portfolio_list[sorted_index[i]]
                      for i in range(top_k)]
            sleeves.append((longs, shorts))
            if len(sleeves) > hold_days:
                sleeves.pop(0)          # the sleeve formed hold_days ago closes

            target = {}
            for sleeve_longs, sleeve_shorts in sleeves:
                for company in sleeve_longs:
                    target[company] = target.get(company, 0.0) + per_name
                for company in sleeve_shorts:
                    target[company] = target.get(company, 0.0) - per_name

            # every name with a position or a target, so names leaving the
            # book are closed rather than left stranded
            for company in self.portfolio_list:
                want = target.get(company, 0.0)
                if want != 0.0 or company.get_share() != 0:
                    cash += company.rebalance_to(want, time)

        for company in self.portfolio_list:
            cash += company.clear_holdings(time)
        self.money_earn = cash
        return self.calculate_return()

    def banded_return(self, entry_rank=10, exit_rank=35):
        """Novy-Marx/Velikov buy-hold spread: enter at a rank, leave at a wider one.

        A name is bought when its predicted rank reaches the top
        `entry_rank` and held until that rank falls past `exit_rank`;
        symmetrically on the short side. Between the two thresholds a name
        is neither bought nor sold, which is the whole point -- it removes
        the trading caused by small day-to-day rank changes at the boundary.

        Held names split the capital equally on each side, so the book is
        long the capital and short the capital when both sides are occupied.
        """
        n = len(self.portfolio_list)
        if exit_rank < entry_rank:
            raise ValueError(
                f"banded_return needs exit_rank >= entry_rank, got "
                f"{exit_rank} < {entry_rank}; the exit threshold is the wider "
                "of the two")
        if 2 * entry_rank > n:
            raise ValueError(
                f"banded_return needs at least {2 * entry_rank} names for the "
                f"long and short entry bands to be disjoint at entry_rank="
                f"{entry_rank}, but the portfolio holds {n}")
        if exit_rank > n:
            raise ValueError(
                f"banded_return needs exit_rank ({exit_rank}) to be within "
                f"the panel ({n} names), otherwise a name entered long can "
                "never fall past it and is held forever")
        self.logger.log(
            f"Banded: enter at rank {entry_rank}, exit past rank {exit_rank}",
            'INFO')
        self.reset()
        cash = self.initial_capital
        testing_period = self.portfolio_list[0].testing_period
        n = len(self.portfolio_list)
        held_long, held_short = set(), set()
        time = 0

        for time in range(testing_period - 1):
            sorted_index, _ = self.get_sorted_return_list(time)
            # rank 1 = highest predicted return
            rank = {}
            for position, idx in enumerate(sorted_index):
                rank[self.portfolio_list[idx]] = n - position

            for company in self.portfolio_list:
                r = rank[company]
                if r <= entry_rank:
                    held_long.add(company)
                    held_short.discard(company)
                elif r > exit_rank:
                    held_long.discard(company)
                if r > n - entry_rank:
                    held_short.add(company)
                    held_long.discard(company)
                elif r <= n - exit_rank:
                    held_short.discard(company)

            target = {}
            if held_long:
                each = self.initial_capital / len(held_long)
                for company in held_long:
                    target[company] = each
            if held_short:
                each = self.initial_capital / len(held_short)
                for company in held_short:
                    target[company] = -each

            for company in self.portfolio_list:
                want = target.get(company, 0.0)
                if want != 0.0 or company.get_share() != 0:
                    cash += company.rebalance_to(want, time)

        for company in self.portfolio_list:
            cash += company.clear_holdings(time)
        self.money_earn = cash
        return self.calculate_return()

    def conditional_long_short_return(self, long_company_number=10,
                                      short_company_number=10):
        """Algorithm 1 of the ONE-NAS paper: daily long-short behind a sign gate.

        Identical to daily_long_short_return except that it rebalances only
        on days when the model separates the cross-section by sign -- the
        `long_company_number`-th ranked prediction must be positive and the
        `short_company_number`-th from the bottom must be negative. On any
        other day the existing positions are held.

        daily_long_short_return has these two conditions in the source but
        commented out, so it trades unconditionally. This is kept separate
        rather than uncommenting them, so results already produced with that
        method stay reproducible.

        Note the gate can only bind on a signed prediction. On rank-normal
        predictions, which is what the ONE-NAS panels carry, roughly half
        the cross-section is negative by construction and the gate fires
        every day, reducing this to a plain daily rebalance.
        """
        n = len(self.portfolio_list)
        if long_company_number + short_company_number > n:
            raise ValueError(
                f"conditional_long_short_return needs at least "
                f"{long_company_number + short_company_number} names for "
                "disjoint long and short sides, but the portfolio holds "
                f"{n}; on a smaller panel the sign gate cannot fire and the "
                "rule silently returns a flat 0%")
        self.logger.log(
            f"Conditional long-short: {long_company_number} long / "
            f"{short_company_number} short, gated on sign separation", 'INFO')
        self.reset()
        cash = self.initial_capital
        testing_period = self.portfolio_list[0].testing_period
        time = 0

        for time in range(testing_period - 1):
            sorted_index, predicted = self.get_sorted_return_list(time)
            kth_long = predicted[sorted_index[-long_company_number]]
            kth_short = predicted[sorted_index[short_company_number - 1]]
            if not (kth_long > 0 and kth_short < 0):
                self.logger.log(
                    f"[time {time}]: gate not satisfied "
                    f"({kth_long:+.6f}, {kth_short:+.6f}), holding", 'DEBUG')
                continue

            for company in self.portfolio_list:
                cash += company.clear_holdings(time)

            quota_long = cash / long_company_number
            quota_short = cash / short_company_number
            for i in range(long_company_number):
                company = self.portfolio_list[sorted_index[-(i + 1)]]
                company.buy_stock(quota_long, time)
                cash -= quota_long
            for i in range(short_company_number):
                company = self.portfolio_list[sorted_index[i]]
                cash += company.short_stock(quota_short, time)

        for company in self.portfolio_list:
            cash += company.clear_holdings(time)
        self.money_earn = cash
        return self.calculate_return()

    def short_term_reversal_return(self, long_company_number=10, short_company_number=10, lookback=1):
        """
        Short-term reversal strategy. The signal for each stock is the
        negative of its own past `lookback`-day return (1-day or 5-day),
        so the biggest recent losers rank highest and the biggest recent
        winners rank lowest. Each day, longs the `long_company_number`
        biggest recent losers and shorts the `short_company_number`
        biggest recent winners, rebalancing daily.
        """
        self.logger.log(f"Short-term reversal ({lookback}-day): longing {long_company_number} biggest losers, shorting {short_company_number} biggest winners", 'INFO')
        self.reset()
        self.current_cash_amount = self.initial_capital
        testing_period = self.portfolio_list[0].testing_period

        longed_stock_yesterday = []
        shorted_stock_yesterday = []
        time = 0
        for time in range(testing_period - 1):
            if time < lookback:
                self.logger.log(f"[time {time}]: not enough history for {lookback}-day lookback, skipping", 'DEBUG')
                continue

            self.logger.log(f"[time {time}]: clearing all holdings", 'DEBUG')
            for company in longed_stock_yesterday:
                self.current_cash_amount += company.clear_holdings(time)
            for company in shorted_stock_yesterday:
                self.current_cash_amount += company.clear_holdings(time)
            self.logger.log(f"[time {time}]: current cash amount after clearing all holdings: {self.current_cash_amount}", 'DEBUG')

            sorted_index, signal_list = self.get_sorted_reversal_signal_list(time, lookback)

            longed_stock_today = []
            for i in range(long_company_number):
                index = -(i + 1)  # highest signal = biggest recent losers
                longed_stock_today.append(self.portfolio_list[sorted_index[index]])

            shorted_stock_today = []
            for i in range(short_company_number):
                shorted_stock_today.append(self.portfolio_list[sorted_index[i]])  # lowest signal = biggest recent winners

            borrow_cash_amount = self.current_cash_amount
            long_quota_per_stock = self.current_cash_amount / long_company_number
            self.logger.log(f"[time {time}]: longing {long_company_number} stocks, quota per stock: {long_quota_per_stock}", 'DEBUG')
            for company in longed_stock_today:
                if self.current_cash_amount > long_quota_per_stock:
                    company.buy_stock(long_quota_per_stock, time)
                    self.current_cash_amount -= long_quota_per_stock
                elif abs(self.current_cash_amount - long_quota_per_stock) < 1:
                    company.buy_stock(self.current_cash_amount, time)
                    self.current_cash_amount = 0
                else:
                    self.logger.log(f"[time {time}]: Can't buy stock {company.get_stock_name()} because of insufficient money", 'DEBUG')

            short_quota_per_stock = borrow_cash_amount / short_company_number
            self.logger.log(f"[time {time}]: shorting {short_company_number} stocks, quota per stock: {short_quota_per_stock}", 'DEBUG')
            for company in shorted_stock_today:
                if borrow_cash_amount >= short_quota_per_stock:
                    credited = company.short_stock(short_quota_per_stock, time)
                    borrow_cash_amount -= short_quota_per_stock
                    self.current_cash_amount += credited
                elif abs(borrow_cash_amount - short_quota_per_stock) < 1:
                    credited = company.short_stock(borrow_cash_amount, time)
                    self.current_cash_amount += credited
                    borrow_cash_amount = 0
                else:
                    self.logger.log(f"[time {time}]: Can't short stock {company.get_stock_name()} because of insufficient money", 'DEBUG')

            longed_stock_yesterday = longed_stock_today
            shorted_stock_yesterday = shorted_stock_today

        for company in self.portfolio_list:
            self.current_cash_amount += company.clear_holdings(time)
        self.money_earn = self.current_cash_amount
        return self.calculate_return()
