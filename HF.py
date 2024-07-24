import backtrader as bt
import numpy as np
import csv
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sec_edgar_downloader import Downloader

class KellyCriterionSizer(bt.Sizer):
    def __init__(self):
        self.win_rate = 0.55  # Initial estimated win rate
        self.payoff_ratio = 1.55  # Initial estimated payoff ratio
        self.kelly_fraction = self.calculate_kelly_fraction()

    def calculate_kelly_fraction(self):
        # Kelly formula
        W = self.win_rate
        R = self.payoff_ratio
        kelly_fraction = W - (1 - W) / R
        return max(0, min(kelly_fraction, 1))  # Bound between 0 and 1

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Calculate position size using Kelly Criterion
        size = (cash * self.kelly_fraction) // data.close[0]
        return size

class HedgeFundTradingStrategy(bt.Strategy):
    params = (
        ('lookback', 50),
        ('exit_after', 15),  # Exit after x bars
        ('stop_loss', 0.02),  # Stop loss as a percentage
        ('take_profit', 0.05),  # Take profit as a percentage
        ('hedge_fund_data', None)
    )

    def __init__(self):
        self.buy_order = None
        self.sell_order = None
        self.bar_executed = 0  # To track when the order was executed
        self.total_trades = 0
        self.total_commissions = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_profit = 0
        self.total_loss = 0
        self.initial_value = 0
        self.final_value = 0

    def notify_trade(self, trade):
        if trade.justopened:
            self.total_trades += 1
        if trade.isclosed:
            self.total_commissions += trade.commission
            pnl = trade.pnlcomm
            if pnl > 0:
                self.total_wins += 1
                self.total_profit += pnl
            else:
                self.total_losses += 1
                self.total_loss += abs(pnl)

    def next(self):
        current_date = self.data.datetime.date(0)
        for _, transaction in self.params.hedge_fund_data.iterrows():
            if transaction['date'].date() == current_date:
                size = self.broker.get_cash() * 0.1 // self.data.close[0]
                if transaction['type'] == 'BUY':
                    self.buy(size=size)
                    self.bar_executed = len(self)
                    self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss)
                    self.take_profit_price = self.data.close[0] * (1 + self.params.take_profit)
                elif transaction['type'] == 'SELL':
                    self.sell(size=size)
                    self.bar_executed = len(self)
                    self.stop_loss_price = self.data.close[0] * (1 + self.params.stop_loss)
                    self.take_profit_price = self.data.close[0] * (1 - self.params.take_profit)

        if self.position:
            if self.position.size > 0:  # Long position
                if self.data.close[0] <= self.stop_loss_price or self.data.close[0] >= self.take_profit_price:
                    self.close()
                elif len(self) >= self.bar_executed + self.params.exit_after:
                    self.close()
            elif self.position.size < 0:  # Short position
                if self.data.close[0] >= self.stop_loss_price or self.data.close[0] <= self.take_profit_price:
                    self.close()
                elif len(self) >= self.bar_executed + self.params.exit_after:
                    self.close()

    def start(self):
        self.initial_value = self.broker.getvalue()

    def stop(self):
        self.final_value = self.broker.getvalue()
        self.win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        self.average_payoff = (self.total_profit / self.total_wins) / (self.total_loss / self.total_losses) if self.total_wins > 0 and self.total_losses > 0 else 0

class SortinoRatio(bt.Analyzer):
    def create_analysis(self):
        self.rets = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.rets.append(trade.pnlcomm / self.strategy.broker.getvalue())

    def get_analysis(self):
        neg_rets = [r for r in self.rets if r < 0]
        sortino_ratio = np.mean(self.rets) / (np.std(neg_rets) if neg_rets else 1)
        return sortino_ratio

class CalmarRatio(bt.Analyzer):
    def __init__(self):
        self.drawdowns = []
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.returns.append(trade.pnlcomm / self.strategy.broker.getvalue())
            self.drawdowns.append(self.strategy.analyzers.drawdown.get_analysis().max.drawdown)

    def get_analysis(self):
        annual_return = np.mean(self.returns) * 252  # Assuming 252 trading days in a year
        max_drawdown = max(self.drawdowns)
        calmar_ratio = annual_return / max_drawdown if max_drawdown else 1
        return calmar_ratio

def fetch_hedge_fund_data(cik, start_date, end_date):
    dl = Downloader(company_name="WMB", email_address="buzz@gmail.com")
    try:
        dl.get("13F-HR", cik)
        filings_path = os.path.join("sec-edgar-filings", cik, "13F-HR")
        all_data = []
        for root, dirs, files in os.walk(filings_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                        data = parse_13f(content)
                        all_data.append(data)
        hedge_fund_data = pd.concat(all_data)
        hedge_fund_data['date'] = pd.to_datetime(hedge_fund_data['date'], errors='coerce')
        hedge_fund_data = hedge_fund_data.dropna(subset=['date'])
        hedge_fund_data = hedge_fund_data[(hedge_fund_data['date'] >= pd.to_datetime(start_date)) & (hedge_fund_data['date'] <= pd.to_datetime(end_date))]
        return hedge_fund_data
    except Exception as e:
        print(f"Error fetching hedge fund data: {e}")
        return None

def parse_13f(content):
    # Custom parsing logic here
    # Return a DataFrame with columns: date, type (BUY/SELL), ticker
    data = {
        'date': [],
        'type': [],
        'ticker': []
    }
    # Parsing logic...
    return pd.DataFrame(data)

def fetch_stock_price_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

if __name__ == '__main__':
    # Create paths for securities and performance folders
    securities_folder = 'securities'
    performance_folder = 'performance'
    os.makedirs(performance_folder, exist_ok=True)

    # Define the date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Define the CIK and ticker symbol
    cik = '0001350694'  # Bridgewater Associates
    ticker = 'GOOGL'  # Example: Alphabet Inc.

    # Fetch hedge fund transaction data
    hedge_fund_data = fetch_hedge_fund_data(cik, start_date, end_date)

    if hedge_fund_data is not None:
        # Fetch historical stock price data
        stock_price_data = fetch_stock_price_data(ticker, start_date, end_date)

        # Save stock price data to CSV
        stock_price_data.to_csv(os.path.join(securities_folder, f'{ticker}.csv'))

        # Load data into backtrader
        data = bt.feeds.GenericCSVData(
            dataname=os.path.join(securities_folder, f'{ticker}.csv'),
            dtformat=('%Y-%m-%d'),
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=6,
            openinterest=-1  # -1 indicates no open interest column in CSV
        )

        # Create Cerebro engine
        cerebro = bt.Cerebro(optreturn=False)

        # Add data to Cerebro
        cerebro.adddata(data)

        # Add strategy to Cerebro with optimization parameters
        cerebro.optstrategy(
            HedgeFundTradingStrategy,
            lookback=range(5, 101, 5),
            exit_after=range(5, 21, 2),
            stop_loss=[0.01, 0.02, 0.03],
            take_profit=[0.03, 0.05, 0.07],
            hedge_fund_data=hedge_fund_data
        )

        # Add Kelly Criterion Sizer to Cerebro
        cerebro.addsizer(KellyCriterionSizer)

        # Set our desired cash start
        cerebro.broker.set_cash(10000)

        # Set the commission
        cerebro.broker.setcommission(commission=0.002)

        # Add analyzers for performance metrics
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(SortinoRatio, _name='sortino')
        cerebro.addanalyzer(CalmarRatio, _name='calmar')

        # CSV setup
        csv_file = os.path.join(performance_folder, 'hedge_fund_strategy_performance.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Lookback', 'Exit After', 'Stop Loss', 'Take Profit', 'Starting Value', 'Ending Value', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Annualized Return', 'Cumulative Return', 'Total Commission Costs', 'Total Trades', 'Win Rate', 'Average Trade Payoff Ratio'])

        # Run the optimization
        results = cerebro.run()  # Use all available CPUs

        # Track the best strategy
        best_strat = None
        best_sharpe = -np.inf  # Use negative infinity to ensure any valid Sharpe ratio will be better

        for strat in results:
            sharpe_ratio = strat[0].analyzers.sharpe.get_analysis()
            if sharpe_ratio['sharperatio'] is not None and sharpe_ratio['sharperatio'] > best_sharpe:
                best_sharpe = sharpe_ratio['sharperatio']
                best_strat = strat[0]

            # Write each strategy's performance to CSV
            drawdown = strat[0].analyzers.drawdown.get_analysis()
            returns = strat[0].analyzers.returns.get_analysis()
            sortino_ratio = strat[0].analyzers.sortino.get_analysis()
            calmar_ratio = strat[0].analyzers.calmar.get_analysis()
            initial_value = strat[0].initial_value
            final_value = strat[0].final_value

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    strat[0].params.lookback,
                    strat[0].params.exit_after,
                    strat[0].params.stop_loss,
                    strat[0].params.take_profit,
                    initial_value,
                    final_value,
                    sharpe_ratio['sharperatio'],
                    sortino_ratio,
                    calmar_ratio,
                    drawdown.max.drawdown,
                    returns['rtot'],
                    returns['rnorm'],
                    (final_value - initial_value) / initial_value,
                    strat[0].total_commissions,
                    strat[0].total_trades,
                    strat[0].win_rate,
                    strat[0].average_payoff
                ])

        # Print out the best strategy's result
        if best_strat is not None:
            sharpe_ratio = best_strat.analyzers.sharpe.get_analysis()
            drawdown = best_strat.analyzers.drawdown.get_analysis()
            returns = best_strat.analyzers.returns.get_analysis()
            sortino_ratio = best_strat.analyzers.sortino.get_analysis()
            calmar_ratio = best_strat.analyzers.calmar.get_analysis()
            
            print('---------------------------------------------------------------')
            print(f'Best Strategy Parameters - Lookback: {best_strat.params.lookback}, Exit After: {best_strat.params.exit_after}, Stop Loss: {best_strat.params.stop_loss}, Take Profit: {best_strat.params.take_profit}')
            print('Starting Portfolio Value: %.2f' % best_strat.initial_value)
            print('Ending Portfolio Value: %.2f' % best_strat.final_value)
            print('Sharpe Ratio:', sharpe_ratio['sharperatio'])
            print('Sortino Ratio:', sortino_ratio)
            print('Calmar Ratio:', calmar_ratio)
            print('Max Drawdown:', drawdown.max.drawdown)
            print('Total Return:', returns['rtot'])
            print('Annualized Return:', returns['rnorm'])
            print('Cumulative Return:', (best_strat.final_value - best_strat.initial_value) / best_strat.initial_value)
            print('Total Commission Costs:', best_strat.total_commissions)
            print('Total Trades:', best_strat.total_trades)
            print('Win Rate:', best_strat.win_rate)
            print('Average Trade Payoff Ratio:', best_strat.average_payoff)
            print('---------------------------------------------------------------')

            # Create a new Cerebro instance for plotting the best strategy
            cerebro_best = bt.Cerebro()

            # Add data to Cerebro
            cerebro_best.adddata(data)

            # Add the best strategy with the best parameters to Cerebro
            cerebro_best.addstrategy(
                HedgeFundTradingStrategy,
                lookback=best_strat.params.lookback,
                exit_after=best_strat.params.exit_after,
                stop_loss=best_strat.params.stop_loss,
                take_profit=best_strat.params.take_profit,
                hedge_fund_data=hedge_fund_data
            )

            # Add Kelly Criterion Sizer to Cerebro
            cerebro_best.addsizer(KellyCriterionSizer)

            # Set our desired cash start
            cerebro_best.broker.set_cash(10000)

            # Set the commission
            cerebro_best.broker.setcommission(commission=0.002)

            # Add analyzers for performance metrics
            cerebro_best.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro_best.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro_best.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro_best.addanalyzer(SortinoRatio, _name='sortino')
            cerebro_best.addanalyzer(CalmarRatio, _name='calmar')

            # Run the best strategy
            cerebro_best.run()

            # Plot the result for the best strategy
            cerebro_best.plot(volume=False)
        else:
            print("No valid strategy found.")
    else:
        print("No hedge fund data found. Exiting.")
