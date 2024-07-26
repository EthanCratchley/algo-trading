import backtrader as bt
import numpy as np
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sec_edgar_downloader import Downloader
import re
from bs4 import BeautifulSoup
import base64
import time

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
        if not self.rets:
            return float('nan')  # Return NaN if no trades
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
            drawdown = self.strategy.analyzers.drawdown.get_analysis().max.drawdown
            self.drawdowns.append(drawdown)

    def get_analysis(self):
        if not self.returns or not self.drawdowns:
            return float('nan')  # Return NaN if no trades or no drawdowns
        annual_return = np.mean(self.returns) * 252  # Assuming 252 trading days in a year
        max_drawdown = max(self.drawdowns) if self.drawdowns else 1  # Avoid empty sequence error
        calmar_ratio = annual_return / max_drawdown if max_drawdown else 1
        return calmar_ratio

# Function to fetch stock price data with retries
def fetch_stock_price_data(ticker, start_date, end_date, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                stock_data.reset_index(inplace=True)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                return stock_data
            else:
                raise Exception("Empty data received from Yahoo Finance")
        except Exception as e:
            print(f"Error fetching stock price data for {ticker} (Attempt {attempt+1}/{retries}): {e}")
            attempt += 1
            time.sleep(2 ** attempt)  # Exponential backoff
    return pd.DataFrame()

# Function to decode PEM content to extract the actual 13F content
def decode_pem_content(content):
    pattern = re.compile(r'-----BEGIN PRIVACY-ENHANCED MESSAGE-----.*?-----END PRIVACY-ENHANCED MESSAGE-----', re.DOTALL)
    match = pattern.search(content)
    if not match:
        print("No PEM encoded content found.")
        return None

    pem_content = match.group(0)
    pem_lines = pem_content.splitlines()
    base64_content = ''.join(pem_lines[4:-2])
    decoded_content = base64.b64decode(base64_content).decode('utf-8')
    
    return decoded_content

# Function to parse 13F filings
def parse_13f(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        decoded_content = decode_pem_content(content)
        if not decoded_content:
            return pd.DataFrame()
        
        soup = BeautifulSoup(decoded_content, 'html.parser')
        info_tables = soup.find_all('infoTable')
        data = []
        for table in info_tables:
            row = {}
            for field in table.find_all():
                row[field.name] = field.text
            data.append(row)
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error parsing 13F file {file_path}: {e}")
        return pd.DataFrame()

# Main function to process 13F filings and fetch stock prices
def process_13f_files(folder_path, ticker, start_date, end_date):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    combined_data = pd.DataFrame()
    for file_path in all_files:
        print(f"Processing file: {file_path}")
        df = parse_13f(file_path)
        if not df.empty:
            combined_data = pd.concat([combined_data, df], ignore_index=True)
    
    if combined_data.empty:
        print("No data found in 13F filings.")
        return

    stock_data = fetch_stock_price_data(ticker, start_date, end_date)
    if stock_data.empty:
        print("No stock data found.")
        return

    # Further processing can be done here based on the fetched data
    print(combined_data.head())
    print(stock_data.head())

def fetch_hedge_fund_data(cik, start_date, end_date):
    company_name = 'WMB'  # Replace with your company name
    email_address = 'buzz@gmail.com'  # Replace with your email address
    # Initialize the downloader
    dl = Downloader(company_name, email_address)
    dl.get("13F-HR", cik, start_date, end_date)
    folder_path = os.path.join("sec-edgar-filings", cik, "13F-HR")
    hedge_fund_data = process_13f_files(folder_path, None, start_date, end_date)
    return hedge_fund_data

if __name__ == '__main__':
    # Create paths for securities and performance folders
    securities_folder = 'securities'
    performance_folder = 'performance'
    os.makedirs(performance_folder, exist_ok=True)

    # Define the date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years ago

    # Define the CIK for Bridgewater Associates
    cik = '0001350694'
    ticker = 'GOOGL'

    hedge_fund_data = fetch_hedge_fund_data(cik, start_date, end_date)

    if hedge_fund_data.empty:
        print("No hedge fund data found. Exiting.")
    else:
        # Fetch historical stock price data
        stock_price_data = fetch_stock_price_data(ticker, start_date, end_date)

        # Save stock price data to CSV
        os.makedirs(securities_folder, exist_ok=True)
        stock_price_data.to_csv(os.path.join(securities_folder, f'{ticker}.csv'), index=False)

        # Load data into backtrader
        data = bt.feeds.GenericCSVData(
            dataname=os.path.join(securities_folder, f'{ticker}.csv'),
            dtformat='%Y-%m-%d',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=6,
            openinterest=-1  # -1 indicates no open interest column in CSV
        )

        # Initialize Backtrader cerebro engine
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(HedgeFundTradingStrategy, hedge_fund_data=hedge_fund_data)
        cerebro.addsizer(KellyCriterionSizer)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(SortinoRatio, _name="sortino")
        cerebro.addanalyzer(CalmarRatio, _name="calmar")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        # Run Backtrader
        initial_value = 10000
        cerebro.broker.set_cash(initial_value)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

        results = cerebro.run()  # Use all available CPUs
        strat = results[0]

        # Output strategy metrics
        print(f"Starting Portfolio Value: {initial_value:.2f}")
        print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")
        print(f"Sharpe Ratio: {strat.analyzers.sharpe.get_analysis()['sharperatio']}")
        print(f"Sortino Ratio: {strat.analyzers.sortino.get_analysis()}")
        print(f"Calmar Ratio: {strat.analyzers.calmar.get_analysis()}")
        print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")
        print(f"Total Trades: {strat.total_trades}")
        print(f"Total Wins: {strat.total_wins}")
        print(f"Total Losses: {strat.total_losses}")
        print(f"Win Rate: {strat.win_rate:.2%}")
        print(f"Average Payoff Ratio: {strat.average_payoff:.2f}")
        print(f"Total Commission Costs: {strat.total_commissions:.2f}")
