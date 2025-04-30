import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the tickers
tickers = ['AMD', 'XLE', '^GSPC']  # Changed tickers, added S&P500 for alpha calculation

# Fetch historical data
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")['Close']  # Get only closing prices

# Ensure that data is not empty
if data.empty:
    print("Error: Could not retrieve historical data. Please check the tickers and date range.")
    exit()

# Define strategy parameters
BB_PERIOD = 20  # Bollinger Band period
BB_STD = 2.0  # Number of standard deviations for Bollinger Bands
POSITION_SIZE_PCT = 0.15  # initial position, changed to 15%
RISK_FREE_RATE = 0.02  # Risk-free rate for alpha calculation

# Calculate Bollinger Bands
def calculate_bollinger_bands(df, period, std, ticker):  # Added ticker argument
    """
    Calculates Bollinger Bands for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column.
        period (int): Period for the moving average.
        std (int): Number of standard deviations.
        ticker (str): the ticker to calculate the bollinger bands on

    Returns:
        pd.DataFrame: DataFrame with 'BB_UPPER', 'BB_MIDDLE', and 'BB_LOWER' columns added.
    """
    df[f'{ticker}_BB_MIDDLE'] = df[ticker].rolling(window=period).mean()
    df[f'{ticker}_BB_STD'] = df[ticker].rolling(window=period).std()
    df[f'{ticker}_BB_UPPER'] = df[f'{ticker}_BB_MIDDLE'] + (std * df[f'{ticker}_BB_STD'])
    df[f'{ticker}_BB_LOWER'] = df[f'{ticker}_BB_MIDDLE'] - (std * df[f'{ticker}_BB_STD'])
    return df


data = calculate_bollinger_bands(data, BB_PERIOD, BB_STD, 'AMD') #calculate bollinger bands on AMD
data = calculate_bollinger_bands(data, BB_PERIOD, BB_STD, 'XLE') #calculate bollinger bands on XLE
data['SPY'] = yf.download('^GSPC', start="2020-01-01", end="2024-01-01")['Close']

# Initialize variables
capital = 100000  # Starting capital
position = 0  # 1 for long, 0 for no position
trades = []  # List to store trade details
initial_capital = capital

# Create a copy to add columns
data_with_signals = data.copy()
data_with_signals['Signal'] = 0 # 0 for hold, 1 for buy, -1 for sell

# Iterate through the data to generate signals and trade
for i in range(BB_PERIOD, len(data)):  # start at bollinger band period
    # Get current prices
    amd_price = data['AMD'].iloc[i]
    xle_price = data['XLE'].iloc[i]

    # Get Bollinger Band values
    amd_bb_lower = data['AMD_BB_LOWER'].iloc[i]
    amd_bb_middle = data['AMD_BB_MIDDLE'].iloc[i]
    amd_bb_upper = data['AMD_BB_UPPER'].iloc[i]

    xle_bb_lower = data['XLE_BB_LOWER'].iloc[i]
    xle_bb_middle = data['XLE_BB_MIDDLE'].iloc[i]
    xle_bb_upper = data['XLE_BB_UPPER'].iloc[i]


    # Check for buy signal
    if position == 0 and amd_price < amd_bb_lower and xle_price < xle_bb_lower: # change buy signal to be below the lower band of XLE as well.
        # Calculate position size, scale with AMD volatility
        position_size = capital * POSITION_SIZE_PCT  # size relative to capital
        units_to_buy = position_size / xle_price
        position = 1
        trades.append({
            'type': 'buy',
            'date': data.index[i],
            'amd_price': amd_price,
            'xle_price': xle_price,
            'units': units_to_buy,
            'capital': capital
        })
        capital -= units_to_buy * xle_price
        data_with_signals.loc[data.index[i], 'Signal'] = 1  # Store signal for analysis

    # Check for sell signal
    elif position == 1 and (amd_price > amd_bb_upper or xle_price > xle_bb_upper): # Change sell signal to be above the upper band
        units_to_sell = units_to_buy
        position = 0
        trades.append({
            'type': 'sell',
            'date': data.index[i],
            'amd_price': amd_price,
            'xle_price': xle_price,
            'units': units_to_sell,
            'capital': capital
        })
        capital += units_to_sell * xle_price
        data_with_signals.loc[data.index[i], 'Signal'] = -1  # Store signal

# Calculate performance metrics
final_capital = capital
total_return = (final_capital - initial_capital) / initial_capital
number_of_trades = len(trades) / 2  # each pair is a trade

# Convert trades to a DataFrame for easier analysis
trades_df = pd.DataFrame(trades)

# Calculate daily returns for Sharpe Ratio
portfolio_value = [initial_capital] # list of portfolio values
cash = initial_capital
position = 0
units = 0
for i in range(BB_PERIOD, len(data)):
    if data_with_signals['Signal'].iloc[i] == 1: # Buy
        position_size = portfolio_value[-1] * POSITION_SIZE_PCT
        units = position_size / data['XLE'].iloc[i]
        cash -= units * data['XLE'].iloc[i]
        position = 1
    elif data_with_signals['Signal'].iloc[i] == -1: # Sell
        cash += units * data['XLE'].iloc[i]
        position = 0
        units = 0
    if position == 1:
        portfolio_value.append(cash + units * data['XLE'].iloc[i])
    else:
        portfolio_value.append(cash)

daily_returns = pd.Series([0.0] * len(portfolio_value))
for i in range(1, len(portfolio_value)):
    daily_returns[i] = (portfolio_value[i] - portfolio_value[i-1]) / portfolio_value[i-1]
annualized_return = np.mean(daily_returns) * 252
annualized_std = np.std(daily_returns) * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0

# # Calculate Alpha
# market_returns = yf.download('^GSPC', start="2020-01-01", end="2024-01-01")['Close'].pct_change().dropna()
# strategy_returns = pd.Series(daily_returns).dropna()

# # Align lengths
# min_length = min(len(market_returns), len(strategy_returns))
# market_returns = market_returns[-min_length:]
# strategy_returns = strategy_returns[-min_length:]
# if len(market_returns) != len(strategy_returns):
#     raise ValueError("Market and Strategy returns must have the same length")

# covariance_matrix = np.cov(strategy_returns, market_returns)
# beta = covariance_matrix[0, 1] / np.var(market_returns)
# expected_return = RISK_FREE_RATE + beta * (market_returns.mean() * 252 - RISK_FREE_RATE)
# alpha = (annualized_return - expected_return) * 100 # in %

# Print the results
print(f"Starting Capital: ${initial_capital:.2f}")
print(f"Final Capital: ${final_capital:.2f}")
print(f"Total Return: {total_return:.2%}")
print(f"Number of Trades: {number_of_trades}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
# print(f"Alpha: {alpha:.2f}%")

print(trades_df)

# Save the trades:
trades_df.to_csv('trades.csv')
data_with_signals.to_csv('data_with_signals.csv')

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['AMD'], label='AMD Price', color='blue') # Changed back to price
plt.plot(data.index, data['AMD_BB_UPPER'], label='AMD Upper Band', color='purple', linestyle='--')
plt.plot(data.index, data['AMD_BB_LOWER'], label='AMD Lower Band', color='purple', linestyle='--')

# Plot buy and sell signals
buy_signals = data_with_signals[data_with_signals['Signal'] == 1]
sell_signals = data_with_signals[data_with_signals['Signal'] == -1]

plt.scatter(buy_signals.index, data['AMD'].loc[buy_signals.index], marker='^', color='green', label='Buy Signal', s=100) #plot at price
plt.scatter(sell_signals.index, data['AMD'].loc[sell_signals.index], marker='v', color='red', label='Sell Signal', s=100) #plot at price

plt.title('AMD Price with Bollinger Bands and Trade Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.figure(figsize=(15, 7))
plt.plot(data.index, data['XLE'], label='XLE Price', color='blue') # Changed back to price
plt.plot(data.index, data['XLE_BB_UPPER'], label='XLE Upper Band', color='purple', linestyle='--')
plt.plot(data.index, data['XLE_BB_LOWER'], label='XLE Lower Band', color='purple', linestyle='--')

# Plot buy and sell signals
buy_signals = data_with_signals[data_with_signals['Signal'] == 1]
sell_signals = data_with_signals[data_with_signals['Signal'] == -1]

plt.scatter(buy_signals.index, data['XLE'].loc[buy_signals.index], marker='^', color='green', label='Buy Signal', s=100) #plot at price
plt.scatter(sell_signals.index, data['XLE'].loc[sell_signals.index], marker='v', color='red', label='Sell Signal', s=100) #plot at price

plt.title('XLE Price with Bollinger Bands and Trade Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
