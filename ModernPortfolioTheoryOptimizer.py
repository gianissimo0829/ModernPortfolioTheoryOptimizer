import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optmz

# on ave there are 252 trading days in a year
NUM_TRADING_DAYS = 252

# we will generate random diff weights
NUM_PORTFOLIOS = 10000

# lets get some stocks we are going to handle
stocks = ["AAPL", "WMT", "TSLA", "GE", "AMZN", "NVDA"]

# lets now get some historical data (define start and end dates)
start_date = '2020-01-01'
end_date = '2025-01-01'

def download_data():
    # name of stock (key) - stock values (2020 - 2025) as the values
    stock_data = {}

    for stock in stocks:
        # get only closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]

    return pd.DataFrame(stock_data)

# let's make a way to visualize the dataset
def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

# lets calculcate the returns
def calculate_return(data):
    # we chose log in order to use NORMALIZATION - to measure all variables in comparable metric
    # the formula for the return is ln(S(t+1)/S(t))
    # you can also read this as log of tomorrow divided by today
    # data.shift(1) just moves all contents of a series/frame by 1
    # 12345 becomes
    #  12345 
    # effectively becoming a time shift at curIndex of data
    log_return = np.log(data/data.shift(1))
    # delete the first row as it will only give us NaN 
    return log_return[1:]

def show_statistics(returns):
    # instead of daily metrics, we are after annual metrics
    # mean of annual return
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weights):
    # we are after the annual return 
    # returns.mean(): Finds the average daily return of each stock.
    # returns.mean() * weights: Adjusts this by how much money you put in each stock.
    # In simple terms: "If the past returns continue, this is how much profit the portfolio could make in a year."
    portfolio_return = np.sum(returns.mean()*weights) * NUM_TRADING_DAYS
    # returns.cov(): Calculates how the stocks move together (risk relationship).
    # returns.cov() * NUM_TRADING_DAYS: Converts daily risk into annual risk.
    # entire inner equation for portfolio_volatility:Computes the overall portfolio risk, considering each stock’s weight.
    # sqrt the equation to get the standard deviation (volatility)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights)))

    print("Expected porfolio mean (return): ", portfolio_return)
    print("Expected porfolio volatility (stdev): ", portfolio_volatility)

def show_portfolios(returns, volatitilies):
    """
    This function creates a scatter plot of different portfolios.
    Each portfolio is represented as a point, where:

    - X-axis: Expected Volatility (Risk)
    - Y-axis: Expected Return (Profit)
    - Color: Sharpe Ratio (Risk-Adjusted Return)
    """
    # Scatter plot where each point represents a portfolio
    plt.figure(figsize=(10, 6)) 
    plt.scatter(volatitilies, returns, c=returns/volatitilies, marker ='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()

def generate_portfolios(returns):
    portfolio_means = [] # Stores portfolio returns (profits)
    portfolio_risks = [] # Stores portfolio volatilities (risk)
    portfolio_weights = [] # Stores weight distributions of stocks in each portfolio

    # Generate multiple random portfolios
    for _ in range(NUM_PORTFOLIOS): 
        w = np.random.random(len(stocks)) # Generate random weights for each stock
        w /= np.sum(w) # Normalize so that total weight sums to 1 (100% invested)
        portfolio_weights.append(w) # Store the generated weights
        # Calculate expected return of portfolio
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        # Calculate portfolio risk (volatility)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()*NUM_TRADING_DAYS, w))))

    # Convert lists to NumPy arrays for easier manipulation
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def statistics(weights, returns):
    """
    This function calculates key statistics for a given portfolio allocation.
    
    - It returns an array containing:
      - `portfolio_return`: Expected return of the portfolio (annualized)
      - `portfolio_volatility`: Expected risk (volatility) of the portfolio (annualized)
      - `Sharpe ratio`: Measures return per unit of risk (higher is better)

    Parameters:
    - `returns`: The historical daily returns of stocks.
    - `weights`: The weight allocation of each stock in the portfolio.

    Returns:
    - A NumPy array: [Expected Return, Expected Volatility, Sharpe Ratio]
    """
    # Calculate expected annual return of the portfolio
    portfolio_return = np.sum(returns.mean()*weights)*NUM_TRADING_DAYS
    # Calculate portfolio volatility (risk)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights)))
    # Return an array with expected return, risk, and Sharpe ratio
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

# The scipy.optimize module is used to find the minimum value of a function.
# Since the highest Sharpe ratio is desired, we need to "flip" the sign of the function (maximize Sharpe Ratio).
# Maximizing a function `f(x)` is the same as minimizing `-f(x)`, so we return the negative Sharpe ratio.
def min_function_sharpe(weights, returns):
    """
    This function returns the negative Sharpe Ratio of a portfolio.
    The optimizer will try to minimize this function, which means it will actually maximize the Sharpe Ratio.

    Parameters:
    - `weights`: The current weight allocation of stocks.
    - `returns`: The historical daily returns of stocks.

    Returns:
    - Negative Sharpe Ratio (since minimizing -SR is the same as maximizing Sharpe Ratio).
    """
    return -statistics(weights, returns)[2] # Minimize the negative Sharpe ratio

# what are the constraints? the sum of weights = 1!!!
# f(x) = 0 this is the function to minimize 
def optimize_portfolio(weights, returns):
    """
    This function optimizes the portfolio allocation to maximize the Sharpe Ratio.
    
    - It does this by **minimizing the negative Sharpe Ratio** (since optimization functions minimize by default).
    - It ensures that the sum of portfolio weights is equal to 1 (100% of capital is invested).
    - It also enforces that each stock's weight is between 0 and 1 (no short selling or leverage).

    Parameters:
    - `weights`: Initial guess for portfolio weights (must sum to 1).
    - `returns`: Historical daily returns of stocks.

    Returns:
    - The optimized portfolio allocation (weights) that maximizes the Sharpe Ratio.
    """
    # Constraint: The sum of all portfolio weights must be equal to 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    # Bounds: Each stock weight must be between 0 and 1 (No short selling, no leverage)
    bounds = tuple((0,1) for _ in range(len(stocks))) # Creates a tuple of (0,1) for each stock in portfolio
    # Optimize the portfolio by minimizing the negative Sharpe Ratio
    return optmz.minimize(
        fun=min_function_sharpe, # Function to minimize (negative Sharpe Ratio)
        x0=weights[0], # Initial weights (first row of input weight matrix)
        args=returns,  # Additional arguments passed to `min_function_sharpe
        method="SLSQP", # Sequential Least Squares Quadratic Programming (a common optimization method)
        bounds=bounds, # Ensure weights are between 0 and 1 (no short selling or leverage)
        constraints=constraints # Ensure sum of weights equals 1 (100% allocation of funds)
    )

def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum["x"].round(3))
    print("Expected return, volatility and Sharpe ratio: ", statistics(optimum["x"].round(3), returns))

def show_optimal_portfolio(optimal_portfolio, returns, portfolio_returns, portfolio_volatilities):
    """
    This function plots all generated portfolios on a risk-return graph and highlights the optimal portfolio.

    - The scatter plot represents different portfolio options:
        - X-axis: Expected volatility (risk)
        - Y-axis: Expected return (profit)
        - Color: Sharpe Ratio (risk-adjusted return)
    - The optimal portfolio is highlighted with a large green star.

    Parameters:
    - `optimal_portfolio`: The best portfolio found by the optimizer.
    - `returns`: The historical daily returns of stocks.
    - `portfolio_returns`: Expected returns of the random portfolios.
    - `portfolio_volatilities`: Expected volatilities of the random portfolios.
    """
    plt.figure(figsize=(10, 6))  # Set figure size
    # Scatter plot of randomly generated portfolios
    plt.scatter(
        portfolio_volatilities,  # X-axis: Risk (volatility)
        portfolio_returns,       # Y-axis: Expected return
        c=portfolio_returns / portfolio_volatilities,  # Color: Sharpe Ratio
        marker='o'  # Use circle markers for each portfolio
    )
    plt.grid(True) # Add grid for better readability
    plt.xlabel("Expected Volatility") # X-axis represents risk
    plt.ylabel("Expected Return") # Y-axis represents profit
    plt.colorbar(label="Sharpe Ratio") # Color indicates the Sharpe Ratio

    # Extract optimal portfolio statistics (expected return and volatility)
    optimal_volatility = statistics(optimal_portfolio['x'], returns)[1]  # Optimal portfolio risk
    optimal_return = statistics(optimal_portfolio['x'], returns)[0]  # Optimal portfolio return
    # Plot the optimal portfolio with a large green star ('g*')
    plt.plot(optimal_volatility, optimal_return, 'g*', markersize=20.0, label="Optimal Portfolio")
    plt.legend()
    plt.show()

# this is the start of the program, the main method
if __name__ == "__main__":
    dataset =  download_data()
    # initial dataset graph print
    show_data(dataset)
    # get the normalized returns of the dataset
    log_daily_returns = calculate_return(dataset)
    # show_statistics(log_daily_returns)

    # define weights means risks via generating portfolios using the daily returns
    pweights, means, risks = generate_portfolios(log_daily_returns)
    # show_portfolios(means, risks)

    # find the optimal portfolio from the generated portfolios
    optimal_portfolio = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimal_portfolio, log_daily_returns)
    # visually show the optimal portfolio
    show_optimal_portfolio(optimal_portfolio, log_daily_returns, means, risks)
