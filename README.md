# ModernPortfolioTheoryOptimizer
This repository contains a Python-based portfolio optimization tool that helps investors construct an optimal stock portfolio using historical stock data. It leverages Modern Portfolio Theory (MPT) to generate and analyze thousands of random portfolios, ultimately identifying the one with the highest Sharpe Ratio (best risk-adjusted return).

Features:

Fetches historical stock data using yfinance
Computes log returns for accurate performance measurement
Generates thousands of random portfolios with different weight distributions
Calculates key portfolio metrics (expected return, volatility, and Sharpe ratio)
Uses scipy.optimize to determine the optimal portfolio allocation
Visualizes risk-return trade-offs with Matplotlib

Technologies Used:

Python
NumPy
Pandas
Matplotlib
SciPy
yFinance

How to Use:

Clone the repository.
Install required dependencies: pip install numpy pandas matplotlib scipy yfinance
Run the script: python portfolio_optimization.py
View generated portfolios and the optimized allocation.
