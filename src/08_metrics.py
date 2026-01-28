import numpy as np
import pandas as pd

def calculate_metrics(portfolio_values):
    """
    Inputs:
        portfolio_values: Pandas Series of account balance over time
    Returns:
        Dictionary of metrics
    """
    # 1. Calculate Returns
    returns = portfolio_values.pct_change().dropna()
    
    # 2. Sharpe Ratio
    # Annualized (Assuming 15-min candles -> 35,040 candles/year)
    risk_free_rate = 0.0 # simplified
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Sharpe = (Mean - RiskFree) / StdDev * Sqrt(Periods per Year)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(35040) if std_return != 0 else 0
    
    # 3. Sortino Ratio (Downside Deviation only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (mean_return / downside_std) * np.sqrt(35040) if downside_std != 0 else 0
    
    # 4. Maximum Drawdown (MDD)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() * 100 # in %
    
    # 5. Win Rate (Need trade logs for exact, but approx via returns)
    positive_periods = len(returns[returns > 0])
    total_periods = len(returns)
    win_rate_period = (positive_periods / total_periods) * 100
    
    return {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": f"{max_drawdown:.2f}%",
        "Win Rate (Period)": f"{win_rate_period:.2f}%"
    }

# --- How to use in backtest.py ---
# After your loop:
# metrics = calculate_metrics(pd.Series(balances))
# print("\n--- Performance Metrics ---")
# for k, v in metrics.items():
#     print(f"{k}: {v}")