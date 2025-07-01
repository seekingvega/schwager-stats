import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)

def annual_rate_to_daily(annual_rate, trading_days = 252) -> float:
    return (1 + annual_rate) ** (1/trading_days) - 1

def sortino_ratio(returns, adjustment_factor=0.0, debug = False) -> float:
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        adjustment_factor : int, float
        Constant daily benchmark return throughout the period.

    Returns
    -------
    sortino_ratio : float

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details.
    """

    # compute annualized return
    returns_risk_adj = np.asanyarray(returns - adjustment_factor)
    mean_annual_return = returns_risk_adj.mean() * 252

    # compute the downside deviation
    downside_diff = np.clip(returns_risk_adj, np.NINF, 0)
    np.square(downside_diff, out=downside_diff)
    annualized_downside_deviation = np.sqrt(downside_diff.mean()) * np.sqrt(252)
    if debug:
        logger.debug(f'avg annual return: {mean_annual_return}')
        logger.debug(f'annualized downside std: {annualized_downside_deviation}')

    return mean_annual_return / annualized_downside_deviation

def calculate_performance_metrics(equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.05, trading_days: int = 252, price_col = "Close",
    exclude_dates:list = [], start_date= None
    ) -> dict:
    """
    Calculate performance metrics for an equity curve.

    Parameters:
    equity_curve (pd.DataFrame): DataFrame with a 'returns' column representing daily returns
    risk_free_rate (float): Annual risk-free rate, default is 2%
    trading_days (int): Number of trading days in a year, default is 252

    Returns:
    dict: A dictionary containing the calculated metrics
    """

    # Ensure 'returns' column exists
    if 'returns' not in equity_curve.columns:
        equity_curve['returns'] = equity_curve[[price_col]].pct_change()
    if exclude_dates:
        equity_curve = equity_curve[~equity_curve.index.isin(exclude_dates)]
    equity_curve = equity_curve[equity_curve.index.date > start_date] if start_date else equity_curve

    # Annualized Return
    total_return = (equity_curve[price_col].iloc[-1] / equity_curve[price_col].iloc[0]) - 1
    years = len(equity_curve) / trading_days
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Sortino Ratio
    downside_returns = equity_curve['returns'][equity_curve['returns'] < 0]
    sr = sortino_ratio(returns = equity_curve['returns'].dropna(),
                       adjustment_factor= annual_rate_to_daily(risk_free_rate, trading_days= trading_days)
                      )

    # Maximum Drawdown
    cumulative_returns = (1 + equity_curve['returns']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()

    # Gain-to-Pain Ratio
    pain = [r for r in equity_curve['returns'].tolist() if r <0]
    gain = [r for r in equity_curve['returns'].tolist() if r >0]
    GPR = sum(gain)/ abs(sum(pain))

    return {
        'Annualized Return': annualized_return,
        'Sortino Ratio': sr, #sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Gain-to-Pain Ratio': GPR
    }

def compute_performance_metrics(df_stocks, risk_free_rate = 0,
        trading_days = 252, price_col = "Close", min_bar_count = 10,
        check_div: bool = False, start_date = None
    ):
    ''' return a dataframe of metrics for each ticker in df_stocks
    Args:
        df_stocks: a ohlc dataframe with data for multiple tickers
    '''
    data = []
    tickers = list(set(df_stocks.columns.get_level_values('Ticker')))
    if check_div:
        import yfinance as yf
        stock_objs = yf.Tickers(tickers)

    for ticker in tqdm(tickers, desc = 'computing metrics'):
        df_t = df_stocks[ticker].dropna() # in case the ETF did not have data
        if len(df_t)<min_bar_count:
            continue

        div_dates = stock_objs.tickers[ticker].dividends.index.date.tolist()if check_div else []
        m = calculate_performance_metrics(equity_curve= df_t,
                                          risk_free_rate= risk_free_rate,
                                          trading_days= trading_days,
                                          price_col= price_col,
                                          exclude_dates= div_dates,
                                          start_date= start_date
                                         )
        m['ticker'] = ticker
        m['bar_count'] = len(df_t)
        if check_div and start_date:
            m['div_count'] = len([d for d in div_dates if d > start_date])
        data.append(m)
    return pd.DataFrame(data)
