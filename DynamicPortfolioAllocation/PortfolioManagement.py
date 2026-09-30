import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import random
import torch
import pickle
import matplotlib.pyplot as plt

from PM_Env2 import PM_Environment2, PM_Environment2_PVM
from RL_Model2 import agent_ppo2, evaluate_model2
from PM_Env_Train_Test import make_train_test_envs

"""List of PorfolioManagement RL Module functions:

1- _clean_na
2- annual_cost_drag
3- _corwin_schultz_spread
4- build_cost_tensor_fixed
5- load_macro_workbook
6- load_firm_data_workbook
7- summarize_rl_results
8- make_test_env
9- run_equal_weight_buy_and_hold_direct
10- run_equal_weight_daily_rebalanced
11- run_markowitz_benchmark
12- set_global_seed
13- run_ppo_seed_experiments
14- save_seed_results
15- load_seed_results
16- run_additional_ppo_seeds
17- results_to_equity_df
18- drawdown_curve
19- history_metric_series
20- plot_seed_band
21- plot_weights_heatmap
22- build_rf_feature_dataset
23- build_dml_dataset"""


#Function 1: Replace text string "#N/A" from Factset datasets with np.nan

def _clean_na(x):
    return pd.to_numeric(x.replace("#N/A", np.nan), errors="coerce")


#####################################################################################################

#Function 2: Compute annual cost drag based on transaction cost & average daily turnover

def annual_cost_drag(transaction_cost, avg_daily_turnover, periods_per_year=252):
    """
    transaction_cost: this is one-way cost per dollar traded, e.g. 0.0005 = 5 bps
    avg_daily_turnover: daily traded notional / portfolio NAV
    """
    return transaction_cost * avg_daily_turnover * periods_per_year


#####################################################################################################

#Function 3: Compute the Corwin-Schultz high-low bid-ask spread estimate

def _corwin_schultz_spread(high, low, window=20):
    """
    Corwin-Schultz high-low bid-ask spread estimator.

    Reference:
    Corwin and Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads
    from Daily High and Low Prices."

    High-Low Spread Estimate (S): Eqn 14,
    Alpha Estimate (independent of S): Eqn 18,
    Beta Estimate (Square of expectation of the Sum_0to1 of ln(Hobs_t+j/Lobs_t_j)): 
    """
    #1: Change data type to a float pandas series for better manipulation
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)

    #2: Declare the log ratio of observed high low prices
    hl = np.log(high / low)
    
    #3 Declare Beta variable, used to compute the Alpha variable / Eqn 7, Corwin Shultz (2012)
    beta = hl.pow(2).rolling(2).sum()

    #4 Declare the Gamma variable, used to compute the Alpha variable / Eqn 10, Corwin Shultz (2012)
    # First declare the max-high and min-low over the 2 days period as done by Corwin Shultz (2012)
    high_2d = high.rolling(2).max()
    low_2d = low.rolling(2).min()
    gamma = np.log(high_2d / low_2d).pow(2)

    #5 Declare the Alpha variable / Eqn 18, Corwin Shultz (2012)
    # Denominator denoted 'k'
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    # Clip alpha for a minimum of 0 spread, as seen later in the below Eqn 14
    alpha = alpha.clip(lower=0)

    # Eqn 14, Corwin Shultz (2012)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # smooth noisy daily estimator
    return spread.rolling(window).median()


#####################################################################################################


#Function 4: Build the cost tensor based on Explicit + Implicit cost modeling

def build_cost_tensor_fixed(
    price_tensor,
    feature_map=None,
    window=20,                # Use a 20-day rolling window (approximately one trading month)
    missing_cost=0.0005,      # 5 bps fallback only for missing data
    min_cost=0.00002,         # 0.2 bps floor
    max_cost=0.02,            # 200 bps cap
    commission_bps=0.5,
    half_spread_floor_bps=0.5,
    participation_rate=0.01,  # Volume(Q) / ADV, e.g. 0.01 = trading 1% of ADV
    impact_bps_per_1pct_adv=5,
    use_corwin_schultz=True,
    shift_costs=True,         # We shift since the first trade in train/test is after observing 1st price
):
    """
        cost[t, i] is one-way transaction cost as fraction of traded notional.
        Example: 0.0005 = 5 bps.
    """

    # Control for proper feature map structure that matches our environment
    if feature_map is None:
        feature_map = {"O": 0, "H": 1, "L": 2, "C": 3, "V": 4}

    # Strip down the tensor for later manipulation
    O = price_tensor[:, :, feature_map["O"]].astype(float)
    H = price_tensor[:, :, feature_map["H"]].astype(float)
    L = price_tensor[:, :, feature_map["L"]].astype(float)
    C = price_tensor[:, :, feature_map["C"]].astype(float)
    V = price_tensor[:, :, feature_map["V"]].astype(float)

    # Extract T: timeframe and N: number of assets from closing price matrix
    T, N = C.shape
    cost = np.full((T, N), np.nan)

    # Loop over each asset in the portfolio
    for i in range(N):
        df = pd.DataFrame({
            "O": O[:, i],
            "H": H[:, i],
            "L": L[:, i],
            "C": C[:, i],
            "V": V[:, i],
        })

        # Add a validity filter from the onset to avoid infinite values
        # This returns TRUE if invalid
        valid = (
            np.isfinite(df["O"])
            & np.isfinite(df["H"])
            & np.isfinite(df["L"])
            & np.isfinite(df["C"])
            & np.isfinite(df["V"])
            & (df["O"] > 0)
            & (df["H"] > 0)
            & (df["L"] > 0)
            & (df["C"] > 0)
            & (df["V"] >= 0)
            & (df["H"] >= df["L"])
        )

        # If validity filter shows invalid, replace price by nan for better manipulation
        df.loc[~valid, ["O", "H", "L", "C", "V"]] = np.nan

        # For each asset, we do the following:
        
        # Compute the daily dollar volume of transactions based on daily closing price
        # --> needed for rolling average daily dollar volume
        df["dollar_volume"] = df["C"] * df["V"]

        # Compute the rolling average daily dollar volume based on closing price
        df["adv"] = df["dollar_volume"].rolling(window).mean()

        
        # Optional high-low volatility proxy, useful as a robustness input.
        #df["parkinson_vol"] = (
        #    np.log(df["H"] / df["L"]).pow(2).rolling(window).mean()
        #    / (4 * np.log(2))
        #).pow(0.5)'''

        # I. Explicit Costs
        # 1. Commission: linear fee per traded notional.
        commission_bps_series = pd.Series(commission_bps, index=df.index)

        # II. Implicit Costs
        # 1. Spread Estimate using Corwin-Schultz High-Low Estimator
        if use_corwin_schultz:
            spread_decimal = _corwin_schultz_spread(df["H"], df["L"], window=window)
        else:
            # In case Crowin Shultz makes no sense:
            # Fallback: do NOT treat full high-low range as spread.
            # Use a small fraction of the range as a crude spread proxy approximation
            # This is the rolling median of 5% of high-low spread fraction
            spread_decimal = 0.05 * ((df["H"] - df["L"]) / df["C"]).rolling(window).median()

        # Half spreads for one-way buy/sell trade
        half_spread_bps = (0.5 * spread_decimal * 10_000).clip(lower=half_spread_floor_bps)

        # 2. Impact Estimate
        # Linear impact assumption:
        # impact_bps = impact_bps_per_1pct_adv * (participation_rate / 0.01)
        # This is intuitive Almgren-Chriss-style temporary impact idea:
        # impact is linear in trading rate / participation.
        
        impact_bps = (
            impact_bps_per_1pct_adv
            * (participation_rate / 0.01)
        )

        impact_bps = pd.Series(impact_bps, index=df.index)
       
        # Safety: if volume is zero or ADV unavailable, impact unavailable.
        impact_bps = impact_bps.where(df["adv"] > 0)

        
        # III. Total Transaction Costs Estimate = Explicit Costs + Implicit Costs
        total_cost_bps = (
            commission_bps_series
            + half_spread_bps
            + impact_bps
        )

        cost[:, i] = total_cost_bps / 10_000

    # If costs are used for decisions made before observing today's H/L/V,
    # shift them by one day to avoid lookahead.
    if shift_costs:
        cost = np.vstack([np.full((1, N), np.nan), cost[:-1]])

    
    # Imputation and Capping of costs
    cost = np.nan_to_num(
        cost,
        nan=missing_cost,
        posinf=max_cost,
        neginf=missing_cost,
    )

    cost = np.clip(cost, min_cost, max_cost)

    # Align with return tensor starting at t=1.
    return cost[1:]

#####################################################################################################


#Function 5: Load macroeconomic data into a Pandas df and perform necessary wrangling

def load_macro_workbook(
    macro_excel_path,
    feature_dates,
    date_col=None,
    value_col=None,
    standardize=True,
):
    """
    
    Loads macro data from an Excel workbook where each sheet contains one macro series.

    Expected sheet format:
        Date | Value

    Returns:
        macro_daily: pd.DataFrame aligned to feature_dates
        macro_tensor: np.ndarray of shape (T, M)
        macro_features: list of macro variable names
    """

    feature_dates = pd.to_datetime(feature_dates)
    feature_dates = pd.DatetimeIndex(feature_dates).sort_values()

    xls = pd.ExcelFile(macro_excel_path)
    xls.close()
    
    macro_series = {}

    for sheet in xls.sheet_names:

        # Skip hidden/system/cache sheets usual in Factset spreadsheets
        if str(sheet).startswith("_") or "CACHE" in str(sheet).upper():
            continue

        df = pd.read_excel(macro_excel_path, sheet_name=sheet)

        # Drop fully empty rows/columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.shape[1] < 2:
            raise ValueError(f"Sheet '{sheet}' must have at least Date and Value columns.")

        # Infer columns if not provided
        dcol = date_col if date_col is not None else df.columns[0]
        vcol = value_col if value_col is not None else df.columns[1]

        temp = df[[dcol, vcol]].copy()
        temp.columns = ["Date", sheet]

        temp["Date"] = pd.to_datetime(temp["Date"], errors="coerce")
        temp[sheet] = pd.to_numeric(temp[sheet], errors="coerce")

        temp = temp.dropna(subset=["Date"])
        temp = temp.sort_values("Date")
        temp = temp.drop_duplicates(subset="Date", keep="last")
        temp = temp.set_index("Date")

        macro_series[sheet] = temp[sheet]

    # Combine all macro series
    macro_df = pd.concat(macro_series.values(), axis=1)
    macro_df.columns = list(macro_series.keys())
    macro_df = macro_df.sort_index()

    # Align to daily RL feature dates and ensure filling according to the latest reported date
    macro_daily = (
    macro_df
    .reindex(
        macro_df.index.union(feature_dates)
    )
    .sort_index()
    .ffill()
    .reindex(feature_dates)
)

    # Handle any initial missing values before first available observation
    macro_daily = macro_daily.bfill()

    # Standardization
    if standardize:
        macro_daily = (macro_daily - macro_daily.mean()) / macro_daily.std(ddof=0)
        macro_daily = macro_daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    macro_features = macro_daily.columns.tolist()
    macro_tensor = macro_daily.values.astype(np.float32)

    return macro_daily, macro_tensor, macro_features

#####################################################################################################

#Function 6: Load firm data into a Pandas df and perform necessary wrangling

def load_firm_data_workbook(
    excel_path,
    feature_dates,
    tickers,
    # The "_Q" suffix was added in the excel sheet names to distinguish quarterly data
    quarterly_suffix="_Q",

    market_cap_sheet="market_cap",
    sector_sheet="sector",

    standardize=True,
):
    feature_dates = pd.DatetimeIndex(pd.to_datetime(feature_dates)).sort_values()
    tickers = list(tickers)

    # 1. Load Market Cap - Daily
    # Raw data: columns = dates, rows = ticker entry

    mcap_raw = pd.read_excel(
        excel_path,
        sheet_name=market_cap_sheet,
        index_col=0 # To use tickers as row labeles
        
    )

    # Change column heads to date-time
    mcap_raw.columns = pd.to_datetime(mcap_raw.columns, errors="coerce")
    # Ensure that we select tickers data only, so we pass "tickers" to .loc
    mcap_raw = mcap_raw.loc[tickers]
    # Apply _clean_na(), function no. 1 in PortfolioManagement module
    # This function replaces text '#N/A' by np.na
    mcap_raw = mcap_raw.apply(lambda col: _clean_na(col))

    # Transpose: Date x Ticker
    # This way the row index becomes a date-time and tickers become column headers
    mcap_df = mcap_raw.T
    # Sort by data
    mcap_df = mcap_df.sort_index()

    # Align with the global daily date index 'feature_dates', using .index.union()
    mcap_daily = (
        mcap_df
        .reindex(mcap_df.index.union(feature_dates)) #.index.union includes "feature_dates"
        # We then sort & apply forward fill
        .sort_index()
        .ffill()
        .reindex(feature_dates) # Ensure index is the feature_dates
    )


    # 2. Load Sector - Categorical
   
    # Raw data: column = sectors, rows = tickers
    sector_raw = pd.read_excel(
        excel_path,
        sheet_name=sector_sheet,
        index_col=0 # To use tickers as row labeles
    )

    # Extract sector from column
    # Store it separately in sector_series
    sector_series = sector_raw.iloc[:, 0].astype(str)
    # Index sectors by tickers
    sector_series = sector_series.reindex(tickers)

    # Extract unique sector categories and sort alphabetically
    sector_categories = sorted(sector_series.dropna().unique())
    # Store sectors alongside ids in a dictionary using a compact enumerate loop
    sector_to_id = {sector: i for i, sector in enumerate(sector_categories)}
    # Extract ids separately
    sector_ids = sector_series.map(sector_to_id)


    # 3. Load Quarterly Fundamentals

    # Store feature names the same way the rows in the worksheets are named
    firm_features = [
        "Earnings Per Share",
        "Sales (Millions)",
        "Total Debt % Equity",
        "Return on Average Total Equity",
    ]

    # Rename to a simpler naming
    renamed_features = [
        "EPS",
        "Revenue",
        "DebtToEquity",
        "ROE",
    ]


    # Extract raw data into a dictionary
    # Raw data format: columns = dates, rows = type of firm data, row index: unique ticker
    firm_daily_by_ticker = {}

    for ticker in tickers:
        sheet = f"{ticker}{quarterly_suffix}"

        q_raw = pd.read_excel(
            excel_path,
            sheet_name=sheet,
            header=None
        )

        # Extract date index from the first row
        # first column = ticker, second column = feature name, remaining columns = release dates
        release_dates = pd.to_datetime(q_raw.iloc[0, 2:], errors="coerce")

        # Copy features x quarterly value
        temp = q_raw.iloc[:, 1:].copy()
        # Name the columns of features x quarterly value
        temp.columns = ["Feature"] + list(release_dates)

        # Set row index to be the "Feature" column
        temp = temp.set_index("Feature")

        # Keep only required rows, which have the firm_feature names
        temp = temp.loc[firm_features]

        # Rename rows to simpler feature names
        temp.index = renamed_features

        # Transpose: Date x Feature
        q_df = temp.T
        # Set row index to dates
        q_df.index = pd.to_datetime(q_df.index, errors="coerce")
        # Keep non nan values and finally sort
        q_df = q_df[~q_df.index.isna()]
        q_df = q_df.sort_index()

        for col in q_df.columns:
            q_df[col] = _clean_na(q_df[col])

        
        # Align quarterly data with daily feature_dates
        # Use union method on index .index.union()
        q_daily = (
            q_df
            .reindex(q_df.index.union(feature_dates))
            .sort_index()
            .ffill()
            .reindex(feature_dates) # This is to ensure perfect alignment with feature calendar
            # Otherwise, there might be misalignment with quarterly announcement dates
        )

        # Add the quarterly announced values to the firm_daily_by_ticker dictionary
        firm_daily_by_ticker[ticker] = q_daily


    # 4. Build Complete Numerical Firm Tensor

    all_firm_features = renamed_features + ["MarketCap"]

    # Declare empty tensor, each matrix for a distinct feature
    # The matrix dimensions shall match the main model's feature tensor
    firm_tensor = np.zeros(
        (len(feature_dates), len(tickers), len(all_firm_features)),
        dtype=np.float32
    )

    # Loop to populate the tensor 1 ticker at a time
    for j, ticker in enumerate(tickers):
        # Get the daily transformed from quarterly
        df = firm_daily_by_ticker[ticker].copy()
        # Get market cap
        df["MarketCap"] = mcap_daily[ticker]

        # Populate each ticker column, across rows and feature matrices in the firm features tensor
        firm_tensor[:, j, :] = df[all_firm_features].values.astype(np.float32)

   
    # 5. Handle infinite values
    
    firm_tensor = np.where(np.isfinite(firm_tensor), firm_tensor, np.nan)

    # Standardization feature-by-feature
    if standardize:
        for k in range(firm_tensor.shape[2]):
            x = firm_tensor[:, :, k]

            mean = np.nanmean(x)
            std = np.nanstd(x)

            if std == 0 or np.isnan(std):
                firm_tensor[:, :, k] = 0.0
            # De-mean and scale by standard deviation
            else:
                firm_tensor[:, :, k] = (x - mean) / std

    # Handle nan & infinite values    
        firm_tensor = np.nan_to_num(firm_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        firm_tensor = np.nan_to_num(firm_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "firm_tensor": firm_tensor,
        "firm_features": all_firm_features,
        "mcap_daily": mcap_daily,
        "sector_series": sector_series,
        "sector_ids": sector_ids,
        "sector_to_id": sector_to_id,
        "firm_daily_by_ticker": firm_daily_by_ticker,
    }


#####################################################################################################

#Function 7: 

def summarize_rl_results(
    name,
    results,
    model=None,
    model_params=None,
    risk_free_rate=0.0,
    periods_per_year=252,
    export_path="experiment_results.xlsx",
):
    equity_curve = np.asarray(results["equity_curve"], dtype=float)

    returns = equity_curve[1:] / equity_curve[:-1] - 1.0

    total_return = equity_curve[-1] / equity_curve[0] - 1.0

    annualized_return = (
        (1 + total_return)
        ** (periods_per_year / len(returns))
        - 1
    )

    volatility = np.std(returns) * np.sqrt(periods_per_year)

    sharpe = (
        (np.mean(returns) * periods_per_year - risk_free_rate)
        / volatility
        if volatility > 0 else np.nan
    )

    running_max = np.maximum.accumulate(equity_curve)

    drawdown = equity_curve / running_max - 1.0

    max_drawdown = np.min(drawdown)

    downside_returns = returns[returns < 0]

    downside_vol = (
        np.std(downside_returns)
        * np.sqrt(periods_per_year)
    )

    sortino = (
        (np.mean(returns) * periods_per_year - risk_free_rate)
        / downside_vol
        if downside_vol > 0 else np.nan
    )

    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0 else np.nan
    )

    history = results.get("history", [])

    turnovers = np.array(
        [h.get("turnover", np.nan) for h in history],
        dtype=float
    )

    costs = np.array(
        [h.get("transaction_cost", np.nan) for h in history],
        dtype=float
    )

    avg_daily_turnover = np.nanmean(turnovers)
    median_daily_turnover = np.nanmedian(turnovers)
    max_daily_turnover = np.nanmax(turnovers)
    total_turnover = np.nansum(turnovers)

    avg_daily_cost_drag = np.nanmean(costs)
    median_daily_cost_drag = np.nanmedian(costs)
    max_daily_cost_drag = np.nanmax(costs)
    total_cost_drag = np.nansum(costs)

    approx_annual_cost_drag = (
        avg_daily_cost_drag * periods_per_year
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "Timestamp": timestamp,

        "Model": name,

        "Final Value": equity_curve[-1],
        "Total Return": total_return,
        "Annualized Return": annualized_return,

        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_drawdown,

        "Avg Daily Turnover": avg_daily_turnover,
        "Median Daily Turnover": median_daily_turnover,
        "Max Daily Turnover": max_daily_turnover,
        "Total Turnover": total_turnover,

        "Avg Daily Cost Drag": avg_daily_cost_drag,
        "Median Daily Cost Drag": median_daily_cost_drag,
        "Max Daily Cost Drag": max_daily_cost_drag,
        "Total Cost Drag": total_cost_drag,
        "Approx Annual Cost Drag": approx_annual_cost_drag,
    }



    if model is not None and hasattr(model, "run_hyperparams"):

        if model_params is None:
            model_params = {}

        model_params.update(model.run_hyperparams)

    if model_params is not None:

        for k, v in model_params.items():

            if isinstance(v, (list, dict)):
                summary[f"param_{k}"] = json.dumps(v)

            else:
                summary[f"param_{k}"] = v

    summary_df = pd.DataFrame([summary])


    # Export
    

    if os.path.exists(export_path):

        existing = pd.read_excel(export_path)

        combined = pd.concat(
            [existing, summary_df],
            ignore_index=True
        )

    else:
        combined = summary_df

    
    base, ext = os.path.splitext(export_path)

    final_export_path = export_path
    version = 2

    while os.path.exists(final_export_path):
        final_export_path = f"{base}_v{version}{ext}"
        version += 1

    combined.to_excel(final_export_path, index=False)

    print(f"Results exported to: {final_export_path}")

    

    return summary_df


#####################################################################################################

#Function 8:

def make_test_env(
    env_class,
    price_tensor,
    cost_tensor,
    feature_tensor,
    feature_mask,
    tickers,
    lookback=30,
    train_ratio=0.8,
    initial_cash=100_000,
):
    T = feature_tensor.shape[0]
    split = int(T * train_ratio)

    test_start = split + lookback
    test_end = T - 1

    return env_class(
        price_tensor=price_tensor,
        cost_tensor=cost_tensor,
        feature_tensor=feature_tensor,
        feature_mask=feature_mask,
        tickers=tickers,
        lookback=lookback,
        initial_cash=initial_cash,
        start_index=test_start,
        end_index=test_end,
    )


#####################################################################################################

#Function 9:

def run_equal_weight_buy_and_hold_direct(
    price_tensor,
    feature_tensor,
    feature_mask,
    cost_tensor,
    start_index,
    end_index,
    initial_cash=100_000,
    log_ret_c_index=3,
):
    N = feature_tensor.shape[1]

    tradable = feature_mask[start_index].astype(float)

    if tradable.sum() == 0:
        raise ValueError("No tradable assets at start_index.")

    weights = tradable / tradable.sum()

    initial_cost = np.sum(weights * cost_tensor[start_index])
    portfolio_value = initial_cash * (1.0 - initial_cost)

    holdings_value = weights * portfolio_value

    values = [portfolio_value]
    weights_history = [np.append(weights, 0.0)]

    history = [{
        "portfolio_value": portfolio_value,
        "turnover": 1.0,
        "transaction_cost": initial_cost,
        "weights": np.append(weights, 0.0),
        "t": start_index,
    }]

    for t in range(start_index, end_index):
        log_ret_c = feature_tensor[t, :, log_ret_c_index]
        asset_growth = np.exp(log_ret_c)
        asset_growth = np.nan_to_num(asset_growth, nan=1.0, posinf=1.0, neginf=1.0)

        holdings_value *= asset_growth
        portfolio_value = holdings_value.sum()

        weights = holdings_value / portfolio_value

        values.append(portfolio_value)
        weights_history.append(np.append(weights, 0.0))

        history.append({
            "portfolio_value": portfolio_value,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "weights": np.append(weights, 0.0),
            "t": t,
        })

    return {
        "equity_curve": np.array(values),
        "weights_history": np.array(weights_history),
        "history": history,
    }


#####################################################################################################

#Function 10:

def run_equal_weight_daily_rebalanced(env):
    obs, info = env.reset()
    done = False

    values = [info["portfolio_value"]]
    weights_history = []

    while not done:
        tradable = env.feature_mask_full[env.t].astype(float)

        if tradable.sum() == 0:
            stock_weights = np.zeros(env.N)
            cash_weight = 1.0
        else:
            stock_weights = tradable / tradable.sum()
            cash_weight = 0.0

        action = np.append(stock_weights, cash_weight)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        values.append(info["portfolio_value"])
        weights_history.append(info["weights"])

    return {
        "equity_curve": np.array(values),
        "weights_history": np.array(weights_history),
        "history": env.history,
    }



#####################################################################################################

#Function 11:

def run_markowitz_benchmark(
    env,
    estimation_window=252,
    rebalance_frequency=21,
    min_assets=2,
    verbose=True,
):
    obs, info = env.reset()
    done = False

    values = [info["portfolio_value"]]
    weights_history = []

    current_action = np.zeros(env.N + 1)
    current_action[-1] = 1.0

    step_count = 0
    failures = 0
    successes = 0

    while not done:

        if step_count % rebalance_frequency == 0:
            end_t = env.t
            start_t = max(env.lookback, end_t - estimation_window)

            hist_log_returns = env.feature_tensor_full[
                start_t:end_t,
                :,
                env.log_ret_c_index
            ]

            hist_returns = np.exp(hist_log_returns) - 1.0
            hist_returns = pd.DataFrame(hist_returns, columns=env.tickers)

            tradable = env.feature_mask_full[env.t].astype(bool)
            tradable_assets = [env.tickers[i] for i in range(env.N) if tradable[i]]

            if len(tradable_assets) >= min_assets:
                hist_returns = hist_returns[tradable_assets]
                hist_returns = hist_returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

                try:
                    mu = expected_returns.mean_historical_return(
                        hist_returns,
                        returns_data=True,
                        frequency=252,
                    )

                    S = risk_models.CovarianceShrinkage(
                        hist_returns,
                        returns_data=True,
                        frequency=252,
                    ).ledoit_wolf()

                    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.25))
                    weights_dict = ef.max_sharpe()
                    cleaned = ef.clean_weights()

                    stock_weights = np.zeros(env.N)

                    for i, ticker in enumerate(env.tickers):
                        stock_weights[i] = cleaned.get(ticker, 0.0)

                    stock_sum = stock_weights.sum()

                    if stock_sum <= 0:
                        raise ValueError("Markowitz produced zero stock allocation.")

                    stock_weights = stock_weights / stock_sum
                    cash_weight = 0.0

                    current_action = np.append(stock_weights, cash_weight)
                    successes += 1

                except Exception as e:
                    failures += 1

                    if verbose:
                        print(f"Markowitz failed at t={env.t}: {e}")

                    stock_weights = tradable.astype(float)
                    stock_weights = stock_weights / stock_weights.sum()
                    current_action = np.append(stock_weights, 0.0)

        obs, reward, terminated, truncated, info = env.step(current_action)
        done = terminated or truncated

        values.append(info["portfolio_value"])
        weights_history.append(info["weights"])

        step_count += 1

    if verbose:
        print(f"Markowitz successes: {successes}")
        print(f"Markowitz failures: {failures}")

    return {
        "equity_curve": np.array(values),
        "weights_history": np.array(weights_history),
        "history": env.history,
        "markowitz_successes": successes,
        "markowitz_failures": failures,
    }


#####################################################################################################

#Function 12:

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#####################################################################################################

#Function 13:

def run_ppo_seed_experiments(
    seeds,
    price_tensor_aligned,
    cost_tensor,
    full_feature_tensor,
    feature_mask,
    tickers,

    base_env_class=PM_Environment2,
    pvm_env_class=PM_Environment2_PVM,

    lookback=30,
    train_ratio=0.8,
    initial_cash=100_000,
    total_timesteps=100_000,
    export_path=None,
    
):
    all_summaries = []
    all_results = {}

    for seed in seeds:
        print(f"\n===== SEED {seed} | PPO BASE =====")
        set_global_seed(seed)

        train_env_base, test_env_base = make_train_test_envs(
            PM_Environment=base_env_class,
            price_tensor=price_tensor_aligned,
            cost_tensor=cost_tensor,
            feature_tensor=full_feature_tensor,
            feature_mask=feature_mask,
            tickers=tickers,
            lookback=lookback,
            train_ratio=train_ratio,
            initial_cash=initial_cash,
        )

        model_base = agent_ppo2(
            train_env_base,
            total_timesteps=total_timesteps,
            seed=seed,
        )

        eval_env_base = make_test_env(
            base_env_class,
            price_tensor=price_tensor_aligned,
            cost_tensor=cost_tensor,
            feature_tensor=full_feature_tensor,
            feature_mask=feature_mask,
            tickers=tickers,
            lookback=lookback,
            train_ratio=train_ratio,
            initial_cash=initial_cash,
        )

        results_base = evaluate_model2(model_base, eval_env_base)

        summary_base = summarize_rl_results(
            name=f"PPO_LOG_RET_OHLCV_Base_seed_{seed}",
            results=results_base,
            model=model_base,
            model_params={"seed": seed, "pvm": False},
            export_path=export_path,
        )

        all_summaries.append(summary_base)
        all_results[f"PPO_Base_{seed}"] = results_base

        print(f"\n===== SEED {seed} | PPO LIGHT PVM =====")
        set_global_seed(seed)

        train_env_pvm, test_env_pvm = make_train_test_envs(
            PM_Environment=pvm_env_class,
            price_tensor=price_tensor_aligned,
            cost_tensor=cost_tensor,
            feature_tensor=full_feature_tensor,
            feature_mask=feature_mask,
            tickers=tickers,
            lookback=lookback,
            train_ratio=train_ratio,
            initial_cash=initial_cash,
        )

        model_pvm = agent_ppo2(
            train_env_pvm,
            total_timesteps=total_timesteps,
            seed=seed,
        )

        eval_env_pvm = make_test_env(
            pvm_env_class,
            price_tensor=price_tensor_aligned,
            cost_tensor=cost_tensor,
            feature_tensor=full_feature_tensor,
            feature_mask=feature_mask,
            tickers=tickers,
            lookback=lookback,
            train_ratio=train_ratio,
            initial_cash=initial_cash,
        )

        results_pvm = evaluate_model2(model_pvm, eval_env_pvm)

        summary_pvm = summarize_rl_results(
            name=f"PPO_LOG_RET_OHLCV_LightPVM_seed_{seed}",
            results=results_pvm,
            model=model_pvm,
            model_params={"seed": seed, "pvm": True},
            export_path=export_path,
        )

        all_summaries.append(summary_pvm)
        all_results[f"PPO_PVM_{seed}"] = results_pvm

    summary_df = pd.concat(all_summaries, ignore_index=True)

    return summary_df, all_results




#####################################################################################################

#Function 14:

def save_seed_results(results_dict, path="ppo_seed_results.pkl"):
    with open(path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Saved seed results to {path}")


#####################################################################################################

#Function 15:

def load_seed_results(path="ppo_seed_results.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            results = pickle.load(f)

        print(f"Loaded existing seed results from {path}")
        return results

    print("No existing seed results found.")
    return {}


#####################################################################################################

#Function 16:

def run_additional_ppo_seeds(
    seeds,
    existing_results_path="ppo_seed_results.pkl",
    export_path="ppo_multiseed_results.xlsx",
    **kwargs
):
    existing_results = load_seed_results(existing_results_path)
    all_summaries = []

    completed_keys = set(existing_results.keys())

    for seed in seeds:

        base_key = f"PPO_Base_{seed}"
        pvm_key = f"PPO_PVM_{seed}"

        if base_key in completed_keys:
            print(f"Skipping existing {base_key}")
        else:
            summary_df, result_dict = run_ppo_seed_experiments(
                seeds=[seed],
                export_path=export_path,
                **kwargs
            )

            existing_results.update(result_dict)
            all_summaries.append(summary_df)

            save_seed_results(existing_results, existing_results_path)

        # run_ppo_seed_experiments already runs both Base and PVM for each seed
        # so after one call both keys should exist.

    if all_summaries:
        new_summary = pd.concat(all_summaries, ignore_index=True)
    else:
        new_summary = pd.DataFrame()

    return new_summary, existing_results


#####################################################################################################

#Function 17:

def results_to_equity_df(results_dict):
    min_len = min(len(v["equity_curve"]) for v in results_dict.values())

    curves = {}

    for name, result in results_dict.items():
        curve = np.asarray(result["equity_curve"], dtype=float)[:min_len]
        curves[name] = curve / curve[0]

    return pd.DataFrame(curves)


#####################################################################################################

#Function 18:

def drawdown_curve(equity_curve):
    equity_curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity_curve)
    return equity_curve / running_max - 1.0


#####################################################################################################

#Function 19:

def history_metric_series(result, metric):
    return pd.Series(
        [h.get(metric, np.nan) for h in result["history"]]
    )


#####################################################################################################

#Function 20:

def plot_seed_band(equity_df, prefix, label):
    cols = [c for c in equity_df.columns if c.startswith(prefix)]

    arr = equity_df[cols].values

    mean = arr.mean(axis=1)
    std = arr.std(axis=1)

    x = np.arange(len(mean))

    plt.plot(x, mean, label=f"{label} Mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)


#####################################################################################################

#Function 21:

def plot_weights_heatmap(result, tickers, title):
    weights = np.asarray(result["weights_history"])

    stock_weights = weights[:, :-1]

    plt.figure(figsize=(14, 7))
    plt.imshow(stock_weights.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Portfolio Weight")
    plt.yticks(np.arange(len(tickers)), tickers)
    plt.xlabel("Test Step")
    plt.ylabel("Asset")
    plt.title(title)
    plt.show()

#####################################################################################################

#Function 22:

def build_rf_feature_dataset(
    full_feature_tensor,
    feature_mask,
    log_ret_c_index=3,
):
    T, N, F = full_feature_tensor.shape

    X_list = []
    y_list = []

    for t in range(T - 1):
        tradable = feature_mask[t].astype(bool)

        if tradable.sum() == 0:
            continue

        # portfolio-level state: average across tradable assets
        x_t = np.nanmean(
            full_feature_tensor[t, tradable, :],
            axis=0
        )

        # target: next-day equal-weight return
        log_ret_next = full_feature_tensor[
            t + 1,
            tradable,
            log_ret_c_index
        ]

        y_t = np.nanmean(np.exp(log_ret_next) - 1.0)

        if np.all(np.isfinite(x_t)) and np.isfinite(y_t):
            X_list.append(x_t)
            y_list.append(y_t)

    return np.asarray(X_list), np.asarray(y_list)

#####################################################################################################

#Function 23:

def build_dml_dataset(
    full_feature_tensor,
    feature_mask,
    feature_names,
    log_ret_c_index=3,
):

    T, N, F = full_feature_tensor.shape

    X_list = []
    y_list = []

    for t in range(T - 1):

        tradable = feature_mask[t].astype(bool)

        if tradable.sum() == 0: # Apply tradability mask, skip 't' if non-tradable
            continue

        #1 Portfolio-level aggregated state at each 't' --> Input variables are the environment faetures
        # Use average value of the portfolio of assets (nanmean)
        # Capture features 'day' panel data
        x_t = np.nanmean(
            full_feature_tensor[t, tradable, :], # Filter tradable assets
            axis=0
        )

        # Next-day equal-weight portfolio return
        # This will be our panel data label
        # Which is the next day's returns on closing price
        # Given the input features in x_t
        log_ret_next = full_feature_tensor[
            t + 1,
            tradable,
            log_ret_c_index
        ]

        #2 Output variable is simple returns --> Data Label
        # Convert from log-returns to simple returns
        # Find an average daily return for an equal weight portfolio (simple a priori application of DML)
        y_t = np.nanmean(np.exp(log_ret_next) - 1.0)

        # Filter against nan & inf values
        if np.all(np.isfinite(x_t)) and np.isfinite(y_t):
            X_list.append(x_t)
            y_list.append(y_t)

    X = np.asarray(X_list) # Each row is a features cross sectional panel
    y = np.asarray(y_list)

    df = pd.DataFrame(X, columns=feature_names)

    # Data label = return on closing price
    df["target_return"] = y

    return df

#####################################################################################################
def plot_metric_bar(
    agg_final,
    metric,
    title,
    ylabel,
    initial_portfolio_value=100_000,
    final_value_plot = False,
    filename=None,
    figsize=(16, 7)
    
):
    means = agg_final[metric]["mean"].sort_values(ascending=False)
    stds = agg_final[metric]["std"].reindex(means.index).fillna(0)

    plt.figure(figsize=figsize)

    plt.bar(
        means.index,
        means.values,
        yerr=stds.values,
        capsize=5
    )

    if final_value_plot == True:

        plt.axhline(
        y=initial_portfolio_value,
        linestyle="--",
        linewidth=2,
        color="red",
        )   

        
        plt.text(
            x=-0.5,
            y=initial_portfolio_value * 1.02,
            s=f"Initial Capital = ${initial_portfolio_value:,.0f}",
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
#####################################################################################################

#####################################################################################################

#####################################################################################################