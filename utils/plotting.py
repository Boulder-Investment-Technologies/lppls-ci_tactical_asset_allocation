from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from seaborn import cubehelix_palette
from sklearn import linear_model

def plot_rolling_stats(rolling_series, title, mean=None):
    rolling_series.plot(figsize=(12,6), title=title)
    if mean:
        plt.axhline(mean, color='r', ls='--', label=f'mean = {mean}')
    plt.ylabel(title)
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
def plot_drawdown_periods(returns, top=5, ax=None, **kwargs):
    ax = plt.gca()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.copy()
    df_cum = pd.Series((1+returns).cumprod())
    df_drawdowns = gen_drawdown_table(returns, top=top)

    df_cum.plot(figsize=(12,6), ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['Peak date', 'Recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])
    ax.set_ylim(lim)
    ax.set_title(f'Top {top} drawdown periods')
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Equal Weight Portfolio Cumulative Return'], loc='upper left',
              frameon=True, framealpha=0.5)
    ax.set_xlabel('')
    plt.grid()
    plt.show()



def gen_drawdown_table(returns, top=10):
    df_cum = (1+returns).cumprod()
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['Net drawdown in %',
                                         'Peak date',
                                         'Valley date',
                                         'Recovery date',
                                         'Duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'Duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak, recovery, freq='M'))
        df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime().strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime().strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'Recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Net drawdown in %'] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
    df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
    df_drawdowns['Recovery date'] = pd.to_datetime(
        df_drawdowns['Recovery date'])

    return df_drawdowns

def get_top_drawdowns(returns, top=10):
    returns = returns.copy()
    df_cum = (1+returns).cumprod()
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1
    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if ((len(returns) == 0)
                or (len(underwater) == 0)
                or (np.min(underwater) == 0)):
            break

    return drawdowns

def get_max_drawdown_underwater(underwater):
    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def plot_drawdown_underwater(returns, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    df_cum = (1+returns).cumprod()
    running_max = np.maximum.accumulate(df_cum)
    underwater = -100 * ((running_max - df_cum) / running_max)
    (underwater).plot(figsize=(12,3), ax=ax, kind='area', color='red', alpha=0.5, **kwargs)
    ax.set_ylabel('Drawdown %')
    ax.set_title('Underwater plot')
    ax.set_xlabel('')
    plt.grid()
    plt.show()
    
def rolling_regression(returns, factor_returns, rolling_window):

    # We need to drop NaNs to regress
    ret_no_na = returns.dropna()
    factor_returns = factor_returns.copy()

    columns = ['alpha'] + factor_returns.columns.tolist()
    rolling_risk = pd.DataFrame(columns=columns, index=ret_no_na.index)

    rolling_risk.index.name = 'dt'

    for beg, end in zip(ret_no_na.index[:-rolling_window], ret_no_na.index[rolling_window:]):
        returns_period = ret_no_na[beg:end]
        factor_returns_period = factor_returns.loc[returns_period.index]
        factor_returns_period_dnan = factor_returns_period.dropna()
        reg = linear_model.LinearRegression(fit_intercept=True).fit(factor_returns_period_dnan, returns_period.loc[factor_returns_period_dnan.index])
        rolling_risk.loc[end, factor_returns.columns] = reg.coef_
        rolling_risk.loc[end, 'alpha'] = reg.intercept_

    return rolling_risk

def plot_rolling_risk_factors(
        returns,
        risk_factors,
        rolling_beta_window,
        legend_loc='best',
        ax=None, **kwargs):
    
    returns = returns.dropna()
    
    ax = plt.gca()
    ax.set_title('Rolling Fama-French Single Factor Betas (6-month)')
    ax.set_ylabel('β')
    
    rolling_beta_SMB = returns.rolling(rolling_beta_window).cov(risk_factors['SMB']) / returns.rolling(rolling_beta_window).var()
    rolling_beta_HML = returns.rolling(rolling_beta_window).cov(risk_factors['HML']) / returns.rolling(rolling_beta_window).var()
    rolling_beta_UMD = returns.rolling(rolling_beta_window).cov(risk_factors['Mom']) / returns.rolling(rolling_beta_window).var()
    
    rolling_beta_SMB.plot(figsize=(12,6), color='steelblue', alpha=0.7, ax=ax, **kwargs)
    rolling_beta_HML.plot(color='orangered', alpha=0.7, ax=ax, **kwargs)
    rolling_beta_UMD.plot(color='forestgreen', alpha=0.7, ax=ax, **kwargs)

    ax.axhline(0.0, color='black', lw=.75)
    ax.legend(['Small-Caps (SMB)',
               'High-Growth (HML)',
               'Momentum (UMD)'],
              loc=legend_loc)

    plt.grid()
    plt.show()
    
def plot_sharpe(many_series):
    ax = plt.gca()
    ax.set_title('Rolling Sharpe')
    
    for (rp, rf, sp, title, color) in many_series:
        # sharpe since inception
        sharpe = (rp - rf).mean() / sp * np.sqrt(252)
        print(f'{title} Sharpe Ratio (Since Inception): {round(sharpe, 2)}')
        # rolling sharpe
        sharpe_roll = (rp - rf).rolling(21 * 6).mean() / rp.rolling(21 * 6).std() * np.sqrt(21 * 6)
        sharpe_roll.rename(f'{title} Rolling Sharpe', inplace=True)
        sharpe_roll.plot(figsize=(12,6), color=color)
        mean_sharpe = round(sharpe_roll.mean(), 2)
        if mean_sharpe:
            plt.axhline(mean_sharpe, color=color, ls='--', label=f'{title} mean = {mean_sharpe}')
        plt.ylabel(title)
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
def plot_sortino(many_series):
    ax = plt.gca()
    ax.set_title('Rolling Sortino')
    
    for (rp, rf, downside, title, color) in many_series:
        # sortino since inception
        sortino = (rp - rf).mean() / downside.std() * np.sqrt(252)
        print(f'{title} Sortino Ratio (Since Inception): {round(sortino, 2)}')
        # rolling sortino
        rolling_downside_std = lambda x: np.std(x[x < 0])
        sortino_roll = (rp - rf).rolling(21 * 6).mean() / rp.rolling(21 * 6).apply(rolling_downside_std) * np.sqrt(21 * 6)
        sortino_roll.rename(f'{title} Rolling Sortino', inplace=True)
        sortino_roll.plot(figsize=(12,6), color=color)
        mean_sortino = round(sortino_roll.mean(), 2)
        if mean_sortino:
            plt.axhline(mean_sortino, color=color, ls='--', label=f'{title} mean = {mean_sortino}')
        plt.ylabel(title)
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
def plot_ir(many_series):
    ax = plt.gca()
    ax.set_title('Rolling Information Ratio')
    
    for (rp, rb, title, color) in many_series:
        # ir since inception
        ir = (rp - rb).mean() / (rp - rb).std() * np.sqrt(252)
        print(f'{title} Information Ratio (Since Inception): {round(ir, 2)}')
        # rolling ir
        ir_roll = (rp - rb).rolling(21 * 6).mean() / (rp - rb).rolling(21 * 6).std() * np.sqrt(21 * 6)
        ir_roll.rename(f'{title} Rolling Information Ratio', inplace=True)
        ir_roll.plot(figsize=(12,6), color=color)
        mean_ir = round(ir_roll.mean(), 2)
        if mean_ir:
            plt.axhline(mean_ir, color=color, ls='--', label=f'{title} mean = {mean_ir}')
        plt.ylabel(title)
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
def plot_beta(many_series):
    ax = plt.gca()
    ax.set_title('Rolling β')
    
    for (r, rp, rb, title, color) in many_series:
        # beta since inception
        beta = r.cov().iloc[0,1] / rb.var()
        print(f'{title} Beta (Since Inception): {round(beta, 2)}')
        # rolling beta
        beta_roll = rp.rolling(21 * 6).cov(rb) / rb.rolling(21 * 6).var()
        beta_roll.rename(f'{title} Rolling Beta', inplace=True)
        beta_roll.plot(figsize=(12,6), color=color)
        mean_beta = round(beta_roll.mean(), 2)
        if mean_beta:
            plt.axhline(mean_beta, color=color, ls='--', label=f'{title} mean = {mean_beta}')
        plt.ylabel(title)
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()    

def plot_pos_confidence_indicators(res_df):
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 6))
    ts = res_df.index
    # plot pos bubbles
    ax1_0 = ax1.twinx()
    ax1.plot(ts, res_df['price'], color='black', linewidth=0.75)
    # ax1_0.plot(compatible_date, pos_lst, label='pos bubbles', color='gray', alpha=0.5)
    ax1_0.plot(ts, res_df['pos_conf'], label='bubble indicator (pos)', color='red', alpha=0.5)
    # set grids
    ax1.grid(which='major', axis='both')
    # set labels
    ax1.set_ylabel('ln(p)')
    ax1_0.set_ylabel('bubble indicator (pos)')
    ax1_0.legend(loc=2)
    plt.title('Positive LPPLS Conf Indicator')
    plt.fill_between(ts, res_df['pos_conf'], color='red', alpha=0.5)
    plt.show()