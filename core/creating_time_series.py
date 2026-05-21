import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def create_features(df, lookback=24, norm_window=24):
    """Создание признаков для временного ряда
    Вход: Log-Return + Z-score
    Выход: Log-Return (будущий)"""
    df = df.copy()

    df['returns'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    rolling_mean = df['log_return'].rolling(window=norm_window).mean()
    rolling_std = df['log_return'].rolling(window=norm_window).std()
    df['norm_return'] = (df['log_return'] - rolling_mean) / (rolling_std + 1e-9)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'norm_return_lag_{lag}'] = df['norm_return'].shift(lag)

    df['norm_return_sma_12'] = df['norm_return'].rolling(12).mean()
    df['norm_return_sma_24'] = df['norm_return'].rolling(24).mean()
    df['norm_return_ema_12'] = df['norm_return'].ewm(span=12).mean()

    df['volatility_returns'] = df['returns'].rolling(window=lookback).std()
    df['volatility_log'] = df['log_return'].rolling(window=lookback).std()
    df['rsi'] = compute_rsi_stable(df['log_return'], period=14)

    long_vol = df['log_return'].rolling(window=lookback*3).std()
    df['vol_ratio'] = df['volatility_log'] / (long_vol + 1e-9)

    for lag in [1, 2, 3]:
        df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)

    forecast_horizon = 24
    future_close = df['Close'].shift(-forecast_horizon)
    current_close = df['Close']
    df['target_return'] = np.log(future_close / current_close.shift(-1))
    df['target_direction'] = (df['target_return'] > 0).astype(int)

    df['volume_ma'] = df['Volume'].rolling(24).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma'] + 1e-9)

    vol_rolling_mean = df['Volume'].rolling(window=norm_window).mean()
    vol_rolling_std = df['Volume'].rolling(window=norm_window).std()
    df['volume_norm'] = (df['Volume'] - vol_rolling_mean) / (vol_rolling_std + 1e-9)

    df.dropna(inplace=True)
    return df

def compute_rsi_stable(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def create_discrete_targets_quantile(df, target_col='target_return', n_bins=5):
    """Дискретизация по квантилям — в каждом бине равное количество токенов"""
    df = df.copy()

    target = df[target_col].clip(
        lower=df[target_col].quantile(0.01),
        upper=df[target_col].quantile(0.99)
    )

    df['target_class'] = pd.qcut(target, q=n_bins, labels=False, duplicates='drop')
    bin_edges = pd.qcut(target, q=n_bins, retbins=True, duplicates='drop')[1]
    labels = []
    mid = n_bins // 2

    for i in range(n_bins):
        if i < mid:
            labels.append(f'down_{mid - i}')
        elif i == mid and n_bins % 2 == 1:
            labels.append('neutral')
        else:
            labels.append(f'up_{i - mid + 1}')

    if n_bins == 5:
        labels = ['strong_down', 'weak_down', 'neutral', 'weak_up', 'strong_up']

    df['target_label'] = pd.cut(
        df[target_col],
        bins=bin_edges,
        labels=labels,
        include_lowest=True
    )

    return df, bin_edges

df = pd.read_csv("btc_1h_data_2020-2025.csv")
df_features = create_features(df)
df_with_classes, edges = create_discrete_targets_quantile(df_features, n_bins=5)