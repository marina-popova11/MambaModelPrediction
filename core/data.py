import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import os

from creating_time_series import create_features, create_discrete_targets_quantile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data_internal(df, lookback=60, n_bins=5, classification_mode=True):
    df = df.sort_values('Open time').reset_index(drop=True)
    df['Open time'] = pd.to_datetime(df['Open time'])
    if classification_mode:
        if 'target_class' not in df.columns:
            raise ValueError("Колонка 'target_class' не найдена. Запустите сначала create_discrete_targets_quantile.")

        target_col_raw = 'target_class'

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['target_encoded'] = le.fit_transform(df[target_col_raw])

        target_col = 'target_encoded'
        class_labels = list(le.classes_) # Сохраняем порядок классов: ['neutral', 'strong_down'...]

        class_medians = df.groupby(target_col)['target_return'].median().to_dict()

    else:
        target_col = 'target_return'
        class_labels = None
        class_medians = None

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'returns', 'log_return',
        'norm_return', 'norm_return_lag_1', 'norm_return_lag_2',
        'norm_return_lag_3', 'norm_return_lag_6', 'norm_return_lag_12',
        'norm_return_lag_24', 'norm_return_sma_12', 'norm_return_sma_24',
        'norm_return_ema_12', 'volatility_returns', 'volatility_log', 'rsi',
        'vol_ratio', 'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3',
        'volume_ma', 'volume_ratio', 'volume_norm'
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    from sklearn.preprocessing import RobustScaler
    scaler_features = RobustScaler(quantile_range=(10.0, 90.0))

    scaler_features.fit(train_df[feature_cols])
    train_scaled = scaler_features.transform(train_df[feature_cols])
    val_scaled = scaler_features.transform(val_df[feature_cols])
    test_scaled = scaler_features.transform(test_df[feature_cols])

    if classification_mode:
        y_train = train_df[target_col].values.astype(np.int64)
        y_val = val_df[target_col].values.astype(np.int64)
        y_test = test_df[target_col].values.astype(np.int64)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler_target = StandardScaler()
        scaler_target.fit(train_df[[target_col]])
        y_train = scaler_target.transform(train_df[[target_col]]).flatten()
        y_val = scaler_target.transform(val_df[[target_col]]).flatten()
        y_test = scaler_target.transform(test_df[[target_col]]).flatten()

    def create_windows(features, targets):
        X, y = [], []
        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            y.append(targets[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64 if classification_mode else np.float32)

    X_train, y_train = create_windows(train_scaled, y_train)
    X_val, y_val = create_windows(val_scaled, y_val)
    X_test, y_test = create_windows(test_scaled, y_test)

    time_train = df['Open time'].iloc[lookback:train_end].values
    time_val = df['Open time'].iloc[train_end+lookback:val_end].values
    time_test = df['Open time'].iloc[val_end+lookback:].values

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    if classification_mode:
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Class distribution in Train: {dict(zip(unique, counts))}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler_features': scaler_features,
        'feat_cols': feature_cols,
        'time_index': {'train': time_train, 'val': time_val, 'test': time_test},
        'classification_mode': classification_mode,
        'class_labels': class_labels,      # Чтобы знать, что класс 0 - это 'neutral' и т.д.
        'class_medians': class_medians,    # Чтобы понимать среднюю доходность класса
        'n_classes': n_bins if classification_mode else None,
        'target_col': target_col
    }

def get_data_loaders(filename, lookback=60, batch_size=128, cache_file='data_cache.pkl',
                     n_bins=5, classification_mode=True, force_reprocess=False):
    cache_key = f"{cache_file}_{lookback}_{n_bins}_{classification_mode}"

    if os.path.exists(cache_key) and not force_reprocess:
        print(f"Загрузка кэшированных данных из {cache_key}...")
        with open(cache_key, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        print(f"Обработка данных из {filename} (это займет время)...")
        df = pd.read_csv(filename)
        data_dict = prepare_data_internal(df, lookback, n_bins, classification_mode)

        with open(cache_key, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Кэш сохранен в {cache_key}")

    def make_loader(X, y, shuffle=False):
        if classification_mode:
            dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        else:
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(data_dict['X_train'], data_dict['y_train'], shuffle=True)
    val_loader = make_loader(data_dict['X_val'], data_dict['y_val'], shuffle=False)
    test_loader = make_loader(data_dict['X_test'], data_dict['y_test'], shuffle=False)

    print(f"Данные готовы: Train={len(data_dict['X_train'])}, Val={len(data_dict['X_val'])}, Test={len(data_dict['X_test'])}")
    print(f"Режим: {'Классификация' if classification_mode else 'Регрессия'}")
    if classification_mode:
        print(f"Количество классов: {data_dict['n_classes']}")
        unique, counts = np.unique(data_dict['y_train'], return_counts=True)
        print(f"Распределение классов в train: {dict(zip(unique, counts))}")

    return data_dict, train_loader, val_loader, test_loader