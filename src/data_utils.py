import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['latitude', 'longitude', 'price'])
    
    coords = df[['latitude', 'longitude']].values.astype(np.float32)
    prices = df[['price']].values.astype(np.float32)

    coord_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()
    
    coords_scaled = coord_scaler.fit_transform(coords)
    prices_scaled = price_scaler.fit_transform(prices)

    return coords_scaled, prices_scaled, coord_scaler, price_scaler

def save_scalers(coord_scaler, price_scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(coord_scaler, os.path.join(output_dir, 'coord_scaler.pkl'))
    joblib.dump(price_scaler, os.path.join(output_dir, 'price_scaler.pkl'))

def split_data(coords, prices, train_ratio=0.7, val_ratio=0.15):
    N = coords.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (coords[train_idx], prices[train_idx]), \
           (coords[val_idx], prices[val_idx]), \
           (coords[test_idx], prices[test_idx])
