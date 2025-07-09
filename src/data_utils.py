import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)[['latitude', 'longitude', 'price']].dropna()
        
    coords = df[['latitude', 'longitude']].values.astype(np.float32)
    prices = df['price'].apply(lambda s: s.removeprefix('$').replace(',','')).values.astype(np.float32).reshape((-1,1))


    coord_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()
    
    coords_scaled = coord_scaler.fit_transform(coords)
    prices_scaled = price_scaler.fit_transform(prices)

    return coords_scaled, prices_scaled, coord_scaler, price_scaler
