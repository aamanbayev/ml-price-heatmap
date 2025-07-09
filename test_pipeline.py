#%%
from src.data_utils import load_and_preprocess_data, split_data
from src.dataset import get_dataloaders

# Step 1: Load and preprocess
coords, prices, coord_scaler, price_scaler = load_and_preprocess_data("data/listings.csv")

# Step 3: Split the data
train_data, val_data, test_data = split_data(coords, prices)

# Step 4: Load into PyTorch Dataloaders
train_dl, val_dl, test_dl = get_dataloaders(train_data, val_data, test_data)

# Step 5: Print some batches
for coords_batch, prices_batch in train_dl:
    print("Coordinates:", coords_batch.shape)
    print("Prices:", prices_batch.shape)
    break
