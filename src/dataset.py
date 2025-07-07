import torch
from torch.utils.data import Dataset, DataLoader

class CoordPriceDataset(Dataset):
    def __init__(self, coords, prices):
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.prices = torch.tensor(prices, dtype=torch.float32)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.prices[idx]

def get_dataloaders(train, val, test, batch_size=64):
    train_ds = CoordPriceDataset(*train)
    val_ds = CoordPriceDataset(*val)
    test_ds = CoordPriceDataset(*test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    return train_dl, val_dl, test_dl
