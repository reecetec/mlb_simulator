import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.data_utils import query_mlb_db

import torch
from torch.utils.data import Dataset


class SQLiteDataset(Dataset):
    """Creates dataset class for model training
    """
    def __init__(self, query):
        print('Querying dataset...')
        self.data = query_mlb_db(query)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor(row[1:].values, dtype=torch.float32)
        label = torch.tensor(row[0], dtype=torch.long)
        return features, label