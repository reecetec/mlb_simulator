import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.data_utils import query_mlb_db

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd


class SQLiteDataset(Dataset):
    """Creates dataset class for model training
    """
    def __init__(self, query):
        
        #load query
        print('Loading dataset...')
        self.raw_data = query_mlb_db(query)
        
        #get categorical columns
        print('Converting labels to integers...')
        self.cat_columns = self.get_object_columns()

        self.one_hot = OneHotEncoder(sparse_output=False)
        self.target_col = self.raw_data.columns[0]
        target_encoded = self.one_hot.fit_transform(self.raw_data[[self.target_col]])

        target_names = self.one_hot.get_feature_names_out([self.target_col])
        self.num_target_labels = len(target_names)

        target_encoded_df = pd.DataFrame(target_encoded, columns=target_names)
        
        #encode categorical values
        self.le = LabelEncoder()
        for col in self.cat_columns:
            if col != self.target_col: 
                self.raw_data[col] = self.le.fit_transform(self.raw_data[col])

        self.encoded_data = pd.concat([target_encoded_df, self.raw_data.drop(self.target_col, axis=1)], axis=1)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        row = self.encoded_data.iloc[idx]
        features = torch.tensor(row[self.num_target_labels:].values, dtype=torch.float32)
        label = torch.tensor(row[0:self.num_target_labels], dtype=torch.long)
        return features, label
    
    #methods to map to labels
    def get_object_columns(self):
        return [col for col in self.raw_data.columns if self.raw_data[col].dtype == 'object']
