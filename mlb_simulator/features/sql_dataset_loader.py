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
import copy


class SQLiteDataset(Dataset):
    """Creates dataset class for model training
    """
    def __init__(self, query):
        
        #load query
        self.raw_data = query_mlb_db(query)
        
        #set target col (should always be first col in query)
        self.target_col = self.raw_data.columns[0]

        #get categorical columns
        self.cat_columns = self.get_object_columns()
        #one hot the target, as well as the pitch type.
        self.one_hot_columns = [self.target_col] #, 'pitch_type']

        #encode categorical values and retain mapping to use later:
        self.label_encoders = {}
        self.one_hot_encoded_dfs = {}

        for col in self.one_hot_columns:
            one_hot = OneHotEncoder(sparse_output=False)
            target_encoded = one_hot.fit_transform(self.raw_data[[col]])
            target_names = one_hot.get_feature_names_out([col])
            target_encoded_df = pd.DataFrame(target_encoded, columns=target_names)

            self.label_encoders[col] = copy.deepcopy(one_hot)

            # save number of classes if this is the target col
            if col == self.target_col:
                self.num_target_classes = len(target_names)

            self.one_hot_encoded_dfs[col] = target_encoded_df
        

        #encode rest of columns regularly
        for col in self.cat_columns:
            if col in self.one_hot_columns: 
                continue
            
            le = LabelEncoder()

            self.raw_data[col] = le.fit_transform(self.raw_data[col])
            self.label_encoders[col] = copy.deepcopy(le)

        #get the encoded data
        self.encoded_data = pd.concat(list(self.one_hot_encoded_dfs.values()) + 
                                 [self.raw_data.drop(self.one_hot_columns, axis=1)]
                                , axis=1)
        #input layer size for input to nn
        self.input_layer_size = len(self.encoded_data.columns) - self.num_target_classes

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        row = self.encoded_data.iloc[idx]
        features = torch.tensor(row.iloc[self.num_target_classes:].values, dtype=torch.float32)
        label = torch.tensor(row.iloc[0:self.num_target_classes], dtype=torch.float32)
        return features, label
    
    #methods to map to labels
    def get_object_columns(self):
        return [col for col in self.raw_data.columns if self.raw_data[col].dtype == 'object']
