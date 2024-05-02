#using functions from data utils, constructs a featureset for a model
#from src.data.data_utils import query_mlb_db
#from src.features.sql_dataset_loader import SQLiteDataset

import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.data_utils import query_mlb_db
from features.sql_dataset_loader import SQLiteDataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_pitch_outcome_dataset(batter_id, batch_size=32, shuffle=False):
    pitch_characteristics = '''pitch_type, release_speed, release_spin_rate, release_extension, release_pos_x, release_pos_y,
                                release_pos_z, pfx_x, pfx_z, plate_x, plate_z, vx0, vy0, vz0, ax, ay, az, zone, sz_top, sz_bot'''
    pitcher_characteristics = 'p_throws'
    game_state = 'pitch_number, strikes, balls, outs_when_up, bat_score - fld_score as score_delta'
    query_str = f"""
    select type, {pitch_characteristics}, {pitcher_characteristics}, {game_state}
    from Statcast 
    where batter={batter_id}
    order by game_date asc
    """

    #create pytorch dataset
    dataset = SQLiteDataset(query_str)

    #ensure shuffle is false -> uses oldest data for training, newest for val.
    train_set, val_set = train_test_split(dataset, test_size=0.25, shuffle=False)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    print('Dataset loaded')

    return train_dataloader, val_dataloader, dataset.num_features, dataset.num_target_classes


def get_pitch_generation_features():
    pass

def get_bat_outcome_features():
    pass

def get_field_outcome_features():
    pass

def feature_class_mapping():
    return 

if __name__ == '__main__':
    train_dataloader, val_dataloader, num_features, num_classes = get_pitch_outcome_dataset(665489,batch_size=2)
    for features, labels in train_dataloader:
        print(f'num features: {num_features} and num classes: {num_classes}')
        print(f'first batch features: {features}\n\n first batch labels: {labels}')
        break