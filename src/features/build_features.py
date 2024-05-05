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

import logging

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

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
    limit 100
    """

    query_str = f"""
        SELECT type, {pitch_characteristics}, {pitcher_characteristics}, {game_state}
        FROM Statcast 
        WHERE batter={batter_id} AND pitch_type IS NOT NULL AND release_speed IS NOT NULL AND release_spin_rate IS NOT NULL AND release_extension IS NOT NULL AND release_pos_x IS NOT NULL AND release_pos_y IS NOT NULL AND release_pos_z IS NOT NULL AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL AND plate_x IS NOT NULL AND plate_z IS NOT NULL AND vx0 IS NOT NULL AND vy0 IS NOT NULL AND vz0 IS NOT NULL AND ax IS NOT NULL AND ay IS NOT NULL AND az IS NOT NULL AND zone IS NOT NULL AND sz_top IS NOT NULL AND sz_bot IS NOT NULL AND p_throws IS NOT NULL AND pitch_number IS NOT NULL AND strikes IS NOT NULL AND balls IS NOT NULL AND outs_when_up IS NOT NULL AND (bat_score - fld_score) IS NOT NULL
        ORDER BY game_date, at_bat_number ASC
    """

    query_str = f"""
        select 
            case
                when description='swinging_strike' or description='swinging_strike_blocked' or description='called_strike' or description='foul_tip' then 'strike'
                when description='foul' then 'foul'
                when description='ball' or description='blocked_ball' then 'ball'
                when description='hit_by_pitch' then 'hit_by_pitch'
                when description='hit_into_play' then 'hit_into_play'
                else NULL
            end as pitch_outcome,
            
            p_throws, pitch_number, strikes, balls, outs_when_up,
            
            case
                when bat_score > fld_score then 1
                when bat_score < fld_score then -1
                else 0
            end as is_winning,
            
            release_speed, 
            release_spin_rate, 
            release_extension,

            release_pos_x,
            release_pos_y,
            release_pos_z,
            
            spin_axis,
            pfx_x, pfx_z, 
            
            vx0, vy0, vz0,
            ax, ay, az,
            plate_x, plate_z
            
        from Statcast
        where batter={batter_id}
        and pitch_outcome & p_throws & pitch_number & strikes & balls & outs_when_up & is_winning &
            release_speed &
            release_spin_rate &
            release_extension &

            release_pos_x &
            release_pos_y &
            release_pos_z &
            
            spin_axis &
            pfx_x & pfx_z &
            
            vx0 & vy0 & vz0 &
            ax & ay & az &
            plate_x & plate_z
        is not null;
    """


    logger.info(f'Loading dataset for {batter_id}')

    #create pytorch dataset
    dataset = SQLiteDataset(query_str)

    logger.info(f'Data successfully queried/transformed for {batter_id}')

    #ensure shuffle is false -> uses oldest data for training, newest for val.
    train_set, val_set = train_test_split(dataset, test_size=0.25, shuffle=False)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    return train_dataloader, val_dataloader, dataset.num_target_classes, dataset.input_layer_size, dataset.label_encoders


def get_pitch_generation_features():
    pass

def get_bat_outcome_features():
    pass

def get_field_outcome_features():
    pass

def feature_class_mapping():
    return 

if __name__ == '__main__':
    train_dataloader, val_dataloader, num_features, num_classes, label_encoders = get_pitch_outcome_dataset(665489,batch_size=2)
    for features, labels in train_dataloader:
        print(f'num features: {num_features} and num classes: {num_classes}')
        print(f'first batch features: {features}\n\n first batch labels: {labels}')
        print(f'label encoder: {label_encoders}')
        break