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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_pitch_outcome_dataset(batter_id, batch_size=32, shuffle=False):

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
        is not null
        order by game_date asc, at_bat_number asc;
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

def get_pitch_outcome_dataset_general(cluster_id, batch_size=32, shuffle=False):
    
    cluster_query = f'select batter from BatterStrikezoneCluster where cluster={cluster_id};'
    
    batter_ids_in_cluster_df = query_mlb_db(cluster_query)
    batter_ids_in_cluster = batter_ids_in_cluster_df['batter'].values

    sql_fmt_ids = ', '.join(map(str, batter_ids_in_cluster))


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
        where batter in ({sql_fmt_ids})
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
        is not null
        order by game_date asc, at_bat_number asc;
    """


    logger.info(f'Loading dataset for cluster {cluster_id}')

    #create pytorch dataset
    dataset = SQLiteDataset(query_str)

    logger.info(f'Data successfully queried/transformed for cluster {cluster_id}')

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
    #train_dataloader, val_dataloader, num_features, num_classes, label_encoders = get_pitch_outcome_dataset(665489,batch_size=2)
    #for features, labels in train_dataloader:
        #print(f'num features: {num_features} and num classes: {num_classes}')
        #print(f'first batch features: {features}\n\n first batch labels: {labels}')
        #print(f'label encoder: {label_encoders}')
        #break
    train_dataloader, val_dataloader, num_classes, num_features, label_encoders = get_pitch_outcome_dataset_general(5,batch_size=2)
    for features, labels in train_dataloader:
            print(f'num features: {num_features} and num classes: {num_classes}')
            print(f'first batch features: {features}\n\n first batch labels: {labels}')
            print(f'label encoder: {label_encoders}')
            print(f'training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}')
            break