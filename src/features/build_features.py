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
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import logging
import torch
import pandas as pd

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PITCH_CHARACTERISITCS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'spin_axis', 'pfx_x', 'pfx_z',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
    'plate_x', 'plate_z'
]

def get_xgb_set(df, target_col, split=False):
    encoders = {} # to store encoders

    le = LabelEncoder()
    y = pd.DataFrame(le.fit_transform(df[target_col]), columns=[target_col])

    encoders[target_col] = deepcopy(le)

    X = df.drop(target_col, axis=1)

    X, encoders = encode_cat_cols(X, encoders)   

    if split:
        X_train, y_train, X_test, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        return X_train, y_train, X_test, y_test, encoders
    
    return X, y, encoders

def encode_cat_cols(X, encoders_dict):
    object_cols = [col for col in X.columns if X[col].dtype == 'object']
    for col in object_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders_dict[col] = deepcopy(le)

    return X, encoders_dict

def get_pitch_outcome_dataset_xgb(batter_id, split=False, backtest_date=''):

    if backtest_date:
       backtest_date = f'and game_date <= "{backtest_date}"' 

    query_str = f"""
        select 
            case
                when description='swinging_strike' or description='swinging_strike_blocked' or description='called_strike' or description='foul_tip' 
                    or description='swinging_pitchout' then 'strike'
                when description='foul' or description='foul_pitchout' then 'foul'
                when description='ball' or description='blocked_ball' or description='pitchout' then 'ball'
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
        {backtest_date}
        order by game_date asc, at_bat_number asc, pitch_number asc;
    """

    df = query_mlb_db(query_str)

    target_col = 'pitch_outcome'
    return get_xgb_set(df, target_col, split)
    
def get_pitch_outcome_dataset(batter_id, batch_size=32, shuffle=False):

    query_str = f"""
        select 
            case
                when description='swinging_strike' or description='swinging_strike_blocked' or description='called_strike' or description='foul_tip' 
                    or description='swinging_pitchout' then 'strike'
                when description='foul' or description='foul_pitchout' then 'foul'
                when description='ball' or description='blocked_ball' or description='pitchout' then 'ball'
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

def get_pitch_outcome_dataset_general(cluster_id, stands, batch_size=32, shuffle=False):
    
    cluster_query = f'select batter from BatterStrikezoneCluster where cluster={cluster_id};'
    
    batter_ids_in_cluster_df = query_mlb_db(cluster_query)
    batter_ids_in_cluster = batter_ids_in_cluster_df['batter'].values

    sql_fmt_ids = ', '.join(map(str, batter_ids_in_cluster))


    query_str = f"""
        select 
            case
                when description='swinging_strike' or description='swinging_strike_blocked' or description='called_strike' or description='foul_tip' 
                    or description='swinging_pitchout' then 'strike'
                when description='foul' or description='foul_pitchout' then 'foul'
                when description='ball' or description='blocked_ball' or description='pitchout' then 'ball'
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
        and stand='{stands}'
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


    # kikuchi: 579328
    query_str = f"""
    SELECT 
        
        release_speed, release_spin_rate, release_extension,
        release_pos_x, release_pos_y, release_pos_z,
        spin_axis, pfx_x, pfx_z,
        vx0, vy0, vz0,
        ax, ay, az,
        plate_x, plate_z,
        
        CASE 
            WHEN stand is 'L' THEN 0
            ELSE 1
        END as stand,
        ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY game_date, at_bat_number, pitch_number) /100 AS cumulative_pitch_number
    FROM 
        Statcast
    WHERE 
        pitcher = {pitcher_id} and pitch_Type = 'FF'
        AND release_speed IS NOT NULL
        AND release_spin_rate IS NOT NULL
        AND release_extension IS NOT NULL
        AND release_pos_x IS NOT NULL
        AND release_pos_y IS NOT NULL
        AND release_pos_z IS NOT NULL
        AND spin_axis IS NOT NULL
        AND pfx_x IS NOT NULL
        AND pfx_z IS NOT NULL
        AND vx0 IS NOT NULL
        AND vy0 IS NOT NULL
        AND vz0 IS NOT NULL
        AND ax IS NOT NULL
        AND ay IS NOT NULL
        AND az IS NOT NULL
        AND plate_x IS NOT NULL
        AND plate_z IS NOT NULL
    ORDER BY 
        game_date ASC, 
        at_bat_number ASC,
        pitch_number ASC;
    """

    pitch_data_df = query_mlb_db(query_str)

    conditioning_cols = ['stand', 'cumulative_pitch_number']


    conditioning_df = pitch_data_df[conditioning_cols]
    non_conditioning_df = pitch_data_df.drop(conditioning_cols, axis=1)

    conditioning_tensor = torch.tensor(conditioning_df.values, dtype=torch.float32)
    non_conditioning_tensor = torch.tensor(non_conditioning_df.values, dtype=torch.float32)

    logger.info(f'Loading pitch dataset for pitcher {pitcher_id}')
    logger.info(f'Data successfully queried/transformed for {pitcher_id}')

    return non_conditioning_tensor, conditioning_tensor

def get_sequencing_dataset(pitcher, backtest_date=''):

    #for backtesting, add date query to only use past data
    if backtest_date:
       backtest_date = f'and game_date <= "{backtest_date}"' 
    
    pitcher_query_str = f"""
        SELECT game_year, pitch_type, batter, pitch_number, strikes, balls, outs_when_up,
            CASE
                when stand='R' then 1
                else 0
            END AS stand,
            CASE
                when on_1b is not null then 1
                else 0
            END AS on_1b,
            CASE
                when on_2b is not null then 1
                else 0
            END AS on_2b,
            CASE
                when on_3b is not null then 1
                else 0
            END AS on_3b,
            CASE
                when fld_score - bat_score > 0 then 1
                when fld_score - bat_score = 0 then 0
                else -1
            END AS is_winning,
            LAG(pitch_type) OVER (PARTITION BY game_pk, pitcher, at_bat_number ORDER BY pitch_number) AS prev_pitch,
            ROW_NUMBER() OVER (PARTITION BY game_pk, pitcher ORDER BY at_bat_number, pitch_number) AS cumulative_pitch_number
        FROM Statcast
        WHERE pitcher = {pitcher}
            AND pitch_type IS NOT NULL
            and pitch_type <> 'PO'
            AND game_type <> 'E' || 'S'
            and game_pk in (
                select distinct game_pk
                from Statcast
                where pitcher = {pitcher}
                    {backtest_date}
                order by game_date desc
                limit 34
            )
        ORDER BY game_year, at_bat_number, pitch_number
    """
    pitcher_df = query_mlb_db(pitcher_query_str).set_index('batter')
    pitch_arsenal = pitcher_df['pitch_type'].unique()

    sql_pitch_arsenal = ', '.join(pitch_arsenal)
    
    #get datasets
    batter_query = lambda table: f"select batter, {sql_pitch_arsenal} from {table}"
    strike_df = query_mlb_db(batter_query('BatterStrikePctByPitchType')).set_index('batter').add_suffix('_strike')
    woba_df = query_mlb_db(batter_query('BatterAvgWobaByPitchType')).set_index('batter').add_suffix('_woba')


    df = pitcher_df.merge(strike_df, left_index=True, right_index=True, how='left')
    df = df.merge(woba_df, left_index=True, right_index=True, how='left')
    df.reset_index(drop=True, inplace=True)

    target_col = 'pitch_type'
    encoders = {} # to store encoders

    le = LabelEncoder()
    y = pd.DataFrame(le.fit_transform(df['pitch_type']), columns=[target_col])

    encoders[target_col] = deepcopy(le)

    X = df.drop(target_col, axis=1)

    X, encoders = encode_cat_cols(X, encoders)   
    
    return X, y, encoders, pitch_arsenal


def get_pitches(pitcher_id, opposing_stance, pitch_type, backtest_date=''):
    if backtest_date:
       backtest_date = f'and game_date <= "{backtest_date}"' 
    pitch_df =  query_mlb_db(f'''select 
        {', '.join(PITCH_CHARACTERISITCS)}, batter, strikes, balls
        from Statcast
        where pitcher={pitcher_id} and
        stand="{opposing_stance}" and
        pitch_type="{pitch_type}"
        and
        {' & '.join(PITCH_CHARACTERISITCS)} 
        is not null
        AND game_type <> 'E' || 'S'
        {backtest_date}
        ''')

    if (l := len(pitch_df)) < 50:
        logger.warning(f'low pitch count ({l}) for {opposing_stance} {pitch_type}')

    sz_df = query_mlb_db('select * from BatterStrikezoneLookup')

    df = pd.merge(pitch_df, sz_df, on='batter', how='left').drop(['batter'], axis=1)

    return df

if __name__ == '__main__':
    #train_dataloader, val_dataloader, num_features, num_classes, label_encoders = get_pitch_outcome_dataset(665489,batch_size=2)
    #for features, labels in train_dataloader:
        #print(f'num features: {num_features} and num classes: {num_classes}')
        #print(f'first batch features: {features}\n\n first batch labels: {labels}')
        #print(f'label encoder: {label_encoders}')
        #break
    #train_dataloader, val_dataloader, num_classes, num_features, label_encoders = get_pitch_outcome_dataset_general(5,stands='R',batch_size=2)
    #for features, labels in train_dataloader:
        #print(f'num features: {num_features} and num classes: {num_classes}')
        #print(f'first batch features: {features}\n\n first batch labels: {labels}')
        #print(f'label encoder: {label_encoders}')
        #print(f'training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}')
       #break
    kukuchi = 579328
    jones = 683003

    pitcher = kukuchi

    all_pitches = get_pitches(pitcher,'L','CU')
    print(all_pitches.head())
