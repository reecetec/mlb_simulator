from mlb_simulator.data.data_utils import query_mlb_db
import pandas as pd

# pitch characteristics to be generated/put into pitch outcome model
_pitch_characteristics = [
    'release_speed', 'release_spin_rate',
    'plate_x', 'plate_z',
]
_pitch_outcome_game_state_features = 'p_throws strikes balls'.split()

PITCH_OUTCOME_FEATURES = _pitch_characteristics + \
                         _pitch_outcome_game_state_features
PITCH_OUTCOME_TARGET_COL = 'pitch_outcome'


def get_pitch_outcome_data(batter_id: int,
                           backtest_date=None) -> tuple[pd.DataFrame, str]:
    f"""Function to get training data for a batter's pitch outcome model

    For a given player, returns the model features for the pitch outcome model 
    {PITCH_OUTCOME_FEATURES} and the pitch_outcome (the target for this model).
    If a backtest date is supplied, will only return data before the backtest
    date.

    Args:
        batter_id (int): the batter's mlb id
        backtest_date (str): get data up to backtest_date (default None)

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: a dataframe containing the query results
            - str: the name of the target col for this model

    Example:
        >>> df = get_pitch_outcome_data(665742, backtest_date='2022-01-01')
        >>> df.head()
    """

    # format input backtest date as query
    if backtest_date is not None:
        backtest_date = f"and game_date <= date('{backtest_date}')"
    else:
        backtest_date = ''

    query_str = f"""
        select 
            case
                when description='swinging_strike' or
                     description='swinging_strike_blocked' or
                     description='called_strike' or description='foul_tip' or
                     description='swinging_pitchout'
                then 'strike'
                when description='foul' or
                     description='foul_pitchout' 
                then 'foul'
                when description='ball' or
                     description='blocked_ball' or
                     description='pitchout' 
                then 'ball'
                when description='hit_by_pitch' then 'hit_by_pitch'
                when description='hit_into_play' then 'hit_into_play'
                else NULL
            end as {PITCH_OUTCOME_TARGET_COL},
            {', '.join(PITCH_OUTCOME_FEATURES)}
        from Statcast
        where batter={batter_id}
        and {' & '.join(PITCH_OUTCOME_FEATURES)} 
        is not null
        {backtest_date}
        order by game_date asc, at_bat_number asc, pitch_number asc;
    """

    df = query_mlb_db(query_str)

    return df, PITCH_OUTCOME_TARGET_COL


#=============================================================================
def get_xgb_set(df, target_col, split=False, test_size=0.1):
    encoders = {} # to store encoders

    le = LabelEncoder()
    y = pd.DataFrame(le.fit_transform(df[target_col]), columns=[target_col])

    encoders[target_col] = deepcopy(le)

    X = df.drop(target_col, axis=1)

    X, encoders = encode_cat_cols(X, encoders)   

    if split:
        X_train, y_train, X_test, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)
        return X_train, X_test, y_train, y_test, encoders
    
    return X, y, encoders

def get_xgb_set_regression(df, target_cols, split=False, test_size=0.1):
    encoders = {}
    y = df[target_cols]
    X = df.drop(target_cols, axis=1)
    X, encoders = encode_cat_cols(X, encoders)

    if split:
        X_train, y_train, X_test, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)
        return X_train, X_test, y_train, y_test, encoders
    
    return X, y, encoders

def get_hit_classification_dataset(split=False, backtest_date=''):
    if backtest_date:
       backtest_date = f'and game_date <= "{backtest_date}"' 
    
    df = query_mlb_db(f"""
    SELECT
        CASE
            WHEN events IN ('single') THEN 'single'
            WHEN events IN ('double') THEN 'double'
            WHEN events IN ('triple') THEN 'triple'
            WHEN events IN ('home_run') THEN 'home_run'
            WHEN events IN ('field_out') THEN 'field_out'
            WHEN events IN ('ground_out', 'force_out') THEN 'ground_out'
            WHEN events IN ('fly_out', 'sac_fly') THEN 'fly_out'
            WHEN events IN ('double_play', 'grounded_into_double_play', 'sac_fly_double_play') THEN 'double_play'
            WHEN events IN ('triple_play') THEN 'triple_play'
            WHEN events IN ('field_error') THEN 'fielding_error'
            WHEN events IN ('fielders_choice') THEN 'fielders_choice'
            ELSE NULL
        END AS simplified_outcome,
        game_pk, batter,
        case when inning_topbot='Top' then home_team else away_team end as 'fielding_team',
        game_year, outs_when_up, stand, 
        case when on_1b is not null then 1 else 0 end as on_1b,
        case when on_2b is not null then 1 else 0 end as on_2b,
        case when on_3b is not null then 1 else 0 end as on_3b,    
        launch_speed, launch_angle, ROUND((-(180 / PI()) * atan2(hc_x - 130, 213 - hc_y) + 90)) as spray_angle
    FROM
        Statcast
    WHERE type='X'
    and game_year > 2020
    and
    simplified_outcome &
    game_year & outs_when_up & of_fielding_alignment &
    launch_speed & launch_angle & spray_angle is not null
    {backtest_date}
    ORDER BY GAME_DATE ASC;
    """)

    speed_df = query_mlb_db('select mlb_id as batter, speed from PlayerSpeed;')
    venue_df = query_mlb_db('select game_pk, venue_name from VenueGamePkMapping;')
    oaa_df = query_mlb_db("""
        select o.year as 'game_year', t.STATCAST as 'fielding_team', o.oaa_rhh_standardized, o.oaa_lhh_standardized
        from TeamOAA o
        left join TeamIdMapping t on o.entity_id = t.TEAMID
    """)
    oaa_df['game_year'] = oaa_df['game_year'].astype(int)

    df = df.merge(venue_df, how='left', on='game_pk')
    df = df.merge(speed_df, how='left', on='batter')
    df = df.merge(oaa_df, how='left', on=['game_year', 'fielding_team'])
    df['speed'] = df['speed'].astype(float)
    df['speed'] = df['speed'].fillna(df['speed'].mean())
    
    
    df['oaa'] = df.apply(lambda row: row['oaa_rhh_standardized'] if row['stand'] =='R' else row['oaa_lhh_standardized'], axis=1)

    df = df.drop(['batter', 'game_pk', 'oaa_rhh_standardized', 'oaa_lhh_standardized'], axis=1)

    target_col = 'simplified_outcome'
    return get_xgb_set(df, target_col=target_col, split=split)


def encode_cat_cols(X, encoders_dict):
    object_cols = [col for col in X.columns if X[col].dtype == 'object']
    for col in object_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders_dict[col] = deepcopy(le)

    return X, encoders_dict

def get_hit_outcome_dataset(batter_id, split=False, backtest_date=''):

    backtest_yr = None
    if backtest_date:
       backtest_date = f'and game_date <= "{backtest_date}"' 
       backtest_yr = backtest_date[:4]

    #CAST(TAN((hc_x - 128) / (208 - hc_y)) * 180 / PI() * 0.75 AS INT) AS spray_angle,
    query_str = f"""
        select 
            game_pk, launch_speed, launch_angle, ROUND((-(180 / PI()) * atan2(hc_x - 130, 213 - hc_y) + 90)) as spray_angle,
            
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
        and description = 'hit_into_play'
        and

            launch_speed &
            launch_angle &
            spray_angle &    
        
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
        and game_pk in (
                select distinct game_pk
                from Statcast
                where batter = {batter_id}
                    {backtest_date}
                order by game_date desc
                limit 162
            )
        order by game_date asc, at_bat_number asc, pitch_number asc;
    """

    df = query_mlb_db(query_str)

    #get park factors that affect hit distance based on backtest yr
    venue_df = query_mlb_db('select * from VenueGamePkMapping')
    park_factors_df = query_mlb_db('select * from ParkFactors')
    
    col_idx = 4
    if backtest_yr:
        for idx, col in enumerate(park_factors_df.columns):
            if backtest_yr in col:
                col_idx = idx
                break
    park_factors_df['distance_factor'] = park_factors_df[park_factors_df.columns[col_idx:col_idx+3]].astype(float).mean(axis=1).fillna(0)
    cur_park_factors_df = park_factors_df[['venue_id', 'venue_name', 'distance_factor']].sort_values(by='distance_factor', ascending=False)


    df = df.merge(venue_df[['game_pk', 'venue_id']], on='game_pk', how='left')

    #venue_df['venue_id'] = venue_df['venue_id'].astype(int)
    #venue_df['game_pk'] = venue_df['game_pk'].astype(int)
    cur_park_factors_df['venue_id'] = cur_park_factors_df['venue_id'].astype(int)
    df['venue_id'] = df['venue_id'].astype(int)

    df = df.merge(cur_park_factors_df[['venue_id', 'distance_factor']], on='venue_id', how='left')

    df.drop(['game_pk', 'venue_id'], axis=1, inplace=True)

    #some spring training fields dont have statcast distance.
    df['distance_factor'] = df['distance_factor'].fillna(0)

    target_cols = ['launch_speed', 'launch_angle', 'spray_angle']
    return get_xgb_set_regression(df, target_cols, split=split)

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
            
            pitch_number, strikes, balls,
            
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
        /* and game_pk in (
                select distinct game_pk
                from Statcast
                where batter = {batter_id}
                    {backtest_date}
                order by game_date desc
                limit 324
        ) */
        and pitch_outcome & p_throws & pitch_number & strikes & balls &
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
        SELECT game_year, pitch_type, batter, pitch_number, strikes, balls, outs_when_up, stand,
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
                limit 48
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
    #kukuchi = 579328
    #jones = 683003

    #pitcher = kukuchi

    #all_pitches = get_pitches(pitcher,'L','CU')
    #print(all_pitches.head())

    df, target_col = get_pitch_outcome_data(665742, backtest_date='2022-01-01')
    print(target_col)
    print(df.head())





