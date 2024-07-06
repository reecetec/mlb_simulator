import warnings
import time
from tqdm import tqdm
import logging
from pathlib import Path
import requests
import pathlib
import os
import sqlite3
import json
from sqlalchemy import inspect, text
import pandas as pd
from datetime import datetime, timedelta
import re
from pybaseball import statcast
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from mlb_simulator.data.data_utils import get_mlb_db_engine, query_mlb_db
from mlb_simulator.data.data_utils import get_db_location, git_clone, git_pull

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

DB_LOCATION = get_db_location()
THIS_FILEPATH = pathlib.Path(__file__).parent.resolve()

def validate_db_and_table(table_name):
    if not Path(DB_LOCATION).exists():
        logger.critical(f'''mlb.db doesnt exist at {DB_LOCATION},
                        something went wrong in the creation''')
        exit()

    # check to see if statcast table exists
    engine = get_mlb_db_engine() 
    if engine is None:
        logger.critical('Error creating engine')
        exit()

    inspector = inspect(engine)
    table_exists = table_name in inspector.get_table_names()

    return engine, table_exists



def update_statcast_table():
    """ Uses pybaseball to download statcast data from 2018 locally.
    """

    # get engine, check if table exists
    engine, statcast_table_exists = validate_db_and_table('Statcast')
    
    # if no data uploaded, get all data from 2018 to today in yr chunks
    # (avoid memory overload)
    if not statcast_table_exists:
        logger.info('Running creation/init upload for Statcast table')

        #create table using schema from ./Statcast_create.sql
        try:
            with open(
                    THIS_FILEPATH / 'sql_scripts' / 'Statcast_create.sql',
                    'r') as f:
                statcast_create_script = f.read()
            scripts = statcast_create_script.split(';')
            with engine.connect() as connection:
                for script in scripts:
                    if script.strip():
                        connection.execute(text(script))
        except Exception as e:
            logger.critical(
                    ('Unable to create table via '),
                    ('mlb_simulator/data/Statcast_create.sql')
                    )
            print(e)
            exit()
            
        for year in range(2018, datetime.now().year + 1):
            df = statcast(f'{year}-01-01', f'{year+1}-01-01', verbose=False)
            if not df.empty:
                # drop duplicates if exist
                df = df.drop(columns=['pitcher.1',
                                      'fielder_2.1'],
                             errors='ignore')
                #make sure sql uses datetime type
                df['game_date'] = pd.to_datetime(df['game_date'])
                # upload
                df.to_sql('Statcast', engine, if_exists='append', index=False,
                        chunksize=1000)
            logger.info((f'\nSuccessfully uploaded statcast data '
                         f'from {year} to {year + 1}\n'))

    # otherwise, get all data from max date in table
    else:
        max_date = pd.read_sql('select max(game_date) from Statcast',
                               engine)['max(game_date)'].iloc[0]
        today = datetime.today()
        max_date = datetime.strptime(max_date, '%Y-%m-%d %H:%M:%S.%f')
        
        # if over 1 year of data missing, query year by year
        if today.year - max_date.year > 1:
            for year in range(max_date.year, today.year + 1):
                df = statcast(f'{year}-01-01',
                              f'{year+1}-01-01',
                              verbose=False)
                if not df.empty:
                    # drop duplicates if exist
                    df = df.drop(columns=['pitcher.1',
                                          'fielder_2.1'],
                                 errors='ignore')
                    #make sure sql uses datetime type
                    df['game_date'] = pd.to_datetime(df['game_date'])

                    #ensure no duplicates for potential overlapping yr
                    if year == max_date.year:
                        cur_keys = pd.read_sql(
                                f'''select game_date, game_pk, at_bat_number,
                                pitch_number
                                from Statcast where game_date >= {year}''',
                                engine)
                        cur_keys['game_date'] = pd.to_datetime(
                                cur_keys['game_date'])
                        merged = pd.merge(
                                df,cur_keys,how='outer',indicator=True)
                        upload_df = merged.loc[merged["_merge"] == "left_only"
                                               ].drop("_merge", axis=1)
                    else:
                        upload_df = df

                    upload_df.to_sql('Statcast', engine, if_exists='append',
                                     index=False, chunksize=1000)
                    
                    logger.info(f'Successfully uploaded {
                        len(upload_df)
                        } rows to table Statcast')


        else:
            # query 2 days of previous keys to ensure no duplicates 
            two_days_prior = (max_date - timedelta(days=2)).strftime('%Y-%m-%d')
            df = statcast(max_date.strftime('%Y-%m-%d'),
                          today.strftime('%Y-%m-%d'),
                          verbose=False)
            if not df.empty:
                # drop garbage
                df.drop(columns=['pitcher.1','fielder_2.1'],
                        errors='ignore', inplace=True)

                # ensure no duplicates
                cur_keys = pd.read_sql(
                        f'''select game_date, game_pk, at_bat_number,
                        pitch_number
                        from Statcast where game_date > {two_days_prior}''',
                        engine)
                cur_keys['game_date'] = pd.to_datetime(cur_keys['game_date'])
                merged = pd.merge(df,cur_keys,how='outer',indicator=True)
                upload_df = merged.loc[merged["_merge"] == "left_only"
                                       ].drop("_merge", axis=1)

                upload_df.to_sql('Statcast', engine, if_exists='append',
                                 index=False, chunksize=1000)
                
                logger.info(f'Successfully uploaded {
                    len(upload_df)
                    } rows to table Statcast')
            else:
                logger.info(f'Statcast query was empty')


def update_woba_strike_tables(
        min_pitch_count=50, min_hit_count=15, backtest_yr=None
        ):
    """ Sets up 2 tables in mlb.db containing the batter id,
        first the batter's strike percentage into a given pitch type,
        second, the batter's average woba given a hit.

        To be used in the pitch_type generator
    """

    # when backtesting, use only data from past seasons.
    if backtest_yr:
        backtest_yr = f'and game_year < {backtest_yr}'
    else:
        backtest_yr = ''

    batter_df = query_mlb_db(f"""
        select batter, pitch_type, type, woba_value
        from Statcast
        where pitch_type <> 'PO' 
            and pitch_type is not null
            and type is not null
            {backtest_yr}
    """)

    if batter_df is None:
        logger.critical('Error querying Statcast')
        exit()

    """
    get each batters strike percentage for each pitch in pitch arsenal given
    they have at least min_pitch_count of the pitch type thrown to them
    standardize across all batters
    """
    filtered_data = batter_df.groupby(['batter', 'pitch_type']).filter(
            lambda x: len(x) >= min_pitch_count
        )
    strike_percentage = filtered_data.groupby(['batter',
                                               'pitch_type'])['type'].apply(
           lambda x: (x == 'S').mean() * 100
       )
    strike_percentage_standardized = strike_percentage.groupby(
            'pitch_type'
        ).transform(zscore).reset_index()    

    strike_df = strike_percentage_standardized.pivot(index='batter',
                                                     columns='pitch_type',
                                                     values='type').fillna(0)

    """
    do the same as above but for avg woba when they hit the ball
    """
    filtered_data = batter_df[batter_df['type'] == 'X'].groupby(
            ['batter', 'pitch_type']
        ).filter(lambda x: (x['type'] == 'X').sum() >= min_hit_count)

    average_woba = filtered_data.groupby(['batter',
                                          'pitch_type'])['woba_value'].mean()
    average_woba_standardized = average_woba.groupby('pitch_type').transform(
            zscore
        ).reset_index()

    woba_df = average_woba_standardized.pivot(
            index='batter', columns='pitch_type', values='woba_value'
        ).fillna(0)

    #upload df's
    engine = get_mlb_db_engine()

    strike_df.to_sql('BatterStrikePctByPitchType', engine, if_exists='replace')
    woba_df.to_sql('BatterAvgWobaByPitchType', engine, if_exists='replace')

def update_chadwick_repo():
    home_dir = pathlib.Path.home()
    repo_save_path = os.path.join(home_dir, 'sports', 'mlb_simulator',
                                  'data', 'raw', 'chadwick')
    
    if os.path.exists(repo_save_path):
        git_pull(repo_save_path)
    else:
        repo_url = 'https://github.com/chadwickbureau/register.git'
        git_clone(repo_url, repo_save_path)

def update_player_name_map(save_path):

    url = 'https://www.smartfantasybaseball.com/PLAYERIDMAPCSV'
    #need this header or request is blocked
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 '
            'Safari/537.36'
        )
    }

    res = requests.get(url, headers=headers)

    if res.status_code == 200:
        # Delete the existing file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Existing file '{save_path}' deleted.")

        # Save the content of the response (CSV file) to the specified path
        open(save_path, 'wb').write(res.content)
        print(f"Name map downloaded and saved to '{save_path}'")
    else:
        print(f"Failed to download CSV file. Status code: {res.status_code}")

def update_similar_sz_table():
    """ Creates cluster of players with similar strikezones. Was used to
        transfer learn neural networks. XGBoost being used make this less
        useful.
    """

    player_sz_data = query_mlb_db('''select batter, avg(sz_top) as sz_top,
                                    avg(sz_bot) as sz_bot
                                    from Statcast
                                    where sz_top & sz_bot is not null
                                    group by batter
                                  ''')
    
    if player_sz_data is None:
        logger.critical('Error querying Statcast')

    X = player_sz_data[['sz_top', 'sz_bot']].values
    
    # normalization or standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # clustering
    kmeans = KMeans(n_clusters=10)  
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_
    
    # add cluster labels 
    player_sz_data['cluster'] = cluster_labels

    engine = get_mlb_db_engine()

    try:
        if engine:
            upload_df = player_sz_data[['batter','cluster']]
            upload_df.to_sql('BatterStrikezoneCluster', engine,
                             if_exists='replace', index=False)
        else:
            raise NameError('Could not create db engine')
    except NameError as e:
        print(e.args)


def update_sz_lookup():
    """Get batter's strikezone for pitch generation.
    """

    player_sz_data = query_mlb_db('''select batter, round(avg(sz_top),3)
                                    as sz_top, round(avg(sz_bot),3) as sz_bot
                                    from Statcast
                                    where sz_top & sz_bot is not null
                                    group by batter''')

    engine = get_mlb_db_engine()

    try:
        if engine:
            upload_df = player_sz_data
            upload_df.to_sql('BatterStrikezoneLookup', engine,
                             if_exists='replace', index=False)
        else:
            raise NameError('Could not create db engine')
    except NameError as e:
        print(e.args)


def update_venue_game_pk_mapping():

    # get engine, check if table exists
    engine, venue_table_exists = validate_db_and_table('VenueGamePkMapping')
    
    # if no data uploaded, get all data from 2018 to today in yr chunks
    # (avoid memory overload)
    if not venue_table_exists:
        logger.info(
                'Running creation/init upload for VenueGamePkMapping table')

        #use init upload file: otherwise will take forever.
        try:
            init_upload_df = pd.read_csv(
                    THIS_FILEPATH / 'init_uploads' / 'VenueMappingInit.csv')
            init_upload_df.drop('Unnamed: 0', axis=1)
            init_upload_df.to_sql('VenueGamePkMapping', engine,
                                  if_exists='replace', index=False)
            logger.info('Successful init upload')

        except Exception as e:
            logger.critical(
                    ('Unable to create initial upload using '),
                    ('./VenueMappingInit.csv')
                    )
            print(e)
            exit()

    # get missing game pks
    game_pks = query_mlb_db('select distinct game_pk from Statcast')
    venue_game_pks = query_mlb_db(
            'select distinct game_pk from VenueGamePkMapping')
    if game_pks is None or venue_game_pks is None:
        exit()

    missing_game_pks = game_pks[
            ~game_pks['game_pk'].isin(
                venue_game_pks['game_pk'])]['game_pk'].to_list()

    if missing_game_pks:
        sleep_time = 1/10
        venue_data = []
        for game_pk in tqdm(missing_game_pks, total=len(missing_game_pks)):
            res = requests.get(
                    f'https://statsapi.mlb.com/api/v1/schedule?gamePk={
                        game_pk}')
            if res.status_code == 200:
                game_json = res.json()
                venue = game_json['dates'][0]['games'][0]['venue']
                venue['game_pk'] = game_pk
                venue_data.append(venue)
                # dont get rate limited... apparently max is 25/sec,
                # 10/sec to be on safe side
                time.sleep(sleep_time)
        df = pd.DataFrame(venue_data)
        upload_df = df.rename(columns={
            "name": "venue_name", "id": "venue_id"}
                              )[['game_pk','venue_name','venue_id']]
        engine = get_mlb_db_engine()
        upload_df.to_sql('VenueGamePkMapping', engine,
                         if_exists='append', index=False)

        logger.info(f'Successfully uploaded {
            len(upload_df)} rows to table VenueGamePkMapping')
    else:
        logger.info(f"No Missing VenueGamePkMapping's")

def update_oaa():
    url = f"https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielding_Team&startYear=2021&endYear={datetime.now().year}&split=yes&team=&range=year&min=q&pos=of&roles=&viz=hide"
    res = requests.get(url)
    
    if res.status_code != 200:
        logger.critical('Error query statcast speed leaderboard')
        exit()
    
    data = re.search(r"data = (.*);", res.text)
    if data is not None:
        data = data.group(1)
        data = json.loads(data)
        df = pd.DataFrame(data)
    else:
        logger.critical('Statcast oaa data not returned')
        exit()

    df = df[['year', 'entity_id', 'entity_name', 'outs_above_average_rhh',
             'outs_above_average_lhh']].sort_values(by=['entity_id', 'year'])

    means = df.groupby('year')[
            ['outs_above_average_rhh', 'outs_above_average_lhh']].mean()
    stds = df.groupby('year')[
            ['outs_above_average_rhh', 'outs_above_average_lhh']].std()

    for col in ('outs_above_average_rhh', 'outs_above_average_lhh'):
        df[f'{col}_standardized'] = df.apply(
                lambda row: (
                    row[col] - means.loc[row['year']][col]
                    ) / stds.loc[row['year']][col], axis=1)
    
    column_renames = {
        'outs_above_average_rhh': 'oaa_rhh',
        'outs_above_average_rhh_standardized': 'oaa_rhh_standardized',
        'outs_above_average_lhh': 'oaa_lhh',
        'outs_above_average_lhh_standardized': 'oaa_lhh_standardized'
    }

    df.rename(columns=column_renames, inplace=True)

    engine = get_mlb_db_engine()
    df.to_sql('TeamOAA', engine, if_exists='replace', index=False)

def update_run_speed():
    url = f"https://baseballsavant.mlb.com/leaderboard/sprint_speed?min_season=2021&max_season={datetime.now().year}&position=&team=&min=5"
    res = requests.get(url)

    if res.status_code != 200:
        logger.critical('Error query statcast speed leaderboard')
        exit()

    data = re.search(r"data = (.*);", res.text)
    if data:
        data = data.group(1)
        data = json.loads(data)
        df = pd.DataFrame(data)[['runner_id', 'r_sprint_speed_top50percent']]
    else:
        logger.critical('Statcast speed data not returned')
        exit()

    column_renames = {
        'runner_id': 'mlb_id',
        'r_sprint_speed_top50percent': 'speed'
    }
    df.rename(columns=column_renames, inplace=True)

    engine = get_mlb_db_engine()
    df.to_sql('PlayerSpeed', engine, if_exists='replace', index=False)

def main():

    logger.info('Starting SQLite db creation/updates')
    # if database location does not exist, create directories
    if not os.path.exists(DB_LOCATION):
        #create dirs
        logger.info(f'Creating database at {DB_LOCATION}')
        os.makedirs(os.path.dirname(DB_LOCATION), exist_ok=True)

        #init empty sqlite db
        conn = sqlite3.connect(DB_LOCATION)
        conn.close()

    # check if player_name_map exists
    name_map_path = pathlib.Path(
            DB_LOCATION).parent.parent / 'raw' / 'name_map.csv'
    if not os.path.exists(name_map_path):
        logger.info(
                f'Creating raw data folder with player map at {name_map_path}')
        os.makedirs(os.path.dirname(name_map_path), exist_ok=True)
        update_player_name_map(name_map_path)

    #################### DAILY UPDATES #####################################

    # update tables
    logger.info(f'Updating Statcast')
    update_statcast_table()

    logger.info(f'Updating VenueGamePkMapping')
    update_venue_game_pk_mapping()

    logger.info(f'Updating TeamOAA')
    update_oaa()

    logger.info(f'Updating PlayerSpeed')
    update_run_speed()

    #################### MONTHLY UPDATES ###################################
    if datetime.now().day == 1:
        logger.info('Running Monthly Updates. This will take a while')

        logger.info('Updating player name mapping')
        name_map_path = pathlib.Path(
            DB_LOCATION).parent.parent / 'raw' / 'name_map.csv'
        update_player_name_map(name_map_path)

        logger.info(
                ('Updating BatterStrikePctByPitchType & '),
                 ('BatterAvgWobaByPitchType')
            )
        update_woba_strike_tables()

        logger.info(f'Updating BatterStrikezoneLookup')
        update_sz_lookup()

    #################### OUTDATED ##########################################

    #logger.info(f'Updating BatterStrikezoneCluster')
    #update_similar_sz_table()

    #logger.info('Updating chadwick repo')
    #update_chadwick_repo()
        
    logger.info('DB creation/updates complete')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
