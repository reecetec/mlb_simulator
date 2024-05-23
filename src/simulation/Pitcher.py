from Player import Player
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from features.build_features import get_sequencing_dataset, get_pitches, PITCH_CHARACTERISITCS
from data.data_utils import query_mlb_db

import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from copy import deepcopy

logging.getLogger('sdv.metadata').setLevel(logging.CRITICAL)
logging.getLogger('sdv.single_table').setLevel(logging.CRITICAL)
logging.getLogger('copulas').setLevel(logging.CRITICAL)
logging.getLogger('sdv.data_processing').setLevel(logging.CRITICAL)
logging.getLogger('rdt.transformers').setLevel(logging.CRITICAL)
logging.getLogger('sdv').propagate = False
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sdv.single_table.base")


from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

class Pitcher(Player):
    def __init__(self, mlb_id=None, rotowire_id=None, backtest_date=''):
        super().__init__(mlb_id, rotowire_id, backtest_date)
        logger.info(f'Starting init for {self.name}')
        #basic...
        self.throws = query_mlb_db(f'select p_throws from Statcast where pitcher={self.mlb_id} and p_throws is not null limit 1')['p_throws'][0]
        #self.throws = 1 if self.throws=='R' else 0
        self.cumulative_pitch_num = 0

        #fit models...
        self.pitch_characteristic_generators = {'L':{},
                                                'R':{} }
        self.fit_pitch_sequencer()
        self.fit_pitch_characteristic_generator()
        logger.info(f'{self.name} throws {self.throws} with arsenal {self.pitch_arsenal}')
        logger.info(f'Init complete for {self.name}')

    def fit_pitch_characteristic_generator(self):
        logging.basicConfig(level=logging.CRITICAL, format=log_fmt)
        for batter_stands in ['L', 'R']:
            for pitch_type in self.pitch_arsenal:
                df = get_pitches(self.mlb_id, batter_stands, pitch_type, self.backtest_date)
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df)
                synthesizer = GaussianCopulaSynthesizer(metadata,
                                        enforce_rounding=True,
                                        enforce_min_max_values=True,
                                        default_distribution='gaussian_kde',
                                        )
                synthesizer.fit(df)
                self.pitch_characteristic_generators[batter_stands][pitch_type] = deepcopy(synthesizer)
        logging.basicConfig(level=logging.INFO, format=log_fmt)

    def fit_pitch_sequencer(self):
        X, y, encoders, pitch_arsenal = get_sequencing_dataset(self.mlb_id, self.backtest_date)
        self.pitch_arsenal = pitch_arsenal
        params = {
            'objective' : 'multi:softprob',
            'eval_metric': 'mlogloss',      
            'learning_rate': 0.01,
            'max_depth': 2, 
            'n_estimators': 500 
         }
        self.pitch_sequencer = xgb.XGBClassifier(**params)
        self.pitch_sequencer.fit(X, y)
        self.pitch_sequencer_encoders = encoders
        self.pitch_sequencer_vars = self.pitch_sequencer.get_booster().feature_names

    def generate_pitch_characteristics(self, pitch_type, batter_stats, game_state):
        cur_data = pd.DataFrame(data=[{
            'strikes': game_state['strikes'],
            'balls': game_state['balls'],
            'sz_top': batter_stats['sz_top'],
            'sz_bot': batter_stats['sz_bot']
        }])

        #stand = 'R' if batter_stats['stand'] == 1 else 'L'
        synthetic_pitch = self.pitch_characteristic_generators[batter_stats['stand']][pitch_type].sample_remaining_columns(
            known_columns = cur_data
        )
        return dict(synthetic_pitch[PITCH_CHARACTERISITCS].iloc[0])

    def generate_pitch_type(self, game_state, pitcher_stats, batter_stats):
        # get df of current game state and batter stats to generate pitch
        #batter_stats['stand'] = self.pitch_sequencer_encoders['stand'].transform([batter_stats['stand']])[0]
        combined_data = {**game_state, **pitcher_stats, **batter_stats}
        df = pd.DataFrame([combined_data])

        #encode cols:
        for col in df.columns:
            if col in self.pitch_sequencer_encoders.keys():
                df[col] = self.pitch_sequencer_encoders[col].transform(df[col])

        #ensure ordering kept the same as when fit
        df = df[self.pitch_sequencer_vars]

        pitch_distribution = self.pitch_sequencer.predict_proba(df)[0]
        pitch_choice_encoded = np.random.choice(len(pitch_distribution), size=1, p=pitch_distribution)
        pitch_choice = self.pitch_sequencer_encoders['pitch_type'].inverse_transform(pitch_choice_encoded)[0]

        return pitch_choice

    def generate_pitch(self, game_state, batter_stats, pitcher_stats):
        pitch_type = self.generate_pitch_type(game_state, batter_stats, pitcher_stats)
        pitch_characteristics = self.generate_pitch_characteristics(pitch_type, batter_stats, game_state)

        return pitch_type, pitch_characteristics


if __name__ == '__main__':
    kukuchi = 579328
    jones = 683003
    gallen = 668678

    pitcher = Pitcher(mlb_id=jones)

    game_state = {
        'game_year': 2023,
        'pitch_number': 1,
        'strikes': 0,
        'balls': 0,
        'outs_when_up': 0,
        'on_1b': 0,
        'on_2b': 0,
        'on_3b': 0,
    }

    pitcher_stats = {
        'is_winning': True,
        'prev_pitch': 'FF',
        'cumulative_pitch_number': 10,
    }

    batter_stats = {
        'FF_strike': 1.0,
        'SL_strike': 0.0,
        'CH_strike': 0.0,
        'CU_strike': 0.5,
        'FF_woba': 0.3,
        'SL_woba': 0.2,
        'CH_woba': 0.4,
        'CU_woba': 0.5,
        'sz_top': 3.438,
        'sz_bot': 1.544,
        'stand': 'R',
    }
        
    #print(pitcher.generate_pitch_type(game_state, pitcher_stats, batter_stats))
    pitch, pitch_char = pitcher.generate_pitch(game_state, batter_stats, pitcher_stats)
    print(pitch)
    print(pitch_char)