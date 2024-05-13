from Player import Player
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from features.build_features import get_sequencing_dataset
from data.data_utils import query_mlb_db

import xgboost as xgb
import pandas as pd
import numpy as np

class Pitcher(Player):
    def __init__(self, mlb_id=None, rotowire_id=None, backtest_date=''):
        super().__init__(mlb_id, rotowire_id, backtest_date)
        print(f'Starting init for {self.name}')
        self.fit_pitch_sequencer()
        self.cumulative_pitch_num = 0
        self.throws = query_mlb_db(f'select p_throws from Statcast where pitcher={self.mlb_id} and p_throws is not null limit 1')['p_throws'][0]
        self.throws = 1 if self.throws=='R' else 0
        print(f'Init complete for {self.name}')

    def fit_pitch_characteristic_generator(self):
        'NEED TO ACCOUNT FOR BATTER STRIKE ZONE...' 
        pass

    def fit_pitch_sequencer(self):
        X, y, encoders = get_sequencing_dataset(self.mlb_id, self.backtest_date)
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

    def generate_pitch_characteristics(self, pitch_type):
        pass

    def generate_pitch_type(self, game_state, batter_stats):
        # get df of current game state and batter stats to generate pitch
        combined_data = {**game_state, **batter_stats}
        df = pd.DataFrame([combined_data])

        #encode cols:
        for col in df.columns:
            if col in self.pitch_sequencer_encoders.keys():
                df[col] = self.pitch_sequencer_encoders[col].transform(df[col])

        #ensure ordering kept the same
        df = df[self.pitch_sequencer_vars]

        pitch_distribution = self.pitch_sequencer.predict_proba(df)[0]
        pitch_choice_encoded = np.random.choice(len(pitch_distribution), size=1, p=pitch_distribution)
        pitch_choice = self.pitch_sequencer_encoders['pitch_type'].inverse_transform(pitch_choice_encoded)[0]

        return pitch_choice

    def generate_pitch(self, game_state, batter_stats):
        pitch_type = self.generate_pitch_type(game_state, batter_stats)
        pitch_characteristics = self.generate_pitch_characteristics(pitch_type)


if __name__ == '__main__':
    kukuchi = 579328
    jones = 683003
    gallen = 668678

    pitcher = Pitcher(mlb_id=kukuchi)

    game_state = {
        'game_year': 2023,
        'pitch_number': 1,
        'strikes': 0,
        'balls': 0,
        'outs_when_up': 0,
        'stand': 1,
        'on_1b': 0,
        'on_2b': 0,
        'on_3b': 0,
        'is_winning': True
    }

    batter_stats = {
        'prev_pitch': 'FF',
        'cumulative_pitch_number': 10,
        'FF_strike': 1.0,
        'SL_strike': 0.0,
        'CH_strike': 0.0,
        'CU_strike': 0.5,
        'FF_woba': 0.3,
        'SL_woba': 0.2,
        'CH_woba': 0.4,
        'CU_woba': 0.5
    }

        
    print(pitcher.generate_pitch_type(game_state, batter_stats))