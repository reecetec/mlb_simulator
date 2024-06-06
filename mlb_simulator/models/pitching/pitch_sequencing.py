"""
This model generates a probability distribution for the next pitch type
based on the pitchers arsenal, game state, and current batter's stats into
each pitch type
"""

from mlb_simulator.models import model_utils as mu
from mlb_simulator.features.build_features import get_pitch_sequencing_data

import xgboost as xgb
import pandas as pd
from random import choices

class PitchSequencer:

    def __init__(self, pitcher_id):

        self.MODEL_NAME = 'pitch_sequencer'
        self.pitcher_id = pitcher_id

    def __repr__(self):
        return f'PitchSequencer({self.pitcher_id})'


    def __call__(self, features):
        """
        Generate a pitch outcome given the features and the fit model
        """

        if not hasattr(self, 'model'):
            print('Trying to use model without first fitting')
            return None

        probs = self.model.predict_proba(features[self.feature_order])[0]
        outcome = choices(range(len(probs)), probs, k=1)
        outcome = self.le.inverse_transform(outcome)

        if outcome is not None:
            return outcome[0]
        else:
            return None


    def fit(self, backtest_date=None):

        query = get_pitch_sequencing_data(self.pitcher_id,
                                          backtest_date=backtest_date)
        dataset, target_col, pitch_arsenal = query

        model, le, X, y = mu.categorical_model_pipeline(xgb.XGBClassifier,
                                                        dataset, target_col)

        hyperparams = mu.get_hyperparams(self.MODEL_NAME, self.pitcher_id,
                                         model, X, y)

        model.set_params(**hyperparams)
        model.fit(X, y)

        feature_order = X.columns

        self.model = model
        self.le = le
        self.feature_order = feature_order
        self.pitch_arsenal = pitch_arsenal

        return model, le, feature_order, pitch_arsenal


if __name__ == "__main__":
    kukuchi = 579328
    gil = 661563
    pitcher_id = gil 

    pitch_seq = PitchSequencer(pitcher_id)
    pitch_seq.fit()

    #print(pitch_seq.feature_order)


    input_f = pd.DataFrame([{'game_year': 2024, 'pitch_number':1, 'strikes':0, 'balls':0,
               'outs_when_up':0, 'stand': 'R', 'on_1b': False, 'on_2b': False,
               'on_3b': False, 'prev_pitch': None, 'cumulative_pitch_number': 1,
               'FF_strike':0, 'CH_strike':0, 'SL_strike':0,
               'FC_strike':0, 'FF_woba':5, 'CH_woba':0, 'SL_woba':0, 'FC_woba':0}])

    model, le, feature_order, pitch_arsenal = pitch_seq.fit()

    print(pitch_seq(input_f))
    


    
                                                               

