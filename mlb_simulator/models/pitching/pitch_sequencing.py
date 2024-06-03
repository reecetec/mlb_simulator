"""
This model generates a probability distribution for the next pitch type
based on the pitchers arsenal, game state, and current batter's stats into
each pitch type
"""

from mlb_simulator.models import model_utils as mu
from mlb_simulator.features.build_features import get_pitch_sequencing_data

import xgboost as xgb


MODEL_NAME = 'pitch_sequencer'

def fit_pitch_sequencer(pitcher_id: int, backtest_date=None):

    query = get_pitch_sequencing_data(pitcher_id, backtest_date=backtest_date)
    dataset, target_col, pitch_arsenal = query

    model, le, X, y = mu.categorical_model_pipeline(xgb.XGBClassifier,
                                                    dataset, target_col)

    hyperparams = mu.get_hyperparams(MODEL_NAME, pitcher_id, model, X, y)

    model.set_params(**hyperparams)
    model.fit(X, y)

    feature_order = X.columns

    return model, le, feature_order, pitch_arsenal


if __name__ == "__main__":
    kukuchi = 579328
    gil = 661563
    pitcher_id = gil 

    model, le, feature_order, pitch_arsenal = fit_pitch_sequencer(pitcher_id)
    print(model)
    print(le)
    print(feature_order)
    print(pitch_arsenal)
    


    
                                                               

