""" 
this model determines the pitch outcome, meaning was the ball hit,
strike, ball, foul, or hit into play?
"""
from mlb_simulator.models import model_utils as mu
from mlb_simulator.features.build_features import get_pitch_outcome_data
import json
import xgboost as xgb


def fit_pitch_outcome_model(batter_id: int, backtest_date=None):
    """Fit pitch outcome model for a given batter id
    
    This function

    Parameters:
        batter_id (int): the desired batter's mlb id
        backtest_date (str, optional): the date to be used if backtesting
    """

    #get the dataset and target col
    dataset, target_col = get_pitch_outcome_data(batter_id,
                                                 backtest_date=backtest_date)

    # check if up to date hyperparams have been fit for this batter
    hyperparam_path = mu.check_for_hyperparams('pitch_outcome', batter_id)

    # if valid hyperparams, load them
    if hyperparam_path:
        with open(hyperparam_path) as f:
            hyperparams = json.load(f)
    # if no valid hyperparams, fit new ones and save them
    else:
        pass

    # get model pipeline (contains encoders). 
    params = {'eval_metric':'mlogloss', **hyperparams}
    model = mu.categorical_model_pipeline(xgb.XGBClassifier,
                                          params,
                                          dataset,
                                          target_col
                                          )


def main():
    pass

if __name__ == "__main__":
    main()
