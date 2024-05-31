""" 
this model determines the pitch outcome, meaning was the ball hit,
strike, ball, foul, or hit into play?
"""
from mlb_simulator.models import model_utils as mu
from mlb_simulator.data.data_utils import get_models_location
from mlb_simulator.features.build_features import get_pitch_outcome_data
import json
import os
import xgboost as xgb
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

MODEL_NAME = 'pitch_outcome'

def fit_pitch_outcome_model(batter_id: int, backtest_date=None):
    """Fit pitch outcome model for a given batter id
    
    This function will obtain the data required to fit the model, check if 
    hyperparameters have previously been optimized for the given batter id. 
    If so, load them, otherwise, fit them and save them. Once ideal hyperparams
    have been fit, fit the model to the dataset.

    Parameters:
        batter_id (int): the desired batter's mlb id
        backtest_date (str, optional): the date to be used if backtesting
    """

    #get the dataset and target col
    dataset, target_col = get_pitch_outcome_data(batter_id,
                                                 backtest_date=backtest_date)

    # get model pipeline 
    model, le, X, y = mu.categorical_model_pipeline(xgb.XGBClassifier,
                                          dataset,
                                          target_col
                                          )

    # check if up to date hyperparams have been fit for this batter
    hyperparam_path = mu.check_for_hyperparams('pitch_outcome', batter_id)

    # if valid hyperparams, load them
    if hyperparam_path:
        with open(hyperparam_path) as f:
            hyperparams = json.load(f)

    # if no valid hyperparams, fit new ones and save them
    else:
        logger.info(
            f'No hyperparams found for {batter_id}, {MODEL_NAME}:\n' + \
            'finding and saving optimal hyperparams...'
        )
        hyperparams = mu.xgb_hyperparam_optimizer(model, X, y)
        save_path = os.path.join(
            get_models_location(),
            MODEL_NAME,
            f'{batter_id}-{datetime.now().strftime('%Y%m%d')}.json'
        )
        with open(save_path, 'w') as f:
            json.dump(hyperparams, f)

    # fit model for use
    model.set_params(**hyperparams)
    model.fit(X, y)

    feature_order = X.columns

    return model, feature_order


    
def main():

    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = soto


    m, f = fit_pitch_outcome_model(batter_id)
    print(m, f)

if __name__ == "__main__":
    main()
