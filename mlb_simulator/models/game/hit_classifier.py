"""
Given the batter's hit characteristics, his speed, and the game state, sample
the event that will occur. E.g., a double, single, field out, sac fly, etc.
Fit with general hit outcomes as a function of speed, hit characteristics,
venue, and game speed.
"""

from mlb_simulator.models import model_utils as mu
from mlb_simulator.features.build_features import get_hit_classification_data

import xgboost as xgb

MODEL_NAME = 'hit_classifier'

def fit_hit_classifier(venue_name):

    dataset, target_col = get_hit_classification_data(venue_name)
    
    model, le, X, y = mu.categorical_model_pipeline(xgb.XGBClassifier,
                                                    dataset, target_col)

    hyperparams = mu.get_hyperparams(MODEL_NAME, venue_name, model, X, y)

    model.set_params(**hyperparams)
    model.fit(X,y)

    feature_order = X.columns

    return model, le, feature_order

if __name__ == '__main__':

    venue = 'Busch Stadium'

    model, le, feature_order = fit_hit_classifier(venue)
    print(model)
    print(feature_order)
