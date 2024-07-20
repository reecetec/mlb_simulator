"""
Given the batter's hit characteristics, his speed, and the game state, sample
the event that will occur. E.g., a double, single, field out, sac fly, etc.
Fit with general hit outcomes as a function of speed, hit characteristics,
venue, and game speed.
"""

from mlb_simulator.models import model_utils as mu
from mlb_simulator.features.build_features import get_hit_classification_data

import xgboost as xgb
import pandas as pd
from random import choices


class HitClassifier:

    def __init__(self, venue_name):
        self.MODEL_NAME = "hit_classifier"
        self.venue_name = venue_name

    def __repr__(self):
        return f"HitClassifier({self.venue_name})"

    def __str__(self):
        if not hasattr(self, "feature_order"):
            return self.__repr__()
        return f"HitClassifier for {self.venue_name} with input features: \n {self.feature_order}"

    def __call__(self, features):
        """
        Generate hit class given input features
        """
        pass
        if not hasattr(self, "model"):
            print(f"Trying to use {self.MODEL_NAME} without first fitting")
            return None

        probs = self.model.predict_proba(features[self.feature_order])[0]
        outcome = choices(range(len(probs)), probs, k=1)
        outcome = self.le.inverse_transform(outcome)

        if outcome is not None:
            return outcome[0]
        else:
            return None

    def fit(self, backtest_date=None):
        dataset, target_col = get_hit_classification_data(
            self.venue_name, backtest_date=backtest_date
        )
        model, le, X, y = mu.categorical_model_pipeline(
            xgb.XGBClassifier, dataset, target_col
        )
        hyperparams = mu.get_hyperparams(self.MODEL_NAME, self.venue_name, model, X, y)

        model.set_params(**hyperparams)
        model.fit(X, y)

        feature_order = X.columns

        self.model = model
        self.le = le
        self.feature_order = feature_order

        return model, le, feature_order


if __name__ == "__main__":

    venue = "Busch Stadium"

    clf = HitClassifier(venue)
    print(clf)

    clf.fit()
    print(clf.feature_order)

    data = {
        "game_year": [2024],
        "outs_when_up": [2],
        "stand": ["R"],
        "on_1b": [True],
        "on_2b": [False],
        "on_3b": [False],
        "launch_speed": [102.5],
        "launch_angle": [35.7],
        "spray_angle": [15.2],
        "speed": [28.5],
        "oaa": [5],
    }

    df = pd.DataFrame(data)

    outcome = clf(df)
    print(outcome)
