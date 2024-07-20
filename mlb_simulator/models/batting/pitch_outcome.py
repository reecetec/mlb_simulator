"""
Module for the pitch outcome model. This model determines the pitch outcome,
meaning it answers the following question: was the ball hit, was there a
strike, ball, foul?
"""

from mlb_simulator.models import model_utils as mu
from mlb_simulator.features.build_features import get_pitch_outcome_data

import logging
import xgboost as xgb
import pandas as pd
from random import choices

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)


class PitchOutcome:

    def __init__(self, batter_id=None):

        self.MODEL_NAME = "pitch_outcome"
        self.batter_id = batter_id

        # if batter id is none, fit default model.

    def __call__(self, features):
        """
        Generate a pitch outcome given the features and the fit model
        """

        if not hasattr(self, "model"):
            print("Trying to use model without first fitting")
            return None

        probs = self.model.predict_proba(features[self.feature_order])[0]
        outcome = choices(range(len(probs)), probs, k=1)
        outcome = self.le.inverse_transform(outcome)

        if outcome is not None:
            return outcome[0]
        else:
            print("error, pitch outcome model returns none")
            return None

    def fit(self, backtest_date=None):
        """
        Fit pitch outcome model for a given batter id.

        This function will obtain the data required to fit the model, check if
        hyperparameters have previously been optimized for the given batter id.
        If so, load them, otherwise, fit them and save them. Once ideal
        hyperparameters have been fit, fit the model to the dataset.

        Args:
            backtest_date (str, optional): The date to be used if backtesting

        Returns:
            model: The fitted model (sklearn.pipeline.Pipeline).
            le: (sklearn.preprocessing.LabelEncoder) The label encoder used for
                encoding target variable.
            feature_order: (list[str]) A list containing the names of features
                in the order they appear in the dataset.
        """

        # get the dataset and target col
        dataset, target_col = get_pitch_outcome_data(self.batter_id, backtest_date)

        # get model pipeline
        model, le, X, y = mu.categorical_model_pipeline(
            xgb.XGBClassifier, dataset, target_col
        )

        # get hyperparams
        hyperparams = mu.get_hyperparams(self.MODEL_NAME, self.batter_id, model, X, y)

        # fit model for use
        model.set_params(**hyperparams)
        model.fit(X, y)

        # return feature ordering
        feature_order = X.columns

        self.model = model
        self.le = le
        self.feature_order = feature_order
        return model, le, feature_order


def main():

    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = soto

    pitch_out = PitchOutcome(batter_id)
    pitch_out.fit()

    pitch_chars = {
        "release_speed": 95,
        "release_spin_rate": 2000,
        "plate_x": 0,
        "plate_z": 2,
        "p_throws": "R",
        "strikes": 0,
        "balls": 0,
    }

    for _ in range(10):
        outcome = pitch_out(pd.DataFrame([pitch_chars]))
        print(outcome)


if __name__ == "__main__":
    main()
