import sys
import os
import logging
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import timedelta, datetime
import statsapi
from copy import deepcopy
import xgboost as xgb
import pandas as pd
import numpy as np

from simulation.Team import Team
from features.build_features import get_hit_classification_dataset

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

class Game:
    def __init__(self, rotowire_lineup, rotowire_time, game_tomorrow=False):
        self.time = rotowire_time
        self.home_team = Team(rotowire_lineup, True)
        self.away_team = Team(rotowire_lineup, False)
        self.game_tomorrow = game_tomorrow 

        #find game venue
        self.venue_id, self.venue_name = self.get_venue()

        #load hit classification model
        logger.info(f'Fitting hit classification model for {self.venue_name}')
        self.fit_hit_classifier()


        #load game state transition dict
        logger.info(f'Loading game state transition probs')

        logger.info('Fitting home team models')
        #self.home_team.fit_models()
        logger.info('Fitting away team models')
        #self.away_team.fit_models()

        logger.info(f'Init complete for {self.away_team.name} @ {self.home_team.name} starting at {self.time} in {self.venue_name}')

    def fit_hit_classifier(self):
        X, y, encoders = get_hit_classification_dataset()
        self.hit_classifier_endoers = deepcopy(encoders)
        X_stad = X[X['venue_name']==self.hit_classifier_encoders['simplified_outcome'].transform([self.venue_name])[0]].drop('venue_name', axis=1)
        y_stad = y.loc[X_stad.index]
        params = {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
        self.hit_outcome_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', **params)
        self.hit_outcome_clf.fit(X_stad, y_stad)
        self.hit_outcome_clf_cols = X_stad.columns
    
    def classify_hit(self, game_state, fielding_team_name, batter_speed, hit_stats, fielding_oaa):
        input_df = {**game_state, **hit_stats}
        input_df['fielding_team'] = self.hit_classifier_encoders['fielding_team'].transform([fielding_team_name])[0]
        input_df['speed'] = batter_speed
        input_df['oaa'] = fielding_oaa

        input_df = input_df[self.hit_outcome_clf_cols]

        pred_probas = self.hit_outcome_clf.predict_proba(input_df)[0]
        hit_class = np.random.choice(list(range(len(pred_probas))), 1, p=pred_probas)
        return self.hit_classifier_endoers['simplified_outcome'].inverse_transform(hit_class)[0]


    def get_venue(self):
        todays_date = (datetime.now() + timedelta(days=int(self.game_tomorrow))).strftime('%Y-%m-%d')
        todays_games = statsapi.schedule(date=todays_date)

        for game in todays_games:
            if game['home_id'] == int(self.home_team.team_id):
                return game['venue_id'], game['venue_name']
        return None, None 
    
    def simulate_game(self):
        pass

    def simulate_inning(self):
        pass

    def simulate_at_bat(self):
        pass

    def simulate_pitch(self):
        pass

