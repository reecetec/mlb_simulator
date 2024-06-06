import logging
from datetime import timedelta, datetime
import statsapi
from copy import deepcopy
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from itertools import product

from mlb_simulator.simulation.Team import Team
from mlb_simulator.simulation.State import State
from mlb_simulator.features.build_features import get_hit_classification_data

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

        logger.info(f'''Running init for:
                    {self.away_team.name} @ {self.home_team.name}
                    starting at {self.time} in {self.venue_name}''')

        #load hit classification model
        logger.info(f'Fitting hit classification model for {self.venue_name}')
        self.fit_hit_classifier()

        #load game state transition dict
        logger.info(f'Loading game state transition probs')
        self.get_game_state_transition_map()
        self.get_deterministic_transition_map()

        logger.info('Fitting home team models')
        self.home_team.fit_models()
        logger.info('Fitting away team models')
        self.away_team.fit_models()

        #initialize fresh state
        logger.info('Initializing Game State')
        self.game_state = State()

        logger.info(f'''Init complete for:
                    {self.away_team.name} @ {self.home_team.name}
                    starting at {self.time} in {self.venue_name}''')


    def simulate_game(self):
        pass

    def simulate_inning(self, fielding_team, batting_team):
        pass

    def simulate_ab(self, fielding_team, batting_team):
        pass

    def simulate_pitch(self, fielding_team, batting_team):
        pass

    def print_game_score(self):
        home_score = self.game_state.home_runs
        away_score = self.game_state.away_runs
        if home_score > away_score:
            print(f'{home_score}-{away_score} {self.home_team.name}-{self.away_team.name}')
        else:
            print(f'{away_score}-{home_score} {self.away_team.name}-{self.home_team.name}')

    def get_game_state_transition_map(self):
        with open('game_state_t_probs.pkl', 'rb') as f:
            self.game_state_transition_map = pickle.load(f)
    
    def get_deterministic_transition_map(self):
        deterministic_transitions = {}
        deterministic_transitions['K'] = {}
        deterministic_transitions['walk'] = {}
        
        # Generate all permutations
        permutations = list(product(range(3), [(False, False, False), (True, False, False), (False, True, False), (False, False, True),
                                            (True, True, False), (False, True, True), (True, False, True), (True, True, True)]))
        for state in permutations:

            first, second, third = state[1]

            k_first = '1b' if first else False
            k_second = '2b' if second else False
            k_third = '3b' if third else False

            deterministic_transitions['K'][state] = (state[0] + 1, (k_first, k_second, k_third), 0)
        
            new_first, new_second, new_third = 'batter', False, False
            runs_scored = 0
            if first and second and third:
                new_second = '1b'
                new_third = '2b'
                runs_scored = 1
            elif first and not second and not third:
                new_second = '1b'
            elif first and second:
                new_second = '1b'
                new_third = '2b'
            elif first and third:
                new_second = '1b'
                new_third = '3b'
            else:
                if second:
                    new_second = '2b'
                if third:
                    new_third = '3b'
        
            deterministic_transitions['walk'][state] = (state[0], (new_first, new_second, new_third), runs_scored)
        self.deterministic_transitions = deterministic_transitions
        
    
    def sample_next_state(self, event, stand, cur_state, print_probs=False):
        if event not in self.game_state_transition_map or stand not in self.game_state_transition_map[event] or cur_state not in self.game_state_transition_map[event][stand]:
            print('ERROR: KEY NOT FOUND')
            return None  # No transitions available for this event and current state

        next_states = list(self.game_state_transition_map[event][stand][cur_state].keys())
        probabilities = list(self.game_state_transition_map[event][stand][cur_state].values())

        if print_probs:
            pprint(self.game_state_transition_map[event][stand][cur_state])
            
        next_state = np.random.choice(len(next_states), p=probabilities)
        return next_states[next_state]
        

    def fit_hit_classifier(self):
        X, y, encoders = get_hit_classification_dataset()
        self.hit_classifier_encoders = deepcopy(encoders)
        X_stad = X[X['venue_name']==self.hit_classifier_encoders['venue_name'].transform([self.venue_name])[0]].drop('venue_name', axis=1)
        y_stad = y.loc[X_stad.index]
        params = {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
        self.hit_outcome_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', **params)
        self.hit_outcome_clf.fit(X_stad, y_stad)
        self.hit_outcome_clf_cols = X_stad.columns
    
    def classify_hit(self, game_state, fielding_team_name, batter_speed, batter_stand, hit_stats, fielding_oaa):
        input_df = {**game_state, **hit_stats}
        input_df['fielding_team'] = self.hit_classifier_encoders['fielding_team'].transform([fielding_team_name])[0]
        input_df['speed'] = batter_speed
        input_df['stand'] = self.hit_classifier_encoders['stand'].transform([batter_stand])[0]
        input_df['oaa'] = fielding_oaa

        input_df = pd.DataFrame([input_df])
        input_df = input_df[self.hit_outcome_clf_cols]

        pred_probas = self.hit_outcome_clf.predict_proba(input_df)[0]
        hit_class = np.random.choice(list(range(len(pred_probas))), 1, p=pred_probas)
        return self.hit_classifier_encoders['simplified_outcome'].inverse_transform(hit_class)[0]


    def get_venue(self):
        todays_date = (datetime.now() + timedelta(days=int(self.game_tomorrow))).strftime('%Y-%m-%d')
        todays_games = statsapi.schedule(date=todays_date)

        for game in todays_games:
            if game['home_id'] == int(self.home_team.team_id):
                return game['venue_id'], game['venue_name']
        return None, None 
    
    
