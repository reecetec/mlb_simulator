"""
Fit a game state transition matrix. Given the outcome of the at bat, find
a probability distribution for how the game evolves.
"""

from mlb_simulator.features.build_features import get_game_state_t_prob_data
from mlb_simulator.models.model_utils import get_models_location

import pandas as pd 
import numpy as np
import pickle
import os
import logging

from collections import namedtuple, Counter
from random import choices
from datetime import datetime, timedelta

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

GameState = namedtuple('GameState', 'outs on_1b on_2b on_3b')
GameStateTransition = namedtuple('GameStateTransition', ['after_outs',
                                                         'after_1b',
                                                         'after_2b',
                                                         'after_3b',
                                                         'runs_scored'])
"""
stores the game state transition. Bases will be '1b', 'batter', etc.
denoting the previous position. So if after_1b is 'batter' then
this means that the batter hit a single
"""

class GameStateTransitionMatrix:

    def __init__(self):
        self.name = 'GameStateTransitionMatrix'

        # check if should be updated. Otherwise, use saved model.
        try:
            self._load()
            logging.info(f'{self.name} loaded successfully')
        except:
            logging.info(f'Refitting {self.name}')
            self._fit()
            self._save()

    def __repr__(self):
        return f'GameStateTransitionMatrix'

    def __str__(self):
        return f'GameStateTransitionMatrx'

    def __call__(self, event: str, stand: str,
                 input_state: GameState) -> GameStateTransition:
        """
        Sample game new game state given input state
        """

        distribution = self.t_matrix.loc[(event, stand, input_state)]
        sampled_event = choices(list(distribution.keys()),
                                weights=distribution.values(),
                                k=1)[0]
        return sampled_event

    def _save(self):
        save_path = os.path.join(get_models_location(), self.name)

        os.makedirs(save_path, exist_ok=True)

        save_path = save_path + f"/{datetime.now().strftime('%Y%m%d')}.pkl"

        with open(save_path, 'wb') as file:
            pickle.dump(self.t_matrix, file)

    def _load(self):
        load_folder = os.path.join(get_models_location(), self.name)
        
        for file in os.listdir(load_folder):
            date_created_str, file_ext = file.split('.')
            if file_ext == 'pkl':
                date_created = datetime.strptime(date_created_str,
                                                 '%Y%m%d')

                file_path = os.path.join(load_folder, file)
                if datetime.now() - date_created > timedelta(days=180):
                    #remove old model
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                    self._fit()
                    self._save()
                else:
                    with open(file_path, 'rb') as f:
                        self.t_matrix = pickle.load(f)
                        return None

        raise Exception("Can't load model")


    def _player_tracker(self, row):
        """
        keep track of where players are before and after:
        map each id to their current position. Can then apply map
        to after state id's to recover the previous position of the runner
        """

        initial_pos = {
            row['batter']: 'batter',
            row['on_1b']: '1b', 
            row['on_2b']: '2b',
            row['on_3b']: '3b'
        }

        after_pos = {
            'on_1b_after_prev_pos': initial_pos.get(row['on_1b_after'], False),
            'on_2b_after_prev_pos': initial_pos.get(row['on_2b_after'], False),
            'on_3b_after_prev_pos': initial_pos.get(row['on_3b_after'], False),
        }

        return after_pos

    def _encode_states(self, row):
        """
        To add state before and state after columns to the dataframe
        """

        state_before = GameState(row['outs_when_up'],
                                 row['on_1b'],
                                 row['on_2b'],
                                 row['on_3b'])
        state_after = GameStateTransition(row['outs_after'],
                                          row['on_1b_after_prev_pos'],
                                          row['on_2b_after_prev_pos'],
                                          row['on_3b_after_prev_pos'],
                                          row['runs_scored'])
        return state_before, state_after

    def _counter_to_probs(self, counter):
        """
        convert transition counts to categorical probability distribution
        """

        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}


    def _fit(self):
        """
        Fit the game state transition prob matrix

        Load Statcast data and encode the game state at each row using the
        GameState namedtuple. Find how the game state changes (runs scored,
        new base positions, new outs) and encode this in a GameStateTransition 
        named tuple. Finally, 
        """

        logging.info('Loading data')
        # query takes a while so save temp file
        df = get_game_state_t_prob_data()
        #df.to_csv('temp_query.csv', index=False)
        #df = pd.read_csv('temp_query.csv')

        logging.info('Transforming data')
        # look to next row to find after state. Fill with 3 outs and empty
        # bases if inning changed
        df['outs_after'] = df.groupby(
                ['game_pk', 'inning', 'inning_topbot']
            )['outs_when_up'].shift(-1, fill_value=3)

        for col in ('on_1b_after', 'on_2b_after', 'on_3b_after'):
            df[col] = df.groupby(
                    ['game_pk', 'inning', 'inning_topbot']
                )[col[:5]].shift(-1, fill_value=np.nan)

        # now that we have before and after state, filter to only 
        # look at hit outcome events
        df = df[~df['simplified_outcome'].isna()].reset_index(drop=True)


        # get columns determining prev position of after state runners 
        after_states = df.apply(self._player_tracker, axis=1)
        after_states_df = pd.DataFrame(after_states.tolist())
        df = pd.concat([df, after_states_df], axis=1)

        
        # get now what we have position changes tracked, don't need player id
        # for the bases and can just represent True for on_1b, False for not.
        for base in range(1,4):
            df[f'on_{base}b'] = df[f'on_{base}b'].notnull().astype(bool)


        logging.info('Getting transition probabilities')
        # encode game states and transitions:
        df['GameState'], df['GameStateTransition'] = zip(
            *df.apply(self._encode_states, axis=1)
        )

        # game state transistions
        df = df[['simplified_outcome', 'stand',
                 'GameState', 'GameStateTransition']]

        # group transitions
        transitions_grouped = df.groupby(['simplified_outcome',
                                          'stand','GameState'],
                                         )

        # initialize counter for each group
        transition_counts = transitions_grouped.apply(
            lambda group: Counter(group['GameStateTransition']),
            include_groups=False
        ) 

        self.t_matrix = transition_counts.apply(
                self._counter_to_probs)


if __name__ == '__main__':
    t_matrix = GameStateTransitionMatrix()

    event = 'home_run'
    stand = 'R'
    outs_before = 0
    on_1b = False
    on_2b = False
    on_3b = False

    g_state = GameState(outs_before,on_1b, on_2b, on_3b)

    for i in range(10):
        print(t_matrix(event, stand, g_state))







