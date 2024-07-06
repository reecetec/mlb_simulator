"""
Pitcher object. Is able to generate a pitch + pitch characteristics as a
function of the game state and opposing batter.
"""

from mlb_simulator.simulation.Player import Player
from mlb_simulator.data.data_utils import query_mlb_db
from mlb_simulator.models.pitching.pitch_sequencing import PitchSequencer
from mlb_simulator.models.pitching.pitch_characteristics \
        import PitchCharacteristics

import logging
import pandas as pd


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

class Pitcher(Player):
    def __init__(self, mlb_id=None, rotowire_id=None, backtest_date=''):
        super().__init__(mlb_id, rotowire_id, backtest_date)
        logger.info(f'Starting init for {self.name}')

        self.throws = query_mlb_db(f'''select p_throws from Statcast where
                                   pitcher={self.mlb_id} and p_throws
                                   is not null limit 1''')['p_throws'][0]
        self.cumulative_pitch_number = 0
        self.prev_pitch = None

        #instantiate and fit models:
        self.sequencer = PitchSequencer(self.mlb_id)
        self.char_generator = PitchCharacteristics(self.mlb_id)
        self.sequencer.fit()
        self.char_generator.fit(self.sequencer.pitch_arsenal)

    def __repr__(self):
        return f'Pitcher(mlb_id={self.mlb_id}, rotowire_id={self.rotowire_id})'

    
    def __str__(self):
        return f'{self.name}'


    def __call__(self, game_state, batter_stats):
        """Generate pitch characteristics"""

        features = pd.DataFrame({**game_state, **batter_stats})
        stand = batter_stats['stand'][0]

        pitch_type = self.sequencer(features)
        pitch_chars = self.char_generator(stand, pitch_type)

        return pitch_chars


        
if __name__=='__main__':
    ids = [683003]
    pitchers = [Pitcher(mlb_id=id) for id in ids]

    game_state = pd.DataFrame([{'game_year': 2024, 'pitch_number':1, 'strikes':0, 'balls':0,
               'outs_when_up':0,  'on_1b': False, 'on_2b': False,
               'on_3b': False, 'prev_pitch': None, 'cumulative_pitch_number': 1,
               }])
    batter_stats = pd.DataFrame([{'stand': 'R','FF_strike':0, 'CH_strike':0, 'SL_strike':0,
               'FC_strike':0, 'FF_woba':5, 'CH_woba':0, 'SL_woba':0, 'FC_woba':0,
                                  'CU_strike':0, 'CU_woba':0}])

    for pitcher in pitchers:
        print(pitcher)
        pitch = pitcher(game_state, batter_stats)
        print(pitch)


    

