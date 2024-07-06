"""
Batter object. Is able to generate hit classes, e.g., single, strike, etc.
based on the pitch thrown.
"""

from mlb_simulator.simulation.Player import Player
from mlb_simulator.data.data_utils import query_mlb_db
from mlb_simulator.models.batting.hit_outcome import HitOutcome
from mlb_simulator.models.batting.pitch_outcome import PitchOutcome

import logging
import pandas as pd
from collections import namedtuple

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

class Batter(Player):

    Hit = namedtuple('Hit', 'pitch_outcome hit_outcome') 


    def __init__(self, mlb_id=None, rotowire_id=None, backtest_date=''):
        super().__init__(mlb_id, rotowire_id, backtest_date)
        logger.info(f'Starting init for {self.name}')

        self._get_stats()

        #instantiate and fit models
        self.pitch_outcome = PitchOutcome(self.mlb_id)
        self.hit_outcome = HitOutcome(self.mlb_id)
        self.pitch_outcome.fit()
        self.hit_outcome.fit()


    def __repr__(self):
        return f'Batter(mlb_id={self.mlb_id}, rotowire_id={self.rotowire_id})'


    def __call__(self, game_state, pitch_chars):
        """Generate hit outcome given a pitch"""

        features = pd.DataFrame({**game_state, **pitch_chars})

        pitch_outcome = self.pitch_outcome(features)

        if pitch_outcome == 'hit_into_play':
            hit_outcome = self.hit_outcome()
            return Batter.Hit(pitch_outcome, hit_outcome)

        return Batter.Hit(pitch_outcome, None)


    def _get_stats(self):
        self.stand = query_mlb_db(f'''
                                  select stand from Statcast
                                  where batter={self.mlb_id}
                                  and stand is not null limit 1
                                  ''')['stand'][0]
        self.pitch_woba = dict(
                query_mlb_db(f'''
                             select * from BatterAvgWobaByPitchType
                             where batter={self.mlb_id}
                             ''').drop('batter', axis=1).iloc[0]
                )
        self.pitch_strike = dict(
                query_mlb_db(f'''
                             select * from BatterStrikePctByPitchType
                             where batter={self.mlb_id}
                             ''').drop('batter', axis=1).iloc[0]
                )

        speed_query = query_mlb_db(f'''
                                   select speed from PlayerSpeed
                                   where mlb_id={self.mlb_id}
                                   ''')['speed'][0]

        if isinstance(speed_query, (int, float, str)):
            self.speed = float(speed_query)
        else:
            logger.critical(f'Error getting batter speed for {self.name}')

        self.sz = dict(
                query_mlb_db(f'''
                             select sz_bot, sz_top from BatterStrikezoneLookup
                             where batter={self.mlb_id}
                             ''').iloc[0]
                )
        self.stats = {f"{k}_strike": v for k, v in self.pitch_strike.items()}
        self.stats.update({f"{k}_woba": v for k, v in self.pitch_woba.items()})
        self.stats.update({f"{k}": v for k, v in self.sz.items()})
        self.stats['speed'] = self.speed
        self.stats['stand'] = self.stand


if __name__=='__main__':
    ids = [665742]
    batters = [Batter(mlb_id=id) for id in ids]

    game_state = pd.DataFrame([{'game_year': 2024, 'pitch_number':1, 'strikes':0, 'balls':0,
               'outs_when_up':0,  'on_1b': False, 'on_2b': False,
               'on_3b': False, 'prev_pitch': None, 'cumulative_pitch_number': 1,
                                'p_throws':'R'
               }])

    pitch_characteristics = {
        'release_speed': 81.5,
        'release_spin_rate': 2541.0,
        'plate_x': 0.77,
        'plate_z': 1.91
    }

    for batter in batters:
        print(batter)
        for i in range(15):
            hit = batter(game_state, pitch_characteristics)
            if hit.hit_outcome is not None:
                print(hit.hit_outcome)
            else:
                print(hit.pitch_outcome)




