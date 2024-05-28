from datetime import datetime

class State:
    def __init__(self, outs=0, strikes=0, balls=0, inning=1, inning_is_top=True,
                 on_1b=None, on_2b=None, on_3b=None,
                 home_runs=0, home_cur_batter=0, 
                 away_runs=0, away_cur_batter=0) -> None:

        self.game_year = datetime.now().year
        self.outs = outs
        self.strikes = strikes
        self.balls = balls
        self.pitch_number = 1
        self.inning = inning
        self.inning_is_top = inning_is_top # away team bats in top of first, home team bats in bottom
        self.bases = {'1b': on_1b, '2b': on_2b, '3b': on_3b}

        self.home_runs = home_runs
        self.home_cur_batter = home_cur_batter

        self.away_runs = away_runs
        self.away_cur_batter = away_cur_batter
    
    def reset_game(self):
        self.reset_count()
        self.outs=0
        self.inning=1
        self.inning_is_top=True
        self.bases = {'1b': None, '2b': None, '3b': None}
        self.home_cur_batter=0
        self.home_runs=0
        self.away_cur_batter=0
        self.away_runs=0
    
    def get_game_state(self):
        return {
            'game_year': self.game_year,
            'strikes': self.strikes,
            'balls': self.balls,
            'pitch_number': self.pitch_number,
            'outs_when_up': self.outs,
            'on_1b': 1 if self.bases['1b'] else 0,
            'on_2b': 1 if self.bases['2b'] else 0,
            'on_3b': 1 if self.bases['3b'] else 0
        }
    
    def pprint_bases(self):
        for key in self.game_state.bases:
            player = self.game_state.bases[key]
            print(f'{key}: {player.name if player is not None else None}')

    def encode_cur_state(self):
        return (self.outs, tuple(self.bases[base] is not None for base in ['1b', '2b', '3b']))

    def process_state_change(self, new_state, batter):
        #(2, ('batter', '1b', False), 0)
        new_outs, new_base_state, runs_scored = new_state
        self.outs = new_outs

        #update runs scored
        if self.inning_is_top:
            self.away_runs += runs_scored
        else:
            self.home_runs += runs_scored
        
        #change players on bases
        new_bases = {'1b': None,
                     '2b': None,
                     '3b': None}
        for base, new_b in enumerate(new_base_state):
            base_string = f'{base+1}b'
            if new_b:
                if new_b=='batter':
                    new_bases[base_string] = batter
                else:
                    new_bases[base_string] = self.bases[new_b]

        #use this to track runs, etc.
        scored_or_out = [self.bases[base] for base in self.bases if self.bases[base] and self.bases[base] not in new_bases.values()]
        
        self.bases = new_bases
                    
    def ab_change(self, cur_pitcher):
        if self.inning_is_top:
            self.away_cur_batter += 1
            self.away_cur_batter = self.away_cur_batter % 9
        else:
            self.home_cur_batter += 1
            self.home_cur_batter = self.home_cur_batter % 9
        
        cur_pitcher.prev_pitch = None
        self.reset_count()

    def change_inning(self):
        if self.inning_is_top:
            self.inning_is_top = False
        else:
            self.inning += 1
            self.inning_is_top = True
        
        self.reset_count()
        self.outs = 0
        self.bases = {'1b': None, '2b': None, '3b': None}
    
    def reset_count(self):
        self.strikes = 0
        self.balls = 0
        self.pitch_number = 1


