import sys
import os
import logging
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import timedelta, datetime
import statsapi

from simulation.Team import Team

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
        print(self.time)
        print(self.venue_name, self.venue_id)
        print(self.away_team.name, self.home_team.name)


        #load hit classification model
        #load game state transition dict

        logger.info('Fitting home team models')
        #self.home_team.fit_models()
        logger.info('Fitting away team models')
        #self.away_team.fit_models()

        logger.info(f'Init complete for {self.away_team.name} @ {self.home_team.name} starting at {self.time}')

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

