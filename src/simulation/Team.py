import sys
import os
import pathlib
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from simulation.Pitcher import Pitcher
from simulation.Batter import Batter
from bs4 import BeautifulSoup
import pandas as pd


class Team:
    def __init__(self, rotowire_lineup, is_home, backtest_date='') -> None:
        self.is_home = is_home
        self.name, self.logo, self.starting_pitcher, self.batting_lineup, self.lineup_confirmed = self.get_rotowire_info(rotowire_lineup, 'home' if self.is_home else 'visit')
        self.models_fit = False
        
        home_dir = pathlib.Path.home()
        team_id_map_dir = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data', 'raw', 'team_id_map.csv')
        team_id_map = pd.read_csv(team_id_map_dir)
        self.team_id = team_id_map.set_index('ROTOWIRETEAM').loc[[self.name]].iloc[0]['TEAMID']
        self.statcast_name = team_id_map.set_index('TEAMID').loc[[self.team_id]].iloc[0]['STATCAST']

    def fit_models(self):
        self.starting_pitcher = Pitcher(rotowire_id=float(self.starting_pitcher))
        self.batting_lineup = [Batter(rotowire_id=float(id)) for id in self.batting_lineup]
        self.models_fit = True

    def get_rotowire_info(self, game, team='home'):

        # get lineup boxes
        lineup_soup = game.find("div", class_=f"lineup__team is-{team}")
        lineup_list_soup = game.find("ul", class_=f"lineup__list is-{team}")

        # get team name, logo, and pitcher
        team_name = lineup_soup.find("div", class_="lineup__abbr").get_text(strip=True)
        team_logo = lineup_soup.find("img")["src"]
        pitcher = lineup_list_soup.find("li", class_="lineup__player-highlight").find("a")["href"].split("-")[-1]

        #get see if lineup is expected or confirmed
        is_confirmed = lineup_list_soup.find("li", class_='lineup__status is-confirmed') 
        is_expected = lineup_list_soup.find("li", class_='lineup__status is-expected')
        if is_confirmed:
            lineup_confirmed = True
        else:
            lineup_confirmed = False

        # get batters
        batters = []
        batters_soup = game.find("ul", class_=f"lineup__list is-{team}").find_all("li", class_="lineup__player")
        for batter in batters_soup:
            batter_rotowireid = batter.find("a")["href"].split("-")[-1]
            batters.append(batter_rotowireid)
            
        return team_name, team_logo, pitcher, batters, lineup_confirmed
    
    
    def print_team_info(self):
        if self.models_fit:
            init = f"""{self.name} ({'home' if self.is_home else 'away'}) lineup:"""
            print(init)
            print('-' * len(init))
            self.starting_pitcher.print_info()
            print('')
            for batter in self.batting_lineup:
                batter.print_info()
        else:
            print('Fit models before running')
            
    #team_name, team_logo, pitcher, batters = get_rotowrite_info(boxes[0], team='visit')

if __name__ == '__main__':
    print('test')

