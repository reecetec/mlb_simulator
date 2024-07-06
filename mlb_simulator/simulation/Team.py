"""
Holds information about a team. The name, starting pitcher, lineup, etc.
"""

from mlb_simulator.data.data_utils import get_team_id_map_location
from mlb_simulator.simulation.Batter import Batter
from mlb_simulator.simulation.Pitcher import Pitcher

import pandas as pd

class Team:
    
    def __init__(self, soup, is_home=True):
        self.is_home = is_home

        self._extract_info(soup)

        self.cur_batter = 0
        self._batters = []


    def __repr__(self):
        return f'Team(is_home={self.is_home})'


    def __str__(self):
        return f'{self.name} (Confirmed Lineup: {self.lineup_is_confirmed})'


    def __len__(self):
        return len(self._batters)


    def __getitem__(self, position):
        return self._batters[position]

    
    def fit(self):
        self.pitcher = Pitcher(rotowire_id=float(self.pitcher_id))
        self._batters = [Batter(rotowire_id=float(batter_id))
                         for batter_id in self.batter_ids]
        # for batter in self._batters:
        #     batter.fit()


    def _extract_info(self, soup):
        """Get the rotowire id's of the teams players, and some other info
        about the team, given the rotowire soup.
        """

        team = 'home' if self.is_home else 'visit'

        # get lineup boxes
        lineup_soup = soup.find("div", class_=f"lineup__team is-{team}")
        lineup_list_soup = soup.find("ul", class_=f"lineup__list is-{team}")

        # get team name, logo, and pitcher
        self.roto_name = lineup_soup.find("div", class_="lineup__abbr") \
                .get_text(strip=True)
        self.logo = lineup_soup.find("img")["src"]
        self.pitcher_id = lineup_list_soup.find("li",
                                        class_="lineup__player-highlight") \
                                                .find("a")["href"] \
                                                .split("-")[-1]

        # get batters
        self.batter_ids = []
        batters_soup = soup.find("ul", class_=f"lineup__list is-{team}") \
                .find_all("li", class_="lineup__player")
        for batter in batters_soup:
            batter_rotowireid = batter.find("a")["href"].split("-")[-1]
            self.batter_ids.append(batter_rotowireid)

        #get see if lineup is expected or confirmed
        is_confirmed = lineup_list_soup \
                .find("li", class_='lineup__status is-confirmed') 
        self.lineup_is_confirmed = True if is_confirmed else False

        # obtain team id information
        team_id_map = pd.read_csv(get_team_id_map_location())
        team_info = team_id_map.set_index('ROTOWIRETEAM') \
                .loc[[self.roto_name]] \
                .iloc[0]
        self.team_id = team_info['TEAMID']
        self.name = team_info['STATCAST']
         
