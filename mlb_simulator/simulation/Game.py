"""
Contains information about the game. A schedule consists of these objects.
Contains the home and away team objects.
"""

from mlb_simulator.simulation.Team import Team

class Game:

    def __init__(self, time, soup):
        self.time = time 

        self.home_team = Team(soup)
        self.away_team = Team(soup, is_home=False)


    def __repr__(self):
        return f'Game({self.time})'


    def __str__(self):
        return f'{self.away_team} @ {self.home_team}, {self.time}'


    def fit(self):
        self.home_team.fit()
        self.away_team.fit()




