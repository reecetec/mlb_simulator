"""
Contains information about the game. A schedule consists of these objects.
Contains the home and away team objects.
"""

from mlb_simulator.simulation.Team import Team
from mlb_simulator.models.game.game_state_transition_probs import (
    GameStateTransitionMatrix,
)
from mlb_simulator.models.game.hit_classifier import HitClassifier


class Game:

    def __init__(self, time, soup):
        self.time = time

        self.home_team = Team(soup)
        self.away_team = Team(soup, is_home=False)

        # Schedule object will find these
        self.venue_id = None
        self.venue_name = None

        # get general game models
        self.game_state_transition_matrix = GameStateTransitionMatrix()

    def __repr__(self):
        return f"Game({self.time})"

    def __str__(self):
        if self.venue_name is not None:
            return (
                f"{self.away_team} @ {self.home_team}, {self.time} at {self.venue_name}"
            )
        return f"{self.away_team} @ {self.home_team}, {self.time}"

    def reset(self):
        self.home_team.cur_batter = 0
        self.away_team.cur_batter = 0

    def fit(self):
        # fit hit classifier (requires venue name)
        if self.venue_id is not None:
            self.hit_classifier = HitClassifier(self.venue_id)
        else:
            print("Error getting venue name")
            exit()
        self.home_team.fit()
        self.away_team.fit()
        self.hit_classifier.fit()
