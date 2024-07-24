"""
Class to run simulations. Takes an instance of the Game object, and optionally
an instance of the GameState object (to allow the simulation to run from
a desired start state). Runs the simulation
"""

from mlb_simulator.simulation.Game import Game
from mlb_simulator.simulation.GameState import GameState
from mlb_simulator.simulation.Schedule import Schedule

import pandas as pd
from tqdm import tqdm


class Simulator:

    def __init__(self, game: Game):
        self.game = game
        self.state = GameState(game.home_team.name, game.away_team.name)

        self.game.fit()

        self.scores = pd.DataFrame(
            columns=pd.Index([game.home_team.name, game.away_team.name])
        )
        self.scores = []

    def __repr__(self):
        return f"Simulator(game=({self.game}), state={self.state})"

    def __call__(self, num_simulations=100, innings=5):
        """
        Run simulation for desired number of innings
        """
        print(f"Running {num_simulations} sims for {self.game}")
        for _ in tqdm(range(num_simulations)):
            try:
                for _ in range(innings):
                    self.simulate_inning()
                #  process score
                self.scores.append(self.state.score)
            except Exception as _:
                # print(f"Exception: {e}")
                pass
            # reset game state for next simulations
            self.game.reset()
            self.state.reset()

        return self.scores

    def simulate_inning(self):
        # home team fields first, then bats
        self.state.inning_is_top = True
        self.simulate_half(self.game.away_team, self.game.home_team)
        self.state.change_half()

        # TODO: check if 9th inning and home team is winning. if so, game is done.
        # irrelevant until model to predict relief pitchers implemented.

        self.simulate_half(self.game.home_team, self.game.away_team)
        self.state.change_half()

    def simulate_half(self, bat_team, field_team):
        while self.state.outs < 3:
            self.simulate_ab(bat_team, field_team)

    def simulate_pitch(self, batter, pitcher, oaa_lhh, oaa_rhh):

        game_state = self.state()
        pitch_type, pitch_chars = pitcher(game_state, pd.DataFrame([batter.stats]))
        game_state.insert(0, "p_throws", pitcher.throws)
        swing = batter(game_state, pitch_chars)

        # print(f"{pitcher.name} threw a {pitch_type} and {batter.name} {swing}")

        # handle batter hitting ball into play
        if swing.pitch_outcome == "hit_into_play":
            game_state = game_state[
                [
                    "game_year",
                    "outs_when_up",
                    "on_1b",
                    "on_2b",
                    "on_3b",
                ]
            ]
            hit_data = {
                "stand": [batter.stats["stand"]],
                "speed": [batter.stats["speed"]],
                "launch_speed": [swing.hit_outcome["launch_speed"]],
                "launch_angle": [swing.hit_outcome["launch_angle"]],
                "spray_angle": [swing.hit_outcome["spray_angle"]],
                "oaa": [oaa_lhh if batter.stats["stand"] == "L" else oaa_rhh],
            }
            hit_features = game_state.assign(**hit_data)
            hit_class = self.game.hit_classifier(hit_features)
            if hit_class is None:
                print("hit class is none")
            return hit_class

        else:
            self.state.count(swing.pitch_outcome)

        self.state.cumulative_pitch_number += 1
        self.prev_pitch = pitch_type

        return self.check_for_ab_event()

    def check_for_ab_event(self):
        if self.state.count.balls > 3:
            return "walk"
        elif self.state.count.strikes > 2:
            return "K"
        return None

    def simulate_ab(self, bat_team, field_team):

        outcome = None
        # until game event occurs, keep throwing pitches.
        while outcome is None:
            outcome = self.simulate_pitch(
                bat_team[bat_team.cur_batter],
                field_team.pitcher,
                field_team.oaa_lhh,
                field_team.oaa_rhh,
            )

        # outcome occured, process it
        # use transition matrix to update game state
        new_state = self.game.game_state_transition_matrix(
            outcome,
            bat_team[bat_team.cur_batter].stats["stand"],
            self.state.get_transition_state(),
        )
        self.state.process_state_change(new_state, bat_team[bat_team.cur_batter])

        # reset count, change batter
        self.state.change_ab()
        bat_team.change_batter()


if __name__ == "__main__":
    s = Schedule(for_today=False)
    some_game = s[0]
    print(some_game)
    game_simulator = Simulator(some_game)
    game_simulator()
