from mlb_simulator.models.game.game_state_transition_probs import (
    TransitionState,
    TransitionStateAfter,
)

from datetime import datetime as dt
import pandas as pd


class GameState:

    def __init__(self, home_team: str, away_team: str):
        self.home_team = home_team
        self.away_team = away_team

        self.inning_is_top = True  # True => away team batting, vice versa
        self.inning = 1

        self.outs = 0

        self.count = Count()
        self.bases = {"1b": None, "2b": None, "3b": None}

        self.cumulative_pitch_number = 1

        self.prev_pitch = None
        self.game_year = dt.now().year

        self.score = pd.DataFrame(
            0,
            index=pd.Index([away_team, home_team]),
            columns=pd.Index([str(i) for i in range(1, 6)]),
        )

    def __repr__(self):
        return "GameState()"

    def __str__(self):
        return f"""{'Top' if self.inning_is_top else 'Bot'} of the {self.inning}, {self.outs} outs, {self.cumulative_pitch_number} pitches in, with bases {self.bases}"""

    def __call__(self):
        """Gets current game state"""
        return pd.DataFrame(
            [
                {
                    "strikes": self.count.strikes,
                    "balls": self.count.balls,
                    "outs_when_up": self.outs,
                    "on_1b": True if self.bases["1b"] is not None else False,
                    "on_2b": True if self.bases["2b"] is not None else False,
                    "on_3b": True if self.bases["3b"] is not None else False,
                    "pitch_number": self.count.pitch_number,
                    "cumulative_pitch_number": self.cumulative_pitch_number,
                    "prev_pitch": self.prev_pitch,
                    "game_year": self.game_year,
                }
            ]
        )

    def reset(self):
        self.change_ab()
        self.inning = 1
        self.inning_is_top = True
        self.bases = {"1b": None, "2b": None, "3b": None}
        self.cumulative_pitch_number = 1
        self.score = pd.DataFrame(
            0,
            index=pd.Index([self.away_team, self.home_team]),
            columns=pd.Index([str(i) for i in range(1, 6)]),
        )

    def change_ab(self):
        self.count.reset()
        self.prev_pitch = None

    def change_half(self):
        if self.inning_is_top:
            self.inning_is_top = False
        else:
            self.inning += 1
            self.inning_is_top = True

        self.count.reset()
        self.outs = 0
        self.bases = {"1b": None, "2b": None, "3b": None}

    def get_transition_state(self):
        return TransitionState(
            self.outs,
            self.bases["1b"] is not None,  # convert to boolean
            self.bases["2b"] is not None,
            self.bases["3b"] is not None,
        )

    def process_state_change(self, new_state, batter):
        self.outs = new_state.after_outs
        # handle runs scored
        if self.inning_is_top:
            # self.score.iloc[0][str(self.inning)] += new_state.runs_scored
            self.score.loc[
                self.score.index[0], str(self.inning)
            ] += new_state.runs_scored
        else:
            # self.score.iloc[1][str(self.inning)] += new_state.runs_scored
            self.score.loc[
                self.score.index[1], str(self.inning)
            ] += new_state.runs_scored

        new_bases = {"1b": None, "2b": None, "3b": None}

        for base in ["1b", "2b", "3b"]:
            after_value = getattr(new_state, f"after_{base}")
            if after_value is None:
                continue
            elif after_value in ["1b", "2b", "3b"]:
                new_bases[base] = self.bases[after_value]
            elif after_value == "batter":
                new_bases[base] = batter

        self.bases = new_bases


class Count:

    def __init__(self):
        self.strikes = 0
        self.balls = 0
        self.pitch_number = 1

    def __repr__(self):
        return "Count()"

    def __str__(self):
        return f"Strikes: {self.strikes}, Balls: {self.balls}, Pitch Num: {self.pitch_number}"

    def __call__(self, event):
        match event:
            case "strike":
                self.strikes += 1
            case "ball":
                self.balls += 1
            case "foul":
                if self.strikes < 2:
                    self.strikes += 1
            case "hit_by_pitch":
                self.balls = 4
            case _:
                # print(f"unknown count event: {event}")
                pass
            # TODO: sometimes pitch outcome is none for some reason
            # figure out why...

        self.pitch_number += 1

    def reset(self):
        self.strikes = 0
        self.balls = 0
        self.pitch_number = 1


if __name__ == "__main__":
    # s = GameState()

    # print(s())
    pass
