"""
Fit a game state transition matrix. Given the outcome of the at bat, find
a probability distribution for how the game evolves.
"""

from mlb_simulator.features.build_features import get_game_state_t_prob_data
from mlb_simulator.models.model_utils import get_models_location

import pandas as pd
import numpy as np
import pickle
import os
import logging

from collections import namedtuple, Counter
from random import choices
from datetime import datetime, timedelta
from itertools import product

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TransitionState = namedtuple("TransitionState", "outs on_1b on_2b on_3b")
TransitionStateAfter = namedtuple(
    "TransitionStateAfter",
    ["after_outs", "after_1b", "after_2b", "after_3b", "runs_scored"],
)
"""
stores the game state transition. Bases will be '1b', 'batter', etc.
denoting the previous position. So if after_1b is 'batter' then
this means that the batter hit a single
"""


class GameStateTransitionMatrix:

    def __init__(self):
        self.name = "GameStateTransitionMatrix"
        self.t_matrix = None

        # get deterministic_transitions
        self._get_deterministic_transition_map()

        # Try loading saved model. If model doesn't exist or is expired,
        # fit/re-fit model
        try:
            self._load()
        except Exception as _:
            logger.info(f"Refitting {self.name}")
            self._fit()
            self._save()

        if self.t_matrix is None:
            raise Exception("Couldn't get transition probs")

    def __repr__(self):
        return "GameStateTransitionMatrix"

    def __str__(self):
        return "GameStateTransitionMatrx"

    def __call__(
        self, event: str, stand: str, input_state: TransitionState
    ) -> TransitionStateAfter:
        """
        Sample a state transition given input state

        Take the event that occured in the game, the batters stand, the
        current game state, and look up the transition distribution in the
        self.t_matrix variable. Then use the distribution of outcomes to
        sample and return a transition
        """

        if event in ("K", "walk"):
            return self.deterministic_transitions[event][input_state]

        if self.t_matrix is None:
            raise Exception("Couldn't get transition probs")

        distribution = self.t_matrix.loc[(event, stand, input_state)]
        sampled_transition = choices(
            list(distribution.keys()), weights=distribution.values(), k=1
        )[0]
        return sampled_transition

    def _save(self):
        """
        Save the fit transition matrix to memory
        """

        save_path = os.path.join(get_models_location(), self.name)

        os.makedirs(save_path, exist_ok=True)

        save_path = save_path + f"/{datetime.now().strftime('%Y%m%d')}.pkl"

        with open(save_path, "wb") as file:
            pickle.dump(self.t_matrix, file)

    def _load(self):
        """
        Try to load the transition matrix from memory
        """

        load_folder = os.path.join(get_models_location(), self.name)

        for file in os.listdir(load_folder):
            date_created_str, file_ext = file.split(".")
            if file_ext == "pkl":
                date_created = datetime.strptime(date_created_str, "%Y%m%d")

                file_path = os.path.join(load_folder, file)
                if datetime.now() - date_created > timedelta(days=180):
                    # remove old model
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(e)
                    self._fit()
                    self._save()
                else:
                    with open(file_path, "rb") as f:
                        self.t_matrix = pickle.load(f)
                        return None

        raise Exception("Can't load model")

    def _player_tracker(self, row):
        """
        keep track of where players are before and after an event:
        map each id to their current position. Can then apply map
        to after state id's to recover the previous position of the runner
        """

        initial_pos = {
            row["batter"]: "batter",
            row["on_1b"]: "1b",
            row["on_2b"]: "2b",
            row["on_3b"]: "3b",
        }

        after_pos = {
            "on_1b_after_prev_pos": initial_pos.get(row["on_1b_after"], False),
            "on_2b_after_prev_pos": initial_pos.get(row["on_2b_after"], False),
            "on_3b_after_prev_pos": initial_pos.get(row["on_3b_after"], False),
        }

        return after_pos

    def _encode_states(self, row):
        """
        To add state before and state after columns to the dataframe
        """

        state_before = TransitionState(
            row["outs_when_up"], row["on_1b"], row["on_2b"], row["on_3b"]
        )
        state_after = TransitionStateAfter(
            row["outs_after"],
            row["on_1b_after_prev_pos"],
            row["on_2b_after_prev_pos"],
            row["on_3b_after_prev_pos"],
            row["runs_scored"],
        )
        return state_before, state_after

    def _counter_to_probs(self, counter):
        """
        convert transition counts to categorical probability distribution
        """

        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}

    def _fit(self):
        """
        Fit the game state transition prob matrix

        Load Statcast data and encode the game state at each row using the
        GameState namedtuple. Find how the game state changes (runs scored,
        new base positions, new outs) and encode this in a GameStateTransition
        named tuple. Finally,
        """

        logger.info("Loading data")
        df = get_game_state_t_prob_data()

        logger.info("Transforming data")
        # look to next row to find after state. Fill with 3 outs and empty
        # bases if inning changed
        df["outs_after"] = df.groupby(["game_pk", "inning", "inning_topbot"])[
            "outs_when_up"
        ].shift(-1, fill_value=3)

        for col in ("on_1b_after", "on_2b_after", "on_3b_after"):
            df[col] = df.groupby(["game_pk", "inning", "inning_topbot"])[col[:5]].shift(
                -1, fill_value=np.nan
            )

        # now that we have before and after state, filter to only
        # look at hit outcome events
        df = df[~df["simplified_outcome"].isna()].reset_index(drop=True)

        # get columns determining prev position of after state runners
        after_states = df.apply(self._player_tracker, axis=1)
        after_states_df = pd.DataFrame(after_states.tolist())
        df = pd.concat([df, after_states_df], axis=1)

        # get now what we have position changes tracked, don't need player id
        # for the bases and can just represent True for on_1b, False for not.
        for base in range(1, 4):
            df[f"on_{base}b"] = df[f"on_{base}b"].notnull().astype(bool)

        logger.info("Getting transition probabilities")
        # encode game states and transitions:
        df["GameState"], df["GameStateTransition"] = zip(
            *df.apply(self._encode_states, axis=1)
        )

        # game state transistions
        df = df[["simplified_outcome", "stand", "GameState", "GameStateTransition"]]

        # group transitions
        transitions_grouped = df.groupby(
            ["simplified_outcome", "stand", "GameState"],
        )

        # initialize counter for each group
        transition_counts = transitions_grouped.apply(
            lambda group: Counter(group["GameStateTransition"]),
            include_groups=False,
        )

        self.t_matrix = transition_counts.apply(self._counter_to_probs)

    def _get_deterministic_transition_map(self):
        deterministic_transitions = {}
        deterministic_transitions["K"] = {}
        deterministic_transitions["walk"] = {}

        # Generate all permutations of transition states
        permutations = list(
            product(
                range(3),
                [
                    (x, y, z)
                    for x in (True, False)
                    for y in (True, False)
                    for z in (True, False)
                ],
            )
        )

        for state in permutations:

            first, second, third = state[1]

            transition_state = TransitionState(state[0], first, second, third)

            k_first = "1b" if first else False
            k_second = "2b" if second else False
            k_third = "3b" if third else False

            deterministic_transitions["K"][transition_state] = TransitionStateAfter(
                transition_state.outs + 1,  # add 1 out
                k_first,  # bases stay same
                k_second,
                k_third,
                0,  # no runs
            )

            # for walks
            new_first, new_second, new_third = "batter", False, False
            runs_scored = 0
            if first and second and third:
                new_second = "1b"
                new_third = "2b"
                runs_scored = 1
            elif first and not second and not third:
                new_second = "1b"
            elif first and second:
                new_second = "1b"
                new_third = "2b"
            elif first and third:
                new_second = "1b"
                new_third = "3b"
            else:
                if second:
                    new_second = "2b"
                if third:
                    new_third = "3b"

            deterministic_transitions["walk"][transition_state] = TransitionStateAfter(
                transition_state.outs,  # outs stay the same
                new_first,  # bases defined by logic above
                new_second,
                new_third,
                runs_scored,  # runs scored 0 unless bases loaded
            )

        self.deterministic_transitions = deterministic_transitions


if __name__ == "__main__":
    t_matrix = GameStateTransitionMatrix()

    event = "home_run"
    stand = "R"
    outs_before = 0
    on_1b = False
    on_2b = False
    on_3b = False

    g_state = TransitionState(outs_before, on_1b, on_2b, on_3b)

    for i in range(10):
        print(t_matrix(event, stand, g_state))
