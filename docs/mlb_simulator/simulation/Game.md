Module mlb_simulator.simulation.Game
====================================

Classes
-------

`Game(rotowire_lineup, rotowire_time, game_tomorrow=False)`
:   

    ### Methods

    `classify_hit(self, game_state, fielding_team_name, batter_speed, batter_stand, hit_stats, fielding_oaa)`
    :

    `fit_hit_classifier(self)`
    :

    `get_deterministic_transition_map(self)`
    :

    `get_game_state_transition_map(self)`
    :

    `get_venue(self)`
    :

    `print_game_score(self)`
    :

    `sample_next_state(self, event, stand, cur_state, print_probs=False)`
    :

    `simulate_ab(self, fielding_team, batting_team)`
    :

    `simulate_game(self)`
    :

    `simulate_inning(self, fielding_team, batting_team)`
    :

    `simulate_pitch(self, fielding_team, batting_team)`
    :