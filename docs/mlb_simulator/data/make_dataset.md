Module mlb_simulator.data.make_dataset
======================================

Functions
---------

    
`main()`
:   

    
`update_chadwick_repo()`
:   

    
`update_oaa()`
:   

    
`update_player_name_map(save_path)`
:   

    
`update_run_speed()`
:   

    
`update_similar_sz_table()`
:   Creates cluster of players with similar strikezones. Was used to
    transfer learn neural networks. XGBoost being used make this less
    useful.

    
`update_statcast_table()`
:   Uses pybaseball to download statcast data from 2018 locally.

    
`update_sz_lookup()`
:   Get batter's strikezone for pitch generation.

    
`update_venue_game_pk_mapping()`
:   

    
`update_woba_strike_tables(min_pitch_count=50, min_hit_count=15, backtest_yr=None)`
:   Sets up 2 tables in mlb.db containing the batter id,
    first the batter's strike percentage into a given pitch type,
    second, the batter's average woba given a hit.
    
    To be used in the pitch_type generator

    
`validate_db_and_table(table_name)`
: