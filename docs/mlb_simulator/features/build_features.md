Module mlb_simulator.features.build_features
============================================

Functions
---------

    
`encode_cat_cols(X, encoders_dict)`
:   

    
`get_hit_classification_dataset(split=False, backtest_date='')`
:   

    
`get_hit_outcome_dataset(batter_id, split=False, backtest_date='')`
:   

    
`get_pitch_outcome_data(batter_id: int, backtest_date=None) ‑> tuple[pandas.core.frame.DataFrame, str]`
:   

    
`get_pitch_outcome_dataset(batter_id, batch_size=32, shuffle=False)`
:   

    
`get_pitch_outcome_dataset_general(cluster_id, stands, batch_size=32, shuffle=False)`
:   

    
`get_pitch_outcome_dataset_xgb(batter_id, split=False, backtest_date='')`
:   

    
`get_pitches(pitcher_id, opposing_stance, pitch_type, backtest_date='')`
:   

    
`get_sequencing_dataset(pitcher, backtest_date='')`
:   

    
`get_xgb_set(df, target_col, split=False, test_size=0.1)`
:   

    
`get_xgb_set_regression(df, target_cols, split=False, test_size=0.1)`
: