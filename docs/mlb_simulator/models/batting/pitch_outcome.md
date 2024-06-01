Module mlb_simulator.models.batting.pitch_outcome
=================================================
Module for the pitch outcome model. This model determines the pitch outcome,
meaning it answers the following question: was the ball hit, was there a
strike, ball, foul?

Functions
---------

    
`fit_pitch_outcome_model(batter_id: int, backtest_date=None)`
:   Fit pitch outcome model for a given batter id
    
    This function will obtain the data required to fit the model, check if 
    hyperparameters have previously been optimized for the given batter id. 
    If so, load them, otherwise, fit them and save them. Once ideal hyperparams
    have been fit, fit the model to the dataset.
    
    Parameters:
        batter_id (int): the desired batter's mlb id
        backtest_date (str, optional): the date to be used if backtesting

    
`main()`
: