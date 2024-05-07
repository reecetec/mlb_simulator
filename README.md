MLB Game Simulator
==============================

Simulate MLB matchup results

- remove all bunting. manually impose batter going to bunt.
    - based on game state, the current batter, is this player going to bunt?
    - if so, use bunting model.
    - Else, use non bunting model.

- pitchouts can happen in both. Need to account for when this happens in game state.
    - pitch model should account for this.

- need to make models "transfer models"
    - train on general player, use this model to adjust for individual players performance.

- in game sim, have probability that the batter is going to try and bunt. Then run according model:
    - model 1: pitch_outcome
    - model 2: pitch_outcome_bunt

- simulation flow:
    
    DECIDE_HITTER
    - Pitcher on mound. 
    - Batter comes to bat. 
    
    SIM_AT_BAT:
    - Pitcher generates pitch based on game state and current batter (MODEL 1)
    - Batter decides to either bunt or hit (MODEL 2)
    - Batter strike, ball, or hits into play (MODEL 3)
    - Any on base steal? (MODEL 5)
    - Based on pitch characteristics, given the batter hits into play, what is hit tajectory? (MODEL 4)
    - Given game state, how does it change? (MODEL 6)
    - adjust game state. 

        AT_BAT_CHANGE:
        - if strikes=3, outs += 1
        - if balls=4, batter.walk()
        - if outs==3, game.change_inning...

- pitcher class:
    - p_throws
    - pitch_arsenal: the pitches he throws often
    - adjustment_factor: How heavily does he change pitch distribution based on batter
    - next_pitch_distribution: what is the next pitch_type to be thrown?
    - pitch_characteristics: generate inputs to batter outcome, based on pitch_type
    
- batter class
    - stands
    - 

TODO:
- Develop each individual model:
    - Pitch generation
    - Pitch outcome
    - Bat outcome
    - Field outcome
- Need to select proper features, obtain likely models
- Once each model is working individually, start to connect
    - Get starting lineups, etc.
    - Can start simulating games based on the models
    - Run lots of simulations for games, generate box score
- Additional models will be needed for lineup swaps, pitcher swaps, etc.
    - Can start with first 5 inning picks...
- Steal model:
    - when this player is on base, find prob of stealing, etc.
    - use general model for new players, or maybe new players don't steal at all.
- Build backtester:
    - https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/1st-half/
    - scrape above page -> has first 5 inning totals, etc.
    - https://github.com/FinnedAI/sportsbookreview-scraper

- get projected lineups:
 - https://github.com/fultoncjb/mlb-scraper/tree/master


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
