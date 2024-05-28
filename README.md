MLB Game Simulator
==============================

Simulate MLB matchup results

Todo:
 - Remove need for SDV (super slow)
 - Improve Batter strike/ball/hit etc. model, maybe use less features
 - Use Copulas for pitch generation
 - Fix game state transition probabilities edge cases
 - Implement simulation logic from notebook 7.0 in simulation class
 - Fix messy imports 

## Getting Started

1. Clone the repo
2. Install the requirements.txt (in a conda env or venv), and then run the following in the project root to create sqlite db
```sh
make data
```
* Note: This downloads Statcast data from 2018. It will take a while.
* Sometimes pybaseball will run into an error, if so, rerun above command to continue setup
* By default, the database will be located at ~/sports/mlb_simulator/data/databases/mlb.db. 
* If you wish to change this, modify the path in /src/data/data_utils.py

Once the database has been set up:

```sh
pip install .
```

## Usage

src/simulation contains objects which interact to simulate an entire baseball 
game. The flow is as follows:

```mermaid
flowchart TD
    A[Current Game State] --> B[Pitcher generates pitch type];
    B --> C[Pitcher generates pitch characteristics];
    C --> D[Batter gets hit/strike/ball/foul/hit by pitch];


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
