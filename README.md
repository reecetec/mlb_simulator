MLB Game Simulator
==============================

Simulate MLB matchup results

Link to documentation: [documentation](https://reecetec.github.io/mlb_simulator/)
 - not exhaustive as of now

Todo:
 - Update doc string format to format seen in data_utils.py
 - write improved pitch characteristic generation model - currently not conditioned on game state
    - more balls will be thrown if count is 0-2 for ex, but not currently a factor
 - add logic to track simulation box scores / player statistics
    - as of now just the game score
 - optimize simulations to increase speed (cprofile)
 - 

## Getting Started

1. Clone the repo
2. Install the package with poetry
```sh
make data
```
* Note: This downloads Statcast data from 2018. It will take a while.
* Sometimes pybaseball will run into an error, if so, rerun above command to continue setup
* The database will be located at ~/sports/mlb_simulator/data/databases/mlb.db. 

Once the database has been set up:

```python
from mlb_simulator.simulation.Schedule import Schedule
from mlb_simulator.simulation.Simulator import Simulator

s = Schedule(for_today=True)

# simulate first game in the schedule
simulator = Simulator(s[0])
simulator(num_simulations=100)

print(simulator.scores)

```

## Usage

mlb_simulator/simulation contains objects which interact to simulate an entire baseball 
game. The flow is as follows:

```mermaid
flowchart TD
    A[Get current GameState] --> B[Pitcher generates pitch type as function of GameState];
    B --> C[Pitcher generates pitch characteristics as function of GameState];
    C --> D[Batter gets hit/strike/ball/foul/hit by pitch];
    D --> E{Did batter hit the ball?};
    E -- yes --> F[Batter generates hit characteristics];
    E -- no --> G[Evolve GameState, check if Batter is out/walks, change at bat, etc.];
    F --> H[Hit classifier runs as a function of the hit characteristics, venue, batter speed];
    H --> I[Evolve remaining bases: if batter hits double, how do base runners react?];
    I --> G;
    G --> A;
```


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
    ├── mlb_simulator      <- Source code for use in this project.
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
