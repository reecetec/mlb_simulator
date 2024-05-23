import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from simulation.Player import Player

from data.data_utils import query_mlb_db
from features.build_features import get_pitch_outcome_dataset_xgb, get_hit_outcome_dataset

import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from data.data_utils import compute_cat_loglik, compute_xgboost_loglik

from scipy.stats import norm, gaussian_kde
from scipy.interpolate import interp1d    

from pprint import pprint

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

class Batter(Player):
    def __init__(self, mlb_id=None, rotowire_id=None, backtest_date=''):
        super().__init__(mlb_id, rotowire_id, backtest_date)
        logger.info(f'Starting init for {self.name}')
        
        #gather batter characteristics to aid in pitch generation
        self.stand = query_mlb_db(f'select stand from Statcast where batter={self.mlb_id} and stand is not null limit 1')['stand'][0]
        self.pitch_woba = dict(query_mlb_db(f'select * from BatterAvgWobaByPitchType where batter={self.mlb_id};').drop('batter', axis=1).iloc[0])
        self.pitch_strike = dict(query_mlb_db(f'select * from BatterStrikePctByPitchType where batter={self.mlb_id};').drop('batter', axis=1).iloc[0])
        self.speed = query_mlb_db(f'select speed from PlayerSpeed where mlb_id={self.mlb_id}')['speed'][0]
        self.sz = dict(query_mlb_db(f'select sz_bot, sz_top from BatterStrikezoneLookup where batter={self.mlb_id}').iloc[0])

        self.stats = {f"{k}_strike": v for k, v in self.pitch_strike.items()}
        self.stats.update({f"{k}_woba": v for k, v in self.pitch_woba.items()})
        self.stats.update({f"{k}": v for k, v in self.sz.items()})
        self.stats['speed'] = self.speed
        self.stats['stand'] = self.stand

        self.fit_pitch_outcome_model()
        self.fit_hit_outcome_model()
        #logger.info(f'{self.name} stands {self.stand}')
        logger.info(f'Init complete for {self.name}')

    # predicts strike, foul, etc.
    def fit_pitch_outcome_model(self):
        X, y, encoders = get_pitch_outcome_dataset_xgb(self.mlb_id, backtest_date=self.backtest_date)
        self.pitch_outcome_encoders = encoders
        self.pitch_outcome_vars = X.columns
        params = {'colsample_bytree': 0.5,
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 100,
            'subsample': 0.5
        }
        self.pitch_outcome_clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', **params)
        self.pitch_outcome_clf.fit(X, y)

        assert(compute_xgboost_loglik(self.pitch_outcome_clf.predict_proba(X),y,'pitch_outcome') > compute_cat_loglik(X, y, y, 'pitch_outcome'))

    # use model
    def generate_pitch_outcome(self, game_state, pitch_characteristics):
        # get df of current game state and batter stats to generate pitch
        combined_data = {**game_state, **pitch_characteristics}
        df = pd.DataFrame([combined_data])

        #encode cols:
        for col in df.columns:
            if col in self.pitch_outcome_encoders.keys():
                df[col] = self.pitch_outcome_encoders[col].transform(df[col])

        #ensure ordering kept the same as when fit
        df = df[self.pitch_outcome_vars]

        outcome_distribution = self.pitch_outcome_clf.predict_proba(df)[0]
        
        outcome_choice_encoded = np.random.choice(len(outcome_distribution), size=1, p=outcome_distribution)
        outcome_choice = self.pitch_outcome_encoders['pitch_outcome'].inverse_transform(outcome_choice_encoded)[0]
        return outcome_choice


    # predicts hit launch speed, angle, etc. 
    def fit_hit_outcome_model(self):
        X, y, encoders = get_hit_outcome_dataset(self.mlb_id, backtest_date=self.backtest_date)

        # Step 1: Estimate the marginal distributions using KDE
        kde_estimators = []
        inverse_cdfs = []

        self.hit_outcome_cols = y.columns

        for column in self.hit_outcome_cols:
            kde = gaussian_kde(y[column])
            kde_estimators.append(kde)
            
            # Compute the inverse CDF using interpolation
            x_vals = np.linspace(y[column].min(), y[column].max(), 1000)
            cdf_vals = np.array([kde.integrate_box_1d(-np.inf, x) for x in x_vals])
            inverse_cdf = interp1d(cdf_vals, x_vals, bounds_error=False, fill_value=(x_vals[0], x_vals[-1]))
            inverse_cdfs.append(inverse_cdf)
        
        self.inverse_cdfs = inverse_cdfs
        
        # Step 2: Transform data to uniform marginals using the CDF of KDEs
        uniform_marginals = np.zeros_like(y.values)
        for i, column in enumerate(y.columns):
            cdf_values = np.array([kde_estimators[i].integrate_box_1d(-np.inf, x) for x in y[column]])
            uniform_marginals[:, i] = cdf_values
        
        # Step 3: Transform uniform marginals to standard normal marginals
        normal_marginals = norm.ppf(uniform_marginals)
        
        # Ensure no infinite values (can occur if uniform marginals are exactly 0 or 1)
        normal_marginals[np.isinf(normal_marginals)] = np.nan
        self.normal_marginals = np.nan_to_num(normal_marginals)

    def generate_hit_outcome(self):
        # Step 4: Generate new samples from a multivariate normal distribution
        mean = np.zeros(len(self.hit_outcome_cols))
        cov = np.corrcoef(self.normal_marginals, rowvar=False)
        new_samples_normal = np.random.multivariate_normal(mean, cov, size=1)
        
        # Step 5: Transform new samples from standard normal to uniform marginals
        new_samples_uniform = norm.cdf(new_samples_normal)
        
        # Step 6: Transform uniform marginals back to original KDE marginals
        new_samples = np.zeros_like(new_samples_uniform)
        for i, column in enumerate(self.hit_outcome_cols):
            new_samples[:, i] = self.inverse_cdfs[i](new_samples_uniform[:, i])
        
        # Create a DataFrame with the new samples
        generated_dataframe = pd.DataFrame(new_samples, columns=self.hit_outcome_cols)

        return dict(generated_dataframe.iloc[0])


if __name__ == '__main__':
    show = 660271
    showtime = Batter(mlb_id=show)
    showtime.print_info()

    pitch_characteristics = {
        'release_speed': 91.5,
        'release_spin_rate': 2541.0,
        'release_extension': 6.3,
        'release_pos_x': -1.1,
        'release_pos_y': 54.2,
        'release_pos_z': 5.71,
        'spin_axis': 201.0,
        'pfx_x': -0.38,
        'pfx_z': 1.0,
        'vx0': 5.52946075120911,
        'vy0': -133.034107193298,
        'vz0': -5.81172219449376,
        'ax': -5.79249929698345,
        'ay': 31.1262248305161,
        'az': -19.2977353984413,
        'plate_x': 0.77,
        'plate_z': 1.91
    }

    game_state = {
        'pitch_number': 1.0,
        'strikes': 0.0,
        'balls': 0.0,
    }

    print(showtime.generate_pitch_outcome(game_state, pitch_characteristics))
    print(showtime.generate_hit_outcome())
    pprint(showtime.stats)