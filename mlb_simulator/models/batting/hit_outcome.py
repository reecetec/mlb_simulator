"""
Module for the hit outcome model. Given the batter has hit the ball, what are 
the characteristics of the hit? What is the launch angle, spray angle, and
launch speed?

TODO: Introduce explanation of some variability though variables such as pitch
speed, bat speed (would need new model to generate bat speed given the pitch,
but data is new and was just introduced in 2024). Then, fit copulas on the 
residual distribution.
"""

from mlb_simulator.features.build_features import get_hit_outcome_data

import logging
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from collections import namedtuple
import pyvinecopulib as pv


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

MODEL_NAME = 'hit_outcome'

def fit_hit_outcome_model(batter_id: int, backtest_date=None):
    
    # get hit data
    hits = get_hit_outcome_data(batter_id, backtest_date=backtest_date)
    targets = hits.columns

    u = pv.to_pseudo_obs(hits)

    cop = pv.Vinecop(d=len(targets))
    cop.select(data=u)

    return cop


def sample_copula(cop, df, n=1):
    """
    Given a fit copula and the data it was fit on, return a sampled value
    """

    u_sim = cop.simulate(n)
    df_scale_sim = np.asarray([np.quantile(df.values[:, i],
                                      u_sim[:, i])
                          for i in range(0, len(df.columns))])
    if n == 1:
        return dict(zip(df.columns, df_scale_sim.flatten()))
    else:
        return pd.DataFrame(df_scale_sim.T, columns = df.columns).describe()


if __name__ == '__main__':
    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = soto

    cop1 = fit_hit_outcome_model(batter_id)
    cop2 = fit_hit_outcome_model(schneider)
    print(sample_copula(cop1, get_hit_outcome_data(batter_id)))
    print(sample_copula(cop2, get_hit_outcome_data(schneider)))

    

    


