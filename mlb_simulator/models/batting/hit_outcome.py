"""
Module for the hit outcome model. Given the batter has hit the ball, what are 
the characteristics of the hit? What is the launch angle, spray angle, and
launch speed?

TODO: Introduce explanation of some variability though variables such as pitch
speed, bat speed (would need new model to generate bat speed given the pitch,
but data sparse as it was just introduced in 2024). Then, fit copulas on the 
residual distribution.
"""

from mlb_simulator.features.build_features import get_hit_outcome_data

import logging
import numpy as np
import pandas as pd
import pyvinecopulib as pv

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class HitOutcome:

    def __init__(self, batter_id):
        self.MODEL_NAME = 'hit_outcome'
        self.batter_id = batter_id

    def fit(self, backtest_date=None):
        
        # get hit data
        self.hits = get_hit_outcome_data(self.batter_id,
                                    backtest_date=backtest_date)
        targets = self.hits.columns
        u = pv.to_pseudo_obs(self.hits)
        cop = pv.Vinecop(d=len(targets))
        cop.select(data=u)
        self.copula = cop
        return cop


    def __call__(self, n=1):
        """
        Given the fit copula and the data it was fit on, return a sampled value
        """
        
        if not hasattr(self, 'copula'):
            print('Trying to use model without first fitting')
            return None

        u_sim = self.copula.simulate(n)
        # map generated uniform distribution back to original scale
        df_scale_sim = np.asarray([np.quantile(self.hits.values[:, i],
                                          u_sim[:, i])
                              for i in range(0, len(self.hits.columns))])
        if n == 1:
            return dict(zip(self.hits.columns, df_scale_sim.flatten()))
        else:
            return pd.DataFrame(df_scale_sim.T,
                                columns = self.hits.columns).describe()


if __name__ == '__main__':
    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = soto

    hit_gen = HitOutcome(batter_id)
    hit_gen.fit()

    for i in range(3):
        print(hit_gen())

    


