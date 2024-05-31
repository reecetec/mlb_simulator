'''
To generate the characteristics of a given pitch type. For example,
generate the features of a fastball. 
'''

#get query mlb db path..
import sys
import os
import itertools
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from data.data_utils import query_mlb_db
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PITCH_CHARACTERISITCS = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'spin_axis', 'pfx_x', 'pfx_z',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
    'plate_x', 'plate_z'
]

def get_pitches(pitcher_id, opposing_stance, pitch_type):
    pitch_df =  query_mlb_db(f'''select 
        {', '.join(PITCH_CHARACTERISITCS)}
        from Statcast
        where pitcher={pitcher_id} and
        stand="{opposing_stance}" and
        pitch_type="{pitch_type}"
        and
        {' & '.join(PITCH_CHARACTERISITCS)} 
        is not null
        ''')

    return pitch_df

def fit_kde(pitcher_id, opposing_stance, pitch_type):

    pitch_df = get_pitches(pitcher_id, opposing_stance, pitch_type)
    X = pitch_df.values
    bandwidth = 0.1

    kde = KernelDensity(bandwidth=bandwidth, kernel='tophat')
    kde.fit(X, sample_weight=None)
    loglik = kde.score(X).sum()
    print(loglik)
    
    return kde

def sample_kde(kde, n=1):
    samples = kde.sample(n)
    sample_df = pd.DataFrame(samples, columns=PITCH_CHARACTERISITCS)
    return sample_df

def visually_inspect_kde(actual_pitches, sample_pitches, variable_pair):

    _, axes = plt.subplots(1, 2, figsize=(20, 6))  
    axes[0].scatter(actual_pitches[variable_pair[0]], actual_pitches[variable_pair[1]], label='Actual Pitch Correlation')
    axes[0].set_title('Scatter Plot for Actual Pitches')
    axes[0].set_xlabel(variable_pair[0])
    axes[0].set_ylabel(variable_pair[1])
    axes[0].legend()
    axes[1].scatter(sample_pitches[variable_pair[0]], sample_pitches[variable_pair[1]], label='Sample Pitch Correlation')
    axes[1].set_title('Scatter Plot for Sample Pitches')
    axes[1].set_xlabel(variable_pair[0])
    axes[1].set_ylabel(variable_pair[1])
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    kukuchi = 579328
    jones = 683003

    pitcher = kukuchi

    actual_pitches = get_pitches(pitcher, 'R', 'FF')
    kde = fit_kde(pitcher, 'R', 'FF')
    kde_sample = sample_kde(kde, len(actual_pitches))

    #for combo in itertools.combinations(PITCH_CHARACTERISITCS, 2):
        #col1, col2 = combo
        #visually_inspect_kde(actual_pitches, kde_sample, (col1, col2))
    for col in PITCH_CHARACTERISITCS:
        visually_inspect_kde(actual_pitches, kde_sample, ('plate_x', col))

