'''
To generate the characteristics of a given pitch type. For example,
generate the features of a fastball. 
'''

from mlb_simulator.data.data_utils import query_mlb_db
from mlb_simulator.features.build_features import _pitch_characteristics
from sklearn.neighbors import KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

def get_pitches(pitcher_id, opposing_stance, pitch_type):
    pitch_df =  query_mlb_db(f'''select 
        {', '.join(_pitch_characteristics)}
        from Statcast
        where pitcher={pitcher_id} and
        stand="{opposing_stance}" and
        pitch_type="{pitch_type}"
        and
        {' & '.join(_pitch_characteristics)} 
        is not null
        ''')

    return pitch_df

def fit_kde(pitcher_id, opposing_stance, pitch_type):

    pitch_df = get_pitches(pitcher_id, opposing_stance, pitch_type)
    X = pitch_df.values
    bandwidth = 0.1

    kde = KernelDensity(bandwidth=bandwidth, kernel='tophat')
    kde.fit(X, sample_weight=None)
    #loglik = kde.score(X).sum()
    
    return kde

def sample_pitch_kde(kde, n=1):
    samples = kde.sample(n)
    sample_df = pd.DataFrame(samples, columns=pd.Index(_pitch_characteristics))
    return sample_df

def visually_inspect_kde(actual_pitches, sample_pitches, variable_pair):

    _, axes = plt.subplots(1, 2, figsize=(20, 6))  
    axes[0].scatter(actual_pitches[variable_pair[0]],
                    actual_pitches[variable_pair[1]],
                    label='Actual Pitch Correlation')
    axes[0].set_title('Scatter Plot for Actual Pitches')
    axes[0].set_xlabel(variable_pair[0])
    axes[0].set_ylabel(variable_pair[1])
    axes[0].legend()
    axes[1].scatter(sample_pitches[variable_pair[0]],
                    sample_pitches[variable_pair[1]],
                    label='Sample Pitch Correlation')
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

    actual_pitches = get_pitches(pitcher, 'L', 'FF')
    kde = fit_kde(pitcher, 'L', 'FF')
    kde_sample = sample_pitch_kde(kde, len(actual_pitches))
    print(kde_sample)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(actual_pitches)
    diagnostic_report = run_diagnostic(
            real_data = actual_pitches,
            synthetic_data = kde_sample,
            metadata=metadata)
    quality_report = evaluate_quality(
        real_data= actual_pitches,
        synthetic_data= kde_sample,
        metadata=metadata)

    print(diagnostic_report)
    print(quality_report)

