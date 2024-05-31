import time
from mlb_simulator.data.data_utils import query_mlb_db

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from copulas.multivariate import VineCopula

# different pitcher mlb id's for testing purposes
PITCHERS = {
    'kukuchi': 579328,
    'yamamoto': 808967,
    'gil': 661563,
    'jones': 683003
}

# values which are known
FEATURES = 'pitch_number stand strikes balls'.split()
# values to be generated
OUTPUTS = 'release_speed release_spin_rate plate_x plate_z'.split()

def get_pitcher_arsenal(pitcher):
    pass

def query_pitches(pitcher, pitch_type):
    pass

def main():
    real_data = sample_trivariate_xyz()

    copula = GaussianMultivariate()
    center = VineCopula('center')
    regular = VineCopula('regular')
    direct = VineCopula('direct')

    #copula.fit(real_data)
    #center.fit(real_data)
    #regular.fit(real_data)
    #direct.fit(real_data)

    #center_samples = center.sample(1000)
    #regular_samples = regular.sample(1000)
    #direct_samples = direct.sample(1000)
    start = time.process_time()
    regular.fit(real_data)
    print(time.process_time() - start)

    print()
    print()
    print()

    start = time.process_time()
    synthetic_data = regular.sample(1)
    print(time.process_time() - start)
    print(synthetic_data)




if __name__ == '__main__':
    main()

