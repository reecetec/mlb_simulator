"""
This module includes functions to help with fitting models, saving models/
model hyperparams, and evaluating models
"""

from mlb_simulator.data.data_utils import get_models_location

import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, log_loss
from scipy.stats import chisquare 
from datetime import datetime, timedelta


def save_hyperparams(model_name, player_id, hyperparams: dict):
    """Save hyperparams to a json file 
    """

    save_path = os.path.join(
        get_models_location(),
        model_name,
        f'{player_id}-{datetime.now().strftime('%Y%m%d')}.json'
    )
    with open(save_path, 'w') as f:
        json.dump(hyperparams, f)


def check_for_hyperparams(model_name, player_id):
    """Check to see if up to date hyperparams exist for desired model/player

    """
    model_folder = os.path.join(get_models_location(), model_name)
    saved_params = os.listdir(model_folder)

    for file in saved_params:

        try:
            parts = file.split('-')
            mlb_id, rest = parts

            # if mlb id found, check if model hyperparams tuned within 90 days
            if mlb_id == str(player_id):
                date_part = rest.replace('.json', '')
                date_created = datetime.strptime(date_part, '%Y%m%d')
                if datetime.now() - date_created > timedelta(days=90):
                    return False
                else:
                    return os.path.join(model_folder, file)
        except Exception as e:
            print(
                (f'Found invalid file format in {model_folder}: {e} \n\n'),
                (f'Files in { model_folder
                    } should be of format: mlbid-%Y%m%d.json \n\n')
            )
            raise

    return False

def categorical_model_pipeline(model, data, target_col):

    #encode target col & split into X, y
    le = LabelEncoder()
    features = data.drop(columns=[target_col])
    target = le.fit_transform(data[target_col])

    numeric_features = features.select_dtypes(
        include=['float64', 'int64']
    ).columns
    categorical_features = features.select_dtypes(
        include=['object']
    ).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model())
    ])

    return deepcopy(model), deepcopy(le), features, target


def sample_predictions(classifier, X):
    """ Given a classifier with a predict_proba method and input features X
        generate a random sample.
    """
    pred_proba = classifier.predict_proba(X)
    sampled_predictions = np.array([
        np.random.choice(classifier.classes_, p=proba) for
            proba in pred_proba
    ])

    return sampled_predictions

def categorical_chisquare(model, label_encoder, X_test, y_test):
    """ Run a hypothesis test that generated data is similar in dist to actual
    """
    hyp_tests = []
    for _ in range(1000):
        sampled_preds = sample_predictions(model, X_test)
        pred_counts = pd.DataFrame(
                label_encoder.inverse_transform(sampled_preds)
            ).value_counts()
        actual_counts = pd.DataFrame(
                label_encoder.inverse_transform(y_test)
            ).value_counts()

        pred_counts = pred_counts.reindex(actual_counts.index, fill_value=0)
        chi2_stat, p_value = chisquare(f_obs=actual_counts, f_exp=pred_counts)
        if p_value < 0.05:
            hyp_tests.append('different')
        else:
            hyp_tests.append('similar')

    sampled_preds = sample_predictions(model, X_test)
    pred_counts = pd.DataFrame(
            label_encoder.inverse_transform(sampled_preds)
        ).value_counts()
    actual_counts = pd.DataFrame(
            label_encoder.inverse_transform(y_test)
        ).value_counts()

    pred_counts = pred_counts.reindex(actual_counts.index, fill_value=0)
    chi2_stat, p_value = chisquare(f_obs=actual_counts, f_exp=pred_counts)
    
    print("Chi-square statistic:", chi2_stat)
    print("p-value:", p_value)
    

    # Interpretation of the result
    if p_value < 0.05:
        print(
            "The distributions are significantly different "
            "(reject the null hypothesis)."
            )
    else:
        print(
            "The distributions are not significantly different "
            "(fail to reject the null hypothesis)."
            )
    print(pd.DataFrame(hyp_tests, columns=['test_results']).value_counts())


def classifier_report(model, label_encoder, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred)))
    print('Log Loss:', log_loss(y_test, y_prob))

    categorical_chisquare(model, label_encoder, X_test, y_test)

def xgb_hyperparam_optimizer(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, shuffle=False)

    grouped_param_grid = [
        {
            'classifier__max_depth':[2, 3, 5],
            'classifier__min_child_weight': [1, 3, 5]
        },
        {
            'classifier__subsample': [0.5, 0.7, 0.9],
            'classifier__colsample_bytree': [0.5, 0.7, 0.9]
        },
        {
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__n_estimators': [50, 100, 200]
        }
    ]

    # use mlogloss always
    all_best_params = {'classifier__eval_metric':'mlogloss'}
    for param_group in grouped_param_grid:
        best_loss = float('inf')
        best_params = {}

        for params in ParameterGrid(param_group):
            #fit model with cur param group params and previous 
            #param group params
            model.set_params(**all_best_params, **params)
            model.fit(X_train, y_train)

            #find model with best log loss on test set
            y_prob = model.predict_proba(X_test)
            loss = log_loss(y_test, y_prob)
            if loss < best_loss:
                best_loss = loss
                best_params = params
        all_best_params = {**all_best_params, **best_params}

    return all_best_params

def compute_model_loglik(y_pred_proba, y_test, target_col):
    loglik = 0
    for idx, target in enumerate(y_test[target_col]):
        loglik += np.log(y_pred_proba[idx, target])
    return loglik

def compute_cat_loglik(X_train, y_train, y_test, target_col):
    df = pd.concat([X_train, y_train], axis=1)
    pitch_cat_prob = (df[target_col].value_counts() / len(df)) 
    loglik = 0
    for target in y_test[target_col]:
        loglik += np.log(pitch_cat_prob.loc[target])
    return loglik

if __name__ == '__main__':
    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = soto

    check_for_hyperparams('pitch_outcome', batter_id)
    
    #params = {'eval_metric':'mlogloss'}
    #data = query_mlb_db(f'''
    #    select 
    #        case
    #            when description='swinging_strike' 
    #                    or description='swinging_strike_blocked' 
    #                    or description='called_strike' 
    #                    or description='foul_tip' 
    #                    or description='swinging_pitchout'
    #                then 'strike'
    #            when description='foul'
    #                    or description='foul_pitchout'
    #                then 'foul'
    #            when description='ball'
    #                    or description='blocked_ball'
    #                    or description='pitchout'
    #                then 'ball'
    #            /* when description='hit_by_pitch' then 'hit_by_pitch' */
    #            when description='hit_into_play' then 'hit_into_play'
    #            else NULL
    #        end as pitch_outcome,
    #        p_throws,
    #        pitch_number, strikes, balls,
    #        release_speed, 
    #        release_spin_rate, 
    #        plate_x, plate_z
    #    from Statcast
    #    where batter={batter_id}
    #    and pitch_outcome & p_throws & pitch_number & strikes & balls &
    #        release_speed &
    #        release_spin_rate &
    #        plate_x & plate_z
    #    is not null
    #    order by game_date asc, at_bat_number asc, pitch_number asc;
    #                    ''')
    #data = data.tail(8000)
    #print(data.columns)

    #target_col = 'pitch_outcome'

    #model, le, X_train, X_test, y_train, y_test = categorical_model_pipeline(
    #        xgb.XGBClassifier,
    #        params,
    #        data,
    #        target_col,
    #        split_data=True
    #    )

    #best_params = param_optim(model, X_train, X_test, y_train, y_test)
    #model.set_params(**best_params)
    #model.fit(X_train, y_train)
    #classifier_report(model, le, X_test, y_test)


