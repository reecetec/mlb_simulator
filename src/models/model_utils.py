import numpy as np
import pandas as pd
from src.data.data_utils import query_mlb_db
import xgboost as xgb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, log_loss
from scipy.stats import chisquare, fisher_exact


def categorical_model_pipeline(model, model_params, data, target_col,
                       split_data=False):

    #encode target col
    le = LabelEncoder()
    features = data.drop(columns=[target_col])
    target = le.fit_transform(data[target_col])

    # if not splitting data, train model on the entire input set.
    X_test, y_test = None, None
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.05, shuffle=False)
    else:
        X_train, y_train = features, target

    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model(**model_params))
    ])

    # Train the model
    model.fit(X_train, y_train)

    if split_data:
        return deepcopy(model), deepcopy(le), X_train, X_test, y_train, y_test

    return deepcopy(model), deepcopy(le)


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

def categorical_fisher_exact(model, label_encoder, X_test, y_test):
    """ Run a hypothesis test that generated data is similar in distribution to actual using Fisher's exact test """
    sampled_preds = sample_predictions(model, X_test)
    pred_counts = pd.DataFrame(label_encoder.inverse_transform(sampled_preds)).value_counts()
    actual_counts = pd.DataFrame(label_encoder.inverse_transform(y_test)).value_counts()

    pred_counts = pred_counts.reindex(actual_counts.index, fill_value=0)
    
    # Prepare the contingency table for Fisher's exact test
    contingency_table = pd.concat([actual_counts, pred_counts], axis=1).fillna(0)
    
    # Perform Fisher's exact test
    odds_ratio, p_value = fisher_exact(contingency_table)
    
    print("Odds ratio:", odds_ratio)
    print("p-value:", p_value)
    
    # Interpretation of the result
    if p_value < 0.05:
        print("The distributions are significantly different (reject the null hypothesis).")
    else:
        print("The distributions are not significantly different (fail to reject the null hypothesis).")


def classifier_report(model, label_encoder, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred)))
    print('Log Loss:', log_loss(y_test, y_prob))

    categorical_chisquare(model, label_encoder, X_test, y_test)

def param_optim(model, X_train, X_test, y_train, y_test):

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

    all_best_params = {}
    for param_group in grouped_param_grid:
        print(f'Cur param group: {list(param_group.keys())}')
        best_loss = float('inf')
        best_params = {}

        for params in ParameterGrid(param_group):
            #fit model with cur param group params and previous 
            #param group params
            model.set_params(**params, **all_best_params)
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

    batter_id = vladdy
    
    params = {'eval_metric':'mlogloss'}
    data = query_mlb_db(f'''
        select 
            case
                when description='swinging_strike' 
                        or description='swinging_strike_blocked' 
                        or description='called_strike' 
                        or description='foul_tip' 
                        or description='swinging_pitchout'
                    then 'strike'
                when description='foul'
                        or description='foul_pitchout'
                    then 'foul'
                when description='ball'
                        or description='blocked_ball'
                        or description='pitchout'
                    then 'ball'
                /* when description='hit_by_pitch' then 'hit_by_pitch' */
                when description='hit_into_play' then 'hit_into_play'
                else NULL
            end as pitch_outcome,
            p_throws,
            pitch_number, strikes, balls,
            release_speed, 
            release_spin_rate, 
            plate_x, plate_z
        from Statcast
        where batter={batter_id}
        and pitch_outcome & p_throws & pitch_number & strikes & balls &
            release_speed &
            release_spin_rate &
            plate_x & plate_z
        is not null
        order by game_date asc, at_bat_number asc, pitch_number asc;
                        ''')
    data = data.tail(8000)


    target_col = 'pitch_outcome'

    model, le, X_train, X_test, y_train, y_test = categorical_model_pipeline(
            xgb.XGBClassifier,
            params,
            data,
            target_col,
            split_data=True
        )

    best_params = param_optim(model, X_train, X_test, y_train, y_test)
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    classifier_report(model, le, X_test, y_test)


