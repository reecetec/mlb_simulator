import numpy as np
import pandas as pd
from src.data.data_utils import query_mlb_db
import xgboost as xgb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
from scipy.stats import chisquare

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
                features, target, test_size=0.1, shuffle=False)
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
        return deepcopy(model), deepcopy(le), X_test, y_test

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
    sampled_preds = sample_predictions(model, X_test)
    pred_counts = pd.DataFrame(
            label_encoder.inverse_transform(sampled_preds)
        ).value_counts()
    actual_counts = pd.DataFrame(
            label_encoder.inverse_transform(y_test)
        ).value_counts()
    
    #print('\nAcutal, Pred:')
    #print(actual_counts, pred_counts)

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

def classifier_report(model, label_encoder, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred)))
    print('Log Loss:', log_loss(y_test, y_prob))

    for _ in range(10):
        categorical_chisquare(model, label_encoder, X_test, y_test)

if __name__ == '__main__':
    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = vladdy

    
    params = {'n_estimators':25, 'eval_metric':'mlogloss'}
    data = query_mlb_db(f'''
        select 
            case
                when description='swinging_strike' or description='swinging_strike_blocked' or description='called_strike' or description='foul_tip' 
                    or description='swinging_pitchout' then 'strike'
                when description='foul' or description='foul_pitchout' then 'foul'
                when description='ball' or description='blocked_ball' or description='pitchout' then 'ball'
                when description='hit_by_pitch' then 'hit_by_pitch'
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
    target_col = 'pitch_outcome'

    model, le, X_test, y_test = categorical_model_pipeline(xgb.XGBClassifier,
                                                           params,
                                                           data, target_col,
                                                           split_data=True
                                     )

    classifier_report(model, le, X_test, y_test)


