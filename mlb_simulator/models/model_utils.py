"""
This module includes functions to help with fitting models, saving models/
model hyperparams, and evaluating models
"""

from mlb_simulator.data.data_utils import get_models_location

import os
import json
import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, log_loss
from scipy.stats import chisquare
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)


def get_hyperparams(model_name, player_id, model, X, y):
    # check if up to date hyperparams have been fit for this batter
    hyperparam_path = check_for_hyperparams(model_name, player_id)

    # if valid hyperparams, load them
    if hyperparam_path:
        with open(hyperparam_path) as f:
            hyperparams = json.load(f)

    # if no valid hyperparams, fit new ones and save them
    else:
        logger.info(
            (
                f"No hyperparams found for {player_id}, {model_name}: "
                "finding and saving optimal hyperparams..."
            )
        )
        hyperparams = xgb_hyperparam_optimizer(model, X, y)
        save_hyperparams(model_name, player_id, hyperparams)

    return hyperparams


def save_hyperparams(model_name, player_id, hyperparams: dict):
    """
    Save hyperparameters to a JSON file.

    :param model_name: The name of the model being saved. Crucial to ensure
        model gets saved to correct path (str).
    :param player_id: The mlbid of the player whose model is being fit (int).
    :param hyperparams: The hyperparameters to save (dict).

    """

    save_path = os.path.join(
        get_models_location(),
        model_name,
        f"{player_id}-{datetime.now().strftime('%Y%m%d')}.json",
    )
    with open(save_path, "w") as f:
        json.dump(hyperparams, f)


def check_for_hyperparams(model_name, player_id):
    """
    Check to see if up-to-date hyperparameters exist for desired model/player.

    Model hyperparameters are saved to /models/model_name as
    player_id-%Y%m%d.json and are set to "expire" every 90 days. Models are
    saved as individual JSON files to avoid loading a huge JSON into memory
    each time this is needed.

    :param model_name: The name of the model, e.g., "pitch_outcome" (str).
    :param player_id: The mlb_id of the player whose model is being fit (int).

    """

    model_folder = os.path.join(get_models_location(), model_name)
    os.makedirs(model_folder, exist_ok=True)

    saved_params = os.listdir(model_folder)

    for file in saved_params:

        try:
            parts = file.split("-")
            mlb_id, rest = parts

            # if mlb id found, check if model hyperparams tuned within 90 days
            if mlb_id == str(player_id):
                date_part = rest.replace(".json", "")
                date_created = datetime.strptime(date_part, "%Y%m%d")
                if datetime.now() - date_created > timedelta(days=90):
                    return False
                else:
                    return os.path.join(model_folder, file)
        except Exception as e:
            print(
                (f"Found invalid file format in {model_folder}: {e} \n\n"),
                (
                    f"Files in { model_folder
                    } should be of format: mlbid-%Y%m%d.json \n\n"
                ),
            )
            raise

    return False


def categorical_model_pipeline(model, data, target_col):
    """
    Creates and returns a machine learning pipeline for categorical data using
    XGBoost or other scikit-learn compatible models.

    This function prepares the data by encoding the target column and applying
    necessary transformations to the features. It constructs a pipeline that
    includes preprocessing steps for numeric and categorical features and a
    specified classification model.

    :param model: The model class to be used for classification. It
        should be compatible with scikit-learn pipelines (have a .fit,
        .pred_proba method, etc) (class).
    :param data: The input dataframe containing features and the
        target column (pd.DataFrame).
    :param target_col: The name of the target column in the dataframe (str).

    :return: A tuple containing the following elements:
        - model_pipeline: A scikit-learn pipeline with preprocessing
            steps and the classifier (Pipeline).
        - label_encoder: The label encoder fitted on the target
            column (LabelEncoder).
        - features: The feature columns after dropping the target
            column (pd.DataFrame).
        - target: The encoded target column (np.ndarray).

    """

    # encode target col & split into X, y
    le = LabelEncoder()
    features = data.drop(columns=[target_col])
    target = le.fit_transform(data[target_col])

    # identify numerical/categorical columns
    numeric_features = features.select_dtypes(include=["float64", "int64"]).columns
    categorical_features = features.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model())])

    return deepcopy(model), deepcopy(le), features, target


def sample_predictions(classifier, X):
    """
    Generates a random sample of predictions given a classifier with a
    predict_proba method and input features X.

    This function uses the predicted probabilities from the classifier to
    generate a sample of class predictions based on the probability
    distribution.

    :param classifier: A trained classifier that has a `predict_proba`
        method (object).
    :param X: The input features for which to generate predictions
        (pd.DataFrame or np.ndarray).

    :return: An array of sampled predictions based on the predicted
        probabilities (np.ndarray).

    """

    pred_proba = classifier.predict_proba(X)
    sampled_predictions = np.array(
        [np.random.choice(classifier.classes_, p=proba) for proba in pred_proba]
    )

    return sampled_predictions


def categorical_chisquare(model, label_encoder, X_test, y_test):
    """Run a hypothesis test that generated data is similar in dist to actual"""
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
            hyp_tests.append("different")
        else:
            hyp_tests.append("similar")

    sampled_preds = sample_predictions(model, X_test)
    pred_counts = pd.DataFrame(
        label_encoder.inverse_transform(sampled_preds)
    ).value_counts()
    actual_counts = pd.DataFrame(label_encoder.inverse_transform(y_test)).value_counts()

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
    # print(pd.DataFrame(hyp_tests, columns=['test_results']).value_counts())


def classifier_report(model, label_encoder, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print(
        classification_report(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(y_pred),
        )
    )
    print("Log Loss:", log_loss(y_test, y_prob))

    categorical_chisquare(model, label_encoder, X_test, y_test)


def balance_classes(X_train, X_test, y_train, y_test):
    """Hit by pitch is a rare class. So, for batters with lower sample size,
    occasionally there does not exist an occurence of this class within either
    the test or training set. This method ensures there exists an instance of
    each class in the training and test set. Of course not ideal, and this
    is an area which can be drastically improved upon.
    """

    col_names = X_train.columns

    unique_train_classes = np.unique(y_train)
    unique_test_classes = np.unique(y_test)

    missing_in_train = np.setdiff1d(unique_test_classes, unique_train_classes)
    missing_in_test = np.setdiff1d(unique_train_classes, unique_test_classes)

    # Add missing classes to the train set
    for missing_class in missing_in_train:
        X_train = np.vstack(
            [X_train, X_test.iloc[[np.where(y_test == missing_class)[0][0]]]]
        )
        X_train = pd.DataFrame(X_train, columns=col_names)
        y_train = np.append(y_train, missing_class)

    # Add missing classes to the test set
    for missing_class in missing_in_test:
        X_test = np.vstack(
            [X_test, X_train.iloc[[np.where(y_train == missing_class)[0][0]]]]
        )
        X_test = pd.DataFrame(X_test, columns=col_names)
        y_test = np.append(y_test, missing_class)

    return X_train, X_test, y_train, y_test


def xgb_hyperparam_optimizer(model, X, y):
    """
    Optimizes hyperparameters for an XGBoost model using a grouped parameter
    grid and log loss as the evaluation metric.

    This function performs hyperparameter optimization for the given XGBoost
    model by splitting the data into training and testing sets, then
    iteratively searching through predefined groups of hyperparameters to find
    the combination that yields the best log loss on the test set.

    :param model: A scikit-learn pipeline object that includes an XGBoost
        classifier (sklearn.pipeline.Pipeline).
    :param X: The input features (pd.DataFrame or np.ndarray).
    :param y: The target variable (pd.Series or np.ndarray).

    :return: A dictionary containing the best hyperparameters found during the
        optimization process (dict).

    """

    # get balanced train, test set
    split_data = train_test_split(X, y, test_size=0.20, shuffle=False)
    X_train, X_test, y_train, y_test = balance_classes(*split_data)

    grouped_param_grid = [
        {
            "classifier__max_depth": [2, 3, 5],
            "classifier__min_child_weight": [1, 3, 5],
        },
        {
            "classifier__subsample": [0.5, 0.7, 0.9],
            "classifier__colsample_bytree": [0.5, 0.7, 0.9],
        },
        {
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__n_estimators": [50, 100, 200],
        },
    ]

    # use mlogloss always
    all_best_params = {"classifier__eval_metric": "mlogloss"}
    for param_group in grouped_param_grid:
        best_loss = float("inf")
        best_params = {}

        for params in ParameterGrid(param_group):
            # fit model with cur param group params and previous
            # param group params
            model.set_params(**all_best_params, **params)
            model.fit(X_train, y_train)

            # find model with best log loss on test set
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
    pitch_cat_prob = df[target_col].value_counts() / len(df)
    loglik = 0
    for target in y_test[target_col]:
        loglik += np.log(pitch_cat_prob.loc[target])
    return loglik


if __name__ == "__main__":
    vladdy = 665489
    soto = 665742
    schneider = 676914
    biggio = 624415
    showtime = 660271
    crowser = 681297

    batter_id = soto

    check_for_hyperparams("pitch_outcome", batter_id)
