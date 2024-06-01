Module mlb_simulator.models.model_utils
=======================================
This module includes functions to help with fitting models, saving models/
model hyperparams, and evaluating models

Functions
---------

    
`categorical_chisquare(model, label_encoder, X_test, y_test)`
:   Run a hypothesis test that generated data is similar in dist to actual

    
`categorical_model_pipeline(model, data, target_col)`
:   Creates and returns a machine learning pipeline for categorical data using
    XGBoost or other scikit-learn compatible models
    
    This function prepares the data by encoding the target column and applying
    necessary transformations to the features. It constructs a pipeline that
    includes preprocessing steps for numeric and categorical features and a
    specified classification model
    
    Parameters:
        model (class): The model class to be used for classification. It
        should be compatible with scikit-learn pipelines (have a .fit,
        .pred_proba method etc)
        data (pd.DataFrame): The input dataframe containing features and the
        target column
        target_col (str): The name of the target column in the dataframe
    
    Returns:
        model_pipeline (Pipeline): A scikit-learn pipeline with preprocessing
        steps and the classifier
        label_encoder (LabelEncoder): The label encoder fitted on the target
        column
        features (pd.DataFrame): The feature columns after dropping the target
        column
        target (np.ndarray): The encoded target column

    
`check_for_hyperparams(model_name, player_id)`
:   Check to see if up to date hyperparams exist for desired model/player
    
    model hyperparams are saved to /models/model_name as player_id-%Y%m%d.json
    and are set to "expire" every 90 days. Models saved as individual json
    files to avoid loading a huge json into memory each time this is needed.
    
    Parameters:
        model_name (str): the name of the model, e.g. pitch_outcome
        player_id (int): the mlb_id of the player whose model is being fit

    
`classifier_report(model, label_encoder, X_test, y_test)`
:   

    
`compute_cat_loglik(X_train, y_train, y_test, target_col)`
:   

    
`compute_model_loglik(y_pred_proba, y_test, target_col)`
:   

    
`sample_predictions(classifier, X)`
:   Generates a random sample of predictions given a classifier with a
    predict_proba method and input features X
    
    This function uses the predicted probabilities from the classifier to
    generate a sample of class predictions based on the probability
    distribution
    
    Parameters:
        classifier (object): A trained classifier that has a `predict_proba`
        method
        X (pd.DataFrame or np.ndarray): The input features for which to
        generate predictions
    
    Returns:
        np.ndarray: An array of sampled predictions based on the predicted
        probabilities

    
`save_hyperparams(model_name, player_id, hyperparams: dict)`
:   Save hyperparams to a json file
    
    Parameters:
        model_name (str): the name of the model being saved. Crutial to ensure 
        model gets saved to correct path
        player_id (int): the mlbid of the player whose model is being fit
        hyperparams (dict): the hyperparams to save

    
`xgb_hyperparam_optimizer(model, X, y)`
:   Optimizes hyperparameters for an XGBoost model using a grouped parameter
    grid and log loss as the evaluation metric
    
    This function performs hyperparameter optimization for the given XGBoost
    model by splitting the data into training and testing sets, then
    iteratively searching through predefined groups of hyperparameters to find
    the combination that yields the best log loss on the test set
    
    Parameters:
        model (sklearn.pipeline.Pipeline): A scikit-learn pipeline object that
        includes an XGBoost classifier
        X (pd.DataFrame or np.ndarray): The input features
        y (pd.Series or np.ndarray): The target variable
    
    Returns:
        dict: A dictionary containing the best hyperparameters found during the
        optimization process.