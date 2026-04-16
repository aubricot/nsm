import time
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

def get_classifier_models() -> Dict[str, Any]:
    """Returns a dictionary mapping classifier names to instantiated MultiOutput models."""
    return {
        'KNN': MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
        'SVM': MultiOutputClassifier(SVC(probability=True, random_state=42)),
        'RandomForest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        'MLP': MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
        'LogisticRegression': MultiOutputClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)))
    }

def train_classifiers(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Trains a suite of classifiers and returns the fitted models along with training times."""
    models = get_classifier_models()
    trained_models = {}
    training_times = {}
    
    if len(y_train) < 2:
        raise ValueError("Training data must contain multiple samples.")
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_times[name] = time.time() - start_time
        trained_models[name] = model
        
    return trained_models, training_times

def predict_classifiers(models: Dict[str, Any], X_test: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Runs prediction on a suite of trained models, returning predictions, probabilities and inference times."""
    predictions = {}
    probabilities = {}
    inference_times = {}
    
    for name, model in models.items():
        start_time = time.time()
        preds = model.predict(X_test)
        inference_times[name] = time.time() - start_time
        predictions[name] = preds

        # Determine probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)
        else:
            probs = None
        probabilities[name] = probs

    return predictions, probabilities, inference_times


# ---------------------------------------------------------------------------
# Position regression models (continuous spine position prediction)
# ---------------------------------------------------------------------------

def get_position_regressors() -> Dict[str, Any]:
    """Returns a dictionary of regressor models for continuous position prediction."""
    return {
        'KNN_Regressor': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf')),
        'RF_Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'MLP_Regressor': make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        ),
    }


def train_position_regressors(
    X_train: np.ndarray, y_train: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Train position regressors on continuous position labels.

    Parameters
    ----------
    X_train : array of shape (n_samples, n_features)
    y_train : array of shape (n_samples,), continuous 0-1 values

    Returns
    -------
    trained_models : dict of name -> fitted model
    training_times : dict of name -> seconds
    """
    models = get_position_regressors()
    trained = {}
    times = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        times[name] = time.time() - start
        trained[name] = model

    return trained, times


def predict_position_regressors(
    models: Dict[str, Any], X_test: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run prediction on trained position regressors.

    Returns
    -------
    predictions : dict of name -> array of predicted positions
    inference_times : dict of name -> seconds
    """
    predictions = {}
    inference_times = {}

    for name, model in models.items():
        start = time.time()
        preds = model.predict(X_test)
        inference_times[name] = time.time() - start
        predictions[name] = np.clip(preds, 0.0, 1.0)

    return predictions, inference_times
