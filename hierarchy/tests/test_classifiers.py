import numpy as np
from hierarchy.classifiers import get_classifier_models, train_classifiers, predict_classifiers

def test_classifiers_training_and_prediction():
    # Mock data equivalent to latents — v2 uses MultiOutputClassifier so y must be 2D
    np.random.seed(42)
    X_train = np.random.rand(100, 512)
    y_species = np.array(['species_A'] * 50 + ['species_B'] * 50)
    y_position = np.array(['Cervical'] * 50 + ['Thoracic'] * 50)
    y_train = np.column_stack((y_species, y_position))

    models, train_times = train_classifiers(X_train, y_train)

    # Check that we trained 5 models
    assert len(models) == 5
    assert len(train_times) == 5

    # Check predictions
    X_test = np.random.rand(10, 512)
    preds, probs, inf_times = predict_classifiers(models, X_test)

    assert 'KNN' in preds
    assert 'SVM' in preds
    assert 'MLP' in preds
    assert 'RandomForest' in preds
    assert 'LogisticRegression' in preds

    # Check output shapes (2 targets)
    for name in models.keys():
        assert preds[name].shape == (10, 2)

    # Check inference time recording
    for name in models.keys():
        assert name in inf_times
        assert isinstance(inf_times[name], float)
