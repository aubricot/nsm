import pytest
import numpy as np
from hierarchy.metric_learning import LatentMetricLearner

def test_lmnn_basic_functionality():
    np.random.seed(42)
    X = np.random.rand(20, 10)
    y = np.array(['A'] * 10 + ['B'] * 10)
    
    learner = LatentMetricLearner(method='LMNN', k=2, max_iter=50)
    
    # Should fit successfully
    X_transformed = learner.fit_transform(X, y)
    
    # Output shape should match input shape
    assert X_transformed.shape == (20, 10)

def test_nca_basic_functionality():
    np.random.seed(42)
    X = np.random.rand(20, 10)
    y = np.array(['A'] * 10 + ['B'] * 10)
    
    learner = LatentMetricLearner(method='NCA', max_iter=50)
    
    X_transformed = learner.fit_transform(X, y)
    assert X_transformed.shape == (20, 10)

def test_unsupported_method():
    with pytest.raises(ValueError, match="Unsupported metric learning method"):
        LatentMetricLearner(method='UNSUPPORTED')

def test_transform_without_fit():
    np.random.seed(42)
    X = np.random.rand(5, 10)
    
    learner = LatentMetricLearner(method='LMNN')
    
    with pytest.raises(RuntimeError, match="Transformer must be fitted"):
        learner.transform(X)

def test_single_class_error():
    np.random.seed(42)
    X = np.random.rand(10, 10)
    y = np.array(['A'] * 10)
    
    learner = LatentMetricLearner(method='LMNN')
    
    with pytest.raises(ValueError, match="multiple classes"):
        learner.fit(X, y)
