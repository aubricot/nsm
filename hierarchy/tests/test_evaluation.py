import pytest
import numpy as np
from hierarchy.evaluation import calculate_metrics, generate_hierarchical_confusion_matrices

def test_calculate_metrics_perfect_accuracy():
    y_true = ['class_A', 'class_B', 'class_C']
    y_pred = ['class_A', 'class_B', 'class_C']
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['instance_accuracy'] == 1.0
    assert metrics['average_class_accuracy'] == 1.0
    assert metrics['precision_macro'] == 1.0
    assert metrics['recall_macro'] == 1.0
    assert metrics['f1_macro'] == 1.0

def test_calculate_metrics_mixed_accuracy():
    y_true = ['A', 'A', 'B', 'C']
    y_pred = ['A', 'B', 'B', 'B']  # Acc: 2/4 = 0.5
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['instance_accuracy'] == 0.5
    # A accuracy: 1/2, B accuracy: 1/1, C accuracy: 0/1. Avg class acc = (0.5+1+0)/3 = 0.5
    assert metrics['average_class_accuracy'] == 0.5

def test_top_k_accuracy():
    y_true = ['A', 'B', 'C']
    y_pred = ['A', 'C', 'B'] # top 1 prediction
    classes = np.array(['A', 'B', 'C'])
    # Probabilities: rows sum to 1.
    y_probs = np.array([
        [0.8, 0.1, 0.1], # A is top
        [0.2, 0.3, 0.5], # C is top, B is second
        [0.3, 0.4, 0.3]  # B is top, A/C tied
    ])
    
    # Top 2 accuracy
    # For A: top 2 are {A}, true A is among top 2 -> correct
    # For B: top 2 are {B, C}, true B is among top 2 -> correct
    # For C: top 2 are {A, B} or {B, C} depending on sort stability. 
    # Let's adjust probs to be unambiguous.
    y_probs[2] = [0.1, 0.8, 0.1] # C's true is C, pred is B. C is not in top 2 (B, A/C)
    y_probs[2] = [0.4, 0.5, 0.1] # Top 2 are B, A. C is not in top 2.
    y_probs[2] = [0.2, 0.7, 0.1] # Top 2: B(0.7), A(0.2). True C(0.1) not in top 2.
    
    # Recalculate correctly:
    # y_probs:
    # 0: [0.8, 0.1, 0.1] -> Top 2: A, C/B. True A is in top 2. (Correct)
    # 1: [0.2, 0.3, 0.5] -> Top 2: C, B. True B is in top 2. (Correct)
    # 2: [0.2, 0.7, 0.1] -> Top 2: B, A. True C is not in top 2. (Incorrect)
    # Expected Top-2: 2/3
    
    metrics = calculate_metrics(y_true, y_pred, y_probs=y_probs, classes=classes, top_k=2)
    assert 'top_2_accuracy' in metrics
    assert metrics['top_2_accuracy'] == 2/3

def test_hierarchical_confusion_matrices():
    y_true = [
        {'family': 'F1', 'genus': 'G1', 'species': 'S1'},
        {'family': 'F1', 'genus': 'G2', 'species': 'S2'}
    ]
    y_pred = [
        {'family': 'F1', 'genus': 'G1', 'species': 'S1'},
        {'family': 'F2', 'genus': 'G2', 'species': 'S3'}
    ]
    
    cm = generate_hierarchical_confusion_matrices(y_true, y_pred)
    
    assert 'family' in cm
    assert 'genus' in cm
    assert 'species' in cm
    
    # Test family CM shape and labels
    # True F: F1, F1. Pred F: F1, F2. Labels: F1, F2.
    assert cm['family']['labels'] == ['F1', 'F2']
    # Matrix:
    #      F1  F2
    # F1   1   1
    # F2   0   0
    assert np.array_equal(cm['family']['matrix'], np.array([[1, 1], [0, 0]]))
