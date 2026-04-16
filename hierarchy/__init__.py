"""
Hierarchy-aware loss, classification, and evaluation for Neural Shape Models.

This package extends NSM with taxonomy-aware contrastive learning,
multi-level classification heads, and hierarchical evaluation metrics
for squamate vertebrae classification.
"""

from .taxonomy import (
    parse_taxonomy_from_filename,
    TaxonomyTree,
    HierarchicalMultiTaskClassifier,
    compute_hierarchical_loss,
)
from .losses import (
    TaxonomyLabelEncoder,
    HierarchyContrastiveLoss,
    TaxonomyClassificationHeads,
    compute_classification_head_loss,
)
from .classifiers import (
    get_classifier_models,
    train_classifiers,
    predict_classifiers,
    get_position_regressors,
    train_position_regressors,
    predict_position_regressors,
)
from .metric_learning import LatentMetricLearner
from .evaluation import (
    calculate_metrics,
    metrics_to_dataframe,
    generate_hierarchical_confusion_matrices,
    generate_position_confusion_matrix,
    calculate_regression_metrics,
)
from .spine_position import SpinePositionMapper
from .run_utils import create_run_directory, write_run_manifest, update_manifest_files_table
