import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

def parse_taxonomy_from_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parses a filename to extract taxonomic hierarchy and spinal position.
    
    Returns a dictionary with keys:
    - family
    - genus
    - species
    - specimen_id
    - position (Cervical, Thoracic, Lumbar)
    - vertebra_number
    """
    # Remove extension if present
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    family, genus, species, specimen_id = "unknown", "unknown", "unknown", "unknown"
    position, vertebra_number = None, None
    
    # Extract vertebra type from end (e.g., -c1, -t12, -l3)
    match_end = re.search(r'-([ctlCTL])(\d+)$', base)
    file_ordinal = None
    if match_end:
        pos_char = match_end.group(1).lower()
        vertebra_number = int(match_end.group(2))
        position = {'c': 'Cervical', 't': 'Thoracic', 'l': 'Lumbar'}.get(pos_char)
        base = base[:match_end.start()] # remove the matched end

        # Now parse the rest, which might have file_num appended (e.g., _26 or -26)
        match_filenum = re.search(r'[-_](\d+)$', base)
        if match_filenum:
            file_ordinal = int(match_filenum.group(1))
            base = base[:match_filenum.start()]
            
        # Split by underscore or hyphen to get taxonomic parts
        # Most files look like: {family}_{genus}_{species}_{specimen_id}
        parts = re.split(r'[-_]', base)
        if len(parts) >= 4:
            family = parts[0]
            genus = parts[1]
            species = parts[2]
            specimen_id = "_".join(parts[3:])
        elif len(parts) == 3:
            family = parts[0]
            genus = parts[1]
            species = parts[2]
        elif len(parts) == 2:
            family = parts[0]
            specimen_id = parts[1]
        elif len(parts) == 1:
            family = parts[0]
            
        return {
            'family': family.lower() if family else "unknown",
            'genus': genus.lower() if genus else "unknown",
            'species': species.lower() if species else "unknown",
            'specimen_id': specimen_id.lower() if specimen_id else "unknown",
            'position': position,
            'vertebra_number': vertebra_number,
            'file_ordinal': file_ordinal,
        }
        
    return None

class TaxonomyTree:
    """Helper class for hierarchy-aware computations."""
    
    @staticmethod
    def get_taxonomic_distance(pred_dict: Dict[str, str], true_dict: Dict[str, str]) -> int:
        """
        Returns a distance based on how wrong the prediction is on the phylogenetic tree:
        0 = Full match (Species level)
        1 = Genus match, Species mismatch
        2 = Family match, Genus mismatch
        3 = Family mismatch (Total failure)
        """
        if pred_dict.get('family', 'pred_unk') != true_dict.get('family', 'true_unk'):
            return 3
        if pred_dict.get('genus', 'pred_unk') != true_dict.get('genus', 'true_unk'):
            return 2
        if pred_dict.get('species', 'pred_unk') != true_dict.get('species', 'true_unk'):
            return 1
        return 0


class HierarchicalMultiTaskClassifier(BaseEstimator, ClassifierMixin):
    """
    Simultaneously predicts multiple targets (e.g., Species and Spinal Position)
    and incorporates a hierarchy-aware penalty during evaluation or ensemble weighting.
    """
    def __init__(self, base_estimator=None):
        if base_estimator is None:
            self.base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.base_estimator = base_estimator

        self.model = MultiOutputClassifier(self.base_estimator)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the multi-output model.
        y should be shape (n_samples, n_targets), e.g. [species, position]
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        return self.model.predict_proba(X)


def compute_hierarchical_loss(y_true_species: List[str], y_pred_species: List[str],
                             taxonomy_dicts: List[Dict[str, str]]) -> float:
    """
    Computes a penalty score based on taxonomic distance.
    Distance scaling: Correct=0, Genus match=1, Family match=2, Total mismatch=3
    """
    total_loss = 0.0
    for i, (pred, true) in enumerate(zip(y_pred_species, y_true_species)):
        true_dict = next((d for d in taxonomy_dicts if d.get('species') == true), {'species': true})
        pred_dict = next((d for d in taxonomy_dicts if d.get('species') == pred), {'species': pred})

        loss = TaxonomyTree.get_taxonomic_distance(pred_dict, true_dict)
        total_loss += loss

    return total_loss / len(y_true_species) if len(y_true_species) > 0 else 0.0
