# hierarchy_loss.py
# =================
# Hierarchy-aware loss components for NSM training.
# Provides:
#   - TaxonomyLabelEncoder: extracts and encodes taxonomy from mesh filenames
#   - HierarchyContrastiveLoss: contrastive loss weighted by taxonomic distance
#   - TaxonomyClassificationHeads: lightweight MLP heads for species/genus/family/position
#   - compute_classification_head_loss: cross-entropy loss across taxonomy levels

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .taxonomy import parse_taxonomy_from_filename


class TaxonomyLabelEncoder:
    """
    Extracts and encodes taxonomy labels from mesh filenames.
    Built once at training startup. Maps dataset indices to integer labels
    for each taxonomy level (family, genus, species, position).
    """

    def __init__(self, list_mesh_paths: List[str]):
        self.list_mesh_paths = list_mesh_paths
        self.n_objects = len(list_mesh_paths)

        # Parse taxonomy from each filename
        self.parsed = []
        for path in list_mesh_paths:
            fname = os.path.basename(path)
            parsed = parse_taxonomy_from_filename(fname)
            if parsed is None:
                parsed = {
                    'family': 'unknown', 'genus': 'unknown',
                    'species': 'unknown', 'position': None,
                }
            self.parsed.append(parsed)

        # Build label encodings for each level
        self.levels = ['family', 'genus', 'species', 'position']
        self.label_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_label: Dict[str, Dict[int, str]] = {}
        self.labels: Dict[str, np.ndarray] = {}

        for level in self.levels:
            unique_labels = sorted(set(
                p.get(level, 'unknown') for p in self.parsed
                if p.get(level) is not None
            ))
            l2i = {label: i for i, label in enumerate(unique_labels)}
            i2l = {i: label for label, i in l2i.items()}

            labels = []
            for p in self.parsed:
                val = p.get(level)
                if val is not None and val in l2i:
                    labels.append(l2i[val])
                else:
                    labels.append(-1)  # sentinel for missing

            self.label_to_idx[level] = l2i
            self.idx_to_label[level] = i2l
            self.labels[level] = np.array(labels, dtype=np.int64)

    def get_batch_labels(
        self, indices: torch.Tensor, level: str, device='cpu'
    ) -> torch.Tensor:
        """
        Given batch object indices, return taxonomy labels for that level.
        Returns tensor of shape (batch_size,) with -1 for missing labels.
        Device can be a string ('cpu', 'cuda:0', 'mps') or a torch.device.
        """
        idx_np = indices.cpu().numpy()
        labels = self.labels[level][idx_np]
        return torch.tensor(labels, dtype=torch.long, device=device)

    def num_classes(self, level: str) -> int:
        return len(self.label_to_idx[level])

    def get_taxonomy_info(self) -> Dict:
        """Returns serializable taxonomy info for saving alongside checkpoints."""
        return {
            level: {
                'label_to_idx': self.label_to_idx[level],
                'idx_to_label': {str(k): v for k, v in self.idx_to_label[level].items()},
                'num_classes': self.num_classes(level),
            }
            for level in self.levels
        }

    @staticmethod
    def taxonomic_distance(a: dict, b: dict) -> int:
        """Taxonomic distance between two parsed taxonomy dicts (0-3 scale)."""
        if a.get('family', 'a_unk') != b.get('family', 'b_unk'):
            return 3
        if a.get('genus', 'a_unk') != b.get('genus', 'b_unk'):
            return 2
        if a.get('species', 'a_unk') != b.get('species', 'b_unk'):
            return 1
        return 0


class HierarchyContrastiveLoss(nn.Module):
    """
    Contrastive loss weighted by taxonomic distance on latent codes.

    For each pair (i, j) in the batch:
    - Same species (dist=0): attract (minimize distance)
    - Same genus, diff species (dist=1): repel with small margin
    - Same family, diff genus (dist=2): repel with medium margin
    - Different family (dist=3): repel with large margin

    Loss per pair:
      attract: d^2
      repel:   max(0, margin - d)^2
    """

    def __init__(self, margins: Optional[Dict] = None):
        super().__init__()
        if margins is None:
            margins = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}
        # Normalize keys to int
        self.margins = {int(k): float(v) for k, v in margins.items()}

    def forward(
        self,
        latent_codes: torch.Tensor,
        object_indices: torch.Tensor,
        taxonomy_encoder: TaxonomyLabelEncoder,
        device: str = 'cpu',
    ) -> torch.Tensor:
        batch_size = latent_codes.shape[0]
        if batch_size < 2:
            return (latent_codes.sum() * 0.0)  # zero with grad, on correct device

        # Normalize for stable distances
        z_norm = F.normalize(latent_codes.float(), p=2, dim=1)

        # Pairwise Euclidean distances — manual computation avoids torch.cdist
        # whose backward is not implemented on MPS
        diff = z_norm.unsqueeze(0) - z_norm.unsqueeze(1)  # (B, B, D)
        dists = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (B, B)

        # Build pairwise taxonomic distance matrix on CPU then move
        idx_np = object_indices.cpu().numpy()
        tax_dists_np = np.zeros((batch_size, batch_size), dtype=np.int64)
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                td = taxonomy_encoder.taxonomic_distance(
                    taxonomy_encoder.parsed[idx_np[i]],
                    taxonomy_encoder.parsed[idx_np[j]],
                )
                tax_dists_np[i, j] = td
                tax_dists_np[j, i] = td
        tax_dists = torch.tensor(tax_dists_np, dtype=torch.long, device=latent_codes.device)

        # Upper triangle mask to avoid double counting
        upper = torch.triu(
            torch.ones(batch_size, batch_size, dtype=torch.bool, device=latent_codes.device),
            diagonal=1,
        )

        loss = latent_codes.sum() * 0.0  # zero with grad, on correct device
        n_pairs = 0

        for tax_dist in [0, 1, 2, 3]:
            mask = (tax_dists == tax_dist) & upper
            if mask.sum() == 0:
                continue

            pair_dists = dists[mask]
            margin = self.margins[tax_dist]

            if tax_dist == 0:
                # Same species: attract
                pair_loss = pair_dists ** 2
            else:
                # Different: repel with taxonomy-scaled margin
                pair_loss = F.relu(margin - pair_dists) ** 2

            loss = loss + pair_loss.sum()
            n_pairs += mask.sum().item()

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss


class TaxonomyClassificationHeads(nn.Module):
    """
    Lightweight MLP heads for predicting taxonomy from latent codes.
    One head per level: species, genus, family, position.
    """

    def __init__(
        self,
        latent_dim: int,
        num_species: int,
        num_genera: int,
        num_families: int,
        num_positions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.heads = nn.ModuleDict({
            'species': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_species),
            ),
            'genus': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_genera),
            ),
            'family': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_families),
            ),
            'position': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_positions),
            ),
        })

    def forward(self, latent_codes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent_codes: (batch_size, latent_dim)
        Returns:
            Dict of level -> logits (batch_size, num_classes)
        """
        return {level: head(latent_codes) for level, head in self.heads.items()}


def compute_classification_head_loss(
    logits: Dict[str, torch.Tensor],
    object_indices: torch.Tensor,
    taxonomy_encoder: TaxonomyLabelEncoder,
    level_weights: Optional[Dict[str, float]] = None,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Cross-entropy loss summed across taxonomy levels.
    Samples with label=-1 (missing) are skipped.
    """
    if level_weights is None:
        level_weights = {
            'species': 1.0, 'genus': 0.5, 'family': 0.25, 'position': 0.75,
        }

    # Derive device from logits tensors to avoid str vs torch.device issues
    any_logit = next(iter(logits.values()))
    actual_device = any_logit.device

    # Start with a zero that carries grad from the logits (never a detached leaf)
    total_loss = any_logit.sum() * 0.0
    loss_dict = {}

    for level, weight in level_weights.items():
        if level not in logits or weight == 0.0:
            continue

        labels = taxonomy_encoder.get_batch_labels(
            object_indices, level, device=actual_device,
        )

        # Mask out missing labels
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            continue

        valid_logits = logits[level][valid_mask].float()
        valid_labels = labels[valid_mask]

        ce_loss = F.cross_entropy(valid_logits, valid_labels)
        total_loss = total_loss + weight * ce_loss
        loss_dict[f'cls_{level}_loss'] = ce_loss.item()

    return total_loss, loss_dict
