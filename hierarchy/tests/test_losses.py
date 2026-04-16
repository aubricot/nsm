import pytest
import torch
import numpy as np
from hierarchy.losses import (
    TaxonomyLabelEncoder,
    HierarchyContrastiveLoss,
    TaxonomyClassificationHeads,
    compute_classification_head_loss,
)


# --- Fixtures ---

SAMPLE_PATHS = [
    "/data/cordylidae_chamaesaura_aenea_uf001-c1.vtk",
    "/data/cordylidae_chamaesaura_aenea_uf002-c2.vtk",
    "/data/cordylidae_cordylus_cordylus_uf003-t1.vtk",
    "/data/gekkonidae_gekko_gecko_uf004-t5.vtk",
    "/data/gekkonidae_gekko_gecko_uf005-l1.vtk",
    "/data/scincidae_scincus_scincus_uf006-c3.vtk",
]


@pytest.fixture
def taxonomy_encoder():
    return TaxonomyLabelEncoder(SAMPLE_PATHS)


# --- TaxonomyLabelEncoder tests ---

def test_taxonomy_encoder_parses_all(taxonomy_encoder):
    assert taxonomy_encoder.n_objects == 6
    assert len(taxonomy_encoder.parsed) == 6


def test_taxonomy_encoder_levels(taxonomy_encoder):
    # 3 families: cordylidae, gekkonidae, scincidae
    assert taxonomy_encoder.num_classes('family') == 3
    # 3 genera: chamaesaura, cordylus, gekko, scincus
    assert taxonomy_encoder.num_classes('genus') == 4
    # 4 species: aenea, cordylus, gecko, scincus
    assert taxonomy_encoder.num_classes('species') == 4
    # 3 positions: Cervical, Thoracic, Lumbar
    assert taxonomy_encoder.num_classes('position') == 3


def test_taxonomy_encoder_batch_labels(taxonomy_encoder):
    indices = torch.tensor([0, 1, 3])  # two cordylidae + one gekkonidae
    family_labels = taxonomy_encoder.get_batch_labels(indices, 'family')
    assert family_labels.shape == (3,)
    # First two should be same family, third different
    assert family_labels[0] == family_labels[1]
    assert family_labels[0] != family_labels[2]


def test_taxonomy_encoder_missing_labels():
    # A file that won't parse (no position suffix) - position should be -1
    paths = ["/data/unknown_file.vtk", "/data/gekkonidae_gekko_gecko_uf001-c1.vtk"]
    enc = TaxonomyLabelEncoder(paths)
    # Position should be -1 for unparseable file (no -c/-t/-l suffix)
    pos_labels = enc.get_batch_labels(torch.tensor([0]), 'position')
    assert pos_labels[0].item() == -1
    # Second file should have a valid position label
    pos_labels_valid = enc.get_batch_labels(torch.tensor([1]), 'position')
    assert pos_labels_valid[0].item() >= 0


def test_taxonomy_encoder_get_info(taxonomy_encoder):
    info = taxonomy_encoder.get_taxonomy_info()
    assert 'family' in info
    assert 'species' in info
    assert 'num_classes' in info['family']
    assert info['family']['num_classes'] == 3


def test_taxonomic_distance():
    a = {'family': 'F1', 'genus': 'G1', 'species': 'S1'}
    b_same = {'family': 'F1', 'genus': 'G1', 'species': 'S1'}
    b_sp = {'family': 'F1', 'genus': 'G1', 'species': 'S2'}
    b_gen = {'family': 'F1', 'genus': 'G2', 'species': 'S3'}
    b_fam = {'family': 'F2', 'genus': 'G3', 'species': 'S4'}

    assert TaxonomyLabelEncoder.taxonomic_distance(a, b_same) == 0
    assert TaxonomyLabelEncoder.taxonomic_distance(a, b_sp) == 1
    assert TaxonomyLabelEncoder.taxonomic_distance(a, b_gen) == 2
    assert TaxonomyLabelEncoder.taxonomic_distance(a, b_fam) == 3


# --- HierarchyContrastiveLoss tests ---

def test_contrastive_loss_same_species(taxonomy_encoder):
    loss_fn = HierarchyContrastiveLoss()
    # Indices 0 and 1 are same species (aenea)
    latents = torch.randn(2, 64)
    indices = torch.tensor([0, 1])
    loss = loss_fn(latents, indices, taxonomy_encoder, device='cpu')
    assert loss.shape == ()
    assert loss.item() >= 0


def test_contrastive_loss_gradient_flows(taxonomy_encoder):
    loss_fn = HierarchyContrastiveLoss()
    latents = torch.randn(4, 64, requires_grad=True)
    indices = torch.tensor([0, 1, 3, 5])
    loss = loss_fn(latents, indices, taxonomy_encoder, device='cpu')
    loss.backward()
    assert latents.grad is not None
    assert not torch.all(latents.grad == 0)


def test_contrastive_loss_single_sample(taxonomy_encoder):
    loss_fn = HierarchyContrastiveLoss()
    latents = torch.randn(1, 64)
    indices = torch.tensor([0])
    loss = loss_fn(latents, indices, taxonomy_encoder, device='cpu')
    assert loss.item() == 0.0


def test_contrastive_loss_custom_margins(taxonomy_encoder):
    margins = {0: 0.0, 1: 0.5, 2: 1.5, 3: 3.0}
    loss_fn = HierarchyContrastiveLoss(margins=margins)
    latents = torch.randn(4, 64)
    indices = torch.tensor([0, 1, 3, 5])
    loss = loss_fn(latents, indices, taxonomy_encoder, device='cpu')
    assert loss.item() >= 0


# --- TaxonomyClassificationHeads tests ---

def test_classification_heads_forward():
    heads = TaxonomyClassificationHeads(
        latent_dim=64, num_species=10, num_genera=5,
        num_families=3, num_positions=3, hidden_dim=32,
    )
    latents = torch.randn(8, 64)
    out = heads(latents)

    assert set(out.keys()) == {'species', 'genus', 'family', 'position'}
    assert out['species'].shape == (8, 10)
    assert out['genus'].shape == (8, 5)
    assert out['family'].shape == (8, 3)
    assert out['position'].shape == (8, 3)


def test_classification_heads_gradient():
    heads = TaxonomyClassificationHeads(
        latent_dim=64, num_species=4, num_genera=4,
        num_families=3, num_positions=3, hidden_dim=32,
    )
    latents = torch.randn(4, 64, requires_grad=True)
    out = heads(latents)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert latents.grad is not None


# --- compute_classification_head_loss tests ---

def test_classification_head_loss(taxonomy_encoder):
    heads = TaxonomyClassificationHeads(
        latent_dim=64,
        num_species=taxonomy_encoder.num_classes('species'),
        num_genera=taxonomy_encoder.num_classes('genus'),
        num_families=taxonomy_encoder.num_classes('family'),
        num_positions=taxonomy_encoder.num_classes('position'),
        hidden_dim=32,
    )
    latents = torch.randn(4, 64)
    indices = torch.tensor([0, 1, 3, 5])
    logits = heads(latents)

    total_loss, loss_dict = compute_classification_head_loss(
        logits, indices, taxonomy_encoder, device='cpu',
    )
    assert total_loss.item() > 0
    assert 'cls_species_loss' in loss_dict
    assert 'cls_family_loss' in loss_dict
    assert 'cls_position_loss' in loss_dict


def test_classification_head_loss_skips_missing():
    # All labels are -1 (unknown) for position
    paths = ["/data/unknown_file.vtk"] * 4
    enc = TaxonomyLabelEncoder(paths)
    heads = TaxonomyClassificationHeads(
        latent_dim=32, num_species=1, num_genera=1,
        num_families=1, num_positions=3, hidden_dim=16,
    )
    latents = torch.randn(4, 32)
    indices = torch.tensor([0, 1, 2, 3])
    logits = heads(latents)

    total_loss, loss_dict = compute_classification_head_loss(
        logits, indices, enc, device='cpu',
    )
    # Position loss should be absent (all labels are -1)
    assert 'cls_position_loss' not in loss_dict


def test_classification_head_loss_custom_weights(taxonomy_encoder):
    heads = TaxonomyClassificationHeads(
        latent_dim=64,
        num_species=taxonomy_encoder.num_classes('species'),
        num_genera=taxonomy_encoder.num_classes('genus'),
        num_families=taxonomy_encoder.num_classes('family'),
        num_positions=taxonomy_encoder.num_classes('position'),
        hidden_dim=32,
    )
    latents = torch.randn(4, 64)
    indices = torch.tensor([0, 1, 3, 5])
    logits = heads(latents)

    # Only species matters
    weights = {'species': 1.0, 'genus': 0.0, 'family': 0.0, 'position': 0.0}
    total_loss, loss_dict = compute_classification_head_loss(
        logits, indices, taxonomy_encoder, level_weights=weights, device='cpu',
    )
    assert total_loss.item() > 0
    assert 'cls_species_loss' in loss_dict
