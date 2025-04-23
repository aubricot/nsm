import os
os.environ['LOC_SDF_CACHE'] = ''
import pytest
import torch
from NSM.reconstruct import reconstruct_latent

# Mock decoder class
class MockDecoder(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]  # Return the first column as a mock prediction

@pytest.fixture
def setup_data(n_pts=100):
    # Create mock decoders
    decoders = [MockDecoder()]

    # Create sample input data
    xyz = torch.rand(n_pts, 3)  # 100 points in 3D space
    sdf_gt = torch.rand(n_pts, 1)  # Corresponding SDF values
    # needs to tell what surface each point is associated with: 
    pts_surface = [0] * n_pts

    return decoders, xyz, sdf_gt, pts_surface

def test_reconstruct_latent_basic(setup_data):
    decoders, xyz, sdf_gt, pts_surface = setup_data

    print(type(xyz))
    print(type(sdf_gt))

    # Call the function with basic parameters
    loss, latent = reconstruct_latent(
        decoders=decoders,
        num_iterations=10,
        latent_size=8,
        xyz=xyz,
        sdf_gt=sdf_gt,
        pts_surface=pts_surface,
        device='cpu'
    )

    # Check the output types
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(latent, torch.Tensor), "Latent should be a tensor"

def test_reconstruct_latent_convergence(setup_data):
    decoders, xyz, sdf_gt, pts_surface = setup_data

    # Call the function with convergence criteria
    loss, latent = reconstruct_latent(
        decoders=decoders,
        num_iterations=100,
        latent_size=8,
        xyz=xyz,
        sdf_gt=sdf_gt,
        pts_surface=pts_surface,
        convergence='overall_loss',
        convergence_patience=5,
        device='cpu'
    )

    # Check if the function converged
    assert loss < 100, "Loss should be less than initial value indicating convergence"

def test_reconstruct_latent_invalid_input(n_pts=100):
    decoders = [MockDecoder()]
    xyz = torch.rand(n_pts, 3)
    sdf_gt = "invalid_input"  # Invalid SDF input
    pts_surface = [0] * n_pts

    with pytest.raises(Exception):
        reconstruct_latent(
            decoders=decoders,
            num_iterations=10,
            latent_size=8,
            xyz=xyz,
            sdf_gt=sdf_gt,
            pts_surface=pts_surface,
            device='cpu'
        )

# Additional tests can be added for different configurations and edge cases