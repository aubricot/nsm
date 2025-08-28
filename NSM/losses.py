"""
Loss functions for neural signed distance functions (SDFs).

This module contains various loss functions that can be used during training
and reconstruction of neural SDF models.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def eikonal_loss(
    sdf_values: torch.Tensor,
    points: torch.Tensor,
    create_graph: bool = True,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute the eikonal loss for signed distance functions.
    
    The eikonal equation states that ||∇f|| = 1 for a valid SDF.
    This loss encourages the gradient magnitude to be close to 1.
    
    Args:
        sdf_values: SDF predictions (B, N_surfaces) or (B,)
        points: 3D coordinates that require gradients (B, 3)
        create_graph: Whether to create computation graph for higher-order derivatives
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Eikonal loss tensor
    """
    if not points.requires_grad:
        raise ValueError("Points must require gradients for eikonal loss computation")
    
    # Handle both single and multi-surface SDFs
    if sdf_values.dim() == 1:
        sdf_values = sdf_values.unsqueeze(-1)  # (B, 1)
    
    batch_size, n_surfaces = sdf_values.shape
    total_loss = 0.0
    
    # Compute eikonal loss for each surface
    for surf_idx in range(n_surfaces):
        surf_sdf = sdf_values[:, surf_idx]  # (B,)
        
        # Compute gradients of SDF w.r.t. points
        gradients = torch.autograd.grad(
            outputs=surf_sdf,
            inputs=points,
            grad_outputs=torch.ones_like(surf_sdf),
            create_graph=create_graph,
            retain_graph=True if surf_idx < n_surfaces - 1 else False,
            only_inputs=True
        )[0]  # (B, 3)
        
        # Compute gradient magnitude
        grad_norm = torch.norm(gradients, dim=-1)  # (B,)
        
        # Eikonal constraint: ||∇f|| = 1
        eikonal_constraint = (grad_norm - 1.0) ** 2
        
        if reduction == "mean":
            surface_loss = eikonal_constraint.mean()
        elif reduction == "sum":
            surface_loss = eikonal_constraint.sum()
        elif reduction == "none":
            surface_loss = eikonal_constraint
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
            
        total_loss += surface_loss
    
    # Average over surfaces if multiple
    if n_surfaces > 1:
        total_loss = total_loss / n_surfaces
        
    return total_loss


def compute_sdf_gradients(
    model: torch.nn.Module,
    latent: torch.Tensor,
    points: torch.Tensor,
    surface_idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SDF values and gradients for eikonal loss computation.
    
    Args:
        model: SDF model
        latent: Latent codes (B, latent_dim)
        points: 3D coordinates (B, 3)
        surface_idx: If specified, only compute for this surface
        
    Returns:
        sdf_values: SDF predictions (B, N_surfaces) or (B,)
        gradients: Spatial gradients (B, 3) or list of gradients for each surface
    """
    # Ensure points require gradients
    points = points.detach().requires_grad_(True)
    
    # Prepare model input
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)  # (1, latent_dim)
    if latent.shape[0] == 1 and points.shape[0] > 1:
        latent = latent.expand(points.shape[0], -1)  # (B, latent_dim)
    
    model_input = torch.cat([latent, points], dim=-1)  # (B, latent_dim + 3)
    
    # Forward pass
    sdf_values = model(model_input)  # (B, N_surfaces)
    
    # Compute gradients
    if surface_idx is not None:
        # Single surface
        if sdf_values.dim() > 1:
            surf_sdf = sdf_values[:, surface_idx]
        else:
            surf_sdf = sdf_values
        
        gradients = torch.autograd.grad(
            outputs=surf_sdf,
            inputs=points,
            grad_outputs=torch.ones_like(surf_sdf),
            create_graph=True,
            retain_graph=False,
            only_inputs=True
        )[0]  # (B, 3)
        
        return surf_sdf, gradients
    else:
        # All surfaces
        if sdf_values.dim() == 1:
            sdf_values = sdf_values.unsqueeze(-1)
        
        all_gradients = []
        n_surfaces = sdf_values.shape[1]
        
        for i in range(n_surfaces):
            surf_sdf = sdf_values[:, i]
            gradients = torch.autograd.grad(
                outputs=surf_sdf,
                inputs=points,
                grad_outputs=torch.ones_like(surf_sdf),
                create_graph=True,
                retain_graph=i < n_surfaces - 1,
                only_inputs=True
            )[0]  # (B, 3)
            all_gradients.append(gradients)
        
        return sdf_values, all_gradients


def combined_sdf_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    points: torch.Tensor,
    model: Optional[torch.nn.Module] = None,
    latent: Optional[torch.Tensor] = None,
    l1_weight: float = 1.0,
    eikonal_weight: float = 0.1,
    loss_type: str = "l1",
    reduction: str = "mean"
) -> Tuple[torch.Tensor, dict]:
    """
    Combined SDF reconstruction loss with optional eikonal regularization.
    
    Args:
        pred_sdf: Predicted SDF values (B, N_surfaces) or (B,)
        gt_sdf: Ground truth SDF values (B, N_surfaces) or (B,)
        points: 3D coordinates (B, 3) - needed for eikonal loss
        model: SDF model - needed for eikonal loss
        latent: Latent codes - needed for eikonal loss  
        l1_weight: Weight for L1/L2 reconstruction loss
        eikonal_weight: Weight for eikonal loss (0 to disable)
        loss_type: "l1" or "l2" for reconstruction loss
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    loss_dict = {}
    
    # Reconstruction loss
    if loss_type == "l1":
        recon_loss = F.l1_loss(pred_sdf, gt_sdf, reduction=reduction)
    elif loss_type == "l2":
        recon_loss = F.mse_loss(pred_sdf, gt_sdf, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    total_loss = l1_weight * recon_loss
    loss_dict["reconstruction"] = recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss
    
    # Eikonal loss (optional)
    if eikonal_weight > 0:
        if model is None or latent is None:
            raise ValueError("Model and latent must be provided for eikonal loss")
        
        # Ensure points require gradients
        if not points.requires_grad:
            points = points.detach().requires_grad_(True)
            
        # Recompute SDF with gradients for eikonal loss
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if latent.shape[0] == 1 and points.shape[0] > 1:
            latent = latent.expand(points.shape[0], -1)
            
        model_input = torch.cat([latent, points], dim=-1)
        sdf_for_gradients = model(model_input)
        
        eik_loss = eikonal_loss(sdf_for_gradients, points, reduction=reduction)
        total_loss += eikonal_weight * eik_loss
        loss_dict["eikonal"] = eik_loss.item() if hasattr(eik_loss, 'item') else eik_loss
    
    return total_loss, loss_dict


# Legacy function aliases for backward compatibility
def l1_loss(pred, target, reduction="mean"):
    """L1 loss function."""
    return F.l1_loss(pred, target, reduction=reduction)


def l2_loss(pred, target, reduction="mean"):
    """L2/MSE loss function."""
    return F.mse_loss(pred, target, reduction=reduction)
