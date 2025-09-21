#!/usr/bin/env python3
"""
Example: Loading a trained NSM model for inference

This example demonstrates how to load a pre-trained Neural Shape Model (NSM)
from the configuration and weights saved during training.
"""

import json
import torch
import argparse
from pathlib import Path

from NSM.models import load_model, list_supported_models


def load_trained_model(experiment_dir, epoch, model_type='triplanar', device=None):
    """
    Load a trained NSM model from experiment directory.
    
    Args:
        experiment_dir (str): Path to experiment directory containing config and model files
        epoch (int): Epoch number to load (e.g., 2000)
        model_type (str): Type of model ('triplanar', 'deepsdf', 'two_stage', 'implicit')
        device (str, optional): Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        torch.nn.Module: Loaded model ready for inference
    """
    experiment_path = Path(experiment_dir)
    
    # 1. Load the configuration from the JSON file saved during training
    config_path = experiment_path / 'model_params_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Determine model weights path
    model_dir = experiment_path / 'model'
    model_path = model_dir / f'{epoch}.pth'
    
    if not model_path.exists():
        # List available model files
        available_models = list(model_dir.glob('*.pth')) if model_dir.exists() else []
        available_epochs = [int(p.stem) for p in available_models]
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Available epochs: {sorted(available_epochs)}"
        )
    
    print(f"Loading model weights from: {model_path}")
    
    # 3. Load the trained model using the config and weights
    model = load_model(
        config=config,
        path_model_state=str(model_path),
        model_type=model_type,
        device=device
    )
    
    print(f"✓ Successfully loaded {model_type} model from epoch {epoch}")
    print(f"✓ Model is on device: {next(model.parameters()).device}")
    print(f"✓ Model is in eval mode: {not model.training}")
    
    return model


def run_inference_example(model, config):
    """
    Example of running inference with the loaded model.
    
    Args:
        model: Loaded NSM model
        config: Model configuration dictionary
    """
    print("\n" + "="*50)
    print("Running inference example...")
    
    # Get model parameters for creating test input
    latent_size = config.get('latent_size', config.get('latent_dim', 256))
    batch_size = 5
    
    # Create test input: [latent_code, xyz_coordinates]
    device = next(model.parameters()).device
    test_latent = torch.randn(batch_size, latent_size, device=device)
    test_xyz = torch.randn(batch_size, 3, device=device) * 0.5  # Points around origin
    test_input = torch.cat([test_latent, test_xyz], dim=1)
    
    print(f"Input shape: {test_input.shape}")
    print(f"  - Latent size: {latent_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - XYZ coordinates: {test_xyz.shape}")
    
    # Run inference
    with torch.no_grad():
        sdf_predictions = model(test_input)
    
    print(f"Output shape: {sdf_predictions.shape}")
    print(f"SDF predictions: {sdf_predictions.cpu().numpy()}")
    print(f"SDF range: [{sdf_predictions.min():.4f}, {sdf_predictions.max():.4f}]")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Load and test a trained NSM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing model_params_config.json and model/ folder"
    )
    
    parser.add_argument(
        "epoch",
        type=int,
        help="Epoch number to load (e.g., 2000)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default='triplanar',
        choices=list_supported_models(),
        help="Type of NSM model to load"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="Device to load model on (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--no-inference",
        action='store_true',
        help="Skip the inference example"
    )
    
    args = parser.parse_args()
    
    print("NSM Model Loader Example")
    print("=" * 50)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Epoch: {args.epoch}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device or 'auto-detect'}")
    print()
    
    try:
        # Load the model
        model = load_trained_model(
            experiment_dir=args.experiment_dir,
            epoch=args.epoch,
            model_type=args.model_type,
            device=args.device
        )
        
        # Load config for inference example
        with open(Path(args.experiment_dir) / 'model_params_config.json', 'r') as f:
            config = json.load(f)
        
        # Run inference example
        if not args.no_inference:
            run_inference_example(model, config)
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
