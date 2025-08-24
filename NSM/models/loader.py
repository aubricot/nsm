"""
Model loader utilities for Neural Shape Models (NSM).

This module provides functions to load pre-trained NSM models from configuration
and state files, supporting multiple model architectures.
"""

import torch
import warnings
from typing import Dict, Any, Union, Optional

from .deep_sdf import Decoder
from .triplanar import TriplanarDecoder
from .two_stage import TwoStageDecoder
from .modulated_periodic_activations import ImplicitDecoder, SirenBlockFactory, LinearBlockFactory


def load_model(
    config: Dict[str, Any],
    path_model_state: str,
    model_type: str = "triplanar",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """
    Loads a pre-trained Neural Shape Model (NSM) from configuration and state files.

    Supports 'triplanar', 'deepsdf', 'two_stage', and 'implicit' model architectures.
    Initializes the model based on parameters in the `config` dictionary, loads the
    learned weights from `path_model_state`, moves the model to the specified device,
    and sets it to evaluation mode.

    Args:
        config (Dict[str, Any]): A dictionary containing model configuration parameters
            (e.g., latent_size, layer_dimensions, activation functions).
        path_model_state (str): Path to the .pt or .pth file containing the
            saved model state_dict.
        model_type (str, optional): The type of NSM architecture to load.
            Supported values are 'triplanar', 'deepsdf', 'two_stage', and 'implicit'.
            Defaults to 'triplanar'.
        device (str or torch.device, optional): Device to load the model on.
            If None, defaults to 'cuda' if available, otherwise 'cpu'.

    Returns:
        torch.nn.Module: The loaded and initialized NSM model, ready for evaluation.

    Raises:
        ValueError: If `model_type` is not one of the supported values.
        FileNotFoundError: If `path_model_state` does not exist.
        KeyError: If required configuration parameters are missing.
    """

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model class and parameters based on model type
    if model_type == "triplanar":
        model_class, params = _get_triplanar_params(config)
    elif model_type == "deepsdf":
        model_class, params = _get_deepsdf_params(config)
    elif model_type == "two_stage":
        model_class, params = _get_two_stage_params(config)
    elif model_type == "implicit":
        model_class, params = _get_implicit_params(config)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: "
            f"triplanar, deepsdf, two_stage, implicit"
        )

    # Initialize model
    try:
        model = model_class(**params)
    except Exception as e:
        raise ValueError(f"Failed to initialize {model_type} model with provided config: {e}")

    # Load model state
    try:
        saved_model_state = torch.load(path_model_state, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model state file not found: {path_model_state}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model state from {path_model_state}: {e}")

    # Handle different save formats
    if isinstance(saved_model_state, dict):
        if "model" in saved_model_state:
            state_dict = saved_model_state["model"]
        elif "state_dict" in saved_model_state:
            state_dict = saved_model_state["state_dict"]
        elif "model_state_dict" in saved_model_state:
            state_dict = saved_model_state["model_state_dict"]
        else:
            # Assume the dict itself is the state_dict
            state_dict = saved_model_state
    else:
        state_dict = saved_model_state

    # Load state dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into model: {e}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model


def _get_triplanar_params(config: Dict[str, Any]) -> tuple:
    """Extract TriplanarDecoder parameters from config."""
    required_keys = ["latent_size"]
    _check_required_keys(config, required_keys, "triplanar")

    params = {
        "latent_dim": config["latent_size"],
        "n_objects": config.get("objects_per_decoder", 1),
        "conv_hidden_dims": config.get("conv_hidden_dims", [512, 512, 512, 512, 512]),
        "conv_deep_image_size": config.get("conv_deep_image_size", 2),
        "conv_norm": config.get("conv_norm", True),
        "conv_norm_type": config.get("conv_norm_type", "batch"),
        "conv_start_with_mlp": config.get("conv_start_with_mlp", True),
        "sdf_latent_size": config.get("sdf_latent_size", 128),
        "sdf_hidden_dims": config.get("sdf_hidden_dims", [512, 512, 512]),
        "sdf_weight_norm": config.get("weight_norm", True),
        "sdf_final_activation": config.get("final_activation", "tanh"),
        "sdf_activation": config.get("activation", "relu"),
        "sdf_dropout_prob": config.get("dropout_prob", 0.0),
        "sum_sdf_features": config.get("sum_conv_output_features", True),
        "conv_pred_sdf": config.get("conv_pred_sdf", False),
        "padding": config.get("padding", 0.1),
    }

    return TriplanarDecoder, params


def _get_deepsdf_params(config: Dict[str, Any]) -> tuple:
    """Extract Decoder (DeepSDF) parameters from config."""
    required_keys = ["latent_size", "layer_dimensions"]
    _check_required_keys(config, required_keys, "deepsdf")

    # Handle deprecated parameters
    if "latent_dropout" in config:
        warnings.warn(
            "latent_dropout is deprecated in config. Use dropout_prob instead.", DeprecationWarning
        )

    params = {
        "latent_size": config["latent_size"],
        "dims": config["layer_dimensions"],
        "n_objects": config.get("objects_per_decoder", 1),
        "dropout": config.get("layers_with_dropout", None),
        "dropout_prob": config.get("dropout_prob", 0.2),
        "norm_layers": config.get("layers_with_norm", ()),
        "latent_in": config.get("layer_latent_in", ()),
        "weight_norm": config.get("weight_norm", True),
        "xyz_in_all": config.get("xyz_in_all", None),
        "activation": config.get("activation", "relu"),
        "final_activation": config.get("final_activation", "tanh"),
        "concat_latent_input": config.get("concat_latent_input", False),
        "progressive_add_depth": config.get("progressive_add_depth", False),
        "layer_split": config.get("layer_split", None),
        "latent_noise_sigma": config.get("latent_noise_sigma", None),
    }

    return Decoder, params


def _get_two_stage_params(config: Dict[str, Any]) -> tuple:
    """Extract TwoStageDecoder parameters from config."""
    required_keys = ["latent_size"]
    _check_required_keys(config, required_keys, "two_stage")

    # Extract triplanar and MLP specific parameters
    triplanar_params = {}
    mlp_params = {}

    # Triplanar parameters
    if "triplanar_params" in config:
        triplanar_params = config["triplanar_params"].copy()
    else:
        # Use default triplanar params with config overrides
        triplanar_params = {
            "conv_hidden_dims": config.get("conv_hidden_dims", [512, 512, 512, 512, 512]),
            "conv_deep_image_size": config.get("conv_deep_image_size", 2),
            "conv_norm": config.get("conv_norm", True),
            "conv_norm_type": config.get("conv_norm_type", "layer"),
            "conv_start_with_mlp": config.get("conv_start_with_mlp", True),
            "sdf_latent_size": config.get("sdf_latent_size", 128),
            "sdf_hidden_dims": config.get("sdf_hidden_dims", [512, 512, 512]),
            "sdf_weight_norm": config.get("weight_norm", True),
            "sdf_final_activation": config.get("final_activation", "tanh"),
            "sdf_activation": config.get("activation", "relu"),
        }

    # MLP parameters
    if "mlp_params" in config:
        mlp_params = config["mlp_params"].copy()
    else:
        # Use default MLP params with config overrides
        mlp_params = {
            "dims": list(config.get("layer_dimensions", (512, 512, 512, 512, 512, 512, 512, 512))),
            "dropout": config.get("layers_with_dropout", None),
            "dropout_prob": config.get("dropout_prob", 0.0),
            "norm_layers": config.get("layers_with_norm", ()),
            "latent_in": config.get("layer_latent_in", ()),
            "weight_norm": config.get("weight_norm", True),
            "xyz_in_all": config.get("xyz_in_all", None),
            "activation": config.get("activation", "relu"),
            "final_activation": config.get("final_activation", "tanh"),
            "concat_latent_input": config.get("concat_latent_input", True),
        }

    params = {
        "latent_size": config["latent_size"],
        "n_objects": config.get("objects_per_decoder", 2),
        "triplanar_params": triplanar_params,
        "mlp_params": mlp_params,
    }

    return TwoStageDecoder, params


def _get_implicit_params(config: Dict[str, Any]) -> tuple:
    """Extract ImplicitDecoder parameters from config."""
    required_keys = ["latent_dim", "hidden_dim", "num_layers"]
    _check_required_keys(config, required_keys, "implicit")

    # Determine block factory
    block_type = config.get("block_type", "linear")
    if block_type == "siren":
        block_factory = SirenBlockFactory(w0=config.get("w0", 30), bias=config.get("bias", True))
    elif block_type == "linear":
        block_factory = LinearBlockFactory(bias=config.get("bias", True))
    else:
        raise ValueError(f"Unknown block_type: {block_type}. Supported: 'siren', 'linear'")

    # Determine final activation
    final_activation = config.get("final_activation", "sigmoid")
    if final_activation == "sigmoid":
        final_activation_fn = torch.sigmoid
    elif final_activation == "tanh":
        final_activation_fn = torch.tanh
    elif final_activation == "linear" or final_activation is None:
        final_activation_fn = None
    else:
        raise ValueError(f"Unknown final_activation: {final_activation}")

    params = {
        "latent_dim": config["latent_dim"],
        "out_dim": config.get("out_dim", 1),
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "block_factory": block_factory,
        "modulation": config.get("modulation", False),
        "dropout": config.get("dropout", 0.0),
        "final_activation": final_activation_fn,
    }

    return ImplicitDecoder, params


def _check_required_keys(config: Dict[str, Any], required_keys: list, model_type: str):
    """Check that all required keys are present in config."""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(
            f"Missing required configuration keys for {model_type} model: {missing_keys}"
        )


def list_supported_models() -> list:
    """Return a list of supported model types."""
    return ["triplanar", "deepsdf", "two_stage", "implicit"]


def get_model_config_template(model_type: str) -> Dict[str, Any]:
    """
    Get a template configuration dictionary for a specific model type.

    Args:
        model_type (str): The model type to get template for.

    Returns:
        Dict[str, Any]: Template configuration with default values and descriptions.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == "triplanar":
        return {
            # Required
            "latent_size": 256,
            # Optional
            "objects_per_decoder": 1,
            "conv_hidden_dims": [512, 512, 512, 512, 512],
            "conv_deep_image_size": 2,
            "conv_norm": True,
            "conv_norm_type": "batch",  # 'batch' or 'layer'
            "conv_start_with_mlp": True,
            "sdf_latent_size": 128,
            "sdf_hidden_dims": [512, 512, 512],
            "weight_norm": True,
            "final_activation": "tanh",  # 'tanh', 'sigmoid', 'linear'
            "activation": "relu",  # 'relu', 'leaky_relu', 'sin', etc.
            "dropout_prob": 0.0,
            "sum_conv_output_features": True,
            "conv_pred_sdf": False,
            "padding": 0.1,
        }

    elif model_type == "deepsdf":
        return {
            # Required
            "latent_size": 256,
            "layer_dimensions": [512, 512, 512, 512, 512, 512, 512, 512],
            # Optional
            "objects_per_decoder": 1,
            "layers_with_dropout": None,  # List of layer indices or None
            "dropout_prob": 0.2,
            "layers_with_norm": (),  # Tuple of layer indices (deprecated)
            "layer_latent_in": (),  # Tuple of layer indices
            "weight_norm": True,
            "xyz_in_all": None,
            "activation": "relu",
            "final_activation": "tanh",
            "concat_latent_input": False,
            "progressive_add_depth": False,
            "layer_split": None,
            "latent_noise_sigma": None,
        }

    elif model_type == "two_stage":
        return {
            # Required
            "latent_size": 512,
            # Optional
            "objects_per_decoder": 2,
            # Can specify nested params or use top-level params
            "triplanar_params": {
                "conv_hidden_dims": [512, 512, 512, 512, 512],
                "conv_deep_image_size": 2,
                "conv_norm": True,
                "conv_norm_type": "layer",
                "conv_start_with_mlp": True,
                "sdf_latent_size": 128,
                "sdf_hidden_dims": [512, 512, 512],
                "sdf_weight_norm": True,
                "sdf_final_activation": "tanh",
                "sdf_activation": "relu",
            },
            "mlp_params": {
                "dims": [512, 512, 512, 512, 512, 512, 512, 512],
                "dropout": None,
                "dropout_prob": 0.0,
                "norm_layers": (),
                "latent_in": (),
                "weight_norm": True,
                "xyz_in_all": None,
                "activation": "relu",
                "final_activation": "tanh",
                "concat_latent_input": True,
            },
        }

    elif model_type == "implicit":
        return {
            # Required
            "latent_dim": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            # Optional
            "out_dim": 1,
            "block_type": "linear",  # 'linear' or 'siren'
            "w0": 30,  # Only for siren blocks
            "bias": True,
            "modulation": False,
            "dropout": 0.0,
            "final_activation": "sigmoid",  # 'sigmoid', 'tanh', 'linear'
        }

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: " f"{list_supported_models()}"
        )
