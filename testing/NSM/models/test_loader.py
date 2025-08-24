"""
Test module for NSM model loader functionality.

Tests the load_model function and related utilities for loading
pre-trained Neural Shape Models from configuration and state files.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from NSM.models import (
    load_model,
    list_supported_models,
    get_model_config_template,
    TriplanarDecoder,
    Decoder,
    TwoStageDecoder,
    ImplicitDecoder,
)


class TestModelLoader:
    """Test class for model loader functionality."""

    def test_list_supported_models(self):
        """Test that supported models list is returned correctly."""
        models = list_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        expected_models = ["triplanar", "deepsdf", "two_stage", "implicit"]
        for model in expected_models:
            assert model in models

    def test_get_model_config_template_all_types(self):
        """Test getting config templates for all supported model types."""
        models = list_supported_models()

        for model_type in models:
            config = get_model_config_template(model_type)
            assert isinstance(config, dict)
            assert len(config) > 0

            # Each config should have some required parameters
            if model_type in ["triplanar", "deepsdf", "two_stage"]:
                assert "latent_size" in config
            elif model_type == "implicit":
                assert "latent_dim" in config
                assert "hidden_dim" in config
                assert "num_layers" in config

    def test_get_model_config_template_invalid_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_config_template("invalid_model_type")

    def test_model_initialization_from_templates(self):
        """Test that all models can be initialized using their config templates."""
        models = list_supported_models()

        for model_type in models:
            config = get_model_config_template(model_type)

            # Import the internal parameter extraction functions
            from NSM.models.loader import (
                _get_triplanar_params,
                _get_deepsdf_params,
                _get_two_stage_params,
                _get_implicit_params,
            )

            if model_type == "triplanar":
                model_class, params = _get_triplanar_params(config)
                assert model_class == TriplanarDecoder
            elif model_type == "deepsdf":
                model_class, params = _get_deepsdf_params(config)
                assert model_class == Decoder
            elif model_type == "two_stage":
                model_class, params = _get_two_stage_params(config)
                assert model_class == TwoStageDecoder
            elif model_type == "implicit":
                model_class, params = _get_implicit_params(config)
                assert model_class == ImplicitDecoder

            # Initialize the model
            model = model_class(**params)
            assert isinstance(model, torch.nn.Module)

    def test_load_model_invalid_type(self):
        """Test that load_model raises ValueError for invalid model type."""
        config = get_model_config_template("triplanar")

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            # Create dummy state dict
            dummy_state = {"model": {}}
            torch.save(dummy_state, tmp_file.name)

            with pytest.raises(ValueError, match="Unknown model type"):
                load_model(config, tmp_file.name, model_type="invalid_type")

    def test_load_model_missing_file(self):
        """Test that load_model raises FileNotFoundError for missing file."""
        config = get_model_config_template("triplanar")

        with pytest.raises(FileNotFoundError):
            load_model(config, "/nonexistent/path/model.pt", model_type="triplanar")

    def test_load_model_missing_config_keys(self):
        """Test that load_model raises KeyError for missing required config keys."""
        # Empty config should fail
        empty_config = {}

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            dummy_state = {"model": {}}
            torch.save(dummy_state, tmp_file.name)

            with pytest.raises(KeyError, match="Missing required configuration keys"):
                load_model(empty_config, tmp_file.name, model_type="triplanar")


@pytest.fixture
def temp_model_files():
    """Create temporary model files with proper state dicts for testing."""
    models_data = {}

    # Create a simple model for each type and save its state
    for model_type in list_supported_models():
        config = get_model_config_template(model_type)

        # Modify configs to be smaller for faster testing
        if model_type == "triplanar":
            config["latent_size"] = 64
            config["conv_hidden_dims"] = [128, 128]
            config["sdf_hidden_dims"] = [128, 128]
            config["sdf_latent_size"] = 32
            model = TriplanarDecoder(
                **{
                    "latent_dim": config["latent_size"],
                    "n_objects": config["objects_per_decoder"],
                    "conv_hidden_dims": config["conv_hidden_dims"],
                    "conv_deep_image_size": config["conv_deep_image_size"],
                    "conv_norm": config["conv_norm"],
                    "conv_norm_type": config["conv_norm_type"],
                    "conv_start_with_mlp": config["conv_start_with_mlp"],
                    "sdf_latent_size": config["sdf_latent_size"],
                    "sdf_hidden_dims": config["sdf_hidden_dims"],
                    "sdf_weight_norm": config["weight_norm"],
                    "sdf_final_activation": config["final_activation"],
                    "sdf_activation": config["activation"],
                    "sdf_dropout_prob": config["dropout_prob"],
                    "sum_sdf_features": config["sum_conv_output_features"],
                    "conv_pred_sdf": config["conv_pred_sdf"],
                    "padding": config["padding"],
                }
            )

        elif model_type == "deepsdf":
            config["latent_size"] = 64
            config["layer_dimensions"] = [128, 128, 128]
            model = Decoder(
                latent_size=config["latent_size"],
                dims=config["layer_dimensions"],
                n_objects=config["objects_per_decoder"],
                dropout=config["layers_with_dropout"],
                dropout_prob=config["dropout_prob"],
                norm_layers=config["layers_with_norm"],
                latent_in=config["layer_latent_in"],
                weight_norm=config["weight_norm"],
                xyz_in_all=config["xyz_in_all"],
                activation=config["activation"],
                final_activation=config["final_activation"],
                concat_latent_input=config["concat_latent_input"],
                progressive_add_depth=config["progressive_add_depth"],
                layer_split=config["layer_split"],
                latent_noise_sigma=config["latent_noise_sigma"],
            )

        elif model_type == "two_stage":
            config["latent_size"] = 128  # Must be even
            config["triplanar_params"]["sdf_hidden_dims"] = [64, 64]
            config["triplanar_params"]["conv_hidden_dims"] = [64, 64]
            config["mlp_params"]["dims"] = [64, 64, 64]
            model = TwoStageDecoder(
                latent_size=config["latent_size"],
                n_objects=config["objects_per_decoder"],
                triplanar_params=config["triplanar_params"],
                mlp_params=config["mlp_params"],
            )

        elif model_type == "implicit":
            config["latent_dim"] = 64
            config["hidden_dim"] = 128
            config["num_layers"] = 3
            from NSM.models.modulated_periodic_activations import LinearBlockFactory

            model = ImplicitDecoder(
                latent_dim=config["latent_dim"],
                out_dim=config["out_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                block_factory=LinearBlockFactory(),
                modulation=config["modulation"],
                dropout=config["dropout"],
                final_activation=torch.sigmoid if config["final_activation"] == "sigmoid" else None,
            )

        # Save model to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        state_dict = {"model": model.state_dict()}
        torch.save(state_dict, temp_file.name)
        temp_file.close()

        models_data[model_type] = {
            "config": config,
            "file_path": temp_file.name,
            "original_model": model,
        }

    yield models_data

    # Cleanup
    for data in models_data.values():
        if os.path.exists(data["file_path"]):
            os.unlink(data["file_path"])


class TestModelLoadingFullWorkflow:
    """Test class for complete model loading workflow with actual models."""

    def test_load_model_full_workflow(self, temp_model_files):
        """Test complete workflow of loading models from saved states."""
        for model_type, data in temp_model_files.items():
            config = data["config"]
            file_path = data["file_path"]
            original_model = data["original_model"]

            # Load the model using our loader
            loaded_model = load_model(config, file_path, model_type=model_type, device="cpu")

            # Verify the loaded model
            assert isinstance(loaded_model, torch.nn.Module)
            assert type(loaded_model) == type(original_model)
            assert not loaded_model.training  # Should be in eval mode

            # Test that the model can perform inference
            if model_type in ["triplanar", "deepsdf", "two_stage"]:
                latent_size = config["latent_size"]
                batch_size = 10

                # Create test input: [latent, xyz]
                test_input = torch.randn(batch_size, latent_size + 3)

                with torch.no_grad():
                    output = loaded_model(test_input)

                assert output.shape[0] == batch_size
                assert output.shape[1] == config.get("objects_per_decoder", 1)

            elif model_type == "implicit":
                latent_size = config["latent_dim"]
                batch_size = 10

                # Create test input: [latent, xyz]
                test_input = torch.randn(batch_size, latent_size + 3)

                with torch.no_grad():
                    output = loaded_model(test_input)

                assert output.shape[0] == batch_size
                assert output.shape[1] == config.get("out_dim", 1)

    def test_different_state_dict_formats(self, temp_model_files):
        """Test loading models with different state dict save formats."""
        # Test with triplanar model
        data = temp_model_files["triplanar"]
        config = data["config"]
        original_model = data["original_model"]

        # Test different save formats
        formats = [
            {"model": original_model.state_dict()},
            {"state_dict": original_model.state_dict()},
            {"model_state_dict": original_model.state_dict()},
            original_model.state_dict(),  # Direct state dict
        ]

        for i, state_format in enumerate(formats):
            with tempfile.NamedTemporaryFile(suffix=f"_format_{i}.pt", delete=False) as tmp_file:
                torch.save(state_format, tmp_file.name)

                try:
                    # Should load successfully regardless of format
                    loaded_model = load_model(
                        config, tmp_file.name, model_type="triplanar", device="cpu"
                    )
                    assert isinstance(loaded_model, TriplanarDecoder)
                    assert not loaded_model.training
                finally:
                    os.unlink(tmp_file.name)

    def test_device_handling(self, temp_model_files):
        """Test that models are loaded to the correct device."""
        data = temp_model_files["deepsdf"]
        config = data["config"]
        file_path = data["file_path"]

        # Test loading to CPU
        model_cpu = load_model(config, file_path, model_type="deepsdf", device="cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Test automatic device detection (should default to CPU in test environment)
        model_auto = load_model(config, file_path, model_type="deepsdf", device=None)
        assert next(model_auto.parameters()).device.type in ["cpu", "cuda"]


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
