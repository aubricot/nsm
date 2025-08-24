![Build Status](https://github.com/gattia/NSM/actions/workflows/build-test.yml/badge.svg?branch=main)<br>
|[Documentation](http://anthonygattiphd.com/NSM/)|



# Introduction

This pacakge is meant to develop generative deep learning models for creating human anatomy. The initial focus is on musculoskeletal tissues, particular of the knee. 

Steps to update this package for new repository: 
4. update `requirements.txt` and `dependencies` in `pyproject.toml`
     - To do - can dependencies read/update from requirements.txt?


# Installation

## Standard Installation

```bash
# Create and activate conda environment
conda create -n nsm python=3.9
conda activate nsm

# Install PyTorch (ensure compatibility with your CUDA version if using GPU)
# See: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install NSM package
pip install -r requirements.txt
pip install .
```

## Development Installation
If you plan to contribute to the development of NSM, install it in editable mode. This means changes you make to the source code will be immediately reflected when you use the package.

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development setup instructions.

Quick setup:
```bash
# Clone the repository
git clone https://github.com/gattia/NSM
cd NSM

# Create and activate conda environment
conda create -n nsm-dev python=3.9
conda activate nsm-dev

# Install all dependencies and NSM in development mode
make install-dev
```

# Usage

## Model Loading

NSM provides a convenient model loader that simplifies loading pre-trained Neural Shape Models. For **real trained models**, you'll typically have:

- `experiment_dir/model_params_config.json` - Configuration saved during training
- `experiment_dir/model/2000.pth` - Model weights at epoch 2000

```python
import json
from NSM.models import load_model

# Load configuration from training
with open('experiment_dir/model_params_config.json', 'r') as f:
    config = json.load(f)

# Load trained model
model = load_model(
    config=config,
    path_model_state='experiment_dir/model/2000.pth',
    model_type='triplanar'  # or 'deepsdf', 'two_stage', 'implicit'
)

# Ready for inference!
model.eval()
```

### Supported Model Types

- `'triplanar'` - TriplanarDecoder for triplanar neural representations
- `'deepsdf'` - Standard DeepSDF decoder  
- `'two_stage'` - Two-stage decoder combining triplanar and MLP
- `'implicit'` - ImplicitDecoder with modulated periodic activations

### Configuration Templates

Get template configurations with sensible defaults:

```python
from NSM.models import get_model_config_template, list_supported_models

# See all supported model types
print(list_supported_models())
# ['triplanar', 'deepsdf', 'two_stage', 'implicit']

# Get configuration template for any model type
config = get_model_config_template('deepsdf')
# Modify parameters as needed
config['latent_size'] = 512
config['layer_dimensions'] = [512, 512, 512, 256, 128]
```

## Examples

### Loading a Trained Model

See [`examples/load_trained_model.py`](examples/load_trained_model.py) for a complete example:

```bash
# Run the example with your trained model
python examples/load_trained_model.py /path/to/experiment_dir 2000 --model-type triplanar

# See all options
python examples/load_trained_model.py --help
```

# Development / Contributing

## Quick Development Commands

The project includes a Makefile for common development tasks:

```bash
# Run all tests
make test

# Run only model loader tests
make test-loader

# Run tests with coverage report
make test-coverage

# Format code with black
make format

# Check code style with flake8
make lint

# Clean up temporary files
make clean
```

## Tests
Run tests with pytest:

```bash
pytest                          # Run all tests
pytest testing/NSM/models/     # Run model tests
make test                       # Use Makefile
```

## Coverage
Generate test coverage reports:
```bash
make test-coverage              # HTML + terminal report
```

## Contributing
If you want to contribute, please read the documentation in `CONTRIBUTING.md` and see `DEVELOPMENT.md` for detailed development setup instructions.

## Documentation

Documentation is planned for future development. Consider using `pdoc` for auto-generated docs:

```bash
# TODO: Set up documentation generation
# pip install pdoc
# pdoc --html --output-dir docs NSM
```


# License

This project is licensed under the terms of the license specified in the [LICENSE](LICENSE) file.