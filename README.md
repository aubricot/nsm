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
conda create -n NSM python=3.8
conda activate NSM

# Install PyTorch (ensure compatibility with your CUDA version if using GPU)
# See: https://pytorch.org/get-started/locally/
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install package requirements
make requirements

# Install the NSM package
pip install .
```

## Development Installation
If you plan to contribute to the development of NSM, install it in editable mode. This means changes you make to the source code will be immediately reflected when you use the package.

```bash
# Clone the repository
git clone https://github.com/gattia/NSM
cd NSM

# Create and activate conda environment
conda create -n NSM python=3.8
conda activate NSM

# Install development dependencies (includes PyTorch, requirements, and dev tools)
make dev

# Install NSM in editable mode
make install-dev
```

# Examples

*Add links to example notebooks or scripts here.*

# Development / Contributing

## Tests
The test can be run by: 

```bash
pytest
```

or 
```bash
make test
```

Inidividual tests can be run by running 

```
python -m pytests path_to_test
```

## Coverage
- Coverage results/info requires `coverage` (`conda install coverage` or `pip install coverage`).
- These should be installed automatically with one of the  `make dev` commands.
- You can get coverage statistics by running: 
    - `coverage run -m pytest`
    or if using make: 
    - `make coverage`
        - This will save an html of the coverage results. 

### note about coverage:
    - Coverage runs by seeing how much of the code-base is covered when you run the command after coverage. 
    In this case, it is looking to see how much of the code-base is covered when we run the tests. 

## Contributing
If you want to contribute, please read over the documentaiton in `CONTRIBUTING.md`

## Docs
To build the docs, run `make docs`. If you want the docs published on gihutb, you need to activate github page.
Go to the `Settings` tab on your github repo, under `Pages` on the left, turn GitHub Pages on, and select the
home dir for the docs to be `/docs` on the `main` branch. Example here:  

![Setup Docs on Github Pages](media/setting_up_docs_automatically.png)


# License

This project is licensed under the terms of the license specified in the [LICENSE](LICENSE) file.