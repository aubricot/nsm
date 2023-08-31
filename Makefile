.PHONY: build install 

requirements:
	python -m pip install -r requirements.txt

requirements-mamba:
	mamba install --file requirements.txt

requirements-conda:
	conda install --file requirements.txt

build:
	python -m build -o wheelhouse

install:
	pip install .

#python setup.py install

install-dev: 
	pip install --editable .

# below is specifically for building / forcing building of cython code. 
# build-cython:
# 	python setup.py build_ext -i --force

dev:
	python -m pip install --upgrade -r requirements-dev.txt

dev-mamba:
	mamba install --file requirements-dev.txt

dev-conda:
	conda install --file requirements-dev.txt

docs:
	pdoc --output-dir docs/ --html --force NSM
	mv docs/NSM/* docs/
	rm -rf docs/NSM


test:
	set -e
	pytest

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .

autoformat:
	set -e
	isort .
	black --config pyproject.toml .

clean:
	rm -rf build dist *.egg-info  

coverage: 
	coverage run -m pytest
	# Make the html version of the coverage results. 
	coverage html 