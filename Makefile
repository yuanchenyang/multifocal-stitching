.PHONY: build test upload install-local

build:
	python -m build

test:
	python -m pytest --doctest-modules --cov-report=html --cov=multifocal_stitching

upload:
	python -m twine upload --repository pypi dist/*

install-local:
	python -m pip install -e .[dev,test,webapp]
