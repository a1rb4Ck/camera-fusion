# Simple makefile to build, clean and test

all: clean test

clean:
	python setup.py clean --all
	rm -rf *.pyc __pycache__ build dist camera_fusion.egg-info
	rm -rf camera_fusion/__pycache__ camera_fusion/units/__pycache__
	rm -rf tests/__pycache__ docs/build .pytest_cache .tox
	rm -rf .coverage *.npy *.xml

test:
	rm -f ./tests/reports/coverage.svg
	tox

upload:
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*