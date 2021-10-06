.PHONY: build stest clean

build: 
	pip install cython==3.0.0a9 numpy==1.21.2 setuptools wheel
	python setup.py build_ext --inplace

test: build
	pip install . 
	pip install -r dev_requirements.txt
	python -m unittest

clean:
	rm -rf stepshift/*.c stepshift/*.so build dist stepshift.egg-info stepshift/*.html
