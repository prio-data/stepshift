.PHONY: build stest clean

build: 
	python setup.py build_ext --inplace

test: build
	python -m unittest

clean:
	rm -r stepshift/*.c stepshift/*.so build dist stepshift.egg-info 
