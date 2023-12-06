#!/bin/bash

git submodule update --init --recursive

pushd docs
rm -rfd ./sphinx/_build &&
sphinx-build . ./sphinx/_build &&
doxygen ./doxygen/Doxyfile.in &&
# Host the docs locally with python
python3 -m http.server --directory ./sphinx/_build
popd