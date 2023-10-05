#!/bin/bash

# This script get executed in `make test_install`. 

# Defines where Resolve is and is set in line #47 in examples/CMakeLists.txt
export ReSolve_DIR=${1}
echo ${ReSolve_DIR}

# Defines where Resolve is and is set in line #47 in examples/CMakeLists.txt
export DATA_DIR=${2}
echo ${DATA_DIR}

# Locate source of the consumer test app
export INSTALL_BUILD_CONSUME=${ReSolve_DIR}/share/examples/resolve_consumer
echo ${INSTALL_BUILD_CONSUME}

mkdir -p ${INSTALL_BUILD_CONSUME}/build

rm -rf ${INSTALL_BUILD_CONSUME}/build/* &&

cmake -B ${INSTALL_BUILD_CONSUME}/build -S ${INSTALL_BUILD_CONSUME} -DRESOLVE_DATA=${DATA_DIR} -DReSolve_DIR=${ReSolve_DIR} &&

cmake --build ${INSTALL_BUILD_CONSUME}/build -- -j 12 &&

cmake --install ${INSTALL_BUILD_CONSUME}/build &&

cd ${INSTALL_BUILD_CONSUME}/build &&

ctest --output-on-failure

exit $?
