#!/bin/bash

module purge

source buildsystem/ascent-env.sh
echo Sourced System-specific variable for ascent

if [ "$(ls -A ./build)" ]; then
   cd build
   ctest -VV
   cd ..
else
	echo RESOLVE NEEDS TO BE BUILT
fi 
