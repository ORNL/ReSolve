#[[

@brief Build ReSolve solver configuration parameter management unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build logger tests
add_executable(runParamTests.exe runParamTests.cpp)
target_link_libraries(runParamTests.exe PRIVATE ReSolve)

# Install tests
set(installable_tests runParamTests.exe)
install(TARGETS ${installable_tests} RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME param_test COMMAND $<TARGET_FILE:runParamTests.exe>)
