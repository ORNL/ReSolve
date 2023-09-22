#[[

@brief Build ReSolve vector (and vector related) unit tests

@author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>

]]

# Build vector handler tests
add_executable(runVectorHandlerTests.exe runVectorHandlerTests.cpp)
target_link_libraries(runVectorHandlerTests.exe PRIVATE ReSolve resolve_vector)

# Install tests
set(installable_tests runVectorHandlerTests.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/unit)
# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/unit)

add_test(NAME vector_handler_test COMMAND $<TARGET_FILE:runVectorHandlerTests.exe>)
