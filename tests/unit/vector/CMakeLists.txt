#[[

@brief Build ReSolve vector (and vector related) unit tests

@author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>

]]

# Build vector tests
add_executable(runVectorTests.exe runVectorTests.cpp)
target_link_libraries(runVectorTests.exe PRIVATE ReSolve resolve_vector)

# Build vector handler tests
add_executable(runVectorHandlerTests.exe runVectorHandlerTests.cpp)
target_link_libraries(runVectorHandlerTests.exe PRIVATE ReSolve resolve_vector)

# Build Gram - Schmidt tests
add_executable(runGramSchmidtTests.exe runGramSchmidtTests.cpp)
target_link_libraries(runGramSchmidtTests.exe PRIVATE ReSolve resolve_vector)

# Install tests
set(installable_tests runVectorTests.exe runVectorHandlerTests.exe runGramSchmidtTests.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME vector_test COMMAND $<TARGET_FILE:runVectorTests.exe>)
add_test(NAME vector_handler_test COMMAND $<TARGET_FILE:runVectorHandlerTests.exe>)
add_test(NAME gram_schmidt_test COMMAND $<TARGET_FILE:runGramSchmidtTests.exe>)
