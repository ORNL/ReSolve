#[[

@brief Build ReSolve memory utilities unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build memory utilities tests
add_executable(runMemoryUtilsTests.exe runMemoryUtilsTests.cpp)
target_link_libraries(runMemoryUtilsTests.exe PRIVATE ReSolve)
message(STATUS "Resolve libraries: ${resolve_backend_hip}")


# Install tests
set(installable_tests runMemoryUtilsTests.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/unit)

# Add tests to run
add_test(NAME memory_test COMMAND $<TARGET_FILE:runMemoryUtilsTests.exe>)
