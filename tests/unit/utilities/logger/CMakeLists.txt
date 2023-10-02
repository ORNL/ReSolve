#[[

@brief Build ReSolve logger unit tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build logger tests
add_executable(runLoggerTests.exe runLoggerTests.cpp)
target_link_libraries(runLoggerTests.exe PRIVATE ReSolve resolve_logger)

# Install tests
set(installable_logger_tests runLoggerTests.exe)
install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/unit)

add_test(NAME logger_test COMMAND $<TARGET_FILE:runLoggerTests.exe>)
