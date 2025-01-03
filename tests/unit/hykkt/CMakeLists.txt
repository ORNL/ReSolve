#[[

@brief Build ReSolve HyKKT unit tests

@author Shaked Regev <regevs@ornl.gov>

]]

# Build matrix Hello World tests
add_executable(runHykktHelloWorldTests.exe runHykktHelloWorld.cpp)
target_link_libraries(runHykktHelloWorldTests.exe PRIVATE ReSolve)

add_test(NAME hykkt_hello_world_test   COMMAND $<TARGET_FILE:runHykktHelloWorldTests.exe>)
