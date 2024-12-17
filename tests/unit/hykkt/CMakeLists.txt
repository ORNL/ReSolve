#[[

@brief Build ReSolve HyKKT unit tests

@author Shaked Regev <regevs@ornl.gov>

]]

add_executable(runHykktHelloWorldTests.exe runHykktHelloWorld.cpp)
target_link_libraries(runHykktHelloWorldTests.exe PRIVATE ReSolve)

add_test(NAME hykkt_hello_world_test   COMMAND $<TARGET_FILE:runHykktHelloWorldTests.exe>)

add_executable(runHykktPermTests.exe runHykktPerm.cpp)
target_link_libraries(runHykktPermTests.exe PRIVATE ReSolve)

add_test(NAME hykkt_perm_test   COMMAND $<TARGET_FILE:runHykktPermTests.exe>)
