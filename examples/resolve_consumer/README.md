## To use the ReSolve library in your own project
Make sure Resolve library is installed (see above)

Below is an example CMakeList.txt file to use ReSolve library in your project
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(ReSolve REQUIRED)

# Build your executable 
add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE ReSolve::ReSolve)
```

## Built in Cmake Tests for Testing Resolve Package
CI is ran per every merge request that makes sure ReSolve can be consumed as a package.

If you follow the [developer guidelines](CONTRIBUTING.md) for building resolve and run make test you will see ReSolve consumed and linked with an example test in Test #1 (resolve_Consume). 

This ReSolve Consume test is executed via a cmake test that exectutes test.sh. This shell script then goes through the cmake build process to ensure that ReSolve can be built from scratch and linked to another cmake project.  
