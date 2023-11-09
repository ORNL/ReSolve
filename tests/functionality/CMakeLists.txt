#[[

@brief Build ReSolve functionality tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build KLU+KLU test
add_executable(klu_klu_test.exe testKLU.cpp)
target_link_libraries(klu_klu_test.exe PRIVATE ReSolve)

if(RESOLVE_USE_CUDA)
        
  # Build KLU+Rf test
  add_executable(klu_rf_test.exe testKLU_Rf.cpp)
  target_link_libraries(klu_rf_test.exe PRIVATE ReSolve)

  # Build KLU+Rf+fgmres test
  add_executable(klu_rf_fgmres_test.exe testKLU_Rf_FGMRES.cpp)
  target_link_libraries(klu_rf_fgmres_test.exe PRIVATE ReSolve)

  # Build KLU+GLU test
  add_executable(klu_glu_test.exe testKLU_GLU.cpp)
  target_link_libraries(klu_glu_test.exe PRIVATE ReSolve)
<<<<<<< HEAD
  
  # Build randSolver test
  add_executable(rand_gmres_cuda_test.exe testRandGMRES_Cuda.cpp)
  target_link_libraries(rand_gmres_cuda_test.exe PRIVATE ReSolve)
=======

  # System solver test with GLU
  add_executable(sys_cuda_test.exe testSysCuda.cpp)
  target_link_libraries(sys_cuda_test.exe PRIVATE ReSolve)

>>>>>>> e627c59 (First stab at system solver.)
endif(RESOLVE_USE_CUDA)


if(RESOLVE_USE_HIP)
        
  # Build KLU+rossolver test
  add_executable(rocsolver_rf_test.exe testKLU_RocSolver.cpp)
  target_link_libraries(rocsolver_rf_test.exe PRIVATE ReSolve)

  # And another one to test FGMRES version
  add_executable(rocsolver_rf_fgmres_test.exe testKLU_RocSolver_FGMRES.cpp)
  target_link_libraries(rocsolver_rf_fgmres_test.exe PRIVATE ReSolve)

  # Build randSolver test
  add_executable(rand_gmres_hip_test.exe testRandGMRES_Rocm.cpp)
  target_link_libraries(rand_gmres_hip_test.exe PRIVATE ReSolve)
  
  # System solver test with rocm rf and iterative refinement
  add_executable(sys_hip_refine.exe testSysHipRefine.cpp)
  target_link_libraries(sys_hip_refine.exe PRIVATE ReSolve)

endif(RESOLVE_USE_HIP)

# Install tests
set(installable_tests klu_klu_test.exe)

if(RESOLVE_USE_CUDA)
  set(installable_tests ${installable_tests}
                        klu_rf_test.exe
                        klu_rf_fgmres_test.exe
                        klu_glu_test.exe
                        rand_gmres_cuda_test.exe)
endif(RESOLVE_USE_CUDA)

if(RESOLVE_USE_HIP)
  set(installable_tests ${installable_tests}
                        rocsolver_rf_test.exe
                        rocsolver_rf_fgmres_test.exe
                        rand_gmres_hip_test.exe)
endif(RESOLVE_USE_HIP)

install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/functionality)

# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/functionality)

set(test_data_dir ${CMAKE_SOURCE_DIR}/tests/functionality/)

add_test(NAME klu_klu_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "${test_data_dir}")
if(RESOLVE_USE_CUDA)
  add_test(NAME klu_rf_test  COMMAND $<TARGET_FILE:klu_rf_test.exe>  "${test_data_dir}")
  add_test(NAME klu_rf_fgmres_test COMMAND $<TARGET_FILE:klu_rf_fgmres_test.exe> "${test_data_dir}")
  add_test(NAME klu_glu_test COMMAND $<TARGET_FILE:klu_glu_test.exe> "${test_data_dir}")
  add_test(NAME rand_gmres_cuda_test COMMAND $<TARGET_FILE:rand_gmres_cuda_test.exe>)
  add_test(NAME sys_cuda_test COMMAND $<TARGET_FILE:sys_cuda_test.exe> "${test_data_dir}")
endif(RESOLVE_USE_CUDA)

if(RESOLVE_USE_HIP)
  add_test(NAME rocsolver_rf_test  COMMAND $<TARGET_FILE:rocsolver_rf_test.exe>  "${test_data_dir}")
  add_test(NAME rocsolver_rf_fgmres_test  COMMAND $<TARGET_FILE:rocsolver_rf_fgmres_test.exe>  "${test_data_dir}")
  add_test(NAME rand_gmres_hip_test COMMAND $<TARGET_FILE:rand_gmres_hip_test.exe>)
  add_test(NAME sys_hip_refine_test  COMMAND $<TARGET_FILE:sys_hip_refine.exe>  "${test_data_dir}")
endif(RESOLVE_USE_HIP)
