#[[

@brief Build ReSolve functionality tests

@author Slaven Peles <peless@ornl.gov>

]]

# Build basic version test
add_executable(version.exe testVersion.cpp) 
target_link_libraries(version.exe PRIVATE ReSolve)

# Build test for Krylov solvers
add_executable(sys_rand_gmres_test.exe testSysRandGMRES.cpp)
target_link_libraries(sys_rand_gmres_test.exe PRIVATE ReSolve)

if(RESOLVE_USE_KLU)
  # Build KLU+KLU test
  add_executable(klu_klu_test.exe testKLU.cpp)
  target_link_libraries(klu_klu_test.exe PRIVATE ReSolve)
endif(RESOLVE_USE_KLU)

if(RESOLVE_USE_LUSOL)
  add_executable(lusol_test.exe testLUSOL.cpp)
  target_link_libraries(lusol_test.exe PRIVATE ReSolve)
endif()

if(RESOLVE_USE_CUDA)
        
  if(RESOLVE_USE_KLU)
    # Build KLU+Rf test
    add_executable(klu_rf_test.exe testKLU_cusolverRf.cpp)
    target_link_libraries(klu_rf_test.exe PRIVATE ReSolve)

    # Build KLU+Rf+fgmres test
    add_executable(klu_rf_fgmres_test.exe testKLU_Rf_FGMRES.cpp)
    target_link_libraries(klu_rf_fgmres_test.exe PRIVATE ReSolve)

    # System solver test with cusolver rf and iterative refinement
    add_executable(sys_refactor_cuda_test.exe testSysRefactor.cpp)
    target_link_libraries(sys_refactor_cuda_test.exe PRIVATE ReSolve)

    # Build KLU+GLU test
    add_executable(klu_glu_test.exe testKLU_GLU.cpp)
    target_link_libraries(klu_glu_test.exe PRIVATE ReSolve)
    
    # System solver test with GLU
    add_executable(sys_glu_test.exe testSysGLU.cpp)
    target_link_libraries(sys_glu_test.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_KLU)

  # Build randSolver test
  add_executable(rand_gmres_cuda_test.exe testRandGMRES_Cuda.cpp)
  target_link_libraries(rand_gmres_cuda_test.exe PRIVATE ReSolve)

endif(RESOLVE_USE_CUDA)


if(RESOLVE_USE_HIP)
        
  if(RESOLVE_USE_KLU)
    # Build KLU+rossolver test
    add_executable(rocsolver_rf_test.exe testKLU_RocSolver.cpp)
    target_link_libraries(rocsolver_rf_test.exe PRIVATE ReSolve)
  
    # And another one to test FGMRES version
    add_executable(rocsolver_rf_fgmres_test.exe testKLU_RocSolver_FGMRES.cpp)
    target_link_libraries(rocsolver_rf_fgmres_test.exe PRIVATE ReSolve)
  
    # System solver test with rocm rf and iterative refinement
    add_executable(sys_refactor_hip_test.exe testSysRefactor.cpp)
    target_link_libraries(sys_refactor_hip_test.exe PRIVATE ReSolve)
  endif(RESOLVE_USE_KLU)

  # Build randSolver test
  add_executable(rand_gmres_hip_test.exe testRandGMRES_Rocm.cpp)
  target_link_libraries(rand_gmres_hip_test.exe PRIVATE ReSolve)
  
endif(RESOLVE_USE_HIP)

set(installable_tests version.exe)

# Install tests
if(RESOLVE_USE_KLU)
  list(APPEND installable_tests klu_klu_test.exe)
endif()

if(RESOLVE_USE_LUSOL)
  list(APPEND installable_tests lusol_test.exe)
endif()

if(RESOLVE_USE_CUDA)
  if(RESOLVE_USE_KLU)
    list(APPEND installable_tests klu_rf_test.exe
                                  klu_rf_fgmres_test.exe
                                  sys_refactor_cuda_test.exe
                                  klu_glu_test.exe
                                  sys_glu_test.exe)
  endif()
  list(APPEND installable_tests rand_gmres_cuda_test.exe)
endif(RESOLVE_USE_CUDA)

if(RESOLVE_USE_HIP)
  if(RESOLVE_USE_KLU)
    list(APPEND installable_tests rocsolver_rf_test.exe
                                  rocsolver_rf_fgmres_test.exe
                                  sys_refactor_hip_test.exe)
  endif()
  list(APPEND installable_tests rand_gmres_hip_test.exe)
endif(RESOLVE_USE_HIP)

install(TARGETS ${installable_tests} 
        RUNTIME DESTINATION bin/resolve/tests/functionality)

# Install directory with data files
install(DIRECTORY data DESTINATION bin/resolve/tests/functionality)

set(test_data_dir ${CMAKE_SOURCE_DIR}/tests/functionality/)

add_test(NAME version COMMAND $<TARGET_FILE:version.exe> "${test_data_dir}")

if(RESOLVE_USE_KLU)
  add_test(NAME klu_klu_test COMMAND $<TARGET_FILE:klu_klu_test.exe> "${test_data_dir}")
endif()

if(RESOLVE_USE_LUSOL)
  add_test(NAME lusol_test COMMAND $<TARGET_FILE:lusol_test.exe> "${test_data_dir}")
endif()

# Krylov solvers tests (FGMRES)
add_test(NAME sys_rand_count_fgmres_CGS2_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>     "-i" "randgmres" "-g" "CGS2" "-s" "count")
add_test(NAME sys_rand_count_fgmres_MGS_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>      "-i" "randgmres" "-g" "MGS" "-s" "count")
add_test(NAME sys_rand_count_fgmres_MGS2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-i" "randgmres" "-g" "MGS_TWO_SYNC" "-s" "count")
add_test(NAME sys_rand_count_fgmres_MGSpm_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>    "-i" "randgmres" "-g" "MGS_PM" "-s" "count")
add_test(NAME sys_rand_fwht_fgmres_CGS2_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>      "-i" "randgmres" "-g" "CGS2" "-s" "fwht")
add_test(NAME sys_rand_fwht_fgmres_MGS_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>       "-i" "randgmres" "-g" "MGS" "-s" "fwht")
add_test(NAME sys_rand_fwht_fgmres_MGS2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>  "-i" "randgmres" "-g" "MGS_TWO_SYNC" "-s" "fwht")
add_test(NAME sys_rand_fwht_fgmres_MGSpm_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>     "-i" "randgmres" "-g" "MGS_PM" "-s" "fwht")
add_test(NAME sys_fgmres_CGS2_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>      "-i" "fgmres" "-g" "CGS2")
add_test(NAME sys_fgmres_MGS_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>       "-i" "fgmres" "-g" "MGS")
add_test(NAME sys_fgmres_MGS2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>  "-i" "fgmres" "-g" "MGS_TWO_SYNC")
add_test(NAME sys_fgmres_MGSpm_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>     "-i" "fgmres" "-g" "MGS_PM")

# Krylov solvers tests (GMRES)
add_test(NAME sys_rand_count_gmres_CGS2_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>     "-x" "no" "-i" "randgmres" "-g" "CGS2" "-s" "count")
add_test(NAME sys_rand_count_gmres_MGS_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>      "-x" "no" "-i" "randgmres" "-g" "MGS" "-s" "count")
add_test(NAME sys_rand_count_gmres_MGS2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe> "-x" "no" "-i" "randgmres" "-g" "MGS_TWO_SYNC" "-s" "count")
add_test(NAME sys_rand_count_gmres_MGSpm_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>    "-x" "no" "-i" "randgmres" "-g" "MGS_PM" "-s" "count")
add_test(NAME sys_rand_fwht_gmres_CGS2_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>      "-x" "no" "-i" "randgmres" "-g" "CGS2" "-s" "fwht")
add_test(NAME sys_rand_fwht_gmres_MGS_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>       "-x" "no" "-i" "randgmres" "-g" "MGS" "-s" "fwht")
add_test(NAME sys_rand_fwht_gmres_MGS2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>  "-x" "no" "-i" "randgmres" "-g" "MGS_TWO_SYNC" "-s" "fwht")
add_test(NAME sys_rand_fwht_gmres_MGSpm_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>     "-x" "no" "-i" "randgmres" "-g" "MGS_PM" "-s" "fwht")
add_test(NAME sys_gmres_CGS2_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>      "-x" "no" "-i" "fgmres" "-g" "CGS2")
add_test(NAME sys_gmres_MGS_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>       "-x" "no" "-i" "fgmres" "-g" "MGS")
add_test(NAME sys_gmres_MGS2sync_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>  "-x" "no" "-i" "fgmres" "-g" "MGS_TWO_SYNC")
add_test(NAME sys_gmres_MGSpm_test COMMAND $<TARGET_FILE:sys_rand_gmres_test.exe>     "-x" "no" "-i" "fgmres" "-g" "MGS_PM")

if(RESOLVE_USE_CUDA)
  if(RESOLVE_USE_KLU)
    add_test(NAME klu_rf_test  COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}")
    add_test(NAME klu_rf_ir_test  COMMAND $<TARGET_FILE:klu_rf_test.exe> "-d" "${test_data_dir}" "-i")
    # add_test(NAME klu_rf_fgmres_test COMMAND $<TARGET_FILE:klu_rf_fgmres_test.exe> "${test_data_dir}")
    add_test(NAME sys_refactor_cuda_test COMMAND $<TARGET_FILE:sys_refactor_cuda_test.exe> "${test_data_dir}")
    add_test(NAME klu_glu_test COMMAND $<TARGET_FILE:klu_glu_test.exe> "${test_data_dir}")
    add_test(NAME sys_glu_test COMMAND $<TARGET_FILE:sys_glu_test.exe> "${test_data_dir}")
  endif(RESOLVE_USE_KLU)
  add_test(NAME rand_gmres_cuda_test COMMAND $<TARGET_FILE:rand_gmres_cuda_test.exe>)
endif(RESOLVE_USE_CUDA)

if(RESOLVE_USE_HIP)
  if(RESOLVE_USE_KLU)
    add_test(NAME rocsolver_rf_test  COMMAND $<TARGET_FILE:rocsolver_rf_test.exe> "-d" "${test_data_dir}")
    add_test(NAME rocsolver_rf_m1_test  COMMAND $<TARGET_FILE:rocsolver_rf_test.exe> "-d" "${test_data_dir}" "-m")
    add_test(NAME rocsolver_rf_ir_test  COMMAND $<TARGET_FILE:rocsolver_rf_test.exe> "-d" "${test_data_dir}" "-i")
    add_test(NAME rocsolver_rf_ir_m1_test  COMMAND $<TARGET_FILE:rocsolver_rf_test.exe> "-d" "${test_data_dir}" "-i" "-m")
    # add_test(NAME rocsolver_rf_fgmres_test  COMMAND $<TARGET_FILE:rocsolver_rf_fgmres_test.exe>  "${test_data_dir}")
    add_test(NAME sys_refactor_hip_test  COMMAND $<TARGET_FILE:sys_refactor_hip_test.exe>  "${test_data_dir}")
  endif(RESOLVE_USE_KLU)
  add_test(NAME rand_gmres_hip_test COMMAND $<TARGET_FILE:rand_gmres_hip_test.exe>)
endif(RESOLVE_USE_HIP)
