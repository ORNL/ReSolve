/**
 * @file runHykktSolverTests.cpp
 * @author Andrew Xu (xua1@ornl.gov)
 * @brief Tests for class hykkt::HyKKTSolver
 *
 */

#include <fstream>
#include <iostream>
#include <string>

#include <resolve/Common.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

#include "HykktSolverTests.hpp"
#include <resolve/vector/Vector.hpp>

/**
 * @brief Run tests with a given backend
 *
 * @param backend - string name of the hardware backend
 * @param result - test results
 */
template <typename WorkspaceType>
void runTests(const std::string& backend, ReSolve::memory::MemorySpace memspace, ReSolve::tests::TestingResults& result)
{
  std::cout << "Running tests on " << backend << " device:\n";

  WorkspaceType workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler           matrix_handler(&workspace);
  ReSolve::VectorHandler           vector_handler(&workspace);
  ReSolve::tests::HykktSolverTests test(memspace, matrix_handler, vector_handler);

  ReSolve::real_type gamma      = 10000.0;
  std::string        source_dir = std::string(SOURCE_DIR);

  result += test.testSolver(
      2278, 490, 1386, 2278, 490, 6784, 980, source_dir + std::string("/HyKKTSolverTestMatrices/block_H_matrix_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Dd_matrix_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_J_matrix_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Jd_matrix_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rx_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rs_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ry_ACTIVSg200_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ryd_ACTIVSg200_AC_09.mtx"), gamma);
  workspace.resetLinAlgWorkspace();
  result += test.testSolver(
      2278, 490, 1386, 2278, 490, 6784, 980, source_dir + std::string("/HyKKTSolverTestMatrices/block_H_matrix_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Dd_matrix_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_J_matrix_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Jd_matrix_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rx_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rs_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ry_ACTIVSg200_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ryd_ACTIVSg200_AC_10.mtx"), gamma);
  workspace.resetLinAlgWorkspace();

  result += test.testSolver(
      25910, 6412, 16933, 38469, 6412, 86139, 12824, source_dir + std::string("/HyKKTSolverTestMatrices/block_H_matrix_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Dd_matrix_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_J_matrix_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Jd_matrix_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rx_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rs_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ry_ACTIVSg2000_AC_09.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ryd_ACTIVSg2000_AC_09.mtx"), gamma);
  workspace.resetLinAlgWorkspace();
  result += test.testSolver(
      25910, 6412, 16933, 38469, 6412, 86139, 12824, source_dir + std::string("/HyKKTSolverTestMatrices/block_H_matrix_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Dd_matrix_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_J_matrix_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_Jd_matrix_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rx_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_rs_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ry_ACTIVSg2000_AC_10.mtx"), source_dir + std::string("/HyKKTSolverTestMatrices/block_ryd_ACTIVSg2000_AC_10.mtx"), gamma);
  workspace.resetLinAlgWorkspace();

  std::cout << "\n";
}

int main(int, char**)
{
  ReSolve::tests::TestingResults result;
  runTests<ReSolve::LinAlgWorkspaceCpu>("CPU", ReSolve::memory::HOST, result);

#ifdef RESOLVE_USE_CUDA
  runTests<ReSolve::LinAlgWorkspaceCUDA>("CUDA", ReSolve::memory::DEVICE, result);
#endif

#ifdef RESOLVE_USE_HIP
  runTests<ReSolve::LinAlgWorkspaceHIP>("HIP", ReSolve::memory::DEVICE, result);
#endif

  return result.summary();
}
