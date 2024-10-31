/**
 * @file testSysHipRefine.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for SystemSolver class
 * @date 2023-12-14
 * 
 * 
 */
#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/SystemSolver.hpp>

#if defined (RESOLVE_USE_CUDA)
#include <resolve/LinSolverDirectCuSolverRf.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
  std::string memory_space("cuda");
#elif defined (RESOLVE_USE_HIP)
#include <resolve/LinSolverDirectRocSolverRf.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceHIP;
  std::string memory_space("hip");
#else
  using workspace_type = ReSolve::LinAlgWorkspaceCpu;
  std::string memory_space("cpu");
#endif

#include <tests/functionality/FunctionalityTestHelper.hpp>


using namespace ReSolve::constants;
using namespace ReSolve::tests;
using namespace ReSolve::colors;
using real_type   = ReSolve::real_type;
using vector_type = ReSolve::vector::Vector;

int verifyResult( AxEqualsRhsProblem &axb,
                   ReSolve::SystemSolver &solver,
                   workspace_type &workspace,
                   double tolerance )
{
  int error_sum = 0;
  ReSolve::matrix::Csr* A = axb.getMatrix();
  vector_type* vec_x   = axb.getVector();
  vector_type* vec_rhs = axb.getRhs();

  // larger tolerance than default 1e-17 because iterative refinement is not applied here
  ReSolve::tests::FunctionalityTestHelper test_helper(tolerance, workspace, axb);

  // Verify relative residual norm computation in SystemSolver
  error_sum += test_helper.checkRelativeResidualNorm(*vec_rhs, *vec_x, solver);

  // Compute norm of scaled residuals
  error_sum += test_helper.checkNormOfScaledResiduals(*A, *vec_rhs, *vec_x, solver);

  // Captain! split this into residual checking and printing
  error_sum += 
  test_helper.checkResult(*A, *vec_rhs, *vec_x, solver, "first matrix");

  return error_sum;
}

void reportFinalResult(int const error_sum)
{
  if (error_sum == 0) {
    std::cout << "Test KLU with Rf solver + IR " << GREEN << "PASSED" << CLEAR <<std::endl<<std::endl;;
  } else {
    std::cout << "Test KLU with Rf solver + IR " << RED << "FAILED" << CLEAR << ", error sum: "<<error_sum<<std::endl<<std::endl;;
  }
}

class FileInputs
{
  public:

  FileInputs(std::string const &data_path)
    :
  matrixFileName1_(data_path + "data/matrix_ACTIVSg2000_AC_00.mtx"),
  matrixFileName2_(data_path + "data/matrix_ACTIVSg2000_AC_02.mtx"),
  rhsFileName1_(data_path + "data/rhs_ACTIVSg2000_AC_00.mtx.ones"),
  rhsFileName2_(data_path + "data/rhs_ACTIVSg2000_AC_02.mtx.ones")
  {
  }

  std::string getMatrixFileName1() const
  {
    return  matrixFileName1_;
  }

  std::string getMatrixFileName2() const
  {
     return matrixFileName2_;
  }

  std::string getRhsFileName1() const
  {
     return rhsFileName1_;
  }

  std::string getRhsFileName2() const
  {
     return rhsFileName2_;
  }

  private:

  // Texas model
  std::string const matrixFileName1_;
  std::string const rhsFileName1_;

  std::string const matrixFileName2_;
  std::string const rhsFileName2_;
};

ReSolve::SystemSolver generateSolver(workspace_type *workspace, const AxEqualsRhsProblem &axb)
{
  ReSolve::SystemSolver solver(workspace);

  // Configure solver (CUDA-based solver needs slightly different
  // settings than HIP-based one)
  // cgs2 = classical Gram-Schmidt
  solver.setRefinementMethod("fgmres", "cgs2");

  solver.getIterativeSolver().setRestart(100);

  if (memory_space == "hip") {
    solver.getIterativeSolver().setMaxit(200);
  }

  if (memory_space == "cuda") {
    solver.getIterativeSolver().setMaxit(400);
    solver.getIterativeSolver().setTol(1e-17);
  }

  // next connect solver to problem:
  int const status = solver.setMatrix(axb.getMatrix());
  
  if(status != 0) {

    std::cout << "solver.setMatrix(axb.getMatrix()) failed!" << std::endl;
    std::exit(status);
  }

  return solver;
}


int main(int argc, char *argv[])
{
  // Input to this code is location of `data` directory where matrix files are stored
  // build filenames for inputs
  const std::string data_path = (argc == 2) ? argv[1] : "./";

  FileInputs const fileInputs( data_path );

  // axb problem construction
  AxEqualsRhsProblem axb(fileInputs.getMatrixFileName1(), 
                         fileInputs.getRhsFileName1());

  // workspace construction
  workspace_type workspace;

  workspace.initializeHandles();

  // solver construction
  ReSolve::SystemSolver solver = generateSolver(&workspace, axb);

  // Error sum must be 0 at the end for test to PASS.
  int error_sum = 0;

  // Solve the first system
  error_sum += solver.analyze();

  error_sum += solver.factorize();

  error_sum += solver.solve(axb.getRhs(), axb.getVector());

  error_sum += verifyResult(axb, solver, workspace, 1e-12);

  // Now prepare the Rf solver
  error_sum += solver.refactorizationSetup();

  // update the Ax=b problem
  axb.updateProblem(fileInputs.getMatrixFileName2(), fileInputs.getRhsFileName2());

  error_sum += solver.refactorize();
  
  error_sum += solver.solve(axb.getRhs(), axb.getVector());

  error_sum += verifyResult(axb, solver, workspace, 1e-12);

  reportFinalResult(error_sum);

  // if not zero, main() exits with problems
  return error_sum;
}
