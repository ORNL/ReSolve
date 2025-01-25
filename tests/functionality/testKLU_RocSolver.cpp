/**
 * @file testKLU_RocSolver.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@amd.com)
 * @brief Functionality test for rocsolver_rf.
 * 
 */
#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using namespace ReSolve::colors;

template <class workspace_type>
class TestHelper
{    
  public:
    /**
     * @brief Test Helper constructor
     * 
     * @param A 
     * @param r 
     * @param x 
     * @param workspace
     * 
     * @pre The linear solver has solved system A * x = r.
     * @pre A, r, and x are all in the same memory space as the workspace.
     */
    TestHelper(ReSolve::matrix::Sparse* A,
               ReSolve::vector::Vector* r,
               ReSolve::vector::Vector* x,
               workspace_type& workspace)
      : A_(A),
        r_(r),
        x_(x),
        mh_(&workspace),
        vh_(&workspace),
        res_(A->getNumRows()),
        x_true_(A->getNumRows())
    {
      if (mh_.getIsCudaEnabled() || mh_.getIsHipEnabled()) {
        memspace_ = ReSolve::memory::DEVICE;
      }

      // Compute rs and residual norms
      res_.copyDataFrom(r_, memspace_, memspace_);
      norm_rhs_ = norm2(*r_, memspace_);
      norm_res_ = computeResidualNorm(*A_, *x_, res_, memspace_);

      // Compute residual norm w.r.t. true solution
      setSolutionVector();
      res_.copyDataFrom(r_, memspace_, memspace_);
      norm_res_true_ = computeResidualNorm(*A_, x_true_, res_, memspace_);

      // Compute residual norm on CPU
      if (memspace_ == ReSolve::memory::DEVICE) {
        A_->syncData(ReSolve::memory::HOST);
        r_->syncData(ReSolve::memory::HOST);
        x_->syncData(ReSolve::memory::HOST);
        res_.copyDataFrom(r_, memspace_, ReSolve::memory::HOST);
        norm_res_cpu_ = computeResidualNorm(*A_, *x_, res_, ReSolve::memory::HOST);
      }

      // Compute vector difference norm
      res_.copyDataFrom(x_, memspace_, memspace_);
      norm_diff_ = computeDiffNorm(x_true_, res_, memspace_);
    }

    ~TestHelper()
    {
      // empty
    }

    ReSolve::real_type getNormResidual()
    {
      return norm_res_;
    }

    ReSolve::real_type getNormResidualScaled()
    {
      return norm_res_/norm_rhs_;
    }

    ReSolve::real_type getNormResidualCpu()
    {
      return norm_res_cpu_;
    }

    ReSolve::real_type getNormResidualTrue()
    {
      return norm_res_true_;
    }

    ReSolve::real_type getNormDiff()
    {
      return norm_diff_;
    }

    ReSolve::real_type getNormDiffScaled()
    {
      return norm_diff_/norm_true_;
    }

    void printSummary()
    {
      std::cout << std::setprecision(16) << std::scientific;
      std::cout << "\t ||b-A*x||               : " << getNormResidual()       << " (residual norm)\n";
      if (memspace_ == ReSolve::memory::DEVICE) {
        std::cout << "\t ||b-A*x|| (CPU)         : " << getNormResidualCpu()    << " (residual norm on CPU)\n";
      }
      std::cout << "\t ||b-A*x||/||b||         : " << getNormResidualScaled() << " (scaled residual norm)\n";
      std::cout << "\t ||x-x_true||            : " << getNormDiff()           << " (solution error)\n";
      std::cout << "\t ||x-x_true||/||x_true|| : " << getNormDiffScaled()     << " (scaled solution error)\n";
      std::cout << "\t ||b-A*x_true||          : " << getNormResidualTrue()   << " (residual norm with exact solution)\n\n";
    }

  private:    
    void setSolutionVector()
    {
      x_true_.allocate(memspace_);
      x_true_.setToConst(static_cast<ReSolve::real_type>(1.0), memspace_);
      x_true_.setDataUpdated(memspace_);
      x_true_.syncData(ReSolve::memory::HOST);
      norm_true_ = norm2(x_true_, memspace_);
    }

    /**
     * @brief Computes residual norm = || A * x - r ||_2
     * 
     * @param[in]     A 
     * @param[in]     x 
     * @param[in,out] r 
     * @return ReSolve::real_type 
     * 
     * @post r is overwritten with residual values
     */
    ReSolve::real_type computeResidualNorm(ReSolve::matrix::Sparse& A,
                                           ReSolve::vector::Vector& x,
                                           ReSolve::vector::Vector& r,
                                           ReSolve::memory::MemorySpace memspace)
    {
      mh_.matvec(&A, &x, &r, &ONE, &MINUSONE, memspace); // r := A * x - r
      return norm2(r, memspace);
    }

    /**
     * @brief Compute vector difference norm = || x - x_true ||_2
     * 
     * @param[in]     x_true 
     * @param[in,out] x 
     * @param[in]     memspace 
     * @return ReSolve::real_type
     * 
     * @post x is overwritten with difference value
     */
    ReSolve::real_type computeDiffNorm(ReSolve::vector::Vector& x_true,
                                       ReSolve::vector::Vector& x,
                                       ReSolve::memory::MemorySpace memspace)
    {
      vh_.axpy(&MINUSONE, &x_true, &x, memspace); // x := -x_true + x
      return norm2(x, memspace);
    }

    ReSolve::real_type norm2(ReSolve::vector::Vector& r,
                             ReSolve::memory::MemorySpace memspace)
    {
      return std::sqrt(vh_.dot(&r, &r, memspace));
    }

  private:
    ReSolve::matrix::Sparse* A_;
    ReSolve::vector::Vector* r_;
    ReSolve::vector::Vector* x_;

    ReSolve::MatrixHandler mh_;
    ReSolve::VectorHandler vh_;

    ReSolve::vector::Vector res_;
    ReSolve::vector::Vector x_true_;

    ReSolve::real_type norm_rhs_{0.0};
    ReSolve::real_type norm_res_{0.0};
    ReSolve::real_type norm_res_cpu_{0.0};
    ReSolve::real_type norm_res_true_{0.0};
    ReSolve::real_type norm_true_{0.0};
    ReSolve::real_type norm_diff_{0.0};

    ReSolve::memory::MemorySpace memspace_{ReSolve::memory::HOST};
};

static int runTest(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  return runTest(argc, argv);
}

int runTest(int argc, char *argv[])
{
  // Use ReSolve data types.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;
  using matrix_type = ReSolve::matrix::Sparse;

  //we want error sum to be 0 at the end
  //that means PASS.
  //otheriwse it is a FAIL.
  int error_sum = 0;
  int status = 0;

  std::cout << "REFACTORING IN PROGRESS!\n";

  ReSolve::LinAlgWorkspaceHIP workspace_HIP;
  workspace_HIP.initializeHandles();
  ReSolve::MatrixHandler matrix_handler(&workspace_HIP);
  ReSolve::VectorHandler vector_handler(&workspace_HIP);

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;

  ReSolve::LinSolverDirectRocSolverRf* Rf = new ReSolve::LinSolverDirectRocSolverRf(&workspace_HIP);
  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  // Read first matrix
  std::ifstream mat1(matrixFileName1);
  if(!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
  A->syncData(ReSolve::memory::DEVICE);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhsFileName1);
  if(!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  vector_type* vec_rhs = new vector_type(A->getNumRows());
  vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->syncData(ReSolve::memory::DEVICE);
  rhs1_file.close();

  // Set reference solution vector (all elements equal to 1)
  vector_type* vec_test = new vector_type(A->getNumRows());
  vec_test->allocate(ReSolve::memory::HOST);
  vec_test->setToConst(1.0, ReSolve::memory::HOST);
  vec_test->setDataUpdated(ReSolve::memory::HOST); // <-- workaround a bug in setToConst
  vec_test->syncData(ReSolve::memory::DEVICE);

  // matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);

  // Compute residual with the exact solution (for control purposes)
  vector_type* vec_r   = new vector_type(A->getNumRows());
  vec_r->allocate(ReSolve::memory::HOST);
  vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  // Compute vec_r := A * vec_test - vec_r
  matrix_handler.matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);
  real_type exactSol_normRmatrix1 = std::sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::HOST));

  // Allocate the solution vector
  vector_type* vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(ReSolve::memory::HOST); //for KLU
  vec_x->allocate(ReSolve::memory::DEVICE);


  // vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  // vec_rhs->syncData(ReSolve::memory::DEVICE);

  // Solve the first system using KLU
  status = KLU->setup(A);
  error_sum += status;

  status = KLU->analyze();
  error_sum += status;

  status = KLU->factorize();
  error_sum += status;

  status = KLU->solve(vec_rhs, vec_x);
  error_sum += status;

  std::cout << "KLU solve status: " << status <<std::endl;      
  TestHelper<ReSolve::LinAlgWorkspaceHIP> th(A, vec_rhs, vec_x, workspace_HIP);

  matrix_type* L = KLU->getLFactor();
  matrix_type* U = KLU->getUFactor();
  if (L == nullptr) {std::cout << "ERROR";}
  index_type* P = KLU->getPOrdering();
  index_type* Q = KLU->getQOrdering();
  vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vec_rhs->setDataUpdated(ReSolve::memory::DEVICE);

  status = Rf->setup(A, L, U, P, Q, vec_rhs); 
  error_sum += status;
  std::cout << "Rf setup status: " << status << std::endl;      

  status = Rf->refactorize();
  error_sum += status;

  // Evaluate residual norm with computed solution
  vec_x->syncData(ReSolve::memory::DEVICE);
  vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  // matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);
  // Compute vec_r := A * vec_x - vec_r
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type normRmatrix1 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  // Norm of the rhs
  real_type normB1 = sqrt(vector_handler.dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));

  //evaluate the residual ON THE CPU using COMPUTED solution
  vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);
  error_sum += status;
  real_type normRmatrix1CPU = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::HOST));

  // Create vec_diff and set all of its elements to 1
  vector_type* vec_diff = new vector_type(A->getNumRows());
  vec_diff->setToConst(1.0, ReSolve::memory::HOST);
  vec_diff->setDataUpdated(ReSolve::memory::HOST); // <-- workaround
  vec_diff->syncData(ReSolve::memory::DEVICE);

<<<<<<< HEAD
  std::cout<<"Results (first matrix): "<<std::endl<<std::endl;
  std::cout << std::scientific << std::setprecision(16);
  std::cout<<"\t ||b-A*x||_2                 : " << normRmatrix1    << " (residual norm)" << std::endl;
  std::cout<<"\t ||b-A*x||_2  (CPU)          : " << normRmatrix1CPU << " (residual norm)" << std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : " << normRmatrix1/normB1   << " (scaled residual norm)"             << std::endl;
  std::cout<<"\t ||x-x_true||_2              : " << normDiffMatrix1       << " (solution error)"                   << std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix1/normXtrue << " (scaled solution error)"        << std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix1 << " (control; residual norm with exact solution)\n\n";
=======
  // Norm of the exact solution
  real_type normXtrue = sqrt(vector_handler.dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));

  // Compute vec_diff := -vec_x + vec_diff
  vector_handler.axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  real_type normDiffMatrix1 = sqrt(vector_handler.dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));

  std::cout << "Results (first matrix): \n\n" << std::setprecision(16) << std::scientific;
  std::cout << "\t ||b-A*x||               : " << normRmatrix1    << " (residual norm)\n";
  std::cout << "\t ||b-A*x||               : " << th.getNormResidual() << " (residual norm)\n";
  std::cout << "\t ||b-A*x|| (CPU)         : " << normRmatrix1CPU << " (residual norm)\n";
  std::cout << "\t ||b-A*x|| (CPU)         : " << th.getNormResidualCpu() << " (residual norm)\n";
  std::cout << "\t ||b-A*x||/||b||         : " << normRmatrix1/normB1   <<             " (scaled residual norm)\n";
  std::cout << "\t ||b-A*x||/||b||         : " << th.getNormResidualScaled()  << " (scaled residual norm)\n";
  std::cout << "\t ||x-x_true||            : " << normDiffMatrix1       <<                   " (solution error)\n";
  std::cout << "\t ||x-x_true||            : " << th.getNormDiff()       <<                   " (solution error)\n";
  std::cout << "\t ||x-x_true||/||x_true|| : " << normDiffMatrix1/normXtrue <<        " (scaled solution error)\n";
  std::cout << "\t ||x-x_true||/||x_true|| : " << th.getNormDiffScaled() <<        " (scaled solution error)\n";
  std::cout << "\t ||b-A*x_true||          : " << exactSol_normRmatrix1 << " (control; residual norm with exact solution)\n";
  std::cout << "\t ||b-A*x_true||          : " << th.getNormResidualTrue() << " (control; residual norm with exact solution)\n\n";
>>>>>>> b21088bb (Functioning test helper prototype.)


  // Load the second matrix
  std::ifstream mat2(matrixFileName2);
  if(!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  A->syncData(ReSolve::memory::DEVICE);
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhsFileName2);
  if(!rhs2_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateArrayFromFile(rhs2_file, &rhs);
  rhs2_file.close();
  vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  // Compute residual with respect to the exact solution
  vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vec_test->setToConst(1.0, ReSolve::memory::DEVICE);
  // Compute vec_r := A * vec_test - vec_r
  matrix_handler.matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE);
  real_type exactSol_normRmatrix2 = std::sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  // this hangs up
  status = Rf->refactorize();
  error_sum += status;
  std::cout << "rocSolverRf refactorization status: " << status << std::endl;      
  
  status = Rf->solve(vec_rhs, vec_x);
  error_sum += status;

  // Compute residual for computed solution
  vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);
  // vec_r := A * vec_x - vec_r
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type normRmatrix2 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  //for testing only - control
  real_type normB2 = sqrt(vector_handler.dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));

  // Compute solution error with respect to the exact solution
  vec_diff->setToConst(1.0, ReSolve::memory::DEVICE);
  vec_diff->setDataUpdated(ReSolve::memory::DEVICE);
  // Compute vec_diff := -vec_x + vec_diff
  vector_handler.axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  real_type normDiffMatrix2 = sqrt(vector_handler.dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));


  std::cout<<"Results (second matrix): "<<std::endl<<std::endl;
  std::cout << std::scientific << std::setprecision(16);
  std::cout<<"\t ||b-A*x||_2                 : "<<normRmatrix2<<" (residual norm)"<<std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : "<<normRmatrix2/normB2<<" (scaled residual norm)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2              : "<<normDiffMatrix2<<" (solution error)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : "<<normDiffMatrix2/normXtrue<<" (scaled solution error)"<<std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : "<<exactSol_normRmatrix2<<" (control; residual norm with exact solution)"<<std::endl<<std::endl;

  if (!std::isfinite(normRmatrix1/normB1) || !std::isfinite(normRmatrix2/normB2)) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((normRmatrix1/normB1 > 1e-16 ) || (normRmatrix2/normB2 > 1e-16)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout << "Test KLU with rocsolverRf refactorization " << GREEN << "PASSED" << CLEAR << std::endl;
  } else {
    std::cout << "Test KLU with rocsolverRf refactorization " << RED << "FAILED" << CLEAR
              << ", error sum: " << error_sum << std::endl;
  }

  //now DELETE
  delete A;
  delete KLU;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  return error_sum;
}

