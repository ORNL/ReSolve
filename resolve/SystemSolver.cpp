#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>

// Available solvers
// #include <resolve/LinSolverDirect.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>

// Available workspaces
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>

// Handlers
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>

// Utilities
#include <resolve/utilities/logger/Logger.hpp>

#include "SystemSolver.hpp"


namespace ReSolve
{
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  SystemSolver::SystemSolver()
  {
    //set defaults:
    factorizationMethod_ = "klu";
    refactorizationMethod_ = "klu";
    solveMethod_ = "klu";
    irMethod_ = "none";
    
    initialize();
  }

  SystemSolver::SystemSolver(LinAlgWorkspaceCUDA* workspace) : workspaceCuda_(workspace)
  {
    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceCuda_);
    vectorHandler_ = new VectorHandler(workspaceCuda_);

    //set defaults:
    factorizationMethod_ = "klu";
    refactorizationMethod_ = "glu";
    solveMethod_ = "glu";
    irMethod_ = "none";
    
    initialize();
  }

  SystemSolver::~SystemSolver()
  {
    //delete the matrix and all the solvers and all their workspace
  }

  int SystemSolver::setMatrix(matrix::Sparse* A)
  {
    A_ = A;
    return 0;
  }

  int SystemSolver::setCudaWorkspace(LinAlgWorkspaceCUDA* workspace)
  {
    workspaceCuda_ = workspace;
    return 0;
  }

  /**
   * @brief Sets up the system solver
   * 
   * This method instantiates components of the system solver based on
   * user inputs. 
   */
  int SystemSolver::initialize()
  {
    // First delete old objects
    if (KLU_)
      delete KLU_;
    if (refactorSolver_)
      delete refactorSolver_;
    
    // Create factorization solver
    if (factorizationMethod_ == "klu") {
      KLU_ = new ReSolve::LinSolverDirectKLU();
      KLU_->setupParameters(1, 0.1, false);
    }

    // Create refactorization solver
    if (refactorizationMethod_ == "klu") {
      // do nothing for now
    } else if (refactorizationMethod_ == "glu") {
      refactorSolver_ = new ReSolve::LinSolverDirectCuSolverGLU(workspaceCuda_);
    } else if (refactorizationMethod_ == "cusolverrf") {
      refactorSolver_ = new ReSolve::LinSolverDirectCuSolverRf();
    } else {
      out::error() << "Refactorization method not recognized ...\n";
      return 1;
    }

    // Create iterative refinement
    // ... some code

    return 0;
  }

  int SystemSolver::analyze()
  {
    if (A_ == nullptr) {
      out::error() << "System matrix not set!\n";
      return 1;
    }

    if (factorizationMethod_ == "klu") {
      // std::cout << "Analysis using KLU ...\n";
      KLU_->setup(A_);
      return KLU_->analyze();
    } 
    return 1;  
  }

  int SystemSolver::factorize()
  {
    if (factorizationMethod_ == "klu") {
      // std::cout << "Factorization using KLU ...\n";

      int status = KLU_->factorize();

      // Get factors and permutation vectors
      L_ = KLU_->getLFactor();
      U_ = KLU_->getUFactor();
      P_ = KLU_->getPOrdering();
      Q_ = KLU_->getQOrdering();

      if (L_ == nullptr) {
        out::warning() << "Factorization failed ...\n";
      }
      return 0;
    } 
    return 1;
  }

  int SystemSolver::refactorize()
  {
    if (refactorizationMethod_ == "klu") {
      return KLU_->refactorize();
    }

    if (refactorizationMethod_ == "glu") {
      // std::cout << "Refactorization using GLU ...\n";
      return refactorSolver_->refactorize();
    }

    return 1;
  }

  int SystemSolver::refactorize_setup()
  {
    if (refactorizationMethod_ == "glu") {
      // std::cout << "Refactorization setup using GLU ...\n";
      return refactorSolver_->setup(A_, L_, U_, P_, Q_);
    } 
    return 1;
  }

  int SystemSolver::solve(vector_type* x, vector_type* rhs)
  {
    if (solveMethod_ == "klu") {
      // std::cout << "Solving with KLU ...\n";
      return KLU_->solve(x, rhs);
    } 

    if (solveMethod_ == "glu") {
      // std::cout << "Solving with GLU ...\n";
      return refactorSolver_->solve(x, rhs);
    } 

    return 1;
  }

  int SystemSolver::refine(vector_type* x, vector_type* rhs)
  {
    return 1;
  }

  void SystemSolver::setFactorizationMethod(std::string method)
  {
    factorizationMethod_ = method;
    // initialize();
  }

  void SystemSolver::setRefactorizationMethod(std::string method)
  {
    refactorizationMethod_ = method;
    // initialize();
  }

  void SystemSolver::setSolveMethod(std::string method)
  {
    solveMethod_ = method;
    // initialize();
  }

  void SystemSolver::setIterativeRefinement(std::string method)
  {
    irMethod_ = method;
    // initialize();
  }

} // namespace ReSolve
