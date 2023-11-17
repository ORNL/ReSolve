#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>

#include <resolve/LinSolverDirectKLU.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#endif

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

#ifdef RESOLVE_USE_CUDA
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
#endif

#ifdef RESOLVE_USE_HIP
  SystemSolver::SystemSolver(LinAlgWorkspaceHIP* workspace) : workspaceHip_(workspace)
  {
    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceHip_);
    vectorHandler_ = new VectorHandler(workspaceHip_);

    //set defaults:
    factorizationMethod_ = "klu";
    refactorizationMethod_ = "rocsolverrf";
    solveMethod_ = "rocsolverrf";
    irMethod_ = "none";
    
    initialize();
  }
#endif

  SystemSolver::~SystemSolver()
  {
    delete dummy_;
    //delete the matrix and all the solvers and all their workspace
  }

  int SystemSolver::setMatrix(matrix::Sparse* A)
  {
    A_ = A;
    dummy_ = new vector_type(A->getNumRows());
    return 0;
  }

#ifdef RESOLVE_USE_CUDA
  int SystemSolver::setCudaWorkspace(LinAlgWorkspaceCUDA* workspace)
  {
    workspaceCuda_ = workspace;
    return 0;
  }
#endif

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
#ifdef RESOLVE_USE_CUDA
    } else if (refactorizationMethod_ == "glu") {
      refactorSolver_ = new ReSolve::LinSolverDirectCuSolverGLU(workspaceCuda_);
    } else if (refactorizationMethod_ == "cusolverrf") {
      refactorSolver_ = new ReSolve::LinSolverDirectCuSolverRf();
#endif
#ifdef RESOLVE_USE_HIP
    } else if (refactorizationMethod_ == "rocsolverrf") {
      refactorSolver_ = new ReSolve::LinSolverDirectRocSolverRf(workspaceHip_);
#endif
    } else {
      out::error() << "Refactorization method " << refactorizationMethod_ 
                   << " not recognized ...\n";
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

      return 0;
    } 
    return 1;
  }

  int SystemSolver::refactorize()
  {
    if (refactorizationMethod_ == "klu") {
      return KLU_->refactorize();
    }

#ifdef RESOLVE_USE_CUDA
    if (refactorizationMethod_ == "glu") {
      // std::cout << "Refactorization using GLU ...\n";
      return refactorSolver_->refactorize();
    }
#endif

#ifdef RESOLVE_USE_HIP
    if (refactorizationMethod_ == "rocsolverrf") {
      std::cout << "Refactorization using RocSolver ...\n";
      return refactorSolver_->refactorize();
    }
#endif

    return 1;
  }

  int SystemSolver::refactorize_setup(vector_type* rhs)
  {
      // Get factors and permutation vectors
      L_ = KLU_->getLFactor();
      U_ = KLU_->getUFactor();
      P_ = KLU_->getPOrdering();
      Q_ = KLU_->getQOrdering();

      if (L_ == nullptr) {
        out::warning() << "Factorization failed ...\n";
      }
#ifdef RESOLVE_USE_CUDA
    if (refactorizationMethod_ == "glu") {
      // std::cout << "Refactorization setup using GLU ...\n";
      isSolveOnDevice_ = true;
      return refactorSolver_->setup(A_, L_, U_, P_, Q_);
    }
#endif

#ifdef RESOLVE_USE_HIP
    if (refactorizationMethod_ == "rocsolverrf") {
      std::cout << "Refactorization setup using rocsolverRf ...\n";
      isSolveOnDevice_ = true;
      auto* Rf = dynamic_cast<LinSolverDirectRocSolverRf*>(refactorSolver_);
      Rf->setSolveMode(1);
//rhs->copyData(memory::HOST, memory::DEVICE);
     return refactorSolver_->setup(A_, L_, U_, P_, Q_, rhs);
    }
#endif
    return 1;
  }

  int SystemSolver::solve(vector_type* x, vector_type* rhs)
  {
    if (solveMethod_ == "klu") {
      // std::cout << "Solving with KLU ...\n";
      return KLU_->solve(x, rhs);
    } 

#ifdef RESOLVE_USE_CUDA
    if (solveMethod_ == "glu") {
      // std::cout << "Solving with GLU ...\n";
      return refactorSolver_->solve(x, rhs);
    } 
#endif

#ifdef RESOLVE_USE_HIP
    if (solveMethod_ == "rocsolverrf") {
      if (isSolveOnDevice_) {
        std::cout << "Solving with RocSolver ...\n";
        return refactorSolver_->solve(x, rhs);
      } else {
        std::cout << "Solving with KLU ...\n";
        return KLU_->solve(x, rhs);
      }
      
    } 
#endif

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
