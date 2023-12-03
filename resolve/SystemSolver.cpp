#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>

#include <resolve/LinSolverDirectKLU.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/GramSchmidt.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/GramSchmidt.hpp>
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
    memspace_ = "cpu";
    factorizationMethod_ = "klu";
    refactorizationMethod_ = "klu";
    solveMethod_ = "klu";
    irMethod_ = "none";
    
    initialize();
  }

#ifdef RESOLVE_USE_CUDA
  SystemSolver::SystemSolver(LinAlgWorkspaceCUDA* workspace, std::string ir) : workspaceCuda_(workspace), irMethod_(ir)
  {
    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceCuda_);
    vectorHandler_ = new VectorHandler(workspaceCuda_);

    //set defaults:
    memspace_ = "cuda";
    factorizationMethod_ = "klu";
    refactorizationMethod_ = "glu";
    solveMethod_ = "glu";
    // irMethod_ = "none";
    gsMethod_ = "cgs2";
    
    initialize();
  }
#endif

#ifdef RESOLVE_USE_HIP
  SystemSolver::SystemSolver(LinAlgWorkspaceHIP* workspace, std::string ir) : workspaceHip_(workspace), irMethod_(ir)
  {
    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceHip_);
    vectorHandler_ = new VectorHandler(workspaceHip_);

    //set defaults:
    memspace_ = "hip";
    factorizationMethod_ = "klu";
    refactorizationMethod_ = "rocsolverrf";
    solveMethod_ = "rocsolverrf";
    // irMethod_ = "none";
    gsMethod_ = "cgs2";
    
    initialize();
  }
#endif

  SystemSolver::~SystemSolver()
  {
    delete resVector_;
    delete factorizationSolver_;
    delete refactorizationSolver_;
    if (irMethod_ != "none") {
      delete iterativeSolver_;
      delete gs_;
    }

    delete matrixHandler_;
    delete vectorHandler_;
  }

  int SystemSolver::setMatrix(matrix::Sparse* A)
  {
    A_ = A;
    resVector_ = new vector_type(A->getNumRows());
    if (memspace_ == "cpu") {
      resVector_->allocate(memory::HOST);
    } else {
      resVector_->allocate(memory::DEVICE);
    }
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
    if (factorizationSolver_)
      delete factorizationSolver_;
    if (refactorizationSolver_)
      delete refactorizationSolver_;
    
    // Create factorization solver
    if (factorizationMethod_ == "klu") {
      factorizationSolver_ = new ReSolve::LinSolverDirectKLU();
    } else {
      out::error() << "Unrecognized factorization " << factorizationMethod_ << "\n";
      return 1;
    }

    // Create refactorization solver
    if (refactorizationMethod_ == "klu") {
      // do nothing for now
#ifdef RESOLVE_USE_CUDA
    } else if (refactorizationMethod_ == "glu") {
      refactorizationSolver_ = new ReSolve::LinSolverDirectCuSolverGLU(workspaceCuda_);
    } else if (refactorizationMethod_ == "cusolverrf") {
      refactorizationSolver_ = new ReSolve::LinSolverDirectCuSolverRf();
#endif
#ifdef RESOLVE_USE_HIP
    } else if (refactorizationMethod_ == "rocsolverrf") {
      refactorizationSolver_ = new ReSolve::LinSolverDirectRocSolverRf(workspaceHip_);
#endif
    } else {
      out::error() << "Refactorization method " << refactorizationMethod_ 
                   << " not recognized ...\n";
      return 1;
    }

#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    if (irMethod_ == "fgmres") {
      // GramSchmidt::GSVariant variant;
      if (gsMethod_ == "cgs2") {
        gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::cgs2);
      } else if (gsMethod_ == "mgs") {
        gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::mgs);
      } else if (gsMethod_ == "mgs_two_synch") {
        gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::mgs_two_synch);
      } else if (gsMethod_ == "mgs_pm") {
        gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::mgs_pm);
      } else if (gsMethod_ == "cgs1") {
        gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::cgs1);
      } else {
        out::warning() << "Gram-Schmidt variant " << gsMethod_ << " not recognized.\n";
        gs_ = nullptr;
      }

      iterativeSolver_ = new LinSolverIterativeFGMRES(irRestart_,
                                                      irTol_,
                                                      irMaxit_,
                                                      irConvCond_,
                                                      matrixHandler_,
                                                      vectorHandler_,
                                                      gs_,
                                                      memspace_);
    }
#endif

    return 0;
  }

  int SystemSolver::analyze()
  {
    if (A_ == nullptr) {
      out::error() << "System matrix not set!\n";
      return 1;
    }

    if (factorizationMethod_ == "klu") {
      factorizationSolver_->setup(A_);
      return factorizationSolver_->analyze();
    } 
    return 1;  
  }

  int SystemSolver::factorize()
  {
    if (factorizationMethod_ == "klu") {
      return factorizationSolver_->factorize();
    } 
    return 1;
  }

  int SystemSolver::refactorize()
  {
    if (refactorizationMethod_ == "klu") {
      return factorizationSolver_->refactorize();
    }

#ifdef RESOLVE_USE_CUDA
    if (refactorizationMethod_ == "glu") {
      return refactorizationSolver_->refactorize();
    }
#endif

#ifdef RESOLVE_USE_HIP
    if (refactorizationMethod_ == "rocsolverrf") {
      return refactorizationSolver_->refactorize();
    }
#endif

    return 1;
  }

  int SystemSolver::refactorizationSetup()
  {
    int status = 0;
    // Get factors and permutation vectors
    L_ = factorizationSolver_->getLFactor();
    U_ = factorizationSolver_->getUFactor();
    P_ = factorizationSolver_->getPOrdering();
    Q_ = factorizationSolver_->getQOrdering();

    if (L_ == nullptr) {
      out::warning() << "Factorization failed ...\n";
      status = 1;
    }

#ifdef RESOLVE_USE_CUDA
    if (refactorizationMethod_ == "glu") {
      isSolveOnDevice_ = true;
      status += refactorizationSolver_->setup(A_, L_, U_, P_, Q_);
    }
#endif

#ifdef RESOLVE_USE_HIP
    if (refactorizationMethod_ == "rocsolverrf") {
      std::cout << "Refactorization setup using rocsolverRf ...\n";
      isSolveOnDevice_ = true;
      auto* Rf = dynamic_cast<LinSolverDirectRocSolverRf*>(refactorizationSolver_);
      Rf->setSolveMode(1);
      status += refactorizationSolver_->setup(A_, L_, U_, P_, Q_, resVector_);
    }
#endif

#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    if (irMethod_ == "fgmres") {
      std::cout << "Setting up FGMRES ...\n";
      gs_->setup(A_->getNumRows(), iterativeSolver_->getRestart()); 
      status += iterativeSolver_->setup(A_); 
    }
#endif

    return status;
  }

  // TODO: First argument in solve function is rhs (const vector) and second is the solution (overwritten)
  int SystemSolver::solve(vector_type* rhs, vector_type* x)
  {
    int status = 1;
    if (solveMethod_ == "klu") {
      status = factorizationSolver_->solve(rhs, x);
    } 

#ifdef RESOLVE_USE_CUDA
    if (solveMethod_ == "glu") {
      if (isSolveOnDevice_) {
        // std::cout << "Solving with GLU ...\n";
        status = refactorizationSolver_->solve(rhs, x);
      } else {
        // std::cout << "Solving with KLU ...\n";
        status = factorizationSolver_->solve(rhs, x);
      }
    } 
#endif

#ifdef RESOLVE_USE_HIP
    if (solveMethod_ == "rocsolverrf") {
      if (isSolveOnDevice_) {
        // std::cout << "Solving with RocSolver ...\n";
        status = refactorizationSolver_->solve(rhs, x);
      } else {
        // std::cout << "Solving with KLU ...\n";
        status = factorizationSolver_->solve(rhs, x);
      }     
    } 
#endif

    return status;
  }

  int SystemSolver::precondition()
  {
    // Not implemented yet
    return 1;
  }

  int SystemSolver::preconditionerSetup()
  {
    // Not implemented yet
    return 1;
  }

  int SystemSolver::refine(vector_type* rhs, vector_type* x)
  {
    int status = 0;
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    status += iterativeSolver_->resetMatrix(A_);
    status += iterativeSolver_->setupPreconditioner("LU", refactorizationSolver_);
    status += iterativeSolver_->solve(rhs, x);
#endif
    return status;
  }

  LinSolverDirect& SystemSolver::getFactorizationSolver()
  {
    return *factorizationSolver_;
  }

  LinSolverDirect& SystemSolver::getRefactorizationSolver()
  {
    return *refactorizationSolver_;
  }

  LinSolverIterative& SystemSolver::getIterativeSolver()
  {
    return *iterativeSolver_;
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

  void SystemSolver::setRefinementMethod(std::string method)
  {
    irMethod_ = method;
    // initialize();
  }

  real_type SystemSolver::getResidualNorm(vector_type* rhs, vector_type* x)
  {
    using namespace ReSolve::constants;
    assert(rhs->getSize() == resVector_->getSize());
    real_type norm_b = 0.0;

    if (memspace_ == "cpu") {
      resVector_->update(rhs, memory::HOST, memory::HOST);
      matrixHandler_->setValuesChanged(true, "cpu");
#ifdef RESOLVE_USE_CUDA
    } else if (memspace_ == "cuda") {
      if (isSolveOnDevice_) {
        resVector_->update(rhs, memory::DEVICE, memory::DEVICE);
      } else {
        resVector_->update(rhs, memory::HOST, memory::DEVICE);
      }
      matrixHandler_->setValuesChanged(true, "cuda");
#endif
#ifdef RESOLVE_USE_HIP
    } else if (memspace_ == "hip") {
      if (isSolveOnDevice_) {
        resVector_->update(rhs, memory::DEVICE, memory::DEVICE);
      } else {
        resVector_->update(rhs, memory::HOST, memory::DEVICE);
      }
      matrixHandler_->setValuesChanged(true, "hip");
#endif
    } else {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    norm_b = std::sqrt(vectorHandler_->dot(resVector_, resVector_, memspace_));
    matrixHandler_->matvec(A_, x, resVector_, &ONE, &MINUSONE, "csr", memspace_);
    real_type resnorm = std::sqrt(vectorHandler_->dot(resVector_, resVector_, memspace_));
    return resnorm/norm_b;
  }

  real_type SystemSolver::getInitResidualNorm()
  {
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    return iterativeSolver_->getInitResidualNorm();
#endif
  }

  real_type SystemSolver::getFinalResidualNorm()
  {
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    return iterativeSolver_->getFinalResidualNorm();
#endif
  }

  int SystemSolver::getNumIter()
  {
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    return iterativeSolver_->getNumIter();
#endif
  }

  const std::string SystemSolver::getFactorizationMethod() const
  {
    return factorizationMethod_;
  }

  void SystemSolver::setMaxIterations(int maxIter)
  {
    iterativeSolver_->setMaxit(maxIter);
  }

  void SystemSolver::setIterationsRestart(int restart)
  {
    iterativeSolver_->setRestart(restart);
  }

} // namespace ReSolve
