#include <cassert>
#include <cmath>

#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/LinSolverDirectSerialILU0.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>

#ifdef RESOLVE_USE_KLU
#include <resolve/LinSolverDirectKLU.hpp>
#endif

#include <resolve/LinSolverIterativeRandFGMRES.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
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

  SystemSolver::SystemSolver(LinAlgWorkspaceCpu* workspaceCpu,
                             std::string factor,
                             std::string refactor,
                             std::string solve,
                             std::string precond,
                             std::string ir) 
    : workspaceCpu_(workspaceCpu),
      factorizationMethod_(factor),
      refactorizationMethod_(refactor),
      solveMethod_(solve),
      precondition_method_(precond),
      irMethod_(ir)
  {
    if ((refactor != "none") && (precond != "none")) {
      out::warning() << "Incorrect input: "
                     << "Refactorization and preconditioning cannot both be enabled.\n"
                     << "Setting both to 'none' ...\n";
      refactorizationMethod_ = "none";
      precondition_method_   = "none";
    }

    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceCpu_);
    vectorHandler_ = new VectorHandler(workspaceCpu_);

    memspace_ = "cpu";
    
    initialize();
  }

#ifdef RESOLVE_USE_CUDA
  SystemSolver::SystemSolver(LinAlgWorkspaceCUDA*  workspaceCuda, 
                             std::string factor,
                             std::string refactor,
                             std::string solve,
                             std::string precond,
                             std::string ir) 
    : workspaceCuda_(workspaceCuda),
      factorizationMethod_(factor),
      refactorizationMethod_(refactor),
      solveMethod_(solve),
      precondition_method_(precond),
      irMethod_(ir)
  {
    if ((refactor != "none") && (precond != "none")) {
      out::warning() << "Incorrect input: "
                     << "Refactorization and preconditioning cannot both be enabled.\n"
                     << "Setting both to 'none' ...\n";
      refactorizationMethod_ = "none";
      precondition_method_   = "none";
    }

    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceCuda_);
    vectorHandler_ = new VectorHandler(workspaceCuda_);

    memspace_ = "cuda";
    
    initialize();
  }
#endif

#ifdef RESOLVE_USE_HIP
  SystemSolver::SystemSolver(LinAlgWorkspaceHIP*  workspaceHip, 
                             std::string factor,
                             std::string refactor,
                             std::string solve,
                             std::string precond,
                             std::string ir) 
    : workspaceHip_(workspaceHip),
      factorizationMethod_(factor),
      refactorizationMethod_(refactor),
      solveMethod_(solve),
      precondition_method_(precond),
      irMethod_(ir)
  {
    if ((refactor != "none") && (precond != "none")) {
      out::warning() << "Incorrect input: "
                     << "Refactorization and preconditioning cannot both be enabled.\n"
                     << "Setting both to 'none' ...\n";
      refactorizationMethod_ = "none";
      precondition_method_   = "none";
    }

    // Instantiate handlers
    matrixHandler_ = new MatrixHandler(workspaceHip_);
    vectorHandler_ = new VectorHandler(workspaceHip_);

    memspace_ = "hip";
    
    initialize();
  }
#endif

  SystemSolver::~SystemSolver()
  {
    delete resVector_;

    if (factorizationMethod_ != "none") {
      delete factorizationSolver_;
    }

    if (refactorizationMethod_ != "none") {
      delete refactorizationSolver_;
    }

    if (irMethod_ != "none") {
      delete iterativeSolver_;
      delete gs_;
    }

    if (precondition_method_ != "none") {
      delete preconditioner_;
    }

    delete matrixHandler_;
    delete vectorHandler_;
  }

  int SystemSolver::setMatrix(matrix::Sparse* A)
  {
    int status = 0;
    A_ = A;
    resVector_ = new vector_type(A->getNumRows());
    if (memspace_ == "cpu") {
      resVector_->allocate(memory::HOST);
    } else {
      resVector_->allocate(memory::DEVICE);
    }

    // If we use iterative solver, we can set it up here
    if (solveMethod_ == "randgmres") {
      auto* rgmres = dynamic_cast<LinSolverIterativeRandFGMRES*>(iterativeSolver_);
      status += rgmres->setup(A_);
      status += gs_->setup(rgmres->getKrand(), rgmres->getRestart());
    } else if (solveMethod_ == "fgmres") {
      auto* fgmres = dynamic_cast<LinSolverIterativeFGMRES*>(iterativeSolver_);
      status += fgmres->setup(A_);
      status += gs_->setup(A_->getNumRows(), fgmres->getRestart()); 
    } else {
      // do nothing
    }

    return status;
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
    if (factorizationSolver_) {
      delete factorizationSolver_;
      factorizationSolver_ = nullptr;
    }
    if (refactorizationSolver_) {
      delete refactorizationSolver_;
      refactorizationSolver_ = nullptr;
    }
    if (preconditioner_) {
      delete preconditioner_;
      preconditioner_ = nullptr;
    }
    if (iterativeSolver_) {
      delete iterativeSolver_;
      iterativeSolver_ = nullptr;
    }
    if (gs_) {
      delete gs_;
      gs_ = nullptr;
    }
    
    // Create factorization solver
    if (factorizationMethod_ == "none") {
      // do nothing
#ifdef RESOLVE_USE_KLU
    } else if (factorizationMethod_ == "klu") {
      factorizationSolver_ = new ReSolve::LinSolverDirectKLU();
#endif
    } else {
      out::error() << "Unrecognized factorization " << factorizationMethod_ << "\n";
      return 1;
    }

    // Create refactorization solver
    if (refactorizationMethod_ == "none") {
      // do nothing
    } else if (refactorizationMethod_ == "klu") {
      // do nothing for now, KLU is the only factorization solver available
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

    // Create iterative refinement
    if (irMethod_ == "fgmres") {
      setGramSchmidtMethod(gsMethod_);
      iterativeSolver_ = new LinSolverIterativeFGMRES(matrixHandler_,
                                                      vectorHandler_,
                                                      gs_);
    }

    // Create preconditioner
    if (precondition_method_ == "none") {
      // do nothing
    } else if (precondition_method_ == "ilu0") {
      if (memspace_ == "cpu") {
        // preconditioner_ = new LinSolverDirectSerialILU0(workspaceCpu_);
        preconditioner_ = new LinSolverDirectCpuILU0(workspaceCpu_);
#ifdef RESOLVE_USE_CUDA
      } else if (memspace_ == "cuda") {
        preconditioner_ = new LinSolverDirectCuSparseILU0(workspaceCuda_);
#endif
#ifdef RESOLVE_USE_HIP
      } else if (memspace_ == "hip") {
        preconditioner_ = new LinSolverDirectRocSparseILU0(workspaceHip_);
#endif
      } else {
        out::error() << "Memory space " << memspace_
                    << " not recognized ...\n";
        return 1;
      }
    } else {
      out::error() << "Preconditioner method " << precondition_method_ 
                   << " not recognized ...\n";
      return 1;
    }

    // Create iterative solver
    if (solveMethod_ == "randgmres") {
      LinSolverIterativeRandFGMRES::SketchingMethod sketch;
      if (sketching_method_ == "count") {
        sketch = LinSolverIterativeRandFGMRES::cs;
      } else if (sketching_method_ == "fwht") {
        sketch = LinSolverIterativeRandFGMRES::fwht;
      } else {
        out::warning() << "Sketching method " << sketching_method_ << " not recognized!\n"
                       << "Using default.\n";
        sketch = LinSolverIterativeRandFGMRES::cs;
      }
      setGramSchmidtMethod(gsMethod_);
      iterativeSolver_ = new LinSolverIterativeRandFGMRES(matrixHandler_,
                                                          vectorHandler_,
                                                          sketch,
                                                          gs_);
    } else if (solveMethod_ == "fgmres") {
      setGramSchmidtMethod(gsMethod_);
      iterativeSolver_ = new LinSolverIterativeFGMRES(matrixHandler_,
                                                      vectorHandler_,
                                                      gs_);
    } else {
      // do nothing
    }

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

    if (refactorizationMethod_ == "glu" || 
        refactorizationMethod_ == "cusolverrf" || 
        refactorizationMethod_ == "rocsolverrf") {
      return refactorizationSolver_->refactorize();
    }

    return 1;
  }

  /**
   * @brief Sets up refactorization.
   * 
   * Extracts factors and permutation vectors from the factorization solver
   * and configures selected refactorization solver. Also configures iterative
   * refinement, if it is enabled by the user.
   * 
   * Also sets flag `is_solve_on_device_` to true signaling to a triangular
   * solver to run on GPU.
   * 
   * @pre Factorization solver exists and provides access to L and U factors,
   * as well as left and right permutation vectors P and Q. Since KLU is
   * the only factorization solver available through ReSolve, the factors are
   * expected in CSC format.
   * 
   * @return int 0 if successful, 1 if it fails
   */
  int SystemSolver::refactorizationSetup()
  {
    int status = 0;
    // Get factors and permutation vectors
    L_ = factorizationSolver_->getLFactor();
    U_ = factorizationSolver_->getUFactor();
    P_ = factorizationSolver_->getPOrdering();
    Q_ = factorizationSolver_->getQOrdering();

    if (L_ == nullptr) {
      out::error() << "Factorization failed, cannot extract factors ...\n";
      status += 1;
    }

#ifdef RESOLVE_USE_CUDA
    if (refactorizationMethod_ == "glu") {
      isSolveOnDevice_ = true;
      status += refactorizationSolver_->setup(A_, L_, U_, P_, Q_);
    }
    if (refactorizationMethod_ == "cusolverrf") {
      matrix::Csc* L_csc = dynamic_cast<matrix::Csc*>(L_);
      matrix::Csc* U_csc = dynamic_cast<matrix::Csc*>(U_);       
      matrix::Csr* L_csr = new matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
      matrix::Csr* U_csr = new matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
      matrixHandler_->csc2csr(L_csc, L_csr, memory::DEVICE);
      matrixHandler_->csc2csr(U_csc, U_csr, memory::DEVICE);
      status += refactorizationSolver_->setup(A_, L_csr, U_csr, P_, Q_);
      delete L_csr;
      delete U_csr;

      LinSolverDirectCuSolverRf* Rf = dynamic_cast<LinSolverDirectCuSolverRf*>(refactorizationSolver_);
      Rf->setNumericalProperties(1e-14, 1e-1);

      isSolveOnDevice_ = true;
    }
#endif

#ifdef RESOLVE_USE_HIP
    if (refactorizationMethod_ == "rocsolverrf") {
      isSolveOnDevice_ = true;
      auto* Rf = dynamic_cast<LinSolverDirectRocSolverRf*>(refactorizationSolver_);
      Rf->setSolveMode(1);
      status += refactorizationSolver_->setup(A_, L_, U_, P_, Q_, resVector_);
    }
#endif

    if (irMethod_ == "fgmres") {
      gs_->setup(A_->getNumRows(), iterativeSolver_->getRestart()); 
      status += iterativeSolver_->setup(A_);
      status += iterativeSolver_->setupPreconditioner("LU", refactorizationSolver_);
    }

    return status;
  }

  /**
   * @brief Calls triangular solver
   * 
   * @param[in]  rhs - Right-hand-side vector of the system
   * @param[out] x   - Solution vector (will be overwritten)
   * @return int status of factorization
   * 
   * @pre Factorization or refactorization has been performed and triangular
   * factors are available. Alternatively, a Krylov solver has been set up.
   * 
   * @todo Make `rhs` a constant vector
   * @todo Need to use `enum`s and `switch` statements here or implement as PIMPL
   */
  int SystemSolver::solve(vector_type* rhs, vector_type* x)
  {
    int status = 0;

    // Use Krylov solver if selected
    if (solveMethod_ == "randgmres" || solveMethod_ == "fgmres") {
      status += iterativeSolver_->resetMatrix(A_);
      status += iterativeSolver_->solve(rhs, x);
      return status;
    }

    if (solveMethod_ == "klu") {
      status += factorizationSolver_->solve(rhs, x);
    } 

    if (solveMethod_ == "glu" || solveMethod_ == "cusolverrf" || solveMethod_ == "rocsolverrf") {
      if (isSolveOnDevice_) {
        status += refactorizationSolver_->solve(rhs, x);
      } else {
        status += factorizationSolver_->solve(rhs, x);
      }
    } 

    if (irMethod_ == "fgmres") {
      if (isSolveOnDevice_) {
        status += refine(rhs, x);
      }
    }
    return status;
  }

  int SystemSolver::preconditionerSetup()
  {
    int status = 0;
    if (precondition_method_ == "ilu0") {
      status += preconditioner_->setup(A_);
      if (memspace_ != "cpu") {
        isSolveOnDevice_ = true;
      }
      iterativeSolver_->setupPreconditioner("LU", preconditioner_);
    }

    return status;
  }

  int SystemSolver::refine(vector_type* rhs, vector_type* x)
  {
    int status = 0;

    status += iterativeSolver_->resetMatrix(A_);
    status += iterativeSolver_->solve(rhs, x);

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
  }

  /**
   * @brief Sets refactorization method to use
   * 
   * @param[in] method - ID for the refactorization method
   * 
   * @post Destroys whatever refactorization solver existed before
   * and sets `refactorization_solver_` pointer to the new
   * refactorization object. Sets refactorization method ID
   * to the value in input parameter `method`.
   */
  void SystemSolver::setRefactorizationMethod(std::string method)
  {
    refactorizationMethod_ = method;
    if (refactorizationSolver_) {
      delete refactorizationSolver_;
      refactorizationSolver_ = nullptr;
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
    }
  }

  /**
   * @brief Sets solve method
   * 
   * @param[in] method - ID of the solve method
   * 
   */
  int SystemSolver::setSolveMethod(std::string method)
  {
    solveMethod_ = method;

    // Remove existing iterative solver and set IR to "none".
    irMethod_ = "none";
    if (iterativeSolver_)
      delete iterativeSolver_;

    if (method == "randgmres") {
      LinSolverIterativeRandFGMRES::SketchingMethod sketch;
      if (sketching_method_ == "count") {
        sketch = LinSolverIterativeRandFGMRES::cs;
      } else if (sketching_method_ == "fwht") {
        sketch = LinSolverIterativeRandFGMRES::fwht;
      } else {
        out::warning() << "Sketching method " << sketching_method_ << " not recognized!\n"
                       << "Using default.\n";
        sketch = LinSolverIterativeRandFGMRES::cs;
      }

      setGramSchmidtMethod(gsMethod_);
      iterativeSolver_ = new LinSolverIterativeRandFGMRES(matrixHandler_,
                                                          vectorHandler_,
                                                          sketch,
                                                          gs_);
    } else if (solveMethod_ == "fgmres") {
      setGramSchmidtMethod(gsMethod_);
      iterativeSolver_ = new LinSolverIterativeFGMRES(matrixHandler_,
                                                      vectorHandler_,
                                                      gs_);
    } else {
      out::error() << "Solve method " << solveMethod_ 
                   << " not recognized ...\n";
      return 1;
    }
    return 0;
  }

  /**
   * @brief Sets iterative refinement method and related orthogonalization. 
   * 
   * @param[in] method   - string ID for the iterative refinement method
   * @param[in] gsMethod - string ID for the orthogonalization method to be used
   * 
   * @todo Iterative refinement temporarily disabled on CPU. Need to fix that.
   */
  void SystemSolver::setRefinementMethod(std::string method, std::string gsMethod)
  {
    if (iterativeSolver_ != nullptr)
      delete iterativeSolver_;

    if (gs_ != nullptr)
      delete gs_;
    
    if (method == "none")
      return;

    if (memspace_ == "cpu") {
      method = "none";
      out::warning() << "Iterative refinement not supported on CPU. "
                     << "Turning off ...\n";
      return;
    }

    gsMethod_ = gsMethod;

    if (method == "fgmres") {
      setGramSchmidtMethod(gsMethod);
      iterativeSolver_ = new LinSolverIterativeFGMRES(matrixHandler_,
                                                      vectorHandler_,
                                                      gs_);
      irMethod_ = method;
    } else {
      out::error() << "Iterative refinement method " << method << " not recognized.\n";
    }
  }

  real_type SystemSolver::getVectorNorm(vector_type* rhs)
  {
    using namespace ReSolve::constants;
    real_type norm_b  = 0.0;
    if (memspace_ == "cpu") {
      norm_b = std::sqrt(vectorHandler_->dot(rhs, rhs, memory::HOST));
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    } else if (memspace_ == "cuda" || memspace_ == "hip") {
      if (isSolveOnDevice_) {
        norm_b = std::sqrt(vectorHandler_->dot(rhs, rhs, memory::DEVICE));
      } else {
        norm_b = std::sqrt(vectorHandler_->dot(rhs, rhs, memory::HOST));
      }
#endif
    } else {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    return norm_b;
  }

  real_type SystemSolver::getResidualNorm(vector_type* rhs, vector_type* x)
  {
    using namespace ReSolve::constants;
    assert(rhs->getSize() == resVector_->getSize());
    real_type norm_b  = 0.0;
    real_type resnorm = 0.0;
    memory::MemorySpace ms = memory::HOST;
    if (memspace_ == "cpu") {
      resVector_->update(rhs, memory::HOST, memory::HOST);
      norm_b = std::sqrt(vectorHandler_->dot(resVector_, resVector_, memory::HOST));
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    } else if (memspace_ == "cuda" || memspace_ == "hip") {
      if (isSolveOnDevice_) {
        resVector_->update(rhs, memory::DEVICE, memory::DEVICE);
        norm_b = std::sqrt(vectorHandler_->dot(resVector_, resVector_, memory::DEVICE));
      } else {
        resVector_->update(rhs, memory::HOST, memory::DEVICE);
        norm_b = std::sqrt(vectorHandler_->dot(resVector_, resVector_, memory::HOST));
        // ms = memory::HOST;
      }
      ms = memory::DEVICE;
#endif
    } else {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    matrixHandler_->setValuesChanged(true, ms);
    matrixHandler_->matvec(A_, x, resVector_, &ONE, &MINUSONE, "csr", ms);
    resnorm = std::sqrt(vectorHandler_->dot(resVector_, resVector_, ms));
    return resnorm/norm_b;
  }

  real_type SystemSolver::getNormOfScaledResiduals(vector_type* rhs, vector_type* x)
  {
    using namespace ReSolve::constants;
    assert(rhs->getSize() == resVector_->getSize());
    real_type norm_x  = 0.0;
    real_type norm_A  = 0.0;
    real_type resnorm = 0.0;
    memory::MemorySpace ms = memory::HOST;
    if (memspace_ == "cpu") {
      resVector_->update(rhs, memory::HOST, memory::HOST);
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    } else if (memspace_ == "cuda" || memspace_ == "hip") {
      if (isSolveOnDevice_) {
        resVector_->update(rhs, memory::DEVICE, memory::DEVICE);
      } else {
        resVector_->update(rhs, memory::HOST, memory::DEVICE);
      }
      ms = memory::DEVICE;
#endif
    } else {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    matrixHandler_->setValuesChanged(true, ms);
    matrixHandler_->matvec(A_, x, resVector_, &ONE, &MINUSONE, "csr", ms);
    resnorm = vectorHandler_->infNorm(resVector_, ms);
    norm_x  = vectorHandler_->infNorm(x, ms);
    matrixHandler_->matrixInfNorm(A_, &norm_A, ms);
    return resnorm / (norm_x * norm_A);
  }

  const std::string SystemSolver::getFactorizationMethod() const
  {
    return factorizationMethod_;
  }

  /**
   * @brief Select sketching method for randomized solvers
   * 
   * This is a brute force method that will delete randomized GMRES solver
   * only to change its sketching function.
   * 
   * @todo This needs to be moved to LinSolverIterative class and accessed from there.
   * 
   * @param[in] sketching_method - string ID of the sketching method
   */
  int SystemSolver::setSketchingMethod(std::string sketching_method)
  {
    if (solveMethod_ != "randgmres") {
      out::warning() << "Trying to set sketching method to an incompatible solver.\n";
      out::warning() << "The setting will be ignored.\n";
      return 1;
    }
    if (sketching_method_ != sketching_method) {
      // For now use a brute force solution and just delete existing iterative solver
      if (iterativeSolver_) {
        delete iterativeSolver_;
        iterativeSolver_ = nullptr;
      }
      sketching_method_ = sketching_method;
      setSolveMethod("randgmres");
    }
    return 0;
  }

  //
  // Private methods
  //

  int SystemSolver::setGramSchmidtMethod(std::string gsMethod)
  {
    if (gs_ != nullptr) {
      gs_->setVariant(gsMethod);
      return 0;
    }

    if (gsMethod == "cgs2") {
      gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::cgs2);
    } else if (gsMethod == "mgs") {
      gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::mgs);
    } else if (gsMethod == "mgs_two_synch") {
      gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::mgs_two_synch);
    } else if (gsMethod == "mgs_pm") {
      gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::mgs_pm);
    } else if (gsMethod == "cgs1") {
      gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::cgs1);
    } else {
      out::warning() << "Gram-Schmidt variant " << gsMethod_ << " not recognized.\n";
      out::warning() << "Using default cgs2 Gram-Schmidt variant.\n";
      gs_ = new GramSchmidt(vectorHandler_, GramSchmidt::cgs2);
      gsMethod_ = "cgs2";
    }

    return 0;
  }

} // namespace ReSolve
