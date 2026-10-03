#include <cassert>
#include <cmath>

#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/PreconditionerLU.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>

#ifdef RESOLVE_USE_KLU
#include <resolve/LinSolverDirectKLU.hpp>
#endif
#include <resolve/LinSolverIterativeRandFGMRES.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#ifdef RESOLVE_USE_CUDSS
#include <resolve/LinSolverDirectCuDssRf.hpp>
#endif
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

// Handlers
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>

// Utilities
#include "SystemSolver.hpp"
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve
{
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  // Helpers below map user-facing string IDs to solver enums. They are kept
  // in an anonymous namespace (still inside ReSolve) so they have internal
  // linkage: they are implementation details of SystemSolver, are not part
  // of the library's exported symbols, and cannot collide with same-named
  // functions in other translation units. If these mappings become useful
  // elsewhere, move them to the classes that own the enums (GramSchmidt and
  // LinSolverIterativeRandFGMRES) as public static methods.
  namespace
  {
    /// Maps string ID to the sketching method enum; warns and defaults to count sketch.
    LinSolverIterativeRandFGMRES::SketchingMethod sketchingMethodFromString(const std::string& method)
    {
      if (method == "count")
      {
        return LinSolverIterativeRandFGMRES::cs;
      }
      if (method == "fwht")
      {
        return LinSolverIterativeRandFGMRES::fwht;
      }
      out::warning() << "Sketching method " << method << " not recognized!\n"
                     << "Using default (count sketch).\n";
      return LinSolverIterativeRandFGMRES::cs;
    }

    /// Maps string ID to the Gram-Schmidt variant enum; warns and defaults to CGS2.
    GramSchmidt::GSVariant gsVariantFromString(const std::string& variant)
    {
      if (variant == "cgs2")
      {
        return GramSchmidt::CGS2;
      }
      if (variant == "mgs")
      {
        return GramSchmidt::MGS;
      }
      if (variant == "mgs_two_sync")
      {
        return GramSchmidt::MGS_TWO_SYNC;
      }
      if (variant == "mgs_pm")
      {
        return GramSchmidt::MGS_PM;
      }
      if (variant == "cgs1")
      {
        return GramSchmidt::CGS1;
      }
      out::warning() << "Gram-Schmidt variant " << variant << " not recognized.\n";
      out::warning() << "Using default CGS2 Gram-Schmidt variant.\n";
      return GramSchmidt::CGS2;
    }

    /// Canonical string ID for a Gram-Schmidt variant (inverse of gsVariantFromString).
    const char* gsVariantName(GramSchmidt::GSVariant variant)
    {
      switch (variant)
      {
      case GramSchmidt::CGS2:
        return "cgs2";
      case GramSchmidt::MGS:
        return "mgs";
      case GramSchmidt::MGS_TWO_SYNC:
        return "mgs_two_sync";
      case GramSchmidt::MGS_PM:
        return "mgs_pm";
      case GramSchmidt::CGS1:
        return "cgs1";
      }
      return "cgs2";
    }

    /// Memory space ID associated with each workspace type.
    const char* memorySpaceName(LinAlgWorkspaceCpu*)
    {
      return "cpu";
    }
#ifdef RESOLVE_USE_CUDA
    const char* memorySpaceName(LinAlgWorkspaceCUDA*)
    {
      return "cuda";
    }
#endif
#ifdef RESOLVE_USE_HIP
    const char* memorySpaceName(LinAlgWorkspaceHIP*)
    {
      return "hip";
    }
#endif
  } // namespace

  /**
   * @brief Shared tail of all constructors.
   *
   * Creates matrix and vector handlers for the given workspace, derives the
   * memory space ID from the workspace type and instantiates solver
   * components via `initialize()`.
   *
   * @tparam Workspace - one of LinAlgWorkspaceCpu, LinAlgWorkspaceCUDA, LinAlgWorkspaceHIP
   * @param[in] workspace - pointer to the workspace (not owned)
   */
  template <class Workspace>
  void SystemSolver::completeSetup(Workspace* workspace)
  {
    matrix_handler_.reset(new MatrixHandler(workspace));
    vector_handler_.reset(new VectorHandler(workspace));
    memspace_ = memorySpaceName(workspace);

    if (initialize() != 0)
    {
      out::error() << "SystemSolver initialization failed with factorization '" << factorization_method_
                   << "', refactorization '" << refactorization_method_
                   << "', solve '" << solve_method_
                   << "', preconditioner '" << precondition_method_
                   << "', iterative refinement '" << ir_method_
                   << "'. Solver is not usable in this state.\n";
    }
  }

  SystemSolver::SystemSolver(LinAlgWorkspaceCpu* workspace_cpu,
                             std::string         factor,
                             std::string         refactor,
                             std::string         solve,
                             std::string         precond,
                             std::string         ir)
    : workspace_cpu_(workspace_cpu),
      factorization_method_(factor),
      refactorization_method_(refactor),
      solve_method_(solve),
      precondition_method_(precond),
      ir_method_(ir)
  {
    completeSetup(workspace_cpu_);
  }

#ifdef RESOLVE_USE_CUDA
  SystemSolver::SystemSolver(LinAlgWorkspaceCUDA* workspace_cuda,
                             std::string          factor,
                             std::string          refactor,
                             std::string          solve,
                             std::string          precond,
                             std::string          ir)
    : workspace_cuda_(workspace_cuda),
      factorization_method_(factor),
      refactorization_method_(refactor),
      solve_method_(solve),
      precondition_method_(precond),
      ir_method_(ir)
  {
    completeSetup(workspace_cuda_);
  }
#endif

#ifdef RESOLVE_USE_HIP
  SystemSolver::SystemSolver(LinAlgWorkspaceHIP* workspace_hip,
                             std::string         factor,
                             std::string         refactor,
                             std::string         solve,
                             std::string         precond,
                             std::string         ir)
    : workspace_hip_(workspace_hip),
      factorization_method_(factor),
      refactorization_method_(refactor),
      solve_method_(solve),
      precondition_method_(precond),
      ir_method_(ir)
  {
    completeSetup(workspace_hip_);
  }
#endif

  /**
   * @brief Destructor
   *
   * All owned components are held in `std::unique_ptr` and are released
   * automatically in reverse declaration order.
   */
  SystemSolver::~SystemSolver() = default;

  int SystemSolver::setMatrix(matrix_type* A)
  {
    int status = 0;
    A_         = A;
    res_vector_.reset(new vector_type(A->getNumRows()));
    if (memspace_ == "cpu")
    {
      res_vector_->allocate(memory::HOST);
    }
    else
    {
      res_vector_->allocate(memory::DEVICE);
      matrix_handler_->setValuesChanged(true, memory::DEVICE);
    }

    // If we use iterative solver, we can set it up here
    if (solve_method_ == "randgmres")
    {
      auto* rgmres = dynamic_cast<LinSolverIterativeRandFGMRES*>(iterative_solver_.get());
      status += rgmres->setup(A_);
    }
    else if (solve_method_ == "fgmres")
    {
      auto* fgmres = dynamic_cast<LinSolverIterativeFGMRES*>(iterative_solver_.get());
      status += fgmres->setup(A_);
    }
    else
    {
      // do nothing
    }

    return status;
  }

  /**
   * @brief Sets up the system solver
   *
   * This method validates the selected method combination (see
   * `validateConfiguration()`) and then instantiates components of the
   * system solver based on user inputs. It can be called again after the
   * configuration has been changed through setters.
   */
  int SystemSolver::initialize()
  {
    // Make sure the method combination is consistent before creating objects
    validateConfiguration();

    // First delete old objects
    iterative_solver_.reset();
    preconditioner_.reset();
    preconditioner_solver_.reset();
    refactorization_solver_.reset();
    factorization_solver_.reset();
    gs_.reset();

    // Create factorization solver
    if (createFactorizationSolver() != 0)
    {
      return 1;
    }

    // Create refactorization solver
    if (createRefactorizationSolver() != 0)
    {
      return 1;
    }

    // Create iterative refinement
    if (ir_method_ != "none")
    {
      if (createIterativeSolver(ir_method_) != 0)
      {
        out::error() << "Iterative refinement method " << ir_method_ << " not recognized.\n";
        return 1;
      }
    }

    // Create preconditioner
    if (precondition_method_ == "none")
    {
      // do nothing
    }
    else if (precondition_method_ == "ilu0")
    {
      if (memspace_ == "cpu")
      {
        preconditioner_solver_.reset(new LinSolverDirectCpuILU0(workspace_cpu_));
        preconditioner_.reset(new PreconditionerLU(preconditioner_solver_.get()));
#ifdef RESOLVE_USE_CUDA
      }
      else if (memspace_ == "cuda")
      {
        preconditioner_solver_.reset(new LinSolverDirectCuSparseILU0(workspace_cuda_));
        preconditioner_.reset(new PreconditionerLU(preconditioner_solver_.get()));
#endif
#ifdef RESOLVE_USE_HIP
      }
      else if (memspace_ == "hip")
      {
        preconditioner_solver_.reset(new LinSolverDirectRocSparseILU0(workspace_hip_));
        preconditioner_.reset(new PreconditionerLU(preconditioner_solver_.get()));
#endif
      }
      else
      {
        out::error() << "Memory space " << memspace_
                     << " not recognized ...\n";
        return 1;
      }
    }
    else
    {
      out::error() << "Preconditioner method " << precondition_method_
                   << " not recognized ...\n";
      return 1;
    }

    // Create iterative solver
    if (solve_method_ == "randgmres" || solve_method_ == "fgmres")
    {
      createIterativeSolver(solve_method_);
    }

    return 0;
  }

  int SystemSolver::analyze()
  {
    if (A_ == nullptr)
    {
      out::error() << "System matrix not set!\n";
      return 1;
    }

    if (factorization_method_ == "klu")
    {
      factorization_solver_->setup(A_);
      return factorization_solver_->analyze();
    }
    return 1;
  }

  int SystemSolver::factorize()
  {
    if (factorization_method_ == "klu")
    {
      is_solve_on_device_ = false;
      return factorization_solver_->factorize();
    }
    return 1;
  }

  int SystemSolver::refactorize()
  {
    if (refactorization_method_ == "klu")
    {
      return factorization_solver_->refactorize();
    }

    if (refactorization_method_ == "glu" || refactorization_method_ == "cusolverrf" || refactorization_method_ == "rocsolverrf"
#ifdef RESOLVE_USE_CUDSS
        || refactorization_method_ == "cudssrf"
#endif
    )
    {
      is_solve_on_device_ = true;
      return refactorization_solver_->refactorize();
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

    L_ = factorization_solver_->getLFactor();
    U_ = factorization_solver_->getUFactor();
    P_ = factorization_solver_->getPOrdering();
    Q_ = factorization_solver_->getQOrdering();

    if (L_ == nullptr)
    {
      out::error() << "Factorization failed, cannot extract factors ...\n";
      status += 1;
    }

#ifdef RESOLVE_USE_CUDA
    if (refactorization_method_ == "glu")
    {
      is_solve_on_device_ = true;
      status += refactorization_solver_->setup(A_, L_, U_, P_, Q_);
    }
    else if (refactorization_method_ == "cusolverrf")
    {
      status += refactorization_solver_->setup(A_, L_, U_, P_, Q_);

      LinSolverDirectCuSolverRf* Rf = dynamic_cast<LinSolverDirectCuSolverRf*>(refactorization_solver_.get());
      Rf->setNumericalProperties(1e-14, 1e-1);

      is_solve_on_device_ = false;
    }
#ifdef RESOLVE_USE_CUDSS
    else if (refactorization_method_ == "cudssrf")
    {
      LinSolverDirectCuDssRf* Rf = dynamic_cast<LinSolverDirectCuDssRf*>(refactorization_solver_.get());
      Rf->setNumericalProperties(1e-14, 1e-1);

      status += refactorization_solver_->setup(A_, L_, U_, P_, Q_);

      is_solve_on_device_ = false;
    }
#endif
#endif

#ifdef RESOLVE_USE_HIP
    if (refactorization_method_ == "rocsolverrf")
    {
      is_solve_on_device_ = false;
      status += refactorization_solver_->setup(A_, L_, U_, P_, Q_, res_vector_.get());
    }
#endif

    if (ir_method_ == "fgmres")
    {
      status += iterative_solver_->setup(A_);

      // The refinement preconditioner applies the LU factors. With "klu"
      // refactorization there is no separate refactorization solver and the
      // factorization solver provides the factors.
      LinSolverDirect* lu_solver = refactorization_solver_ ? refactorization_solver_.get()
                                                           : factorization_solver_.get();
      preconditioner_.reset(new PreconditionerLU(lu_solver));
      status += iterative_solver_->setPreconditioner(preconditioner_.get());
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
    if (solve_method_ == "randgmres" || solve_method_ == "fgmres")
    {
      status += iterative_solver_->resetMatrix(A_);
      status += iterative_solver_->solve(rhs, x);
      return status;
    }

    if (solve_method_ == "klu")
    {
      status += factorization_solver_->solve(rhs, x);
    }

    if (solve_method_ == "glu" || solve_method_ == "cusolverrf" || solve_method_ == "rocsolverrf"
#ifdef RESOLVE_USE_CUDSS
        || solve_method_ == "cudssrf"
#endif
    )
    {
      if (is_solve_on_device_)
      {
        status += refactorization_solver_->solve(rhs, x);
      }
      else
      {
        status += factorization_solver_->solve(rhs, x);
      }
    }

    // Iterative refinement is applied once the LU solver wrapped by its
    // preconditioner is ready: after refactorizationSetup() on the host KLU
    // path, or once the device refactorization solver has taken over.
    if (ir_method_ == "fgmres" && preconditioner_)
    {
      if (is_solve_on_device_ || refactorization_method_ == "klu")
      {
        status += refine(rhs, x);
      }
    }
    return status;
  }

  /**
   * @brief Sets up the preconditioner for the system solver
   *
   * Initializes and attaches the preconditioner to the iterative solver.
   *
   * @return int 0 if successful, 1 if it fails
   */
  int SystemSolver::preconditionerSetup(std::string side)
  {
    int status = 0;

    if (preconditioner_ == nullptr)
    {
      out::error() << "Preconditioner not initialized!\n";
      status += 1;
    }

    if (iterative_solver_ == nullptr)
    {
      out::error() << "Iterative solver not initialized!\n";
      status += 1;
    }

    if (status != 0)
    {
      return status;
    }

    Preconditioner::Side prec_side;
    if (side == "left")
    {
      prec_side = Preconditioner::LEFT;
    }
    else if (side == "right")
    {
      prec_side = Preconditioner::RIGHT;
    }
    else
    {
      out::error() << "Preconditioning side '" << side
                   << "' not recognized. Use 'left' or 'right'.\n";
      return 1;
    }

    status += preconditioner_->setSide(prec_side);
    status += preconditioner_->setup(A_);

    if (memspace_ != "cpu")
    {
      is_solve_on_device_ = true;
    }
    status += iterative_solver_->setPreconditioner(preconditioner_.get());

    return status;
  }

  /**
   * @brief Reset the preconditioner with a new matrix.
   *
   * Assumes the matrix sparsity pattern does not change.
   *
   * @param[in] A New sparse matrix (values updated).
   *
   * @return int 0 if successful, 1 if it fails
   */
  int SystemSolver::resetPreconditioner(matrix_type* A)
  {
    int status = 0;
    A_         = A;

    if (preconditioner_ == nullptr)
    {
      out::error() << "Preconditioner not initialized!\n";
      return 1;
    }

    status += preconditioner_->reset(A);

    return status;
  }

  int SystemSolver::refine(vector_type* rhs, vector_type* x)
  {
    int status = 0;

    status += iterative_solver_->resetMatrix(A_);
    status += iterative_solver_->solve(rhs, x);

    return status;
  }

  LinSolverDirect& SystemSolver::getFactorizationSolver()
  {
    return *factorization_solver_;
  }

  LinSolverDirect& SystemSolver::getRefactorizationSolver()
  {
    return *refactorization_solver_;
  }

  LinSolverDirect& SystemSolver::getPreconditionerSolver()
  {
    return *preconditioner_solver_;
  }

  LinSolverIterative& SystemSolver::getIterativeSolver()
  {
    return *iterative_solver_;
  }

  Preconditioner& SystemSolver::getPreconditioner()
  {
    return *preconditioner_;
  }

  /**
   * @brief Sets factorization method to use
   *
   * @param[in] method - ID for the factorization method
   *
   * @post Destroys the existing factorization solver together with the
   * factors and permutation vectors it owned, and creates a new solver
   * of the requested type. If refactorization method is "klu", the
   * refactorization path reuses the factorization solver, so
   * `refactorization_solver_` is reset as well. The caller must run
   * `analyze()`, `factorize()` and, if applicable, `refactorizationSetup()`
   * again before the next solve.
   *
   * @return int 0 if successful, 1 if method is not recognized or an
   * iterative solve method is active
   */
  int SystemSolver::setFactorizationMethod(std::string method)
  {
    if (solve_method_ == "fgmres" || solve_method_ == "randgmres")
    {
      out::error() << "Factorization method cannot be set while iterative solve method '"
                   << solve_method_ << "' is active. Keeping '" << factorization_method_ << "'.\n";
      return 1;
    }

    factorization_method_ = method;
    factorization_solver_.reset();

    // Factors and permutations were owned by the old solver.
    L_                  = nullptr;
    U_                  = nullptr;
    P_                  = nullptr;
    Q_                  = nullptr;
    is_solve_on_device_ = false;

    // KLU refactorization reuses the factorization solver; drop any stale state.
    if (refactorization_method_ == "klu")
    {
      refactorization_solver_.reset();
    }

    return createFactorizationSolver();
  }

  /**
   * @brief Sets refactorization method to use
   *
   * @param[in] method - ID for the refactorization method
   *
   * @post Destroys whatever refactorization solver existed before
   * and sets `refactorization_solver_` pointer to the new
   * refactorization object. Sets refactorization method ID
   * to the value in input parameter `method`. Resets
   * `is_solve_on_device_` since the new solver has not been set up yet.
   * If iterative refinement is enabled, its preconditioner wraps the
   * refactorization solver and is destroyed as well; `refactorizationSetup()`
   * recreates it.
   *
   * @return int 0 if successful, 1 if method is not recognized
   */
  int SystemSolver::setRefactorizationMethod(std::string method)
  {
    refactorization_method_ = method;

    // Iterative refinement preconditioner holds a non-owning pointer to the
    // refactorization solver, so it must not outlive it.
    if (ir_method_ != "none")
    {
      preconditioner_.reset();
    }
    refactorization_solver_.reset();
    is_solve_on_device_ = false;

    return createRefactorizationSolver();
  }

  /**
   * @brief Sets solve method
   *
   * @param[in] method - ID of the solve method
   *
   */
  int SystemSolver::setSolveMethod(std::string method)
  {
    solve_method_ = method;

    // Remove existing iterative solver and set IR to "none".
    ir_method_ = "none";
    iterative_solver_.reset();

    if (createIterativeSolver(method) != 0)
    {
      out::error() << "Solve method " << solve_method_
                   << " not recognized ...\n";
      return 1;
    }
    return 0;
  }

  /**
   * @brief Sets iterative refinement method and related orthogonalization.
   *
   * @param[in] method   - string ID for the iterative refinement method
   * @param[in] gs_method - string ID for the orthogonalization method to be used
   */
  void SystemSolver::setRefinementMethod(std::string method, std::string gs_method)
  {
    // With an iterative solve method the Krylov solver belongs to the solve
    // path, not to iterative refinement, so leave it untouched.
    if (solve_method_ == "randgmres" || solve_method_ == "fgmres")
    {
      if (method != "none")
      {
        out::warning() << "Iterative refinement cannot be enabled together with an "
                       << "iterative solve method ('randgmres' or 'fgmres'). "
                       << "Keeping refinement method 'none'.\n";
      }
      ir_method_ = "none";
      return;
    }

    // Direct solve path: the Krylov solver, if any, is the refinement solver.
    iterative_solver_.reset();
    gs_.reset();
    ir_method_ = "none";

    if (method == "none")
    {
      return;
    }

    gs_method_ = gs_method;

    if (method == "fgmres" && createIterativeSolver(method) == 0)
    {
      ir_method_ = method;
    }
    else
    {
      out::error() << "Iterative refinement method " << method << " not recognized.\n";
    }
  }

  real_type SystemSolver::getVectorNorm(vector_type* rhs)
  {
    using namespace ReSolve::constants;
    real_type norm_b = 0.0;
    if (memspace_ == "cpu")
    {
      norm_b = std::sqrt(vector_handler_->dot(rhs, rhs, memory::HOST));
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    }
    else if (memspace_ == "cuda" || memspace_ == "hip")
    {
      if (is_solve_on_device_)
      {
        norm_b = std::sqrt(vector_handler_->dot(rhs, rhs, memory::DEVICE));
      }
      else
      {
        norm_b = std::sqrt(vector_handler_->dot(rhs, rhs, memory::HOST));
      }
#endif
    }
    else
    {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    return norm_b;
  }

  real_type SystemSolver::getResidualNorm(vector_type* rhs, vector_type* x)
  {
    using namespace ReSolve::constants;
    assert(rhs->getSize() == res_vector_->getSize());
    real_type           norm_b  = 0.0;
    real_type           resnorm = 0.0;
    memory::MemorySpace ms      = memory::HOST;
    if (memspace_ == "cpu")
    {
      res_vector_->copyFromExternal(rhs, memory::HOST, memory::HOST);
      norm_b = std::sqrt(vector_handler_->dot(res_vector_.get(), res_vector_.get(), memory::HOST));
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    }
    else if (memspace_ == "cuda" || memspace_ == "hip")
    {
      if (is_solve_on_device_)
      {
        res_vector_->copyFromExternal(rhs, memory::DEVICE, memory::DEVICE);
        norm_b = std::sqrt(vector_handler_->dot(res_vector_.get(), res_vector_.get(), memory::DEVICE));
      }
      else
      {
        res_vector_->copyFromExternal(rhs, memory::HOST, memory::DEVICE);
        res_vector_->syncData(memory::HOST);
        norm_b = std::sqrt(vector_handler_->dot(res_vector_.get(), res_vector_.get(), memory::HOST));
        // ms = memory::HOST;
      }
      ms = memory::DEVICE;
#endif
    }
    else
    {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    matrix_handler_->setValuesChanged(true, ms);
    matrix_handler_->matvec(A_, x, res_vector_.get(), &ONE, &MINUS_ONE, ms);
    resnorm = std::sqrt(vector_handler_->dot(res_vector_.get(), res_vector_.get(), ms));
    return resnorm / norm_b;
  }

  real_type SystemSolver::getNormOfScaledResiduals(vector_type* rhs, vector_type* x)
  {
    using namespace ReSolve::constants;
    assert(rhs->getSize() == res_vector_->getSize());
    real_type           norm_x  = 0.0;
    real_type           norm_A  = 0.0;
    real_type           resnorm = 0.0;
    memory::MemorySpace ms      = memory::HOST;
    if (memspace_ == "cpu")
    {
      res_vector_->copyFromExternal(rhs, memory::HOST, memory::HOST);
#if defined(RESOLVE_USE_HIP) || defined(RESOLVE_USE_CUDA)
    }
    else if (memspace_ == "cuda" || memspace_ == "hip")
    {
      if (is_solve_on_device_)
      {
        res_vector_->copyFromExternal(rhs, memory::DEVICE, memory::DEVICE);
      }
      else
      {
        res_vector_->copyFromExternal(rhs, memory::HOST, memory::DEVICE);
      }
      ms = memory::DEVICE;
#endif
    }
    else
    {
      out::error() << "Unrecognized device " << memspace_ << "\n";
      return -1.0;
    }
    matrix_handler_->setValuesChanged(true, ms);
    matrix_handler_->matvec(A_, x, res_vector_.get(), &ONE, &MINUS_ONE, ms);
    resnorm = vector_handler_->amax(res_vector_.get(), ms);
    norm_x  = vector_handler_->amax(x, ms);
    matrix_handler_->matrixInfNorm(A_, &norm_A, ms);
    return resnorm / (norm_x * norm_A);
  }

  const std::string& SystemSolver::getFactorizationMethod() const
  {
    return factorization_method_;
  }

  const std::string& SystemSolver::getRefactorizationMethod() const
  {
    return refactorization_method_;
  }

  const std::string& SystemSolver::getSolveMethod() const
  {
    return solve_method_;
  }

  const std::string& SystemSolver::getRefinementMethod() const
  {
    return ir_method_;
  }

  const std::string& SystemSolver::getGramSchmidtMethod() const
  {
    return gs_method_;
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
    if (solve_method_ != "randgmres")
    {
      out::warning() << "Trying to set sketching method to an incompatible solver.\n";
      out::warning() << "The setting will be ignored.\n";
      return 1;
    }

    LinSolverIterativeRandFGMRES::SketchingMethod tmp = sketchingMethodFromString(sketching_method);

    sketching_method_ = sketching_method;

    // At this point iterative solver, if created, can only be LinSolverIterativeRandFGMRES
    if (iterative_solver_)
    {
      // TODO: Use cast here as a temporary solution; will be replaced by parameter setting framework
      auto* sol = dynamic_cast<LinSolverIterativeRandFGMRES*>(iterative_solver_.get());
      sol->setSketchingMethod(tmp);
    }

    return 0;
  }

  /**
   * @brief Sets Gram-Schmidt orthogonalization variant.
   *
   * Records the variant in `gs_method_` so that it survives re-creation of
   * the Krylov solver, and applies it to the existing `GramSchmidt` object
   * or creates one if none exists yet. An unrecognized string ID falls back
   * to CGS2 with a warning, and `gs_method_` is set to "cgs2" accordingly.
   *
   * @param[in] variant - string ID of the Gram-Schmidt variant
   *
   * @return int 0 on success
   */
  int SystemSolver::setGramSchmidtMethod(std::string variant)
  {
    GramSchmidt::GSVariant gs_variant = gsVariantFromString(variant);

    // Store the canonical name so that the stored ID always matches the
    // variant actually in use.
    gs_method_ = gsVariantName(gs_variant);

    if (gs_)
    {
      gs_->setVariant(gs_variant);
    }
    else
    {
      gs_.reset(new GramSchmidt(vector_handler_.get(), gs_variant));
    }

    return 0;
  }

  //
  // Private methods
  //

  /**
   * @brief Enforces supported combinations of user-selected methods.
   *
   * Two configurations are supported:
   * - Iterative solver ("fgmres" or "randgmres" as the solve method):
   *   preconditioner may be set; factorization, refactorization and
   *   iterative refinement must be "none".
   * - Direct solver (any other solve method): factorization and
   *   refactorization are set, iterative refinement is optional, and the
   *   preconditioner must be "none".
   *
   * Offending settings are reset to "none" with a warning.
   */
  void SystemSolver::validateConfiguration()
  {
    const bool is_iterative_solve = (solve_method_ == "fgmres" || solve_method_ == "randgmres");

    if (is_iterative_solve)
    {
      if (factorization_method_ != "none")
      {
        out::warning() << "Incorrect input: factorization method '" << factorization_method_
                       << "' cannot be used with iterative solve method '" << solve_method_
                       << "'. Setting factorization to 'none' ...\n";
        factorization_method_ = "none";
      }
      if (refactorization_method_ != "none")
      {
        out::warning() << "Incorrect input: refactorization method '" << refactorization_method_
                       << "' cannot be used with iterative solve method '" << solve_method_
                       << "'. Setting refactorization to 'none' ...\n";
        refactorization_method_ = "none";
      }
      if (ir_method_ != "none")
      {
        out::warning() << "Incorrect input: iterative refinement cannot be enabled "
                       << "together with iterative solve method '" << solve_method_
                       << "'. Setting refinement method to 'none' ...\n";
        ir_method_ = "none";
      }
    }
    else
    {
      if (precondition_method_ != "none")
      {
        out::warning() << "Incorrect input: preconditioner '" << precondition_method_
                       << "' can only be used with an iterative solve method ('fgmres' or 'randgmres'). "
                       << "Setting preconditioner to 'none' ...\n";
        precondition_method_ = "none";
      }
    }
  }

  /**
   * @brief Instantiates Krylov solver of the requested type.
   *
   * Used both for iterative solve methods and for iterative refinement.
   * Configures Gram-Schmidt orthogonalization according to `gs_method_`
   * and, for randomized GMRES, sketching according to `sketching_method_`.
   *
   * @param[in] method - "fgmres" or "randgmres"
   * @post `iterative_solver_` points to a new solver.
   *
   * @return int 0 if successful, 1 if method is not recognized
   */
  int SystemSolver::createIterativeSolver(const std::string& method)
  {
    if (method == "randgmres")
    {
      setGramSchmidtMethod(gs_method_);
      iterative_solver_.reset(new LinSolverIterativeRandFGMRES(matrix_handler_.get(),
                                                               vector_handler_.get(),
                                                               sketchingMethodFromString(sketching_method_),
                                                               gs_.get()));
    }
    else if (method == "fgmres")
    {
      setGramSchmidtMethod(gs_method_);
      iterative_solver_.reset(new LinSolverIterativeFGMRES(matrix_handler_.get(),
                                                           vector_handler_.get(),
                                                           gs_.get()));
    }
    else
    {
      return 1;
    }
    return 0;
  }

  /**
   * @brief Instantiates factorization solver selected by `factorization_method_`.
   *
   * Shared by `initialize()` and `setFactorizationMethod()` so the list of
   * supported backends is maintained in one place.
   *
   * @pre `factorization_solver_` is null.
   * @post `factorization_solver_` points to a new solver, or stays null for
   * method "none".
   *
   * @return int 0 if successful, 1 if method is not recognized
   */
  int SystemSolver::createFactorizationSolver()
  {
    if (factorization_method_ == "none")
    {
      // do nothing
#ifdef RESOLVE_USE_KLU
    }
    else if (factorization_method_ == "klu")
    {
      factorization_solver_.reset(new ReSolve::LinSolverDirectKLU());
#endif
    }
    else
    {
      out::error() << "Factorization method " << factorization_method_
                   << " not recognized ...\n";
      return 1;
    }
    return 0;
  }

  /**
   * @brief Instantiates refactorization solver selected by `refactorization_method_`.
   *
   * Shared by `initialize()` and `setRefactorizationMethod()` so the list of
   * supported backends is maintained in one place.
   *
   * @pre `refactorization_solver_` is null.
   * @post `refactorization_solver_` points to a new solver, or stays null for
   * methods "none" and "klu" (KLU refactorization reuses the factorization
   * solver).
   *
   * @return int 0 if successful, 1 if method is not recognized
   */
  int SystemSolver::createRefactorizationSolver()
  {
    if (refactorization_method_ == "none")
    {
      // do nothing
    }
    else if (refactorization_method_ == "klu")
    {
      // do nothing for now, KLU is the only factorization solver available
#ifdef RESOLVE_USE_CUDA
    }
    else if (refactorization_method_ == "glu")
    {
      refactorization_solver_.reset(new ReSolve::LinSolverDirectCuSolverGLU(workspace_cuda_));
    }
    else if (refactorization_method_ == "cusolverrf")
    {
      refactorization_solver_.reset(new ReSolve::LinSolverDirectCuSolverRf());
#ifdef RESOLVE_USE_CUDSS
    }
    else if (refactorization_method_ == "cudssrf")
    {
      refactorization_solver_.reset(new ReSolve::LinSolverDirectCuDssRf());
#endif
#endif
#ifdef RESOLVE_USE_HIP
    }
    else if (refactorization_method_ == "rocsolverrf")
    {
      refactorization_solver_.reset(new ReSolve::LinSolverDirectRocSolverRf(workspace_hip_));
#endif
    }
    else
    {
      out::error() << "Refactorization method " << refactorization_method_
                   << " not recognized ...\n";
      return 1;
    }
    return 0;
  }

} // namespace ReSolve
