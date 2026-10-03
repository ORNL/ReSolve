#pragma once

#include <memory>
#include <string>

#include <resolve/Common.hpp>
#include <resolve/Preconditioner.hpp>

// this is to solve the system, can call different linear solvers if necessary
namespace ReSolve
{
  class LinSolverDirectKLU;
  class LinSolverDirect;
  class LinSolverIterative;
  class GramSchmidt;
  class LinAlgWorkspaceCUDA;
  class LinAlgWorkspaceHIP;
  class LinAlgWorkspaceCpu;
  class MatrixHandler;
  class VectorHandler;

  namespace vector
  {
    class Vector;
  }

  namespace matrix
  {
    class Sparse;
  }

  class SystemSolver
  {
  public:
    using vector_type = vector::Vector;
    using matrix_type = matrix::Sparse;

    SystemSolver(LinAlgWorkspaceCpu* workspace_cpu,
                 std::string         factor   = "klu",
                 std::string         refactor = "klu",
                 std::string         solve    = "klu",
                 std::string         precond  = "none",
                 std::string         ir       = "none");
    SystemSolver(LinAlgWorkspaceCUDA* workspace_cuda,
                 std::string          factor   = "klu",
                 std::string          refactor = "cusolverrf",
                 std::string          solve    = "cusolverrf",
                 std::string          precond  = "none",
                 std::string          ir       = "none");
    SystemSolver(LinAlgWorkspaceHIP* workspace_hip,
                 std::string         factor   = "klu",
                 std::string         refactor = "rocsolverrf",
                 std::string         solve    = "rocsolverrf",
                 std::string         precond  = "none",
                 std::string         ir       = "none");

    ~SystemSolver();

    int initialize();
    int setMatrix(matrix_type* A);
    int analyze();   //    symbolic part
    int factorize(); //  numeric part
    int refactorize();
    int refactorizationSetup();
    int preconditionerSetup(std::string side);
    int resetPreconditioner(matrix_type* A);
    int solve(vector_type* rhs, vector_type* x);  // for direct and iterative
    int refine(vector_type* rhs, vector_type* x); // for iterative refinement

    // we update the matrix once it changed
    int updateMatrix(std::string format, int* ia, int* ja, double* a);

    LinSolverDirect&    getFactorizationSolver();
    LinSolverDirect&    getRefactorizationSolver();
    LinSolverDirect&    getPreconditionerSolver();
    LinSolverIterative& getIterativeSolver();
    Preconditioner&     getPreconditioner();

    real_type getVectorNorm(vector_type* rhs);
    real_type getResidualNorm(vector_type* rhs, vector_type* x);
    real_type getNormOfScaledResiduals(vector_type* rhs, vector_type* x);

    // Get solver parameters
    const std::string getFactorizationMethod() const;
    const std::string getRefactorizationMethod() const;
    const std::string getSolveMethod() const;
    const std::string getRefinementMethod() const;
    const std::string getOrthogonalizationMethod() const;

    // Set solver parameters
    void setFactorizationMethod(std::string method);
    int  setRefactorizationMethod(std::string method);
    int  setSolveMethod(std::string method);
    void setRefinementMethod(std::string method, std::string gs = "cgs2");
    int  setSketchingMethod(std::string method);
    int  setGramSchmidtMethod(std::string gs_method);

  private:
    template <class Workspace>
    void completeSetup(Workspace* workspace);
    void validateConfiguration();
    int  createRefactorizationSolver();
    int  createIterativeSolver(const std::string& method);

    std::unique_ptr<MatrixHandler> matrix_handler_;
    std::unique_ptr<VectorHandler> vector_handler_;

    // Owned solver components. Declaration order matters for destruction:
    // members are destroyed in reverse order, so dependents are listed
    // after the objects they reference.
    std::unique_ptr<LinSolverDirect>    factorization_solver_;
    std::unique_ptr<LinSolverDirect>    refactorization_solver_;
    std::unique_ptr<LinSolverDirect>    preconditioner_solver_;
    std::unique_ptr<GramSchmidt>        gs_;
    std::unique_ptr<Preconditioner>     preconditioner_;
    std::unique_ptr<LinSolverIterative> iterative_solver_;

    LinAlgWorkspaceCUDA* workspace_cuda_{nullptr};
    LinAlgWorkspaceHIP*  workspace_hip_{nullptr};
    LinAlgWorkspaceCpu*  workspace_cpu_{nullptr};

    bool is_solve_on_device_{false};

    matrix_type* L_{nullptr};
    matrix_type* U_{nullptr};

    index_type* P_{nullptr};
    index_type* Q_{nullptr};

    std::unique_ptr<vector_type> res_vector_;

    matrix::Sparse* A_{nullptr};

    // Configuration parameters
    std::string factorization_method_{"none"};
    std::string refactorization_method_{"none"};
    std::string solve_method_{"none"};
    std::string precondition_method_{"none"};
    std::string ir_method_{"none"};
    std::string gs_method_{"cgs2"};
    std::string sketching_method_{"count"}; ///< @todo move this to LinSolverIterative class

    std::string memspace_;
  };
} // namespace ReSolve
