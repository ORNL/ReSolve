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

    /// @brief Temporary until abstract preconditioner class is created
    using precond_type = LinSolverDirect;

    SystemSolver(LinAlgWorkspaceCpu* workspaceCpu,
                 std::string         factor   = "klu",
                 std::string         refactor = "klu",
                 std::string         solve    = "klu",
                 std::string         precond  = "none",
                 std::string         ir       = "none");
    SystemSolver(LinAlgWorkspaceCUDA* workspaceCuda,
                 std::string          factor   = "klu",
                 std::string          refactor = "cusolverrf",
                 std::string          solve    = "cusolverrf",
                 std::string          precond  = "none",
                 std::string          ir       = "none");
    SystemSolver(LinAlgWorkspaceHIP* workspaceHip,
                 std::string         factor   = "klu",
                 std::string         refactor = "rocsolverrf",
                 std::string         solve    = "rocsolverrf",
                 std::string         precond  = "none",
                 std::string         ir       = "none");

    ~SystemSolver();

    int initialize();
    int setMatrix(matrix::Sparse* A);
    int analyze();   //    symbolic part
    int factorize(); //  numeric part
    int refactorize();
    int refactorizationSetup();
    int preconditionerSetup();
    int solve(vector_type* rhs, vector_type* x);  // for direct and iterative
    int refine(vector_type* rhs, vector_type* x); // for iterative refinement

    // we update the matrix once it changed
    int updateMatrix(std::string format, int* ia, int* ja, double* a);

    LinSolverDirect&    getFactorizationSolver();
    LinSolverDirect&    getRefactorizationSolver();
    LinSolverIterative& getIterativeSolver();

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
    void setRefactorizationMethod(std::string method);
    int  setSolveMethod(std::string method);
    void setRefinementMethod(std::string method, std::string gs = "cgs2");
    int  setSketchingMethod(std::string method);
    int  setGramSchmidtMethod(std::string gs_method);

  private:
    LinSolverDirect*    factorizationSolver_{nullptr};
    LinSolverDirect*    refactorizationSolver_{nullptr};
    LinSolverIterative* iterativeSolver_{nullptr};
    GramSchmidt*        gs_{nullptr};

    precond_type* preconditioner_{nullptr};

    LinAlgWorkspaceCUDA* workspaceCuda_{nullptr};
    LinAlgWorkspaceHIP*  workspaceHip_{nullptr};
    LinAlgWorkspaceCpu*  workspaceCpu_{nullptr};

    MatrixHandler* matrixHandler_{nullptr};
    VectorHandler* vectorHandler_{nullptr};

    bool is_solve_on_device_{false};

    matrix_type* L_{nullptr};
    matrix_type* U_{nullptr};

    index_type* P_{nullptr};
    index_type* Q_{nullptr};

    vector_type* resVector_{nullptr};

    matrix::Sparse* A_{nullptr};

    // Configuration parameters
    std::string factorizationMethod_{"none"};
    std::string refactorizationMethod_{"none"};
    std::string solveMethod_{"none"};
    std::string precondition_method_{"none"};
    std::string irMethod_{"none"};
    std::string gsMethod_{"cgs2"};
    std::string sketching_method_{"count"}; ///< @todo move this to LinSolverIterative class

    std::string memspace_;
  };
} // namespace ReSolve
